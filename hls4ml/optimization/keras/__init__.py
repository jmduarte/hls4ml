import os
import sys
import time
import numpy as np
import tensorflow as tf
import hls4ml.optimization.keras.utils as utils
from hls4ml.optimization.config import SUPPORTED_STRUCTURES
from hls4ml.optimization.keras.reduction import reduce_model
from hls4ml.optimization.keras.masking import get_model_masks
from hls4ml.optimization.scheduler import OptimizationScheduler
from hls4ml.optimization.keras.builder import build_optimizable_model, remove_custom_regularizers
from hls4ml.optimization.keras.config import SUPPORTED_LAYERS, SUPPORTED_METRICS, TMP_DIRECTORY

# Enables printing of loss tensors during Eager execution
from tensorflow.python.ops.numpy_ops import np_config
np_config.enable_numpy_behavior()

EPOCHS_REWINDING = 1
CUTOFF_BAD_TRIALS = 3

def optimize_model(
    model, model_attributes, objective, scheduler, X_train, y_train, X_val, y_val, batch_size, epochs,
    optimizer, loss_fn, validation_metric, increasing, rtol, ranking_metric='l2', local=False, verbose=False,
    early_stopping_epochs=None, early_stopping_delta=0, directory=TMP_DIRECTORY, tuner='Bayesian', knapsack_solver='CBC_MIP',
    regularization_range=np.logspace(-6, -2, num=15).tolist(), learning_rate_range=np.logspace(-6, -3, num=10).tolist()
  ):
    '''
    Top-level function for optimizing a Keras model, given objectives 

    Args:
    - model (keras.Model): Model to be optimized
    - model_attributes (dict): Layer-wise model attributes, obtained from hls4ml.optimization.get_attributes_from_keras_model(...)
    - objective (hls4ml.optimization.objectives.ObjectiveEstimator): Parameter, hardware or user-defined objective of optimization
    - scheduler (hls4ml.optimization.schduler.OptimizationScheduler): Sparsity scheduler, choose between constant, polynomial and binary
    - X_train (np.array): Training inputs
    - y_train (np.array): Training labels
    - X_val (np.array): Validation inputs
    - y_val (np.array): Validation labels
    - batch_size (int): Batch size during training
    - epochs (int): Maximum number of epochs to fine-tune model, in one iteration of pruning
    - optimizer (keras.optimizers.Optimizer): Optimizer used during training
    - loss_fn (keras.losses.Loss): Loss function used during training
    - validation_metric (keras.metrics.Metric): Validation metric, used as a baseline
    - increasing (boolean): If the metric improves with increased values; e.g. accuracy -> increasing = True, MSE -> increasing = False
    - rtol (float): Relative tolerance; pruning stops when pruned_validation_metric < (or >) rtol * baseline_validation_metric
    
    Kwargs:
    - ranking_metric (string): Metric used for rannking weights and structures; currently supported l1, l2, saliency and Oracle
    - local (boolean): Layer-wise or global pruning
    - verbose (boolean): Display debug logs during model optimization
    - early_stopping_epochs (int): Number of epochs during the pruning stage after which the model should stop if there is no improvement in validation loss
    - early_stopping_delta (float): Minimum change in validation loss (absolute) to be considered as an improvement
    - directory (string): Directory to store temporary results
    - tuner (str): Tuning alogorithm, choose between Bayesian and Hyperband
    - knapsack_solver (str): Algorithm to solve Knapsack problem when optimizing; default usually works well; for very large networks, greedy algorithm might be more suitable
    - regularization_range (list): List of suitable hyperparameters for weight decay
    - learning_rate_range (list): List of suitable hyperparameters for learning rate

    Notes:
    - Early stopping can be used to reduce the number of epochs taken during the fine-tuning stage of pruning. This is particularly useful when the sparsity is low.
      However, enabling early stopping will evaluate the model on the validation set, after every epoch; so the overall savings in time might be negligible
      Without early stopping, the model is only evaluated on the validation set after trained to convergence
    '''

    if not isinstance(scheduler, OptimizationScheduler):
        raise Exception('Scheduler must be an instance of from hls4ml.optimization.scheduler.OptimizationScheduler' +\
                        'If you provided string description (e.g. \'constant\'), please use an object instance (i.e. ConstantScheduler())'
                        'For a full list of supported schedulers and their description, refer to hls4ml.optimization.scheduler.'
                      )

    if not isinstance(optimizer, (tf.keras.optimizers.Optimizer, tf.keras.optimizers.legacy.Optimizer)):
       raise Exception('Optimizer must be an instance of keras.optimizers.Optimizer.' +\
                       'If you provided string description (e.g. \'adam\'), please use an object instance (i.e. Adam())'
                      )
    
    if not isinstance(loss_fn, tf.keras.losses.Loss):
       raise Exception('Loss function must be an instance of tensorflow.keras.losses.Loss.' +\
                       'If you provided string description (e.g. \'mse\'), please use an object instance (i.e. MeanSquaredError())'
                      )
    
    if not isinstance(validation_metric, tf.keras.metrics.Metric):
       raise Exception('Validation metric must be an instance of tensorflow.keras.losses.Loss.' +\
                       'If you provided string description (e.g. \'accuracy\'), please use an object instance (i.e. CategoricalAccuracy())'
                      )

    if epochs <= EPOCHS_REWINDING:
       raise Exception('Please increase the number of epochs. \
                       The current epoch number is too small to perform effective pruning & weight rewinding'
                      )
  
    if ranking_metric not in SUPPORTED_METRICS:
        raise Exception('Unknown metric for ranking weights')
    
    # Identify optimizable layers, given the current objectives
    last_optimizable_layer = utils.get_last_layer_with_weights(model)
    for i, layer in enumerate(model.layers):
        if isinstance(layer, SUPPORTED_LAYERS): 
          optimizable, optimization_attributes = objective.is_layer_optimizable(model_attributes[layer.name])
          model_attributes[layer.name].optimizable = optimizable
          model_attributes[layer.name].optimization_attributes = optimization_attributes
          
          # In the last layer, structured pruning can't be applied, as it removes output labels
          # Weight sharing, as well as all other types of pruning (unstructured, block etc.) are applicable
          if i >= last_optimizable_layer and optimization_attributes.structure_type == SUPPORTED_STRUCTURES.STRUCTURED and optimization_attributes.pruning:
            model_attributes[layer.name].optimization_attributes.pruning = False
            model_attributes[layer.name].optimizable = model_attributes[layer.name].optimization_attributes.weight_sharing
        
        else: 
          model_attributes[layer.name].optimizable = False
          model_attributes[layer.name].optimization_attributes = None
        
    # Evaluate baseline performance
    baseline_performance = utils.evaluate_model(model, X_val, y_val, validation_metric, verbose)
    if verbose:
      print('Baseline performance on validation set: {}'.format(baseline_performance))
    
    # Save best weights
    # Always save weights to a file, to reduce RAM utilization
    if not os.path.isdir(directory):
      os.mkdir(directory)
    if not os.path.isdir(f'{directory}/optimization'):
      os.mkdir(f'{directory}/optimization')
    model.save_weights(f'{directory}/optimization/best_weights.h5')

    # Add regularization loss to optimizable layers
    optimizable_model = build_optimizable_model(
      model, model_attributes, optimizer, loss_fn, validation_metric, 
      increasing, X_train, y_train, X_val, y_val, batch_size, epochs // 2, 
      verbose=verbose, directory=directory, tuner=tuner,
      regularization_range=regularization_range, 
      learning_rate_range=learning_rate_range
    )

    # Split data set into batches
    batch_X = np.split(X_train, range(batch_size, X_train.shape[0], batch_size))
    batch_y = np.split(y_train, range(batch_size, y_train.shape[0], batch_size))
    assert(len(batch_X) == len(batch_y))

    # In certain cases, the model might underperform at the current sparsity level, but perform better at a higher sparsity
    # Therefore, monitor the models performance over several sparsity levels and only stop pruning after high loss over several trials
    bad_trials = 0  
    sparsity_conditions = True
    target_sparsity = scheduler.get_sparsity()

    while sparsity_conditions:
      gradients = utils.get_model_gradients(optimizable_model, loss_fn, X_train, y_train) if ranking_metric == 'gradients' else {}
      hessians = utils.get_model_hessians(optimizable_model, loss_fn, X_train, y_train) if ranking_metric == 'saliency' else {}
      masks, offsets = get_model_masks(optimizable_model, model_attributes, target_sparsity, objective, metric=ranking_metric, local=local, gradients=gradients, hessians=hessians, knapsack_solver=knapsack_solver)

      # Mask weights
      for layer in optimizable_model.layers:
        if isinstance(layer, SUPPORTED_LAYERS) and model_attributes[layer.name].optimizable:
          layer_weights = layer.get_weights()
          layer_weights[0] = np.multiply(layer_weights[0], masks[layer.name]) + offsets[layer.name]
          layer.set_weights(layer_weights)

      # Before training the model at the next sparsity level, reset internal states
      # Furthemore, modern optimizers (e.g. Adam) accumulate gradients during backprop
      # Therefore, even if the gradient for a weight is zero, it might be updated, due to previous gradients
      # Avoid this by resetting the internal variables of an optimizer
      optimizable_model.reset_states()
      for x in optimizable_model.optimizer.variables():
        x.assign(tf.zeros_like(x))
      
      # Train model with weight freezing [pruning]
      if verbose:
        print(f'Pruning with a target sparsity of {target_sparsity * 100.0}% [relative to objective]')
      
      val_loss_best = sys.float_info.max
      early_stopping_counter = 0
      for epoch in range(epochs - EPOCHS_REWINDING):
        start_time = time.time()
        epoch_loss_avg = tf.keras.metrics.Mean()
        
        # Training
        for i in range(len(batch_X)):
          loss_value = utils.masked_backprop(optimizable_model, batch_X[i], batch_y[i], loss_fn, model_attributes, masks)
          epoch_loss_avg.update_state(loss_value)

        # Early stopping
        if early_stopping_epochs is not None:
          val_loss_current = tf.keras.backend.get_value(loss_fn(y_val, optimizable_model(X_val, training=False)))
          if np.abs(val_loss_best - val_loss_current) >= early_stopping_delta:
            val_loss_best = val_loss_current
          else:
            early_stopping_counter += 1
          if early_stopping_counter >= early_stopping_epochs:
            break
          if verbose:
            print(f'Epoch: {epoch + 1} - Time: {time.time() - start_time}s - Training Loss: {round(epoch_loss_avg.result(), 3)} - Validation Loss: {round(val_loss_current, 3)}')
        else:
          if verbose:
            print(f'Epoch: {epoch + 1} - Time: {time.time() - start_time}s - Training Loss: {round(epoch_loss_avg.result(), 3)}')
      
      # Evaluate pruned model performance
      pruned_performance =  utils.evaluate_model(optimizable_model, X_val, y_val, validation_metric, verbose)            
      if verbose:
        print(f'Performance on validation set: {pruned_performance} with a target sparsity: {target_sparsity} [relative to objective]')
      
      if __compare__(pruned_performance, rtol * baseline_performance, not increasing):
        bad_trials = 0
        sparsity_conditions, target_sparsity = scheduler.update_step()
        optimizable_model.save_weights(f'{directory}/optimization/best_weights.h5')
      else:
        bad_trials += 1
        sparsity_conditions, target_sparsity = scheduler.repair_step()
              
      # If the model performed poorly over several sparsity levels, stop optimization [maximum sparsity reached]
      if bad_trials > CUTOFF_BAD_TRIALS:
        break

      # Train model without weight freezing [rewinding] 
      if verbose:
        print(f'Starting weight rewinding for {EPOCHS_REWINDING} epochs')
      for epoch in range(EPOCHS_REWINDING):
        start_time = time.time()
        epoch_loss_avg = tf.keras.metrics.Mean()

        for i in range(len(batch_X)):
          loss_value = utils.backprop(optimizable_model, batch_X[i], batch_y[i], loss_fn)
          epoch_loss_avg.update_state(loss_value)

        if verbose:
          print(f'Epoch: {epoch + 1} - Time: {time.time() - start_time}s - Training Loss: {round(epoch_loss_avg.result(), 3)}')

    # Load best weights 
    optimizable_model.load_weights(f'{directory}/optimization/best_weights.h5')
    
    # Remove regularizers and save best model
    optimizable_model = remove_custom_regularizers(optimizable_model)
    with open(f'{directory}/optimization/best_model.json', 'w') as json_file:
      json_file.write(optimizable_model.to_json())

    # In GPU FLOP Optimization, remove structures to achieve speed-up & fine-tune the smaller architecture
    if objective.__name__ == 'GPUFLOPEstimator':
      optimizable_model = reduce_model(optimizable_model)
      optimizable_model.compile(optimizer, loss_fn)
      optimizable_model.fit(X_train, y_train, epochs = epochs, batch_size = batch_size, validation_data = (X_val, y_val))

      with open(f'{directory}/optimization/reduced_model.json', 'w') as json_file:
        json_file.write(optimizable_model.to_json())
      optimizable_model.save_weights(f'{directory}/optimization/reduced_model_weights.h5')

    # Evaluate final optimized model [purely for debugging / informative purposes]
    if verbose:
      pruned_performance =  utils.evaluate_model(optimizable_model, X_val, y_val, validation_metric, verbose)      
      print(f'Optimized model performance on validation set: {pruned_performance}')

    return optimizable_model

def __compare__(x, y, leq=False):
   '''
   Helper function for comparing two values, x & y
   Sometimes, we use the >= sign - e.g. pruned_accuracy >= tolerance * baseline_accuracy [ 0 <= tolerance <= 1]
   Other times, use the <= sign - e.g. pruned_mse <= tolerance * baseline_mse [tolerance >= 1]
   '''
   if leq:
      return x <= y
   else:
      return x >= y