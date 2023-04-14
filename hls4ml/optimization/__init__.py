import numpy as np
from hls4ml.optimization.keras import optimize_model
from hls4ml.optimization.attributes import get_attributes_from_keras_model_and_hls4ml_config

def optimize_keras_for_hls4ml(
    keras_model, hls_config, objective, scheduler, X_train, y_train, X_val, y_val, batch_size, epochs,
    optimizer, loss_fn, validation_metric, increasing, rtol, ranking_metric='l2', local=False, verbose=False,
    early_stopping_epochs=None, early_stopping_delta=0, directory='hls4ml-optimization', tuner='Bayesian', knapsack_solver='CBC_MIP',
    regularization_range=np.logspace(-6, -2, num=15).tolist(), learning_rate_range=np.logspace(-6, -3, num=10).tolist()
  
):
    '''
    Top-level function for optimizing a Keras model, given hls4ml config and a hardware objective(s)

    Args:
    - keras_model (keras.Model): Model to be optimized
    - hls_config (dict): hls4ml configuration, obtained from hls4ml.utils.config.config_from_keras_model(...)
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
    - rtol (float): Relative tolerance; pruning stops when pruned_validation_metric < rtol * baseline_validation_metric

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
    - Early stopping can be used to reduces the number of epochs taken during the fine-tuning stage of pruning. This is particularly useful when the sparsity is low.
      However, enabling early stopping will evaluate the model on the validation set, after every epoch; so the overall savings in time might be negligible
    '''

    # Extract model attributes
    model_attributes = get_attributes_from_keras_model_and_hls4ml_config(keras_model, hls_config)

    # Optimize model
    return optimize_model(
        keras_model, model_attributes, objective, scheduler,
        X_train, y_train, X_val, y_val, batch_size, epochs, 
        optimizer, loss_fn, validation_metric, increasing, rtol, 
        ranking_metric=ranking_metric, local=local, verbose=verbose, 
        early_stopping_epochs=early_stopping_epochs, early_stopping_delta=early_stopping_delta,
        directory=directory, tuner=tuner, knapsack_solver=knapsack_solver,
        regularization_range=regularization_range, learning_rate_range=learning_rate_range
    )