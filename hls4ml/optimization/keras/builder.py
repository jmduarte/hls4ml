import re
import numpy as np
import tensorflow as tf
import keras_tuner as kt
from qkeras import QDense, QConv2D
from tensorflow.keras.layers import Dense, Conv2D
from tensorflow.keras.callbacks import EarlyStopping
from hls4ml.optimization.keras.config import SUPPORTED_LAYERS, TMP_DIRECTORY
from hls4ml.optimization.keras.regularizers import DenseRegularizer, Conv2DRegularizer

class HyperOptimizationModel(kt.HyperModel):
    '''
    Helper class for Keras Tuner

    Args:
        - model (keras.Model): Baseline model
        - attributes (dict): Layer-wise dictionary of attributes
        - optimizer (keras.optimizers.Optimizer): Model optimizer
        - loss_fn (keras.losses.Loss): Model loss function
        - validation_metric (keras.metrics.Metric): Model validation metric
        - regularization_range (list): List of suitable hyperparameters for weight decay
        - learning_rate_range (list): List of suitable hyperparameters for learning rate
    '''
    
    def __init__(self, model, attributes, optimizer, loss_fn, validation_metric, regularization_range, learning_rate_range):
        self.model = model
        self.attributes = attributes
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.validation_metric = validation_metric
        self.regularization_range = regularization_range
        self.learning_rate_range = learning_rate_range
    
    def build(self, hp):
        model_to_prune = tf.keras.models.clone_model(self.model)

        default_regularizaton = self.regularization_range[len(self.regularization_range) // 2]
        default_learning_rate = self.learning_rate_range[len(self.learning_rate_range) // 2]

        # Make regularization loss a tunable hyperparameter
        for layer in model_to_prune.layers:
            if isinstance(layer, SUPPORTED_LAYERS) and self.attributes[layer.name].optimizable:        
                structure_type = self.attributes[layer.name].optimization_attributes.structure_type
                block_shape = self.attributes[layer.name].optimization_attributes.block_shape
                pattern_offset = self.attributes[layer.name].optimization_attributes.pattern_offset
                consecutive_patterns = self.attributes[layer.name].optimization_attributes.consecutive_patterns
                
                pruning = self.attributes[layer.name].optimization_attributes.pruning
                weight_sharing = self.attributes[layer.name].optimization_attributes.weight_sharing         
            
                alpha = hp.Choice(f'{layer.name}_alpha', values=self.regularization_range, default=default_regularizaton) if pruning else 0
                beta = hp.Choice(f'{layer.name}_beta', values=self.regularization_range, default=default_regularizaton) if weight_sharing else 0

                if isinstance(layer, (Dense, QDense)) and self.attributes[layer.name].optimizable:
                    layer.kernel_regularizer = DenseRegularizer(alpha, beta, norm=1, structure_type=structure_type, block_shape=block_shape, pattern_offset=pattern_offset, consecutive_patterns=consecutive_patterns)
                elif isinstance(layer, (Conv2D, QConv2D)) and self.attributes[layer.name].optimizable:
                    layer.kernel_regularizer = Conv2DRegularizer(alpha, beta, norm=1, structure_type=structure_type, pattern_offset=pattern_offset, consecutive_patterns=consecutive_patterns)

        # Make learning rate a tunable hyperparameter
        self.optimizer.learning_rate.assign(hp.Choice(f'lr', values=self.learning_rate_range, default=default_learning_rate))
        
        # Rebuild model graph
        model_to_prune = tf.keras.models.model_from_json(model_to_prune.to_json())
        model_to_prune.set_weights(self.model.get_weights())
        model_to_prune.compile(optimizer=self.optimizer, loss=self.loss_fn, metrics=[self.validation_metric]) 

        return model_to_prune

def build_optimizable_model(model, attributes, optimizer, loss_fn, validation_metric, increasing,
                            X_train, y_train, X_val, y_val, batch_size, epochs, verbose=False, directory=TMP_DIRECTORY, tuner='Bayesian',
                            regularization_range=np.logspace(-6, -2, num=15).tolist(), learning_rate_range=np.logspace(-6, -3, num=10).tolist()
                            ):

    '''
    Function identifying optimizable layers and adding a regularization loss

    Args:
    - model (keras.Model): Model to be optimized
    - attributes (dict): Layer-wise model attributes, obtained from hls4ml.optimization.get_attributes_from_keras_model(...)
    - optimizer (keras.optimizers.Optimizer): Optimizer used during training
    - loss_fn (keras.losses.Loss): Loss function used during training
    - validation_metric (keras.metrics.Metric): Validation metric, used as a baseline
    - X_train (np.array): Training inputs
    - y_train (np.array): Training labels
    - X_val (np.array): Validation inputs
    - y_val (np.array): Validation labels
    - batch_size (int): Batch size during training
    - epochs (int): Maximum number of epochs to fine-tune model, in one iteration of pruning

    Kwargs:
    - verbose (bool): Whether to log tuner outputs to the console
    - directory (string): Directory to store tuning results
    - tuner (str): Tuning alogorithm, choose between Bayesian and Hyperband
    - regularization_range (list): List of suitable hyperparameters for weight decay
    - learning_rate_range (list): List of suitable hyperparameters for learning rate

    Notes:
    - In general, the regularization and learning rate ranges do not need to be provided, as the implementation sets a generic enough range.
      However, if the user has an idea on the possible range on hyperparameter ranges (eg. VGG-16 weight decay ~10^-5), the tuning will complete faster
    - The default tuner is Bayesian & when coupled with the correct ranges of hyperparameters, it performs quite well, fast. However, older version of Keras Tuner had a crashing bug with Bayesian Tuner
    - In general, the directory does not need to be specified. However, if pruning several models simultaneously, to avoid conflicting intermediate results, it is useful to specify directory
    '''

    objective_direction = 'max' if increasing else 'min'
    objective_name = re.sub(r'(?<!^)(?=[A-Z])', '_', validation_metric.__class__.__name__).lower()
    if tuner == 'Bayesian':
        tuner = kt.BayesianOptimization(
            hypermodel = HyperOptimizationModel(model, attributes, optimizer, loss_fn, validation_metric, regularization_range, learning_rate_range),
            objective = kt.Objective(objective_name, objective_direction),
            max_trials = 10,
            overwrite = True,
            directory = directory + '/tuning',
        )
    elif tuner == 'Hyperband':
        tuner = kt.Hyperband(
            hypermodel = HyperOptimizationModel(model, attributes, optimizer, loss_fn, validation_metric, regularization_range, learning_rate_range),
            objective = kt.Objective(objective_name, objective_direction),
            max_epochs = epochs,
            factor = 3,
            hyperband_iterations = 1,
            overwrite = True,
            directory = directory + '/tuning',
        )
    else:
        raise Exception('Unknown tuner; possible options are Bayesian and Hyperband')

    if verbose:
        tuner.search_space_summary()
    
    tuner.search(
                X_train, y_train, 
                epochs = epochs, 
                batch_size = batch_size, 
                validation_data = (X_val, y_val),
                callbacks = [ EarlyStopping(monitor='val_loss', patience=1) ]
            )
    
    if verbose:
        tuner.results_summary()

    return tuner.get_best_models(num_models=1)[0]

def remove_custom_regularizers(model):
    '''
    Helper function to remove custom regularizers (DenseRegularizer & Conv2DRegularizer)
    This makes it possible to load the model in a different environment without hls4ml installed

    Args:
        - model (keras.Model): Baseline model
    '''
    weights = model.get_weights()
    for layer in model.layers:
        if hasattr(layer, 'kernel_regularizer'):
            if isinstance(layer.kernel_regularizer, (DenseRegularizer, Conv2DRegularizer)):
                layer.kernel_regularizer = None

    model = tf.keras.models.model_from_json(model.to_json())
    model.set_weights(weights)
    return model
