import logging

import numpy as np

from hls4ml.optimization.attributes import OptimizationAttributes
from hls4ml.optimization.config import SUPPORTED_STRUCTURES
from hls4ml.optimization.objectives import ObjectiveEstimator


class VivadoDSPEstimator(ObjectiveEstimator):
    @classmethod
    def is_layer_optimizable(self, layer_attributes):
        if not layer_attributes.weight_shape or layer_attributes.args['hls4ml_attributes'].weight_precision.width < 9:
            return False, None
        else:
            return True, OptimizationAttributes(
                SUPPORTED_STRUCTURES.PATTERN,
                pruning=True,
                weight_sharing=False,
                pattern_offset=np.prod(layer_attributes.weight_shape)
                // layer_attributes.args['hls4ml_attributes'].reuse_factor,
                consecutive_patterns=1,
            )

    @classmethod
    def layer_resources(self, layer_attributes):
        if not layer_attributes.weight_shape or layer_attributes.args['hls4ml_attributes'].weight_precision.width < 9:
            return [0]
        else:
            return [
                np.prod(layer_attributes.weight_shape)
                // layer_attributes.args['hls4ml_attributes'].reuse_factor
                * layer_attributes.args['hls4ml_attributes'].parallelization_factor
            ]

    @classmethod
    def layer_savings(self, layer_attributes):
        if not layer_attributes.weight_shape or layer_attributes.args['hls4ml_attributes'].weight_precision.width < 9:
            return [0]

        structure_type = layer_attributes.optimization_attributes.structure_type

        if structure_type == SUPPORTED_STRUCTURES.UNSTRUCTURED:
            if layer_attributes.args['hls4ml_attributes'].reuse_factor == 1:
                return 1
            else:
                return 0
        elif structure_type == SUPPORTED_STRUCTURES.STRUCTURED:
            if layer_attributes.args['hls4ml_attributes'].reuse_factor == layer_attributes.args['hls4ml_attributes'].n_out:
                layer_attributes.args['hls4ml_attributes'].n_in
            else:
                return [0]
        elif structure_type == SUPPORTED_STRUCTURES.PATTERN:
            pattern_offset = layer_attributes.optimization_attributes.pattern_offset
            number_of_patterns = np.prod(layer_attributes.weight_shape) // pattern_offset

            if number_of_patterns == layer_attributes.args['hls4ml_attributes'].reuse_factor:
                return [layer_attributes.optimization_attributes.consecutive_patterns]
            else:
                return [0]
        elif structure_type == SUPPORTED_STRUCTURES.BLOCK:
            logging.warn('hls4ml does not support block sparsity patterns...setting layer savings to zero')
            return [0]


class VivadoBRAMEstimator(ObjectiveEstimator):
    @classmethod
    def is_layer_optimizable(self, layer_attributes):
        pass

    @classmethod
    def layer_resources(self, layer_attributes):
        pass

    @classmethod
    def layer_savings(self, layer_attributes):
        pass


class VivadoFFEstimator(ObjectiveEstimator):
    @classmethod
    def is_layer_optimizable(self, layer_attributes):
        if not layer_attributes.weight_shape:
            return False, None

        # Resource strategy and I/O type io_stream store both weights and activation tensors in BRAM; skipping
        if (
            layer_attributes.args['hls4ml_attributes'].io_type == 'io_stream'
            and layer_attributes.args['hls4ml_attributes'].strategy.lower() == 'resource'
        ):
            logging.warn('FFs are at minimum utilization with io_stream and Resource strategy')
            return False, None

        # With io_stream in Latency, weight are stored in FFs, so unstructured pruning will benefit the most
        if (
            layer_attributes.args['hls4ml_attributes'].io_type == 'io_stream'
            and layer_attributes.args['hls4ml_attributes'].strategy.lower() == 'latency'
        ):
            return True, OptimizationAttributes(SUPPORTED_STRUCTURES.UNSTRUCTURED, pruning=True, weight_sharing=False)

        # In io_parallel with Resource, weights are stored in BRAM but activation tensors in FFs, so structured pruning is the most suitable, it reduces the size out output before compile-time
        if (
            layer_attributes.args['hls4ml_attributes'].io_type == 'io_parallel'
            and layer_attributes.args['hls4ml_attributes'].strategy.lower() == 'resource'
        ):
            return True, OptimizationAttributes(SUPPORTED_STRUCTURES.STRUCTURED, pruning=True, weight_sharing=False)

        # TODO
        # In io_parallel with Latency, weights and latency are all stored in FFs, so it is equivalent to unstructured, high sparsity pruning
        if (
            layer_attributes.args['hls4ml_attributes'].io_type == 'io_parallel'
            and layer_attributes.args['hls4ml_attributes'].strategy.lower() == 'latency'
        ):
            return True, OptimizationAttributes(SUPPORTED_STRUCTURES.STRUCTURED, pruning=True, weight_sharing=False)

    # TODO - This method is inaccurate (accross all cases); in general, estimating FFs is hard, but as long as it is consistent(ly wrong), it should not matter for the pruning
    @classmethod
    def layer_resources(self, layer_attributes):
        if not layer_attributes.weight_shape:
            return [0]

        # Resource strategy and I/O type io_stream store both weights and activation tensors in BRAM; minimal FF utilization
        if (
            layer_attributes.args['hls4ml_attributes'].io_type == 'io_stream'
            and layer_attributes.args['hls4ml_attributes'].strategy.lower() == 'resource'
        ):
            return [0]

        # With io_stream in Latency, weight are stored in FFs, so FF ~ number_of_weights x weight_precision
        if (
            layer_attributes.args['hls4ml_attributes'].io_type == 'io_stream'
            and layer_attributes.args['hls4ml_attributes'].strategy.lower() == 'latency'
        ):
            return [
                np.prod(layer_attributes.weight_shape) * layer_attributes.args['hls4ml_attributes'].weight_precision.width
            ]

        # In io_parallel with Resource, weights are stored in BRAM but activation tensors in FFs, so FF ~ number_of_outputs x weight_precision
        if (
            layer_attributes.args['hls4ml_attributes'].io_type == 'io_parallel'
            and layer_attributes.args['hls4ml_attributes'].strategy.lower() == 'resource'
        ):
            return [
                np.prod(layer_attributes.output_shape) * layer_attributes.args['hls4ml_attributes'].output_precision.width
            ]

        # In io_parallel with Latency, weights and latency are all stored in FFs, so it is equivalent to the sum of the above two cases
        if (
            layer_attributes.args['hls4ml_attributes'].io_type == 'io_parallel'
            and layer_attributes.args['hls4ml_attributes'].strategy.lower() == 'latency'
        ):
            return [
                np.prod(layer_attributes.weight_shape) * layer_attributes.args['hls4ml_attributes'].weight_precision.width
                + np.prod(layer_attributes.output_shape) * layer_attributes.args['hls4ml_attributes'].output_precision.width
            ]

    @classmethod
    def layer_savings(self, layer_attributes):
        if not layer_attributes.weight_shape:
            return [0]

        structure_type = layer_attributes.optimization_attributes.structure_type
        pruning = layer_attributes.optimization_attributes.pruning
        weight_sharing = layer_attributes.optimization_attributes.weight_sharing

        if weight_sharing:
            logging.warn(
                'Weight sharing does not decrease the number of parameters. \
                         It is recommened to use the default attributes, returned from is_layer_optimizable(...)'
            )
            return [0]

        if not pruning:
            logging.warn(
                'Pruning needs to be enabled to decrease the number of parameters. \
                         It is recommened to use the default attributes, returned from is_layer_optimizable(...)'
            )
            return [0]

        # Resource strategy and I/O type io_stream store both weights and activation tensors in BRAM; minimal FF utilization
        if (
            layer_attributes.args['hls4ml_attributes'].io_type == 'io_stream'
            and layer_attributes.args['hls4ml_attributes'].strategy.lower() == 'resource'
        ):
            return [0]

        # With io_stream in Latency, weight are stored in FFs, so any type of pruning will help:
        if (
            layer_attributes.args['hls4ml_attributes'].io_type == 'io_stream'
            and layer_attributes.args['hls4ml_attributes'].strategy.lower() == 'latency'
        ):
            if structure_type == SUPPORTED_STRUCTURES.UNSTRUCTURED:
                return [layer_attributes.args['hls4ml_attributes'].weight_precision.width]
            elif structure_type == SUPPORTED_STRUCTURES.STRUCTURED:
                return [
                    layer_attributes.args['hls4ml_attributes'].n_in
                    * layer_attributes.args['hls4ml_attributes'].weight_precision.width
                ]
            elif structure_type == SUPPORTED_STRUCTURES.PATTERN:
                number_of_patterns = (
                    np.prod(layer_attributes.weight_shape) // layer_attributes.optimization_attributes.pattern_offset
                )
                return [
                    number_of_patterns
                    * layer_attributes.optimization_attributes.consecutive_patterns
                    * layer_attributes.args['hls4ml_attributes'].weight_precision.width
                ]
            elif structure_type == SUPPORTED_STRUCTURES.BLOCK:
                return [
                    np.prod(layer_attributes.optimization_attributes.block_shape)
                    * layer_attributes.args['hls4ml_attributes'].weight_precision.width
                ]

        # In io_parallel with Resource, weights are stored in BRAM but activation tensors in FFs, so only structured pruning helps
        if (
            layer_attributes.args['hls4ml_attributes'].io_type == 'io_parallel'
            and layer_attributes.args['hls4ml_attributes'].strategy.lower() == 'resource'
        ):
            if structure_type == SUPPORTED_STRUCTURES.STRUCTURED:
                return [layer_attributes.args['hls4ml_attributes'].output_precision.width]
            else:
                return [0]

        # In io_parallel with Latency, weights and latency are all stored in FFs, so any type of pruning helps
        if (
            layer_attributes.args['hls4ml_attributes'].io_type == 'io_parallel'
            and layer_attributes.args['hls4ml_attributes'].strategy.lower() == 'latency'
        ):
            # This is a significant under-estimate, as some savings are incurred due to less intermediate results
            if structure_type == SUPPORTED_STRUCTURES.UNSTRUCTURED:
                return [layer_attributes.args['hls4ml_attributes'].weight_precision.width]
            elif structure_type == SUPPORTED_STRUCTURES.STRUCTURED:
                print('here')
                return [
                    layer_attributes.args['hls4ml_attributes'].n_in
                    * layer_attributes.args['hls4ml_attributes'].weight_precision.width
                    + layer_attributes.args['hls4ml_attributes'].output_precision.width
                ]
            elif structure_type == SUPPORTED_STRUCTURES.PATTERN:
                number_of_patterns = (
                    np.prod(layer_attributes.weight_shape) // layer_attributes.optimization_attributes.pattern_offset
                )
                return [
                    number_of_patterns
                    * layer_attributes.optimization_attributes.consecutive_patterns
                    * layer_attributes.args['hls4ml_attributes'].weight_precision.width
                ]
            elif structure_type == SUPPORTED_STRUCTURES.BLOCK:
                return [
                    np.prod(layer_attributes.optimization_attributes.block_shape)
                    * layer_attributes.args['hls4ml_attributes'].weight_precision.width
                ]
