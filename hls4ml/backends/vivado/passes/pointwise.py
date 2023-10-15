from copy import copy

import numpy as np

from hls4ml.backends.fpga.fpga_layers import PointwiseConv1D, PointwiseConv2D
from hls4ml.backends.vivado.passes.convolution_templates import (
    Conv1DConfigTemplate,
    Conv1DFunctionTemplate,
    Conv2DConfigTemplate,
    Conv2DFunctionTemplate,
    conv_mult_config_template,
)
from hls4ml.model.layers import register_layer
from hls4ml.model.optimizer import OptimizerPass

pointwise_conv1d_config_template = """struct config{index} : nnet::conv1d_config {{
    static const unsigned pad_left = {pad_left};
    static const unsigned pad_right = {pad_right};
    static const unsigned in_width = {in_width};
    static const unsigned n_chan = {n_chan};
    static const unsigned filt_width = {filt_width};
    static const unsigned kernel_size = filt_width;
    static const unsigned n_filt = {n_filt};
    static const unsigned stride_width = {stride_width};
    static const unsigned dilation = {dilation};
    static const unsigned out_width = {out_width};
    static const unsigned reuse_factor = {reuse};
    static const unsigned n_zeros = {nzeros};
    static const bool store_weights_in_bram = false;
    static const unsigned strategy = nnet::{strategy};
    static const nnet::conv_implementation implementation = nnet::conv_implementation::{implementation};
    static const unsigned min_width = {min_width};
    static const ap_uint<filt_width> pixels[min_width];
    static const unsigned n_partitions = {n_partitions};
    static const unsigned n_pixels = out_width / n_partitions;
    template<class data_T, class CONFIG_T>
    using fill_buffer = nnet::{fill_fn}<data_T, CONFIG_T>;
    typedef {accum_t.name} accum_t;
    typedef {bias_t.name} bias_t;
    typedef {weight_t.name} weight_t;
    typedef {config_t} mult_config;
    template<unsigned K, unsigned S, unsigned W>
    using scale_index = nnet::{scale_index_type}<K, S, W>;
    template<class data_T, class res_T, class CONFIG_T>
    using pointwise_conv = nnet::{pointwise_fn}<data_T, res_T, CONFIG_T>;
}};
const ap_uint<config{index}::filt_width> config{index}::pixels[] = {{{instructions}}};\n"""

pointwise_conv2d_config_template = """struct config{index} : nnet::conv2d_config {{
    static const unsigned pad_top = {pad_top};
    static const unsigned pad_bottom = {pad_bottom};
    static const unsigned pad_left = {pad_left};
    static const unsigned pad_right = {pad_right};
    static const unsigned in_height = {in_height};
    static const unsigned in_width = {in_width};
    static const unsigned n_chan = {n_chan};
    static const unsigned filt_height = {filt_height};
    static const unsigned filt_width = {filt_width};
    static const unsigned kernel_size = filt_height * filt_width;
    static const unsigned n_filt = {n_filt};
    static const unsigned stride_height = {stride_height};
    static const unsigned stride_width = {stride_width};
    static const unsigned out_height = {out_height};
    static const unsigned out_width = {out_width};
    static const unsigned reuse_factor = {reuse};
    static const unsigned n_zeros = {nzeros};
    static const unsigned multiplier_limit =
        DIV_ROUNDUP(kernel_size * n_chan * n_filt, reuse_factor) - n_zeros / reuse_factor;
    static const bool store_weights_in_bram = false;
    static const unsigned strategy = nnet::{strategy};
    static const nnet::conv_implementation implementation = nnet::conv_implementation::{implementation};
    static const unsigned min_height = {min_height};
    static const unsigned min_width = {min_width};
    static const ap_uint<filt_height * filt_width> pixels[min_height * min_width];
    static const unsigned n_partitions = {n_partitions};
    static const unsigned n_pixels = out_height * out_width / n_partitions;
    template<class data_T, class CONFIG_T>
    using fill_buffer = nnet::{fill_fn}<data_T, CONFIG_T>;
    typedef {accum_t.name} accum_t;
    typedef {bias_t.name} bias_t;
    typedef {weight_t.name} weight_t;
    typedef {config_t} mult_config;
    template<unsigned K, unsigned S, unsigned W>
    using scale_index_height = nnet::{scale_index_height_type}<K, S, W>;
    template<unsigned K, unsigned S, unsigned W>
    using scale_index_width = nnet::{scale_index_width_type}<K, S, W>;
    template<class data_T, class res_T, class CONFIG_T>
    using pointwise_conv = nnet::{pointwise_fn}<data_T, res_T, CONFIG_T>;
}};
const ap_uint<config{index}::filt_height * config{index}::filt_width> config{index}::pixels[] = {{{instructions}}};\n"""

pointwise_conv1d_function_template = (
    'nnet::pointwise_conv_1d_{data_format}<{input_t}, {output_t}, {config}>({input}, {output}, {w}, {b});'
)
pointwise_conv2d_function_template = (
    'nnet::pointwise_conv_2d_{data_format}<{input_t}, {output_t}, {config}>({input}, {output}, {w}, {b});'
)

sepconv1d_include_list = ['nnet_utils/nnet_conv1d.h', 'nnet_utils/nnet_sepconv1d_stream.h']
sepconv2d_include_list = ['nnet_utils/nnet_conv2d.h', 'nnet_utils/nnet_sepconv2d_stream.h']


class PointwiseConv1DConfigTemplate(Conv1DConfigTemplate):
    def __init__(self):
        super(Conv1DConfigTemplate, self).__init__(PointwiseConv1D)
        self.template = pointwise_conv1d_config_template
        self.mult_template = conv_mult_config_template


class PointwiseConv1DFunctionTemplate(Conv1DFunctionTemplate):
    def __init__(self):
        super(Conv1DFunctionTemplate, self).__init__(PointwiseConv1D, include_header=sepconv1d_include_list)
        self.template = pointwise_conv1d_function_template


class PointwiseConv2DConfigTemplate(Conv2DConfigTemplate):
    def __init__(self):
        super(Conv2DConfigTemplate, self).__init__(PointwiseConv2D)
        self.template = pointwise_conv2d_config_template
        self.mult_template = conv_mult_config_template


class PointwiseConv2DFunctionTemplate(Conv2DFunctionTemplate):
    def __init__(self):
        super(Conv2DFunctionTemplate, self).__init__(PointwiseConv2D, include_header=sepconv2d_include_list)
        self.template = pointwise_conv2d_function_template


def register_pointwise(backend):
    # Register the layer types to the layer map
    register_layer('PointwiseConv1D', PointwiseConv1D)
    register_layer('PointwiseConv2D', PointwiseConv2D)

    # Register the optimization passes
    backend.register_pass('optimize_pointwise_conv', OptimizePointwiseConv)

    # Register template passes
    backend.register_template(PointwiseConv1DConfigTemplate)
    backend.register_template(PointwiseConv1DFunctionTemplate)
    backend.register_template(PointwiseConv2DConfigTemplate)
    backend.register_template(PointwiseConv2DFunctionTemplate)


class OptimizePointwiseConv(OptimizerPass):
    def match(self, node):
        return (
            node.class_name in ('Conv1D', 'Conv2D')
            and node.get_attr('filt_height', 1) == 1
            and node.get_attr('filt_width') == 1
        )

    def transform(self, model, node):
        dim = node.__class__.__name__[-2:]  # '1D' or '2D'
        pw_node = model.make_node('PointwiseConv' + dim, node.name, copy(node.attributes), node.inputs.copy())
        if len(node.weights['weight'].data.shape) == 2:  # This can happen if we assign weights of Dense layer to 1x1 Conv2D
            expand_axis = tuple(range(int(dim[0])))
            pw_node.weights['weight'].data = np.expand_dims(node.weights['weight'].data, axis=expand_axis)
        pw_node.weights['bias'].data = node.weights['bias'].data
        # Set strategy to ensure lowercase string is passed to the template
        if model.config.is_resource_strategy(pw_node):
            pw_node.set_attr('strategy', 'resource')
        else:
            pw_node.set_attr('strategy', 'latency')
        model.replace_node(node, pw_node)

        return True
