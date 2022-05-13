#ifndef NNET_CONV1D_LATENCY_H_
#define NNET_CONV1D_LATENCY_H_

#include "nnet_common.h"
#include <cstdlib>

namespace nnet {

//Computes multiplier limit
//This function should not be synthesized into firmware
template<typename CONFIG_T>
int compute_multiplier_limit(
    typename CONFIG_T::weight_t  weights[CONFIG_T::filt_width * CONFIG_T::n_chan * CONFIG_T::n_filt]
)
{
    int n_mult = 0;
    for(int ii = 0; ii < CONFIG_T::out_width; ii++) {
        for(int ff = 0; ff < CONFIG_T::n_filt; ff++){
            for(int cc = 0; cc < CONFIG_T::n_chan; cc++){
                for(int jj = 0; jj < CONFIG_T::filt_width; jj++){

                    int index_weight = jj*CONFIG_T::n_chan*CONFIG_T::n_filt + cc*CONFIG_T::n_filt + ff;

                    if((ii*CONFIG_T::stride_width+jj) < CONFIG_T::pad_left || (ii*CONFIG_T::stride_width+jj) >= (CONFIG_T::pad_left + CONFIG_T::in_width)){
                        //padded -- do nothing
                        continue;
                    } else {
                        //need to tune this cut?
                        if( weights[index_weight] > 1e-20 || weights[index_weight] < -1e-20 ){
                            n_mult++;
                        }//end if nonzero weight
                    }//end not padding
                }//end loop accross filter
            }//end channel loop
        }//end filter loop
    }//end output loop

    return ceil( float(n_mult) / float(CONFIG_T::reuse_factor) );

}//end compute_n_mult


template<class data_T, class res_T, typename CONFIG_T>
void conv_1d_latency_cl(
    data_T data[CONFIG_T::in_width * CONFIG_T::n_chan],
    res_T  res[CONFIG_T::out_width * CONFIG_T::n_filt],
    typename CONFIG_T::weight_t weights[CONFIG_T::filt_width * CONFIG_T::n_chan * CONFIG_T::n_filt],
    typename CONFIG_T::bias_t   biases[CONFIG_T::n_filt]
)
{

    typename CONFIG_T::accum_t mult[CONFIG_T::out_width * CONFIG_T::n_filt * CONFIG_T::n_chan * CONFIG_T::filt_width];
    typename CONFIG_T::accum_t acc[CONFIG_T::out_width][CONFIG_T::n_filt];

    #pragma HLS ARRAY_PARTITION variable=mult complete dim=0
    #pragma HLS ARRAY_PARTITION variable=acc complete dim=0

    // Use a function_instantiate in case it helps to explicitly optimize unchanging weights/biases
    #pragma HLS function_instantiate variable=weights,biases

    // Parallel mode
    #pragma HLS PIPELINE
    #pragma HLS ARRAY_PARTITION variable=biases complete dim=0

    // Limit multipliers to control parallelization
    //const int multiplier_limit = compute_multiplier_limit<CONFIG_T>(weights);
    //#pragma HLS ALLOCATION instances=mul limit=multiplier_limit operation

    // Convolve, saving all multiplication results to accumulate later
    ConvOut: for(int ii = 0; ii < CONFIG_T::out_width; ii++) {
        ConvFilt: for(int ff = 0; ff < CONFIG_T::n_filt; ff++){
            ConvChan: for(int cc = 0; cc < CONFIG_T::n_chan; cc++){
                ConvMult: for(int jj = 0; jj < CONFIG_T::filt_width; jj++){

                    int index_mult   = ii*CONFIG_T::n_filt*CONFIG_T::n_chan*CONFIG_T::filt_width + ff*CONFIG_T::n_chan*CONFIG_T::filt_width + cc*CONFIG_T::filt_width + jj;
                    int index_weight = jj*CONFIG_T::n_chan*CONFIG_T::n_filt + cc*CONFIG_T::n_filt + ff;
                    int index_data   = (ii*CONFIG_T::stride_width+jj-CONFIG_T::pad_left) * CONFIG_T::n_chan + cc;

                    if((ii*CONFIG_T::stride_width+jj) < CONFIG_T::pad_left || (ii*CONFIG_T::stride_width+jj) >= (CONFIG_T::pad_left + CONFIG_T::in_width)){
                        mult[index_mult] = 0;
                    }
                    else {
                        mult[index_mult] = data[index_data] * weights[index_weight];
                    }
                }
            }//end channel loop
        }//end filter loop
    }//end output loop


    // Initialize accumulator with input biases
    for(int ii = 0; ii < CONFIG_T::out_width; ii++) {
        for(int ff = 0; ff < CONFIG_T::n_filt; ff++) {
            acc[ii][ff]=biases[ff];
        }
    }


    // Accumulate multiplication result
    AccumOut: for(int ii = 0; ii < CONFIG_T::out_width; ii++) {
        AccumFilt: for(int ff = 0; ff < CONFIG_T::n_filt; ff++) {
            //Do "dot product" sum within filter and sum over channels
            AccumChan: for(int cc = 0; cc < CONFIG_T::n_chan; cc++){
                AccumDot: for(int jj = 0; jj < CONFIG_T::filt_width; jj++){
                    int index_mult = ii*CONFIG_T::n_filt*CONFIG_T::n_chan*CONFIG_T::filt_width + ff*CONFIG_T::n_chan*CONFIG_T::filt_width + cc*CONFIG_T::filt_width + jj;
                    acc[ii][ff] += mult[index_mult];
                }//end dot product loop
            }//end channel loop
        }//end filter loop
    }//end output loop


    // Cast to "res_t" type
    for(int ii = 0; ii < CONFIG_T::out_width; ii++) {
        for(int ff = 0; ff < CONFIG_T::n_filt; ff++) {
            res[ii * CONFIG_T::n_filt + ff] = (res_T)(acc[ii][ff]);
        }
    }
}

template<class data_T, class res_T, typename CONFIG_T>
void pointwise_conv_1d_latency_cl(
    data_T data[CONFIG_T::in_width * CONFIG_T::n_chan/CONFIG_T::reuse_factor],
    res_T  res[CONFIG_T::out_width * CONFIG_T::n_filt/CONFIG_T::reuse_factor],
    typename CONFIG_T::weight_t weights[CONFIG_T::n_chan * CONFIG_T::n_filt],
    typename CONFIG_T::bias_t   biases[CONFIG_T::n_filt])
{
    assert(CONFIG_T::filt_width == 1);

    typename CONFIG_T::accum_t mult[CONFIG_T::out_width * CONFIG_T::n_filt * CONFIG_T::n_chan/CONFIG_T::reuse_factor];
    typename CONFIG_T::accum_t acc[CONFIG_T::out_width/CONFIG_T::reuse_factor][CONFIG_T::n_filt];

    #pragma HLS ARRAY_PARTITION variable=mult complete dim=0
    #pragma HLS ARRAY_PARTITION variable=acc complete dim=0

    // Use a function_instantiate in case it helps to explicitly optimize unchanging weights/biases
    #pragma HLS function_instantiate variable=weights,biases

    // Parallel mode
    #pragma HLS PIPELINE II=CONFIG_T::reuse_factor
    #pragma HLS ARRAY_PARTITION variable=biases complete dim=0

    // Limit multipliers to control parallelization
    //const int multiplier_limit = compute_multiplier_limit<CONFIG_T>(weights);
    //#pragma HLS ALLOCATION instances=mul limit=multiplier_limit operation

    // Convolve, saving all multiplication results to accumulate later
    ConvOut: for(int ii = 0; ii < CONFIG_T::out_width/CONFIG_T::reuse_factor; ii++) {
        ConvFilt: for(int ff = 0; ff < CONFIG_T::n_filt; ff++) {
            ConvChan: for(int cc = 0; cc < CONFIG_T::n_chan; cc++) {
                int index_mult   = ii*CONFIG_T::n_filt*CONFIG_T::n_chan + ff*CONFIG_T::n_chan + cc;
                int index_weight = cc*CONFIG_T::n_filt + ff;
                int index_data   = (ii*CONFIG_T::stride_width-CONFIG_T::pad_left) * CONFIG_T::n_chan + cc;

                if((ii*CONFIG_T::stride_width) < CONFIG_T::pad_left || (ii*CONFIG_T::stride_width) >= (CONFIG_T::pad_left + CONFIG_T::in_width)){
                    mult[index_mult] = 0;
                }
                else {
                    mult[index_mult] = data[index_data] * weights[index_weight];
                }
            }//end channel loop
        }//end filter loop
    }//end output loop


    // Initialize accumulator with input biases
    for(int ii = 0; ii < CONFIG_T::out_width/CONFIG_T::reuse_factor; ii++) {
        for(int ff = 0; ff < CONFIG_T::n_filt; ff++) {
            acc[ii][ff]=biases[ff];
        }
    }


    // Accumulate multiplication result
    AccumOut: for(int ii = 0; ii < CONFIG_T::out_width/CONFIG_T::reuse_factor; ii++) {
        AccumFilt: for(int ff = 0; ff < CONFIG_T::n_filt; ff++) {
            //Do "dot product" sum within filter and sum over channels
            AccumChan: for(int cc = 0; cc < CONFIG_T::n_chan; cc++) {
                int index_mult = ii*CONFIG_T::n_filt*CONFIG_T::n_chan + ff*CONFIG_T::n_chan + cc;
                acc[ii][ff] += mult[index_mult];
            }//end channel loop
        }//end filter loop
    }//end output loop


    // Cast to "res_t" type
    for(int ii = 0; ii < CONFIG_T::out_width/CONFIG_T::reuse_factor; ii++) {
        for(int ff = 0; ff < CONFIG_T::n_filt; ff++) {
            res[ii * CONFIG_T::n_filt + ff] = (res_T)(acc[ii][ff]);
        }
    }
}

template<class data_T, class res_T, typename CONFIG_T>
void pointwise_conv_1d_latency_cl_split2(
    data_T data[CONFIG_T::in_width * CONFIG_T::n_chan],
    res_T  res[CONFIG_T::out_width * CONFIG_T::n_filt],
    typename CONFIG_T::weight_t weights[CONFIG_T::n_chan * CONFIG_T::n_filt],
    typename CONFIG_T::bias_t   biases[CONFIG_T::n_filt])
{

    data_T data_tmp[2][CONFIG_T::in_width*CONFIG_T::n_chan/2];
    #pragma HLS ARRAY_PARTITION variable=data_tmp complete dim=0
    res_T res_tmp[2][CONFIG_T::out_width*CONFIG_T::n_filt/2];
    #pragma HLS ARRAY_PARTITION variable=res_tmp complete dim=0
    
    for(int jj = 0; jj < 2; jj++) {
        for(int ii = 0; ii < CONFIG_T::in_width*CONFIG_T::n_chan/2; ii++) {
            #pragma HLS UNROLL
            data_tmp[jj][ii] = data[jj*CONFIG_T::in_width*CONFIG_T::n_chan/2+ii];
        }
    }

    pointwise_conv_1d_latency_cl<data_T, res_T, CONFIG_T>(data_tmp[0], res_tmp[0], weights, biases);
    pointwise_conv_1d_latency_cl<data_T, res_T, CONFIG_T>(data_tmp[1], res_tmp[1], weights, biases);

    for(int jj = 0; jj < 2; jj++) {
        for(int ii = 0; ii < CONFIG_T::out_width * CONFIG_T::n_filt/2; ii++) {
            #pragma HLS UNROLL
            res[jj*CONFIG_T::out_width*CONFIG_T::n_filt/2+ii] = res_tmp[jj][ii];
        }
    }
}

template<class data_T, class res_T, typename CONFIG_T>
void pointwise_conv_1d_latency_cl_split4(
    data_T data[CONFIG_T::in_width * CONFIG_T::n_chan],
    res_T  res[CONFIG_T::out_width * CONFIG_T::n_filt],
    typename CONFIG_T::weight_t weights[CONFIG_T::n_chan * CONFIG_T::n_filt],
    typename CONFIG_T::bias_t   biases[CONFIG_T::n_filt])
{

    data_T data_tmp[4][CONFIG_T::in_width*CONFIG_T::n_chan/4];
    #pragma HLS ARRAY_PARTITION variable=data_tmp complete dim=0
    res_T res_tmp[4][CONFIG_T::out_width*CONFIG_T::n_filt/4];
    #pragma HLS ARRAY_PARTITION variable=res_tmp complete dim=0
    
    for(int jj = 0; jj < 4; jj++) {
        for(int ii = 0; ii < CONFIG_T::in_width*CONFIG_T::n_chan/4; ii++) {
            #pragma HLS UNROLL
            data_tmp[jj][ii] = data[jj*CONFIG_T::in_width*CONFIG_T::n_chan/4+ii];
        }
    }

    pointwise_conv_1d_latency_cl<data_T, res_T, CONFIG_T>(data_tmp[0], res_tmp[0], weights, biases);
    pointwise_conv_1d_latency_cl<data_T, res_T, CONFIG_T>(data_tmp[1], res_tmp[1], weights, biases);
    pointwise_conv_1d_latency_cl<data_T, res_T, CONFIG_T>(data_tmp[2], res_tmp[2], weights, biases);
    pointwise_conv_1d_latency_cl<data_T, res_T, CONFIG_T>(data_tmp[3], res_tmp[3], weights, biases);

    for(int jj = 0; jj < 4; jj++) {
        for(int ii = 0; ii < CONFIG_T::out_width * CONFIG_T::n_filt/4; ii++) {
            #pragma HLS UNROLL
            res[jj*CONFIG_T::out_width*CONFIG_T::n_filt/4+ii] = res_tmp[jj][ii];
        }
    }
}

template<class data_T, class res_T, typename CONFIG_T>
void pointwise_conv_1d_latency_cl_split6(
    data_T data[CONFIG_T::in_width * CONFIG_T::n_chan],
    res_T  res[CONFIG_T::out_width * CONFIG_T::n_filt],
    typename CONFIG_T::weight_t weights[CONFIG_T::n_chan * CONFIG_T::n_filt],
    typename CONFIG_T::bias_t   biases[CONFIG_T::n_filt])
{

    data_T data_tmp[6][CONFIG_T::in_width*CONFIG_T::n_chan/6];
    #pragma HLS ARRAY_PARTITION variable=data_tmp complete dim=0
    res_T res_tmp[6][CONFIG_T::out_width*CONFIG_T::n_filt/6];
    #pragma HLS ARRAY_PARTITION variable=res_tmp complete dim=0
    
    for(int jj = 0; jj < 6; jj++) {
        for(int ii = 0; ii < CONFIG_T::in_width*CONFIG_T::n_chan/6; ii++) {
            #pragma HLS UNROLL
            data_tmp[jj][ii] = data[jj*CONFIG_T::in_width*CONFIG_T::n_chan/6+ii];
        }
    }

    pointwise_conv_1d_latency_cl<data_T, res_T, CONFIG_T>(data_tmp[0], res_tmp[0], weights, biases);
    pointwise_conv_1d_latency_cl<data_T, res_T, CONFIG_T>(data_tmp[1], res_tmp[1], weights, biases);
    pointwise_conv_1d_latency_cl<data_T, res_T, CONFIG_T>(data_tmp[2], res_tmp[2], weights, biases);
    pointwise_conv_1d_latency_cl<data_T, res_T, CONFIG_T>(data_tmp[3], res_tmp[3], weights, biases);
    pointwise_conv_1d_latency_cl<data_T, res_T, CONFIG_T>(data_tmp[4], res_tmp[4], weights, biases);
    pointwise_conv_1d_latency_cl<data_T, res_T, CONFIG_T>(data_tmp[5], res_tmp[5], weights, biases);

    for(int jj = 0; jj < 6; jj++) {
        for(int ii = 0; ii < CONFIG_T::out_width * CONFIG_T::n_filt/6; ii++) {
            #pragma HLS UNROLL
            res[jj*CONFIG_T::out_width*CONFIG_T::n_filt/6+ii] = res_tmp[jj][ii];
        }
    }
}

template<class data_T, class res_T, typename CONFIG_T>
void pointwise_conv_1d_latency_cl_split8(
    data_T data[CONFIG_T::in_width * CONFIG_T::n_chan],
    res_T  res[CONFIG_T::out_width * CONFIG_T::n_filt],
    typename CONFIG_T::weight_t weights[CONFIG_T::n_chan * CONFIG_T::n_filt],
    typename CONFIG_T::bias_t   biases[CONFIG_T::n_filt])
{

    data_T data_tmp[8][CONFIG_T::in_width*CONFIG_T::n_chan/8];
    #pragma HLS ARRAY_PARTITION variable=data_tmp complete dim=0
    res_T res_tmp[8][CONFIG_T::out_width*CONFIG_T::n_filt/8];
    #pragma HLS ARRAY_PARTITION variable=res_tmp complete dim=0
    
    for(int jj = 0; jj < 8; jj++) {
        for(int ii = 0; ii < CONFIG_T::in_width*CONFIG_T::n_chan/8; ii++) {
            #pragma HLS UNROLL
            data_tmp[jj][ii] = data[jj*CONFIG_T::in_width*CONFIG_T::n_chan/8+ii];
        }
    }

    pointwise_conv_1d_latency_cl<data_T, res_T, CONFIG_T>(data_tmp[0], res_tmp[0], weights, biases);
    pointwise_conv_1d_latency_cl<data_T, res_T, CONFIG_T>(data_tmp[1], res_tmp[1], weights, biases);
    pointwise_conv_1d_latency_cl<data_T, res_T, CONFIG_T>(data_tmp[2], res_tmp[2], weights, biases);
    pointwise_conv_1d_latency_cl<data_T, res_T, CONFIG_T>(data_tmp[3], res_tmp[3], weights, biases);
    pointwise_conv_1d_latency_cl<data_T, res_T, CONFIG_T>(data_tmp[4], res_tmp[4], weights, biases);
    pointwise_conv_1d_latency_cl<data_T, res_T, CONFIG_T>(data_tmp[5], res_tmp[5], weights, biases);
    pointwise_conv_1d_latency_cl<data_T, res_T, CONFIG_T>(data_tmp[6], res_tmp[6], weights, biases);
    pointwise_conv_1d_latency_cl<data_T, res_T, CONFIG_T>(data_tmp[7], res_tmp[7], weights, biases);

    for(int jj = 0; jj < 8; jj++) {
        for(int ii = 0; ii < CONFIG_T::out_width * CONFIG_T::n_filt/8; ii++) {
            #pragma HLS UNROLL
            res[jj*CONFIG_T::out_width*CONFIG_T::n_filt/8+ii] = res_tmp[jj][ii];
        }
    }
}

}
#endif
