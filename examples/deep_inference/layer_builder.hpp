#ifndef LAYERS_BUILDER_MODELS_
#define LAYERS_BUILDER_MODELS_
#include <algorithm>
#include <cmath>
#include <iostream>
#include <string>
#include <vector>

#include <assert.h>
#include <malloc.h>


#include <chrono>
#include <vector>
#include <unordered_map>

#include "oneapi/dnnl/dnnl.hpp"
#include "utils.hpp"
#include "check_error.hpp"

//#include "flatbuffers/flatbuffers.h"
//#include "schema_generated.h"

using namespace dnnl;

namespace BrixLab
{
    using tag = memory::format_tag;
    using dt = memory::data_type;
    template<typename DType>
    void OP_convolution_inference_forward(const layerNode<DType> &node, graphState<DType> &state){
        
        node.conv_bottom_memory = state.input;
        if (node.convolution_pdesc.src_desc() != node.src_bottom_memory.get_desc()) {
            node.conv_bottom_memory = memory(node.convolution_pdesc.src_desc(), BrixLab::graph_eng);
            std::unordered_map<int, memory> op_arg = {{DNNL_ARG_FROM, node.src_bottom_memory},
                    {DNNL_ARG_TO, node.conv_bottom_memory}};
            reorder(node.src_bottom_memory, node.conv_bottom_memory).execute(BrixLab::graph_stream, op_arg);
        }
        if (node.convolution_pdesc.weights_desc() != node.src_weight_memory.get_desc()) {
            node.conv_weights_memory = memory(node.convolution_pdesc.weights_desc(), BrixLab::graph_eng);
            reorder(node.src_weights_memory, node.conv_weights_memory)
                    .execute(BrixLab::graph_stream, node.src_weights_memory, node.conv_weights_memory);
        }
        convolution_forward(node.convolution_pdesc).execute(BrixLab::graph_stream, node.op_args);
    }

    template<typename DType>
    void OP_convolution_layer_setup(const layerWeightsParam<DType> &param, graphState<DType> &g_state){
        layerNode<DType> node = {(OP_type)0};
        int k_w = param.k_w;
        int k_h = param.k_h;
        int k_c = param.k_c;
        int k_s = param.strides;
        int k_pad = param.padding;
        int data_formate = param.formate;
        int inHeight = param.inHeight;
        int inChannel = param.inChannel;
        int inWidth = param.inWidth;
        int inBatch = param.inBatch;
        node.layer_h = inHeight;
        node.layer_c = inChannel;
        node.layer_w = inWidth;
        node.layer_n = inBatch;
        node.bottom_shape = {inBatch, inChannel, inHeight, inWidth};
        node.groups = param.groups;
        if(node.groups > 1){
            assert(inChannel % node.groups == 0);
            assert(k_c % node.groups == 0);
            int ICg = inChannel / node.groups;
            int OCg = k_c / node.groups;
            node.weights_shape = {node.groups, OCg, ICg, k_w, k_h};
        }else if(node.groups == 1){
            node.weights_shape = {k_c, inChannel, k_w, k_h};
        }
        node.conv_strides = {k_s, k_s};
        node.conv_padding = {k_pad, k_pad};
        node.dialited_rate = param.dialited_rate;
        int diat_rate = param.dialited_rate;
        if(node.hasBias)
            node.bias_shape = {k_c};
        
        // src bottom_md
        node.src_bottom_md = memory::desc({node.bottom_shape}, dt::f32, tag::any);
        // weights & bias
        node.src_weights_memory = memory({{node.weights_shape}, dt::f32, tag::oihw}, BrixLab::graph_eng);
        write_to_dnnl_memory<DType>(param.conv_weights, node.src_weights_memory);
        node.conv_weights_md = memory::desc({node.weights_shape}, dt::f32, tag::any);
        if(node.hasBias){
            node.conv_bias_memory = memory({{node.bias_shape}, dt::f32, tag::x}, BrixLab::graph_eng);
            write_to_dnnl_memory<DType>(param.conv_bias, node.conv_bias_memory);
            node.conv_bias_md = memory::desc({node.bias_shape}, dt::f32, tag::any);
        }

        int outWidth = (inWidth - (1 + (k_w - 1) * (diat_rate + 1)) + 2 * k_pad) / k_s + 1;
        int outHeight = (inHeight - ((1 + (k_h - 1) * (diat_rate + 1))) + 2 * k_pad) / k_s + 1;
        node.top_shape = {inBatch, k_c, outHeight, outWidth};
        
        node.layer_top_md = memory::desc({node.top_shape}, dt::f32, tag::any);

        if(param.dialited_rate == 0){ 
            if(node.hasBias){
                node.convolution_desc = convolution_forward::desc(prop_kind::forward_inference,
                                        algorithm::convolution_direct, node.src_bottom_md, node.conv_weights_md,
                                        node.conv_bias_md, node.layer_top_md, node.conv_strides, 
                                        node.conv_padding, node.conv_padding);
            }else{
                node.convolution_desc = convolution_forward::desc(prop_kind::forward_inference, 
                                        algorithm::convolution_direct, node.src_bottom_md, 
                                        node.conv_weights_md, node.layer_top_md, 
                                        node.conv_strides, node.conv_padding, node.conv_padding); 
            }
            
        }else if(node.dialited_rate >= 1){
            memory::dims conv_dilates = {node.dialited_rate, node.dialited_rate};
            if(node.hasBias){
                node.convolution_desc = convolution_forward::desc(prop_kind::forward_inference,
                                          algorithm::convolution_direct,node.src_bottom_md, node.conv_weights_md,
                                          node.conv_bias_md, node.layer_top_md, node.conv_strides, conv_dilates,
                                          node.conv_padding, node.conv_padding);
            }else{
                node.convolution_desc = convolution_forward::desc(prop_kind::forward_inference,
                                          algorithm::convolution_direct,node.src_bottom_md, node.conv_weights_md,
                                          node.layer_top_md, node.conv_strides, conv_dilates,
                                          node.conv_padding, node.conv_padding); 
            }
            
        }

        node.convolution_pdesc = convolution_forward::primitive_desc(node.convolution_desc, BrixLab::graph_eng);

        node.conv_weights_memory = node.src_weights_memory;

        node.layer_top_memory = memory(node.convolution_pdesc.dst_desc(), BrixLab::graph_eng);
        node.op_args = {{DNNL_ARG_SRC, node.conv_bottom_memory},
                {DNNL_ARG_WEIGHTS, node.conv_weights_memory},
                {DNNL_ARG_BIAS, node.conv_bias_memory},
                {DNNL_ARG_DST, node.layer_top_memory}};
        node.inference_forward = OP_convolution_inference_forward; 
        graph_insert(g_state, &node);
    }
    

    template<typename DType>
    void OP_batchnorm_inference_forward(const layerNode<DType> &node, graphState<DType> &state){
        node.batchnorm_bottom_memory = state.input;
        batch_normalization_forward(node.batchnorm_pdesc).execute(BrixLab::graph_stream, node.op_args);
    }
    template<typename DType>
    void OP_batchnorm_layer_setup(const layerWeightsParam<DType> &param, 
                                                            const graphState<DType> &g_state){
        layerNode<DType> node = {(OP_type)0};
        int inHeight = param.inHeight;
        int inChannel = param.inChannel;
        int inWidth = param.inWidth;
        int inBatch = param.inBatch;
        node.layer_h = inHeight;
        node.layer_c = inChannel;
        node.layer_w = inWidth;
        node.layer_n = inBatch;
        node.bottom_shape = {inBatch, inChannel, inHeight, inWidth};
        node.scale_shift_shape = {2, inChannel};

        node.batchnorm_bottom_md = memory::desc(node.bottom_shape, dt::f32, tag::nchw);
        node.batchnorm_scale_shift_md = memory::desc(node.scale_shift_shape, dt::f32, tag::nc);

        node.batchnorm_scale_shift_memory = memory(node.batchnorm_scale_shift_md, BrixLab::graph_eng);
        
        
        write_to_dnnl_memory<DType>(param.b_shift_scale, node.batchnorm_scale_shift_memory);

        // Create operation descriptor.
        node.batchnorm_desc = batch_normalization_forward::desc(
                                    prop_kind::forward_inference, node.batchnorm_bottom_md, 1.e-10f,
                                    normalization_flags::use_scale_shift
                                    | normalization_flags::use_global_stats);

        // Create primitive descriptor.
        node.batchnorm_pdesc = batch_normalization_forward::primitive_desc(node.batchnorm_desc, BrixLab::graph_eng);

        // Create memory objects using memory descriptors created by the primitive
        // descriptor: mean, variance.
        
        node.batchnorm_mean_memory = memory(node.batchnorm_pdesc.mean_desc(), BrixLab::graph_eng);
        node.batchnorm_variance_memory = memory(node.batchnorm_pdesc.variance_desc(), BrixLab::graph_eng);
        write_to_dnnl_memory<DType>(param.b_means, node.batchnorm_mean_memory);
        write_to_dnnl_memory<DType>(param.b_variance, node.batchnorm_variance_memory);

        node.batchnorm_desc = batch_normalization_forward::desc(
                                    prop_kind::forward_training, node.batchnorm_bottom_md, 1.e-10f,
                                    normalization_flags::use_scale_shift
                                    | normalization_flags::fuse_norm_relu);

        node.op_args ={{DNNL_ARG_SRC, node.batchnorm_bottom_memory},
                            {DNNL_ARG_MEAN, node.batchnorm_mean_memory},
                            {DNNL_ARG_VARIANCE, node.batchnorm_variance_memory},
                            {DNNL_ARG_SCALE_SHIFT, node.batchnorm_scale_shift_memory},
                            {DNNL_ARG_DST, node.batchnorm_bottom_memory}};
        node.layer_top_memory = node.batchnorm_bottom_memory;
        node.inference_forward = OP_batchnorm_inference_forward;
        graph_insert(g_state, &node);
    }

    template<typename DType>
    void OP_pooling_inference_forward(const layerNode<DType> &node, graphState<DType> &state){
         node.pooling_bottom_memory = state.input;
        if(node.p_dialiated == 0){
            pooling_forward(node.pooling_pdesc_without_d).execute(BrixLab::graph_stream, node.op_args);
        }else if(node.p_dialiated >= 1){
            pooling_v2_forward(node.pooling_pdesc).execute(BrixLab::graph_stream, node.op_args);
        }
    }

    template<typename DType>
    void OP_pooling_layer_setup(const layerWeightsParam<DType> &param, 
                                                        const graphState<DType> &g_state){
        layerNode<DType> node = {(OP_type)0};
        int inHeight = param.inHeight;
        int inChannel = param.inChannel;
        int inWidth = param.inWidth;
        int inBatch = param.inBatch;
        node.layer_h = inHeight;
        node.layer_c = inChannel;
        node.layer_w = inWidth;
        node.layer_n = inBatch;
        int k_w = param.p_kw;
        int k_h = param.p_kh;
        int k_s = param.p_strides;
        int k_pad = param.p_padding;
        node.bottom_shape = {inBatch, inChannel, inHeight, inWidth};
        int dialiated_rate = param.p_diliated;
        node.p_dialiated = param.p_diliated;
        PoolingType p_type = param.p_type;
        node.pooling_type = get_op_mapped_type(p_type);
        int outHight = 0;
        int outWidth = 0;
        if(dialiated_rate == 0){
            outHight = (inHeight - k_h + 2 * k_pad) / k_s + 1;
            outWidth = (inWidth - k_w + 2 * k_pad) / k_s + 1;
        }else if(dialiated_rate > 0){
            outHight = (inHeight - ((k_h - 1) * dialiated_rate + k_h) +2 *k_pad) / k_s + 1;
            outWidth = (inWidth - ((k_w - 1) * dialiated_rate + k_w) +2 *k_pad) / k_s + 1;
        }
        
        node.pooling_kernel = {k_w, k_h};
        node.pooling_strides = {k_s, k_s};
        node.pooling_padding = {k_pad, k_pad};
        node.pooling_dialiate = {dialiated_rate, dialiated_rate};
        
        node.top_shape = {inBatch, inChannel, outHight, outWidth};
        node.layer_top_md = memory::desc(node.top_shape, dt::f32, tag::nchw);
        node.layer_top_memory = memory(node.layer_top_md, BrixLab::graph_eng);
       
        node.batchnorm_bottom_md = memory::desc(node.bottom_shape, dt::f32, tag::nchw);
        if(dialiated_rate > 0){
            node.pooling_desc = pooling_v2_forward::desc(prop_kind::forward_inference,
                                            node.pooling_type, node.pooling_bottom_md, 
                                            node.layer_top_md,
                                            node.pooling_strides, node.pooling_kernel,
                                            node.pooling_dialiate, node.pooling_padding, 
                                            node.pooling_padding);
            node.pooling_pdesc = pooling_v2_forward::primitive_desc(node.pooling_desc, BrixLab::graph_eng);
        }else if(dialiated_rate == 0){
            node.pooling_desc_without_d = pooling_forward::desc(prop_kind::forward_inference, 
                                                node.pooling_type, node.pooling_bottom_md, 
                                                node.layer_top_md, node.pooling_strides, 
                                                node.pooling_kernel, node.pooling_dialiate, 
                                                node.pooling_padding,node.pooling_padding);
            node.pooling_pdesc_without_d = pooling_forward::primitive(node.pooling_desc_without_d, BrixLab::graph_eng);
        }
        node.op_args = {{DNNL_ARG_SRC, node.pooling_bottom_memory},
                        {DNNL_ARG_DST, node.layer_top_memory}
                        };
        node.inference_forward = OP_pooling_inference_forward;
        graph_insert(g_state, &node);
    }

    template<typename DType>
    void OP_concat_inference_forward(const layerNode<DType> &node, graphState<DType> &state){
        concat(node.concat_pdesc).execute(BrixLab::graph_stream, node.op_args);
    }

    template<typename DType>
    void OP_concat_layer_setup(const layerWeightsParam<DType> &param, 
                                                                const graphState<DType> &g_state){
        layerNode<DType> node = {(OP_type)0};
        int inHeight = param.inHeight;
        int inChannel = param.inChannel;
        int inWidth = param.inWidth;
        int inBatch = param.inBatch;
        node.bottom_shape = {inBatch, inChannel, inHeight, inWidth};
        node.concat_num = param.concat_num;
        node.concat_axis = param.concat_axis;
        node.concat_index = (int*)xcalloc(node.concat_num, sizeof(int));
        node.op_type = param.op_type;
        for(int ii = 0; ii < node.concat_num; ii++){
            int layer_index = node.concat_index[ii]->top;
            node.concat_bottom_md.push_back(g_state[ii]->layer_top_md);
            node.concat_bottom_memory.push_back(g_state[ii]->layer_top_memory);
            node.op_args.insert({DNNL_ARG_MULTIPLE_SRC + ii, g_state[ii]->layer_top_memory});
        }
        
        node.op_args.insert({{DNNL_ARG_DST, node.layer_top_memory}});
        node.concat_pdesc = concat::primitive_desc(node.concat_axis, node.concat_bottom_md, BrixLab::graph_eng);
        node.layer_top_md = node.concat_pdesc.dst_desc();
        node.layer_top_memory = memory(node.layer_top_md, BrixLab::graph_eng);
        node.inference_forward = OP_concat_inference_forward;
        graph_insert(g_state, &node);
    }

    template<typename DType>
    void OP_sum_inference_forward(const layerNode<DType> &node, graphState<DType> &state){
        sum(node.sum_pdesc).execute(node.op_type, BrixLab::graph_stream);
    }

    template<typename DType>
    void OP_sum_layer_setup(const layerWeightsParam<DType> &param, 
                                                const graphState<DType> &g_state){
        layerNode<DType> node = {(OP_type)0};
        node.sum_num = param.sum_num;
        node.sum_index = (int*)xcalloc(node.sum_num, sizeof(int));
        int inHeight = g_state[node.sum_index[0]]->layer_h;
        int inWidth = g_state[node.sum_index[0]]->layer_w;
        int inChannel = g_state[node.sum_index[0]]->layer_c;
        int inBatch = g_state[node.sum_index[0]]->layer_n;
        memory::dims S_Shape = {inBatch, inChannel, inWidth, inHeight};
        for(int ii = 0; ii < node.sum_num; ii++){
            node.sum_index[ii] = param.sum_index[ii];
            node.sum_scale[ii] = DType(1.0);
            checK_equal_dims(S_Shape, g_state[ii]->top_shape);
            node.sum_bottom_memory.push_back(g_state[ii]->layer_top_memory);
            node.sum_bottom_md.push_back(g_state[ii]->layer_top_md);
            node.op_type.insert({DNNL_ARG_MULTIPLE_SRC + ii, node.sum_bottom_memory[ii]});
        }
        node.sum_pdesc = sum::primitive_desc(node.sum_scale, node.sum_bottom_md, BrixLab::graph_eng);
        node.top_shape = {inBatch, inChannel, inWidth, inHeight};
        node.layer_top_md = memory::desc(node.top_shape, dt::f32, tag::nchw);
        node.layer_top_memory = memory(node.sum_pdesc.dst_desc(), BrixLab::graph_eng);
        node.op_type.insert({DNNL_ARG_DST, node.layer_top_memory});
        for(int ii = 0; ii < node.sum_num; ii++){
            node.op_type.insert({DNNL_ARG_MULTIPLE_SRC + ii, node.sum_bottom_memory[ii]});
        }
        node.inference_forward = OP_sum_inference_forward;
        graph_insert(g_state, &node);
    }

    template<typename DType>
    void OP_resample_inference_forward(const layerNode<DType> &node, graphState<DType> &state){
        node.resample_bottom_memory = state.input;
        resampling_forward(node.resample_pdesc).execute(node.op_args, BrixLab::graph_stream);
    }

    template<typename DType>
    void OP_resample_layer_setup(const layerWeightsParam<DType> &param, 
                                                const graphState<DType> &g_state){
        int inHeight = param.inHeight;
        int inWidth = param.inWidth;
        int inChannel = param.inChannel;
        int inBatch = param.inBatch;
        layerNode<DType> node = {0};
        node.bottom_shape = {inBatch, inChannel, inHeight, inWidth};
        node.op_type = param.op_type;
        node.resample_bottom_md = memory::desc(node.bottom_shape, dt::f32, tag::nchw);
        node.adjust_scale = param.adjust_scale;
        int outHeight = int(inHeight * node.adjust_scale);
        int outWidth = int(inWidth * node.adjust_scale);
        node.top_shape = {inBatch, inChannel, outHeight, outWidth};
        node.layer_top_md = memory::desc(node.top_shape, dt::f32, tag::nchw);
        node.layer_top_memory = memory(node.layer_top_md, BrixLab::graph_eng);

        node.resample_desc = resampling_forward::desc(prop_kind::forward_inference,
            algorithm::resampling_linear, node.resample_bottom_md, node.layer_top_md);
        node.resample_pdesc = resampling_forward::primitive_desc(node.resample_desc, BrixLab::graph_eng);

        node.op_args.insert({DNNL_ARG_SRC, node.resample_bottom_memory});
        node.op_args.insert({DNNL_ARG_DST, node.layer_top_memory});
        node.inference_forward = OP_resample_inference_forward;
        graph_insert(g_state, &node);
    }

    template<typename DType>
    void OP_deconvolution_inference_forward(const layerNode<DType> &node, graphState<DType> &state){
        node.src_bottom_memory = state.input;
        node.deconv_bottom_memory = node.src_bottom_memory;
        if (node.deconv_pdesc.src_desc() != node.src_bottom_memory.get_desc()) {
            node.deconv_bottom_memory = memory(node.deconv_pdesc.src_desc(), BrixLab::graph_eng);
            std::unordered_map<int, memory> op_arg = {{DNNL_ARG_FROM, node.src_bottom_memory},
                    {DNNL_ARG_TO, node.deconv_bottom_memory}};
            reorder(node.src_bottom_memory, node.deconv_bottom_memory).execute(BrixLab::graph_stream, op_arg);
        }
        if (node.deconv_pdesc.weights_desc() != node.src_weight_memory.get_desc()) {
            node.deconv_weights_memory = memory(node.deconv_pdesc.weights_desc(), BrixLab::graph_eng);
            reorder(node.src_weights_memory, node.deconv_weights_memory)
                    .execute(BrixLab::graph_stream, node.src_weights_memory, node.deconv_weights_memory);
        }
        deconvolution_forward(node.deconv_pdesc).execute(BrixLab::graph_stream, node.op_args);
    }

    template<typename DType>
    void OP_deconvolution_layer_setup(const layerWeightsParam<DType> &param, 
                                                const graphState<DType> &g_state){
        layerNode<DType> node = {(OP_type)0};
        int k_w = param.k_w;
        int k_h = param.k_h;
        int k_c = param.k_c;
        int k_s = param.strides;
        int k_pad = param.padding;
        int data_formate = param.formate;
        int inHeight = param.inHeight;
        int inChannel = param.inChannel;
        int inWidth = param.inWidth;
        int inBatch =param.inBatch;
        
        node.bottom_shape = {inBatch, inChannel, inHeight, inWidth};
        node.weights_shape = {k_c, inChannel, k_w, k_h};
        node.deconv_strides = {k_s, k_s};
        node.deconv_padding = {k_pad, k_pad};
        node.dedialited_rate = param.dialited_rate;
        node.hasdeBias = param.hasBias;
        if(node.hasdeBias)
            node.bias_shape = {k_c};
        // src bottom data
        node.src_bottom_md = memory::desc({node.bottom_shape}, dt::f32, tag::any);
        // weights & bias
        node.src_weights_memory = memory({{node.weights_shape}, dt::f32, tag::oihw}, BrixLab::graph_eng);
        write_to_dnnl_memory<DType>(param.transposed_weights, node.src_weights_memory);
        node.src_weights_md = memory::desc({node.weights_shape}, dt::f32, tag::any);
        if(node.hasdeBias){
            node.deconv_bias_memory = memory({{node.bias_shape}, dt::f32, tag::x}, BrixLab::graph_eng);
            write_to_dnnl_memory<DType>(param.transposed_bias, node.deconv_bias_memory);
            node.deconv_bias_md = memory::desc({node.bias_shape}, dt::f32, tag::any);
        }

        int D_kHeight = 1 + (k_h -  1) * (node.dedialited_rate + 1);
        int D_kWidth = 1 + (k_w - 1) * (node.dedialited_rate_rate + 1);

        int outWidth = int((inWidth - 1) * k_s + D_kWidth - 2 * k_pad);
        int outHeight = int((inHeight - 1) * k_s + D_kHeight - 2 * k_pad);
        node.top_shape = {inBatch, k_c, outHeight, outWidth};
        node.layer_top_md = memory::desc({node.top_shape}, dt::f32, tag::nchw);

        if(node.dedialited_rate > 0){
            memory::dims deconv_deliatd = {node.dedialited_rate, node.dedialited_rate};
            if(node.hasdeBias)
                node.deconv_desc  = desc(prop_kind::forward_inference, algorithm::deconvolution_direct,
                                            node.src_bottom_md, node.src_weights_md,
                                            node.deconv_bias_md, node.layer_top_md,
                                            node.deconv_strides, deconv_deliatd,
                                            node.deconv_padding,node.deconv_padding);
            else{
                node.deconv_desc  = desc(prop_kind::forward_inference, algorithm::deconvolution_direct,
                                            node.src_bottom_md, node.src_weights_md,
                                            node.layer_top_md,
                                            node.deconv_strides, deconv_deliatd,
                                            node.deconv_padding, node.deconv_padding);
            }
        }else if(node.dedialited_rate == 0){
            if(node.hasdeBias)
                node.deconv_desc  = desc(prop_kind::forward_inference, algorithm::deconvolution_direct,
                                            node.src_bottom_md, node.src_weights_md,
                                            node.deconv_bias_md, node.layer_top_md,
                                            node.deconv_strides, node.deconv_padding,
                                            node.deconv_padding);
            else{
                node.deconv_desc  = desc(prop_kind::forward_inference, algorithm::deconvolution_direct,
                                            node.src_bottom_md, node.src_weights_md,
                                            node.layer_top_md,
                                            node.deconv_strides, node.deconv_padding,
                                            node.deconv_padding);
            }
        }

        node.deconv_pdesc = deconvolution_forward::primitive_desc(node.deconv_desc, BrixLab::graph_eng);

        node.deconv_weights_memory = node.src_weights_memory;

        node.layer_top_memory = memory(node.convolution_pdesc.dst_desc(), BrixLab::graph_eng);
        node.op_args = {{DNNL_ARG_SRC, node.deconv_bottom_memory},
                {DNNL_ARG_WEIGHTS, node.deconv_weights_memory},
                {DNNL_ARG_BIAS, node.deconv_bias_memory},
                {DNNL_ARG_DST, node.layer_top_memory}};
        node.inference_forward = OP_deconvolution_inference_forward; 
        graph_insert(g_state, &node);
    }

    template<typename DType>
    void OP_innerproduct_inference_forward(const layerNode<DType> &node, graphState<DType> &state){
        // Reorder the data in case the weights memory layout generated by the
        // primitive and the one provided by the user are different. In this case,
        // we create additional memory objects with internal buffers that will
        // contain the reordered data.
        node.src_bottom_memory = state.input;
        if (node.inner_pdesc.weights_desc() != node.src_weights_memory.get_desc()) {
            node.inner_weights_memory = memory(node.inner_pdesc.weights_desc(), BrixLab::graph_eng);
            reorder(node.src_weights_memory, node.inner_weights_memory).execute(BrixLab::graph_stream, 
                                    node.src_bottom_memory, node.inner_weights_memory);
        }
        inner_product_forward(node.inner_pdesc).execute(BrixLab::graph_stream, node.op_args);
    }

    template<typename DType>
    void OP_innerproduct_layer_setup(const layerWeightsParam<DType> &param, 
                                                const graphState<DType> &g_state){
        layerNode<DType> node = {(OP_type)0};
        int k_c = param.inner_out;
        int data_formate = param.formate;
        int inHeight = param.inHeight;
        int inChannel = param.inChannel;
        int inWidth = param.inWidth;
        int inBatch = param.inBatch;
        node.layer_c = k_c;
        node.layer_n = inBatch;
        node.bottom_shape = {inBatch, inChannel, inHeight, inWidth};
        node.weights_shape = {k_c, inChannel, inHeight, inWidth};
        node.bias_shape = {k_c};
        // src bottom data
        
        node.src_bottom_md = memory::desc({node.bottom_shape}, dt::f32, tag::nchw);
        // weights & bias
        node.src_weights_memory =  memory({node.weights_shape, dt::f32, tag::oihw}, BrixLab::graph_eng);
        write_to_dnnl_memory<DType>(param.inner_weights, node.src_weights_memory);
        node.inner_weights_md = memory::desc({node.weights_shape}, dt::f32, tag::any);

        node.inner_bias_memory = memory({{node.bias_shape}, dt::f32, tag::x}, BrixLab::graph_eng);
        write_to_dnnl_memory<DType>(param.inner_bias, node.inner_bias_memory);
        node.inner_bias_md = memory::desc({node.bias_shape}, dt::f32, tag::any);

        node.top_shape = {inBatch, k_c};
        
        node.layer_top_md = memory::desc({node.top_shape}, dt::f32, tag::any);

        node.inner_desc = inner_product_forward::desc(prop_kind::forward_inference, node.src_bottom_md,
                    node.inner_weights_md, node.inner_bias_md, node.layer_top_md);
        
        // Create primitive post-ops (ReLU).
        const float scale = 1.0f;
        const float alpha = 0.f;
        const float beta = 0.f;
        post_ops inner_product_ops;
        inner_product_ops.append_eltwise( scale, algorithm::eltwise_relu, alpha, beta);
        primitive_attr inner_product_attr;
        inner_product_attr.set_post_ops(inner_product_ops);

        node.inner_weights_memory = node.src_weights_memory;

        node.inner_pdesc = inner_product_forward::primitive_desc(
                                                node.inner_desc, inner_product_attr, BrixLab::graph_eng);

        node.layer_top_memory = memory(node.top_shape, BrixLab::graph_eng);

        node.op_args = {{DNNL_ARG_SRC, node.src_bottom_memory},
                {DNNL_ARG_WEIGHTS, node.inner_weights_memory},
                {DNNL_ARG_BIAS, node.inner_bias_memory},
                {DNNL_ARG_DST, node.layer_top_memory}};
        node.inference_forward = OP_innerproduct_inference_forward; 
        graph_insert(g_state, &node);
    }

    template<typename DType>
    void OP_activation_inference_forward(const layerNode<DType> &node, graphState<DType> &state){
        node.src_bottom_memory = state.input;
        eltwise_forward(node.eltwise_pdesc).execute(BrixLab::graph_stream, node.op_args);
    }

    template<typename DType>
    void OP_activation_layer_setup(const layerWeightsParam<DType> &param, 
                                                const graphState<DType> &g_state){
        layerNode<DType> node = {(OP_type)0};
        int inHeight = param.inHeight;
        int inChannel = param.inChannel;
        int inWidth = param.inWidth;
        int inBatch = param.inBatch;
        node.layer_c = inChannel;
        node.layer_h = inHeight;
        node.layer_w = inWidth;
        node.layer_n = inBatch;
        //bottom_src memory
        node.bottom_shape = {inBatch, inChannel, inHeight, inWidth};
        node.src_bottom_md = memory::desc(node.bottom_shape, dt::f32, tag::nchw);
        

        // top &memory
        node.top_shape = {inBatch, inChannel, inHeight, inWidth};
        node.layer_top_md = memory::desc(node.top_shape, dt::f32, tag::nchw);
        node.layer_top_memory = memory(node.layer_top_md, BrixLab::graph_stream);

        //op
        node.activate_type = get_op_mapped_type(param.activate_type);
        node.alpha = param.alpha;
        node.beta = param.beta;
        node.eltwise_desc = eltwise_forward::desc(prop_kind::forward_inference,
                                node.activate_type, node.src_bottom_md, node.alpha, node.beta);
        node.eltwise_pdesc = eltwise_forward::primitive_desc(node.eltwise_desc, BrixLab::graph_eng);

        node.op_args = {
            {DNNL_ARG_SRC, node.src_bottom_memory},
            {DNNL_ARG_DST, node.layer_top_memory}
        };
        graph_insert(g_state, &node);
    }

    #define NODE_INTO_GRPAH(operation, param, graph) \
            OP_operation_layer_setup<int>(param, graph)

    #define LAYER_NODE_INFERENCE(operation, node, graph) \
            OP_operation_inference_forward(node, graph)

    template<typename DType>
    class NetGraph{
        public:
        int get_Graphsize() const;
        int get_GraphinWidth() const;
        int get_GraphinHeight() const;
        layerNode<DType> *getGraphOutput();
        NetGraph(const int &inH, const int &inW, const int &size);
        ~NetGraph();
        static engine graph_eng;
        static stream graph_stream;

        void network_predict();

        void make_graph(const std::string modelFile);
        private:
        int input_w;
        int input_h;
        graphState<DType> graph_state;
        int graph_size;
    };
} // namespace BrixLab

#endif