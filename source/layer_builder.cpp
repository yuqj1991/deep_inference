#include "layer_builder.hpp"
#include <assert.h>
namespace BrixLab
{
    template<typename DType>
    void OP_convolution_inference_forward(layerNode<DType> *node, graphSet<DType> &g_net){
        node->src_bottom_memory = g_net.input;
        if (node->conv_pdesc.src_desc() != node->src_bottom_memory.get_desc()) {
            auto temp_memory = memory(node->conv_pdesc.src_desc(), BrixLab::graph_eng);
            LOG_CHECK(temp_memory.get_data_handle() != nullptr, "varilfy the empty pointer") << "temp_memory.get_data_handle() should not be nullptr";
            std::unordered_map<int, memory> op_arg = {{DNNL_ARG_FROM, node->src_bottom_memory},
                                                      {DNNL_ARG_TO, temp_memory}};
            reorder(node->src_bottom_memory, temp_memory).execute(BrixLab::graph_stream, op_arg);
            node->src_bottom_memory = temp_memory;
        }
        node->op_args = {{DNNL_ARG_SRC, node->src_bottom_memory},
                        {DNNL_ARG_WEIGHTS, node->src_weights_memory},
                        {DNNL_ARG_BIAS, node->src_bias_memory},
                        {DNNL_ARG_DST, node->layer_top_memory}};
        convolution_forward(node->conv_pdesc).execute(BrixLab::graph_stream, node->op_args);
        LOG(DEBUG_INFO, "test_convolution_finished")<<"done!";
    }

    template<typename DType>
    layerNode<DType> OP_convolution_layer_setup(const layerWeightsParam<DType> &param){
        layerNode<DType> node(OP_type::CONVOLUTION);
        int k_w = param.k_w;
        int k_h = param.k_h;
        int k_c = param.k_c;
        int k_sX = param.stridesX;
        int k_sY = param.stridesY;
        int k_padXL = 0;
        int k_padXR = 0;
        int k_padYT = 0;
        int k_padYB = 0;
        QUANITIZED_TYPE quantized_type = param.quantized_type;
        memory::data_type dnnDataType = dt::f32;
        memory::data_type dnnDataBiasType = dt::f32;
        if(quantized_type == BrixLab::QUANITIZED_TYPE::UINT8_QUANTIZED){
            dnnDataType = dt::u8;
            dnnDataBiasType = dt::s32;
        }
        TENSOR_FORMATE data_formate = param.formate;
        int inHeight = param.inHeight;
        int inChannel = param.inChannel;
        int inWidth = param.inWidth;
        int inBatch = param.inBatch;
        node.dilateX = param.dilateX;
        node.dilateY = param.dilateY;
        int outWidth = floor((inWidth - (1 + (k_w - 1) * (node.dilateX + 1)) + k_padXL + k_padXR) / k_sX) + 1;
        int outHeight = floor((inHeight - ((1 + (k_h - 1) * (node.dilateX + 1))) + k_padYB + k_padYT) / k_sY) + 1; 
        if(param.padMode == PaddingType::PaddingSAME){
            outHeight    = std::ceil((inHeight) / k_sY); // oh = ceil(ih / stride)
            outWidth    = std::ceil((inWidth) / k_sX); // ow = ceil(iw / stride)
            int pad_width = ARGSMAX(0, ((outWidth - 1) * k_sX + ((k_w - 1) * (node.dilateX + 1) + 1) - inWidth) / 2);
            int pad_height = ARGSMAX(0, ((outHeight - 1) * k_sY + ((k_h - 1) * (node.dilateY + 1) + 1) - inHeight) / 2);
            k_padYT = std::floor(pad_height / 2);
            k_padXL = std::floor(pad_width / 2);
            k_padYB = pad_height - k_padXL;
            k_padXR = pad_width - k_padXR;
        }
        node.hasBias = param.hasBias;
        node.bottom_shape = {inBatch, inChannel, inHeight, inWidth};
        node.groups = param.groups >= 1 ? param.groups : 1;
        if(node.groups > 1){
            assert(inChannel % node.groups == 0);
            assert(k_c % node.groups == 0);
            int ICg = inChannel / node.groups;
            int OCg = k_c / node.groups;
            node.weights_shape = {node.groups, OCg, ICg, k_h, k_w};
        }else if(node.groups == 1){
            node.weights_shape = {k_c, inChannel, k_h, k_w};
        }
        node.conv_strides = {k_sY, k_sX};
        node.conv_paddingL = {k_padYT, k_padXL};
        node.conv_paddingR = {k_padYB, k_padXR};
        
        if(node.hasBias)
            node.bias_shape = {k_c};
        
        // src bottom_md
        node.src_bottom_md = memory::desc({node.bottom_shape}, dnnDataType, tag::nchw);
        // weights & bias
        node.src_weights_md = memory::desc({node.weights_shape}, dnnDataType, tag::any);
        node.src_weights_memory = memory({{node.weights_shape}, dnnDataType, tag::oihw}, BrixLab::graph_eng);
        write_to_dnnl_memory(param.conv_weights, node.src_weights_memory);
        if(node.hasBias){
            node.src_bias_md = memory::desc({node.bias_shape}, dnnDataBiasType, tag::any);
            node.src_bias_memory = memory({{node.bias_shape}, dnnDataBiasType, tag::x}, BrixLab::graph_eng);
            if(quantized_type == QUANITIZED_TYPE::UINT8_QUANTIZED){
                write_to_dnnl_memory(param.quantized_bias, node.src_bias_memory);
            }else if(quantized_type == QUANITIZED_TYPE::FLOAT32_REGULAR){
                write_to_dnnl_memory(param.conv_bias, node.src_bias_memory);
            }
        }
        // output feature shape
        
        node.top_shape = {inBatch, k_c, outHeight, outWidth};
        node.layer_top_md = memory::desc({node.top_shape}, dnnDataType, tag::any);
        node.layer_h = outHeight;
        node.layer_c = k_c;
        node.layer_w = outWidth;
        node.layer_n = inBatch;
        printf("[%s][line %d]layer info: outWidth: %d, outHeight: %d\n",__FUNCTION__, __LINE__, outWidth, outHeight);
        int dilate = ARGSMAX(node.dilateX, node.dilateY);
        if(dilate == 0){ 
            if(node.hasBias){
                auto convolution_desc = convolution_forward::desc(prop_kind::forward_inference,
                                        algorithm::convolution_direct, node.src_bottom_md, node.src_weights_md,
                                        node.src_bias_md, node.layer_top_md, node.conv_strides, 
                                        node.conv_paddingL, node.conv_paddingR);
                node.conv_pdesc = convolution_forward::primitive_desc(convolution_desc, BrixLab::graph_eng);
                printf("[%s][line %d] \n", __FUNCTION__, __LINE__);
            }else{
                auto convolution_desc = convolution_forward::desc(prop_kind::forward_inference, 
                                        algorithm::convolution_direct, node.src_bottom_md, 
                                        node.src_weights_md, node.layer_top_md, 
                                        node.conv_strides, node.conv_paddingL, node.conv_paddingR);
                node.conv_pdesc = convolution_forward::primitive_desc(convolution_desc, BrixLab::graph_eng);
            }
            
        }else if(dilate >= 1){
            memory::dims conv_dilates = {node.dilateX, node.dilateX};
            if(node.hasBias){
                auto convolution_desc = convolution_forward::desc(prop_kind::forward_inference,
                                          algorithm::convolution_direct,node.src_bottom_md, node.src_weights_md,
                                          node.src_bias_md, node.layer_top_md, node.conv_strides, conv_dilates,
                                          node.conv_paddingL, node.conv_paddingR);
                node.conv_pdesc = convolution_forward::primitive_desc(convolution_desc, BrixLab::graph_eng);
            }else{
                auto convolution_desc = convolution_forward::desc(prop_kind::forward_inference,
                                          algorithm::convolution_direct,node.src_bottom_md, node.src_weights_md,
                                          node.layer_top_md, node.conv_strides, conv_dilates,
                                          node.conv_paddingL, node.conv_paddingR);
                node.conv_pdesc = convolution_forward::primitive_desc(convolution_desc, BrixLab::graph_eng);
            }
            
        }

        if (node.conv_pdesc.weights_desc() != node.src_weights_memory.get_desc()) {
            auto temp_memory = memory(node.conv_pdesc.weights_desc(), BrixLab::graph_eng);
            reorder(node.src_weights_memory, temp_memory)
                    .execute(BrixLab::graph_stream, node.src_weights_memory, temp_memory);
            node.src_weights_memory = temp_memory;
        }

        node.layer_top_memory = memory(node.conv_pdesc.dst_desc(), BrixLab::graph_eng);
        node.inference_forward = OP_convolution_inference_forward; 
        return node;
    }
    INSTANCE_LAYEROP(convolution);
    
    template<typename DType>
    void OP_batchnorm_inference_forward(layerNode<DType> *node, graphSet<DType> &g_net){
        node->src_bottom_memory = g_net.input;
        batch_normalization_forward(node->batchnorm_pdesc).execute(BrixLab::graph_stream, node->op_args);
    }
        
    template<typename DType>
    layerNode<DType> OP_batchnorm_layer_setup(const layerWeightsParam<DType> &param){
        layerNode<DType> node(OP_type::BATCHNORM);
        int inHeight = param.inHeight;
        int inChannel = param.inChannel;
        int inWidth = param.inWidth;
        int inBatch = param.inBatch;
        node.layer_h = inHeight;
        node.layer_c = inChannel;
        node.layer_w = inWidth;
        node.layer_n = inBatch;
        node.bottom_shape = {inBatch, inChannel, inHeight, inWidth};
        node.batchnorm_scale_shift_shape = {2, inChannel};

        node.src_bottom_md = memory::desc(node.bottom_shape, dt::f32, tag::nchw);
        node.batchnorm_scale_shift_md = memory::desc(node.batchnorm_scale_shift_shape, dt::f32, tag::nc);

        node.batchnorm_scale_shift_memory = memory(node.batchnorm_scale_shift_md, BrixLab::graph_eng);
        
        
        write_to_dnnl_memory(param.b_shift_scale, node.batchnorm_scale_shift_memory);

        // Create operation descriptor.
        dnnl::batch_normalization_forward::desc batchnorm_desc = batch_normalization_forward::desc(
                                    prop_kind::forward_inference, node.src_bottom_md, 1.e-10f,
                                    normalization_flags::use_scale_shift
                                    | normalization_flags::use_global_stats);

        // Create primitive descriptor.
        ;
        node.batchnorm_pdesc = batch_normalization_forward::primitive_desc(batchnorm_desc, BrixLab::graph_eng);

        // Create memory objects using memory descriptors created by the primitive
        // descriptor: mean, variance.
        
        node.batchnorm_mean_memory = memory(node.batchnorm_pdesc.mean_desc(), BrixLab::graph_eng);
        node.batchnorm_variance_memory = memory(node.batchnorm_pdesc.variance_desc(), BrixLab::graph_eng);
        write_to_dnnl_memory(param.b_means, node.batchnorm_mean_memory);
        write_to_dnnl_memory(param.b_variance, node.batchnorm_variance_memory);

        node.op_args ={{DNNL_ARG_SRC, node.src_bottom_memory},
                            {DNNL_ARG_MEAN, node.batchnorm_mean_memory},
                            {DNNL_ARG_VARIANCE, node.batchnorm_variance_memory},
                            {DNNL_ARG_SCALE_SHIFT, node.batchnorm_scale_shift_memory},
                            {DNNL_ARG_DST, node.src_bottom_memory}};
        node.layer_top_memory = node.src_bottom_memory;
        node.inference_forward = OP_batchnorm_inference_forward;
        return node;
    }
    INSTANCE_LAYEROP(batchnorm);

    template<typename DType>   
    void OP_pooling_inference_forward(layerNode<DType> *node, graphSet<DType> &g_net){
         node->src_bottom_memory = g_net.input;
        if(node->p_dialiated == 0){
            pooling_forward(node->pooling_pdesc_without_d).execute(BrixLab::graph_stream, node->op_args);
        }else if(node->p_dialiated >= 1){
            pooling_v2_forward(node->pooling_pdesc).execute(BrixLab::graph_stream, node->op_args);
        }
    }
    template<typename DType>    
    layerNode<DType> OP_pooling_layer_setup(const layerWeightsParam<DType> &param){
        layerNode<DType> node(OP_type::POOLING);
        int inHeight = param.inHeight;
        int inChannel = param.inChannel;
        int inWidth = param.inWidth;
        int inBatch = param.inBatch;
        node.layer_h = inHeight;
        node.layer_c = inChannel;
        node.layer_w = inWidth;
        node.layer_n = inBatch;
        int k_w = param.p_kernelsX;
        int k_h = param.p_kernelsY;
        int k_s = param.p_stridesX;
        int k_padX = param.p_paddingX;
        int k_padY = param.p_paddingY;
        node.bottom_shape = {inBatch, inChannel, inHeight, inWidth};
        int dilatedX = param.p_dilatedX;
        int dilatedY = param.p_dilatedY;
        int dilated = ARGSMIN(dilatedX, dilatedY);
        PoolingType p_type = param.pooling_type;
        node.pooling_type = get_op_mapped_pooling_type(p_type);
        int outHight = 0;
        int outWidth = 0;
        if(dilated == 0){
            outHight = (inHeight - k_h + 2 *k_padX) / k_s + 1;
            outWidth = (inWidth - k_w + 2 * k_padY) / k_s + 1;
        }else if(dilated > 0){
            outHight = (inHeight - ((k_h - 1) * dilatedY + k_h) +2 *k_padX) / k_s + 1;
            outWidth = (inWidth - ((k_w - 1) * dilatedX + k_w) +2 *k_padY) / k_s + 1;
        }
        
        node.pooling_kernel = {k_w, k_h};
        node.pooling_strides = {k_s, k_s};
        node.pooling_padding = {k_padX, k_padY};
        node.pooling_dialiate = {dilatedX, dilatedY};
        
        node.top_shape = {inBatch, inChannel, outHight, outWidth};
        node.layer_top_md = memory::desc(node.top_shape, dt::f32, tag::nchw);
        node.layer_top_memory = memory(node.layer_top_md, BrixLab::graph_eng);
        node.src_bottom_md = memory::desc(node.bottom_shape, dt::f32, tag::nchw);
        if(dilated > 0){
            dnnl::pooling_v2_forward::desc pooling_desc = pooling_v2_forward::desc(prop_kind::forward_inference,
                                            node.pooling_type, node.src_bottom_md, 
                                            node.layer_top_md,
                                            node.pooling_strides, node.pooling_kernel,
                                            node.pooling_dialiate, node.pooling_padding, 
                                            node.pooling_padding);
            node.pooling_pdesc = pooling_v2_forward::primitive_desc(pooling_desc, BrixLab::graph_eng);
        }else if(dilated == 0){
            dnnl::pooling_forward::desc pooling_desc_without_d = pooling_forward::desc(prop_kind::forward_inference, 
                                                node.pooling_type, node.src_bottom_md, 
                                                node.layer_top_md, node.pooling_strides, 
                                                node.pooling_kernel,
                                                node.pooling_padding,node.pooling_padding);
            node.pooling_pdesc_without_d = pooling_forward::primitive_desc(pooling_desc_without_d, BrixLab::graph_eng);
        }
        node.op_args = {{DNNL_ARG_SRC, node.src_bottom_memory},
                        {DNNL_ARG_DST, node.layer_top_memory}};
        node.inference_forward = OP_pooling_inference_forward;
        return node;
    }
    INSTANCE_LAYEROP(pooling);

    template<typename DType>     
    void OP_concat_inference_forward(layerNode<DType> *node, graphSet<DType> &g_net){
        concat(node->concat_pdesc).execute(BrixLab::graph_stream, node->op_args);
    }
    template<typename DType>
    layerNode<DType> OP_concat_layer_setup(const layerWeightsParam<DType> &param){
        layerNode<DType> node(OP_type::CONCAT);
        int inHeight = param.inHeight;
        int inChannel = param.inChannel;
        int inWidth = param.inWidth;
        int inBatch = param.inBatch;
        node.bottom_shape = {inBatch, inChannel, inHeight, inWidth};
        node.concat_num = param.concat_num;
        node.concat_axis = param.concat_axis;
        node.concat_index = (int*)xcalloc(node.concat_num, sizeof(int));
        node.op_type = param.op_type;
        /*for(int ii = 0; ii < node.concat_num; ii++){
            node.concat_index[ii] = param.concat_index[ii];
            node.concat_bottom_md.push_back(g_net[ii]->layer_top_md);
            node.concat_bottom_memory.push_back(g_net[ii]->layer_top_memory);
            node.op_args.insert({DNNL_ARG_MULTIPLE_SRC + ii, g_net[ii]->layer_top_memory});
        }*/
        
        node.op_args.insert({{DNNL_ARG_DST, node.layer_top_memory}});
        node.concat_pdesc = concat::primitive_desc(node.concat_axis, node.concat_bottom_md, BrixLab::graph_eng);
        node.layer_top_md = node.concat_pdesc.dst_desc();
        node.layer_top_memory = memory(node.layer_top_md, BrixLab::graph_eng);
        node.inference_forward = OP_concat_inference_forward;
        return node;
    }
    INSTANCE_LAYEROP(concat);

    template<typename DType>    
    void OP_sum_inference_forward(layerNode<DType> *node, graphSet<DType> &g_net){
        sum(node->sum_pdesc).execute(BrixLab::graph_stream, node->op_args);
    }

    template<typename DType>    
    layerNode<DType> OP_sum_layer_setup(const layerWeightsParam<DType> &param){
        layerNode<DType> node(OP_type::ELTWISE);
        node.sum_num = param.sum_num;
        node.sum_index = (int*)xcalloc(node.sum_num, sizeof(int));
        /*
        int inHeight = g_net[param.sum_index[0]]->layer_h;
        int inWidth = g_net[param.sum_index[0]]->layer_w;
        int inChannel = g_net[param.sum_index[0]]->layer_c;
        int inBatch = g_net[param.sum_index[0]]->layer_n;
        memory::dims S_Shape = {inBatch, inChannel, inWidth, inHeight};
        for(int ii = 0; ii < node.sum_num; ii++){
            node.sum_index[ii] = param.sum_index[ii];
            node.sum_scale[ii] = DType(1.0);
            checK_equal_dims(S_Shape, g_net[ii]->top_shape);
            node.sum_bottom_memory.push_back(g_net[ii]->layer_top_memory);
            node.sum_bottom_md.push_back(g_net[ii]->layer_top_md);
            node.op_args.insert({DNNL_ARG_MULTIPLE_SRC + ii, node.sum_bottom_memory[ii]});
        }
        node.sum_pdesc = sum::primitive_desc(node.sum_scale, node.sum_bottom_md, BrixLab::graph_eng);
        node.top_shape = {inBatch, inChannel, inWidth, inHeight};
        node.layer_top_md = memory::desc(node.top_shape, dt::f32, tag::nchw);
        node.layer_top_memory = memory(node.sum_pdesc.dst_desc(), BrixLab::graph_eng);
        node.op_args.insert({DNNL_ARG_DST, node.layer_top_memory});
        for(int ii = 0; ii < node.sum_num; ii++){
            node.op_args.insert({DNNL_ARG_MULTIPLE_SRC + ii, node.sum_bottom_memory[ii]});
        }
        node.inference_forward = OP_sum_inference_forward;
        */
        return node;
    }
    INSTANCE_LAYEROP(sum);
    
    template<typename DType>
    void OP_resample_inference_forward(layerNode<DType> *node, graphSet<DType> &g_net){
        node->src_bottom_memory = g_net.input;
        resampling_forward(node->resample_pdesc).execute(BrixLab::graph_stream, node->op_args);
    }
    template<typename DType>
    layerNode<DType> OP_resample_layer_setup(const layerWeightsParam<DType> &param){
        int inHeight = param.inHeight;
        int inWidth = param.inWidth;
        int inChannel = param.inChannel;
        int inBatch = param.inBatch;
        layerNode<DType> node(OP_type::RESAMPLING);
        node.bottom_shape = {inBatch, inChannel, inHeight, inWidth};
        node.op_type = param.op_type;
        node.src_bottom_md = memory::desc(node.bottom_shape, dt::f32, tag::nchw);
        int outHeight = int(inHeight * param.adjust_height_scale);
        int outWidth = int(inWidth * param.adjust_width_scale);
        node.top_shape = {inBatch, inChannel, outHeight, outWidth};
        node.layer_top_md = memory::desc(node.top_shape, dt::f32, tag::nchw);
        node.layer_top_memory = memory(node.layer_top_md, BrixLab::graph_eng);

        dnnl::resampling_forward::desc resample_desc = resampling_forward::desc(prop_kind::forward_inference,
            algorithm::resampling_linear, node.src_bottom_md, node.layer_top_md);
        node.resample_pdesc = resampling_forward::primitive_desc(resample_desc, BrixLab::graph_eng);

        node.op_args.insert({DNNL_ARG_SRC, node.src_bottom_memory});
        node.op_args.insert({DNNL_ARG_DST, node.layer_top_memory});
        node.inference_forward = OP_resample_inference_forward;
        return node;
    }
    INSTANCE_LAYEROP(resample);
    
    template<typename DType>
    void OP_deconvolution_inference_forward(layerNode<DType> *node, graphSet<DType> &g_net){
        node->src_bottom_memory = g_net.input;
        
        if (node->deconv_pdesc.src_desc() != node->src_bottom_memory.get_desc()) {
            auto temp_memory = memory(node->deconv_pdesc.src_desc(), BrixLab::graph_eng);
            std::unordered_map<int, memory> op_arg = {{DNNL_ARG_FROM, node->src_bottom_memory},
                    {DNNL_ARG_TO, temp_memory}};
            reorder(node->src_bottom_memory, temp_memory).execute(BrixLab::graph_stream, op_arg);
            node->src_bottom_memory = temp_memory;
            printf("[OP_deconvolution_inference_forward] reorder bottom memory! \n");
        }
        node->op_args = {{DNNL_ARG_SRC, node->src_bottom_memory},
                        {DNNL_ARG_WEIGHTS, node->src_weights_memory},
                        {DNNL_ARG_BIAS, node->src_bias_memory},
                        {DNNL_ARG_DST, node->layer_top_memory}};
        deconvolution_forward(node->deconv_pdesc).execute(BrixLab::graph_stream, node->op_args);
        printf("[OP_deconvolution_inference_forward] done!\n");
    }    

    template<typename DType>
    layerNode<DType> OP_deconvolution_layer_setup(const layerWeightsParam<DType> &param){
        printf("start deconvolution set_up!\n");
        layerNode<DType> node(OP_type::DECONVOLUTION);
        int k_w = param.k_w;
        int k_h = param.k_h;
        int k_c = param.k_c;
        int k_sX = param.stridesX;
        int k_sY = param.stridesY;
        int k_padXL = 0;
        int k_padXR = 0;
        int k_padYT = 0;
        int k_padYB = 0;
        TENSOR_FORMATE data_formate = param.formate;
        QUANITIZED_TYPE quantized_type = param.quantized_type;
        memory::data_type dnnDataType = dt::f32;
        memory::data_type dnnDataBiasType = dt::f32;
        if(quantized_type == BrixLab::QUANITIZED_TYPE::UINT8_QUANTIZED){
            dnnDataType = dt::u8;
            dnnDataBiasType = dt::s32;
        }
        int inHeight = param.inHeight;
        int inChannel = param.inChannel;
        int inWidth = param.inWidth;
        int inBatch =param.inBatch;
        
        node.bottom_shape = {inBatch, inChannel, inHeight, inWidth};
        node.weights_shape = {k_c, inChannel, k_w, k_h};
        node.deconv_strides = {k_sX, k_sY};
        node.deconv_paddingL = {k_padYT, k_padXL};
        node.deconv_paddingL = {k_padYB, k_padXR};
        node.dilateX = param.dilateX;
        node.dilateY = param.dilateY;
        node.hasBias = param.hasBias;
        if(node.hasBias)
            node.bias_shape = {k_c};
        // src bottom data
        node.src_bottom_md = memory::desc({node.bottom_shape}, dnnDataType, tag::any);
        // weights & bias
        node.src_weights_memory = memory({{node.weights_shape}, dnnDataType, tag::oihw}, BrixLab::graph_eng);
        printf("start to writo to memory!\n");
        write_to_dnnl_memory(param.transposed_weights, node.src_weights_memory);
        node.src_weights_md = memory::desc({node.weights_shape}, dnnDataType, tag::any);
        if(node.hasBias){
            node.src_bias_memory = memory({{node.bias_shape}, dnnDataBiasType, tag::x}, BrixLab::graph_eng);
            node.src_bias_md = memory::desc({node.bias_shape}, dnnDataBiasType, tag::any);
            if(quantized_type == QUANITIZED_TYPE::UINT8_QUANTIZED){
                write_to_dnnl_memory(param.quantized_bias, node.src_bias_memory);
            }else if(quantized_type == QUANITIZED_TYPE::FLOAT32_REGULAR){
                write_to_dnnl_memory(param.transposed_bias, node.src_bias_memory);
            }
        }

        int D_kHeight = 1 + (k_h -  1) * (node.dilateX + 1);
        int D_kWidth = 1 + (k_w - 1) * (node.dilateY + 1);

        int outWidth = int((inWidth - 1) * k_sX + D_kWidth - (k_padXL + k_padXR));
        int outHeight = int((inHeight - 1) * k_sY + D_kHeight - (k_padYT + k_padYB));
        node.top_shape = {inBatch, k_c, outHeight, outWidth};

        node.layer_c = k_c;
        node.layer_h = outHeight;
        node.layer_n = inBatch;
        node.layer_w = outWidth;

        node.layer_top_md = memory::desc({node.top_shape}, dt::f32, tag::nchw);
        int dilate = ARGSMAX(node.dilateX, node.dilateY);
        if(dilate > 0){
            memory::dims deconv_deliatd = {node.dilateX, node.dilateX};
            if(node.hasBias){
                auto deconv_desc  = deconvolution_forward::desc(prop_kind::forward_inference, algorithm::deconvolution_direct,
                                            node.src_bottom_md, node.src_weights_md,
                                            node.src_bias_md, node.layer_top_md,
                                            node.deconv_strides, deconv_deliatd,
                                            node.deconv_paddingL,node.deconv_paddingR);
                node.deconv_pdesc = deconvolution_forward::primitive_desc(deconv_desc, BrixLab::graph_eng);

            }else{
                auto deconv_desc  = deconvolution_forward::desc(prop_kind::forward_inference, algorithm::deconvolution_direct,
                                            node.src_bottom_md, node.src_weights_md,
                                            node.layer_top_md,
                                            node.deconv_strides, deconv_deliatd,
                                            node.deconv_paddingL, node.deconv_paddingR);
                node.deconv_pdesc = deconvolution_forward::primitive_desc(deconv_desc, BrixLab::graph_eng);
            }
        }else if(dilate == 0){
            if(node.hasBias){
                auto deconv_desc  = deconvolution_forward::desc(prop_kind::forward_inference, algorithm::deconvolution_direct,
                                            node.src_bottom_md, node.src_weights_md,
                                            node.src_bias_md, node.layer_top_md,
                                            node.deconv_strides, node.deconv_paddingL,
                                            node.deconv_paddingR);
                node.deconv_pdesc = deconvolution_forward::primitive_desc(deconv_desc, BrixLab::graph_eng);
            }else{
                auto deconv_desc  = deconvolution_forward::desc(prop_kind::forward_inference, algorithm::deconvolution_direct,
                                            node.src_bottom_md, node.src_weights_md,
                                            node.layer_top_md,
                                            node.deconv_strides, node.deconv_paddingL,
                                            node.deconv_paddingR);
                node.deconv_pdesc = deconvolution_forward::primitive_desc(deconv_desc, BrixLab::graph_eng);
            }
        }

        if (node.deconv_pdesc.weights_desc() != node.src_weights_memory.get_desc()) {
            auto temp_memory = memory(node.deconv_pdesc.weights_desc(), BrixLab::graph_eng);
            reorder(node.src_weights_memory, temp_memory)
                    .execute(BrixLab::graph_stream, node.src_weights_memory, temp_memory);
            node.src_weights_memory = temp_memory;
        }
       
        node.layer_top_memory = memory(node.deconv_pdesc.dst_desc(), BrixLab::graph_eng);
        
        node.inference_forward = OP_deconvolution_inference_forward;
        return node;
    }
    INSTANCE_LAYEROP(deconvolution);
    
    template<typename DType>
    void OP_innerproduct_inference_forward(layerNode<DType> *node, graphSet<DType> &g_net){
        // Reorder the data in case the weights memory layout generated by the
        // primitive and the one provided by the user are different. In this case,
        // we create additional memory objects with internal buffers that will
        // contain the reordered data.
        node->src_bottom_memory = g_net.input;
        if (node->inner_pdesc.weights_desc() != node->src_weights_memory.get_desc()) {
            node->src_weights_memory = memory(node->inner_pdesc.weights_desc(), BrixLab::graph_eng);
            reorder(node->src_weights_memory, node->src_weights_memory).execute(BrixLab::graph_stream, 
                                    node->src_bottom_memory, node->src_weights_memory);
        }
        inner_product_forward(node->inner_pdesc).execute(BrixLab::graph_stream, node->op_args);
    }
   
    template<typename DType>
    layerNode<DType> OP_innerproduct_layer_setup(const layerWeightsParam<DType> &param){
        layerNode<DType> node(OP_type::INNERPRODUCT);
        int k_c = param.k_c;
        TENSOR_FORMATE data_formate = param.formate;
        int inHeight = param.inHeight;
        int inChannel = param.inChannel;
        int inWidth = param.inWidth;
        int inBatch = param.inBatch;
        node.layer_c = k_c;
        node.layer_n = inBatch;
        node.bottom_shape = {inBatch, inChannel, inHeight, inWidth};
        node.weights_shape = {k_c, inChannel, inHeight, inWidth};
        node.bias_shape = {k_c};
        QUANITIZED_TYPE quantized_type = param.quantized_type;
        memory::data_type dnnDataType = dt::f32;
        memory::data_type dnnDataBiasType = dt::f32;
        if(quantized_type == BrixLab::QUANITIZED_TYPE::UINT8_QUANTIZED){
            dnnDataType = dt::u8;
            dnnDataBiasType = dt::s32;
        }
        // src bottom data
        
        node.src_bottom_md = memory::desc({node.bottom_shape}, dnnDataType, tag::nchw);
        // weights & bias
        node.src_weights_memory =  memory({node.weights_shape, dnnDataType, tag::oihw}, BrixLab::graph_eng);
        write_to_dnnl_memory(param.innerWeights, node.src_weights_memory);
        node.src_weights_md = memory::desc({node.weights_shape}, dnnDataType, tag::any);

        node.src_bias_memory = memory({{node.bias_shape}, dnnDataBiasType, tag::x}, BrixLab::graph_eng);
        node.src_bias_md = memory::desc({node.bias_shape}, dnnDataBiasType, tag::any);

        if(quantized_type == QUANITIZED_TYPE::UINT8_QUANTIZED){
            write_to_dnnl_memory(param.quantized_bias, node.src_bias_memory);
        }else if(quantized_type == QUANITIZED_TYPE::FLOAT32_REGULAR){
            write_to_dnnl_memory(param.innerBias, node.src_bias_memory);
        }

        node.top_shape = {inBatch, k_c};
        
        node.layer_top_md = memory::desc({node.top_shape}, dt::f32, tag::any);

        dnnl::inner_product_forward::desc inner_desc = inner_product_forward::desc(prop_kind::forward_inference, node.src_bottom_md,
                    node.src_weights_md, node.src_bias_md, node.layer_top_md);
        
        // Create primitive post-ops (like ReLU).
        const DType scale = 1.0f;
        const DType alpha = 0.f;
        const DType beta = 0.f;
        post_ops inner_product_ops;
        inner_product_ops.append_eltwise( scale, algorithm::eltwise_relu, alpha, beta);
        primitive_attr inner_product_attr;
        inner_product_attr.set_post_ops(inner_product_ops);

        node.src_weights_memory = node.src_weights_memory;

        node.inner_pdesc = inner_product_forward::primitive_desc(
                                                inner_desc, inner_product_attr, BrixLab::graph_eng);

        node.layer_top_memory = memory(node.inner_pdesc.dst_desc(), BrixLab::graph_eng);

        node.op_args = {{DNNL_ARG_SRC, node.src_bottom_memory},
                {DNNL_ARG_WEIGHTS, node.src_weights_memory},
                {DNNL_ARG_BIAS, node.src_bias_memory},
                {DNNL_ARG_DST, node.layer_top_memory}};
        node.inference_forward = OP_innerproduct_inference_forward; 
        return node;
    }
    INSTANCE_LAYEROP(innerproduct);

    template<typename DType>
    void OP_activation_inference_forward(layerNode<DType> *node, graphSet<DType> &g_net){
        node->src_bottom_memory = g_net.input;
        eltwise_forward(node->eltwise_pdesc).execute(BrixLab::graph_stream, node->op_args);
    }
    template<typename DType>
    layerNode<DType> OP_activation_layer_setup(const layerWeightsParam<DType> &param){
        layerNode<DType> node(OP_type::ACTIVITION);
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
        node.layer_top_memory = memory(node.layer_top_md, BrixLab::graph_eng);

        //op
        node.activate_type = get_op_mapped_activition_type(param.activate_type);
        node.alpha = param.alpha;
        node.beta = param.beta;
        dnnl::eltwise_forward::desc eltwise_desc = eltwise_forward::desc(prop_kind::forward_inference,
                                node.activate_type, node.src_bottom_md, node.alpha, node.beta);
        node.eltwise_pdesc = eltwise_forward::primitive_desc(eltwise_desc, BrixLab::graph_eng);

        node.op_args = {
            {DNNL_ARG_SRC, node.src_bottom_memory},
            {DNNL_ARG_DST, node.layer_top_memory}
        };
        return node;
    }
    INSTANCE_LAYEROP(activation);

    template<typename DType>
    NetGraph<DType>::NetGraph(const int &inH, const int &inW, const int &size, 
                const std::string &tflite_path, const memory &input):input_w(inW), input_h(inH), 
                graph_state(graphSet<DType>(0, 0, input)), graph_size(size),_tflite_model(nullptr), 
                tflite_file(tflite_path){
    }

    template<typename DType>
    int NetGraph<DType>::get_Graphsize() const{
        return graph_size;
    }

    template<typename DType>
    int NetGraph<DType>::get_GraphinWidth() const{
        return input_w;
    }

    template<typename DType>
    int NetGraph<DType>::get_GraphinHeight() const{
        return input_h;
    }

    template<typename DType>
    void NetGraph<DType>::network_predict(){
        //int size = graph_state.graphSize;
        int layer_count = 0;
        layerNode<DType> *layer_node = graph_state.head;
        while(layer_node != nullptr){
            OP_type type= layer_node->op_type;
            std::string OP_name = get_mapped_op_string(type);
            layer_node->inference_forward(layer_node, graph_state);
            graph_state.input = layer_node->layer_top_memory;
            layer_node = layer_node->next;
            layer_count++;
        }
    }

    template<typename DType>
    layerNode<DType>* NetGraph<DType>::getGraphOutput(){
        if(graph_state.current_index <= graph_size){
            
        }
        return graph_state.current;
    }

    template<typename DType>
    void NetGraph<DType>::make_graph(const std::vector<layerWeightsParam<DType> > &params, const int &layer_size){

    }

    template<typename DType>
    void NetGraph<DType>::make_netParamfromTflite(const std::string &tflite_file){

        std::ifstream inputFile(tflite_file, std::ios::binary);
        inputFile.seekg(0, std::ios::end);
        const auto size = inputFile.tellg();
        inputFile.seekg(0, std::ios::beg);

        char* buffer = new char[size];
        inputFile.read(buffer, size);
        inputFile.close();

        // verify model
        flatbuffers::Verifier verify((uint8_t*)buffer, size);
        if (!tflite::VerifyModelBuffer(verify)) {
            std::cout << "TFlite model version ERROR!";
        }

        _tflite_model = tflite::UnPackModel(buffer);
        delete[] buffer;
    }
    template<typename DType>
    NetT<DType> NetGraph<DType>::tfliteConvertGraphList(){
        if(_tflite_model == nullptr){
            make_netParamfromTflite(tflite_file);
        }
        NetT<DType> g_net;
        const auto& tfliteOpSet = _tflite_model->operator_codes;
        const auto subGraphsSize      = _tflite_model->subgraphs.size();
        const auto& tfliteModelBuffer = _tflite_model->buffers;

        // check whether this tflite model is quantization model
        // use the weight's data type of Conv2D|DepthwiseConv2D to decide quantizedModel mode
        bool quantizedModel = true;
        for (unsigned int i = 0; i < subGraphsSize; ++i) {
            const auto& ops     = _tflite_model->subgraphs[i]->operators;
            const auto& tensors = _tflite_model->subgraphs[i]->tensors;
            const int opNums    = ops.size();
            for (int j = 0; j < opNums; ++j) {
                const int opcodeIndex = ops[j]->opcode_index;
                const auto opCode     = tfliteOpSet[opcodeIndex]->builtin_code;
                if (opCode == tflite::BuiltinOperator_CONV_2D || opCode == tflite::BuiltinOperator_DEPTHWISE_CONV_2D) {
                    const int weightIndex    = ops[j]->inputs[1];
                    const auto& weightTensor = tensors[weightIndex];
                    quantizedModel           = weightTensor->type == tflite::TensorType_UINT8;
                    if(weightTensor->type == tflite::TensorType_INT8){
                        LOG(FATAL_ERROR, "varify_Tflite_UINT*")<< "***DO NOT SUPPORT Tflite [INT8] quantized model now***";
                    }
                    if (!quantizedModel)
                        break;
                }
            }
        }
        auto& buffers = _tflite_model->buffers;

        for (unsigned int i = 0; i < subGraphsSize; ++i) {
            const auto& ops     = _tflite_model->subgraphs[i]->operators;
            const auto& tensors = _tflite_model->subgraphs[i]->tensors;
            // set const
            std::vector<bool> extractedTensors(_tflite_model->subgraphs[i]->tensors.size(), false);
            // set input, maybe the inputs size should be 1 in one subgraphs.
            for (const auto index : _tflite_model->subgraphs[i]->inputs) {
                layerWeightsParam<DType> input_OpT;
                const auto& inputTensor = tensors[index];
                input_OpT.node_name           = inputTensor->name;
                input_OpT.op_type           = OP_type::DATA_INPUTS;
                input_OpT.formate = TENSOR_FORMATE::NHWC;
                input_OpT.inBatch = inputTensor->shape[0];
                input_OpT.inChannel = inputTensor->shape[3];
                input_OpT.inHeight = inputTensor->shape[1];
                input_OpT.inWidth = inputTensor->shape[2];
                g_net.layer_ops.emplace_back(input_OpT);
            }
            // set output names
            for (unsigned int k = 0; k < _tflite_model->subgraphs[i]->outputs.size(); ++k) {
                g_net.output_name.push_back(tensors[_tflite_model->subgraphs[i]->outputs[k]]->name);
            }
            // tensor names
            for (const auto& tensor : tensors) {
                g_net.tensorName.push_back(tensor->name);
            }

            const int opNums = ops.size();
            for (int j = 0; j < opNums; ++j) {
                const int opcodeIndex = ops[j]->opcode_index;
                const auto opCode     = tfliteOpSet[opcodeIndex]->builtin_code;
                #if 0
                if (needExtractInput(opCode)) {
                    for (auto input : ops[j]->inputs) {
                        if (extractedTensors[input]) {
                            continue;
                        }
                        extractedTensors[input] = true;
                        auto& tensor = _tflite_model->subgraphs[i]->tensors[input];
                        auto& buffer = buffers[tensor->buffer];
                        if (buffer->data.empty()) {
                            continue;
                        }
                        std::unique_ptr<OpT> newOp(new OpT);
                        newOp->type = OpType_Const;
                        newOp->name = tensor->name;
                        newOp->outputIndexes = {input};
                        newOp->main.type = OpParameter_Blob;
                        newOp->main.value = new BlobT;
                        auto blob = newOp->main.AsBlob();
                        blob->dims = tensor->shape;
                        blob->dataFormat = DATA_FORMATE::NHWC;
                        blob->dataType = tflite_dataTypeMap(tensor->type);
                        int size = 1;
                        for (auto s : blob->dims) {
                            size *= s;
                        }
                        void* dst = nullptr;
                        switch (blob->dataType) {
                            case DataType_DT_DType:
                                blob->DType32s.resize(size);
                                dst = blob->DType32s.data();
                                break;
                            case DataType_DT_INT32:
                                blob->int32s.resize(size);
                                dst = blob->int32s.data();
                                break;
                            case DataType_DT_INT8:
                                blob->int8s.resize(size);
                                dst = blob->int8s.data();
                                break;
                            case DataType_DT_UINT8:
                                blob->uint8s.resize(size);
                                dst = blob->uint8s.data();
                                break;
                            default:
                                break;
                        }
                        ::memcpy(dst, buffer->data.data(), buffer->data.size());
                        MNNNetT->oplists.emplace_back(std::move(newOp));
                    }
                }
                #endif
                #if 0
                layerWeightsParam<DType> New_OP;
                auto creator = liteOpConvertMapKit::get()->search(opCode);
                DCHECK(creator) << "NOT_SUPPORTED_OP: [ " << tflite::EnumNameBuiltinOperator(opCode) << " ]";

                // tflite op to MNN op
                op->name      = tensors[ops[j]->outputs[0]]->name;
                op->type      = creator->opType(quantizedModel);
                op->main.type = creator->type(quantizedModel);
                // set default input output index
                op->inputIndexes.resize(ops[j]->inputs.size());
                op->outputIndexes.resize(ops[j]->outputs.size());
                for (int i = 0; i < ops[j]->inputs.size(); i++) {
                    op->inputIndexes[i] = ops[j]->inputs[i];
                }
                for (int i = 0; i < ops[j]->outputs.size(); i++) {
                    op->outputIndexes[i] = ops[j]->outputs[i];
                }
                // Run actual conversion
                creator->run(op, ops[j], tensors, tfliteModelBuffer, tfliteOpSet, quantizedModel);
                MNNNetT->oplists.emplace_back(op);
                #endif
            }
        }
        
        return g_net;
    }

    INSTANEC_CLASSNET(NetGraph);



    LayerSetup getSetupFunc(const std::string &func_name){
        if(func_name == "OP_convolution_layer_setup"){
            return OP_convolution_layer_setup;
        }else if(func_name == "OP_deconvolution_layer_setup"){
            return OP_deconvolution_layer_setup;
        }
    }

} // namespace BrixLab

