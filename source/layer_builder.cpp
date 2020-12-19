#include "layer_builder.hpp"
#include <assert.h>
namespace BrixLab
{
    template<typename DType>
    void OP_convolution_inference_forward(layerNode<DType> *node, graphSet<DType> &g_net){
        node->src_bottom_memory = g_net.input;
        if (node->conv_pdesc.src_desc() != node->src_bottom_memory.get_desc()) {
            auto temp_memory = memory(node->conv_pdesc.src_desc(), BrixLab::graph_eng);
            LOG_CHECK(temp_memory.get_data_handle() != nullptr, "empty pointer") << "temp_memory should not be nullptr";
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
        int k_w     = param.k_w;
        int k_h     = param.k_h;
        int k_c     = param.k_c;
        int k_sX    = param.stridesX;
        int k_sY    = param.stridesY;
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
        LOG_CHECK(param.in_shapes.size()==1, "CHECK_INPUTS");
        LOG_CHECK(param.out_shapes.size()==1, "CHECK_OUTPUTS");
        node.in_shapes.resize(1);
        int inHeight    = param.in_shapes[0].Height;
        int inChannel   = param.in_shapes[0].Channel;
        int inWidth     = param.in_shapes[0].Width;
        int inBatch     = param.in_shapes[0].Batch;
        int dilateX     = param.dilateX;
        int dilateY     = param.dilateY;
        bool dilate     = false;
        LOG_CHECK(dilateX >= 0, "CHECK dilateX >= 0");
        LOG_CHECK(dilateY >= 0, "CHECK dilateY >= 0");
        
        if(dilateX == 0 && dilateY == 0){
            dilate = false;
        }else{
            dilate = true;
        }
        int DkernelX        = 1 + (k_w - 1) * (node.dilateX + 1);
        int DkernelY        = 1 + (k_h - 1) * (node.dilateY + 1);
        int outWidth        = floor((inWidth - DkernelX + k_padXL + k_padXR) / k_sX) + 1;
        int outHeight       = floor((inHeight - DkernelY + k_padYB + k_padYT) / k_sY) + 1; 
        node.out_shapes.resize(1);
        int paramOutHeight  = param.out_shapes[0].Height;
        int paramOutWidth   = param.out_shapes[0].Width;
        if(param.padMode == PaddingType::PaddingSAME){
            outHeight       = std::ceil((inHeight) / k_sY); // oh = ceil(ih / stride)
            outWidth        = std::ceil((inWidth) / k_sX); // ow = ceil(iw / stride)
            int pad_width   = ARGSMAX(0, (outWidth - 1) * k_sX + DkernelX - inWidth);
            int pad_height  = ARGSMAX(0, (outHeight - 1) * k_sY + DkernelY - inHeight);
            k_padYT         = std::floor(pad_height / 2);
            k_padXL         = std::floor(pad_width / 2);
            k_padYB         = pad_height - k_padXL;
            k_padXR         = pad_width - k_padXR;
        }
        LOG_CHECK(outHeight == paramOutHeight, "CHECK PADDING CONV OUTHEIGHT");
        LOG_CHECK(outWidth  == paramOutWidth, "CHECK PADDING CONV OUTWIDTH");
        node.in_shapes[0]   = param.in_shapes[0];
        node.out_shapes[0]  = param.out_shapes[0];
        node.hasBias        = param.hasBias;
        node.bottom_shape   = {inBatch, inChannel, inHeight, inWidth};
        node.groups         = param.groups >= 1 ? param.groups : 1;
        if(node.groups > 1){
            LOG_CHECK(inChannel % node.groups == 0, "DepthWise_Conv inChannel");
            LOG_CHECK(k_c % node.groups == 0, "DepthWise Conv outChannel");
            int ICg             = inChannel / node.groups;
            int OCg             = k_c / node.groups;
            node.weights_shape  = {node.groups, OCg, ICg, k_h, k_w};
        }else if(node.groups == 1){
            node.weights_shape  = {k_c, inChannel, k_h, k_w};
        }
        node.conv_strides       = {k_sY, k_sX};
        node.conv_paddingL      = {k_padYT, k_padXL};
        node.conv_paddingR      = {k_padYB, k_padXR};
        
        if(node.hasBias)
            node.bias_shape     = {k_c};
        
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
        node.top_shape      = {inBatch, k_c, outHeight, outWidth};
        node.layer_top_md   = memory::desc({node.top_shape}, dnnDataType, tag::any);
        
        if(!dilate){ 
            if(node.hasBias){
                auto convolution_desc = convolution_forward::desc(prop_kind::forward_inference,
                                        algorithm::convolution_direct, node.src_bottom_md, node.src_weights_md,
                                        node.src_bias_md, node.layer_top_md, node.conv_strides, 
                                        node.conv_paddingL, node.conv_paddingR);
                if(param.fused_ops){
                    node.conv_post_op = get_posts_opsMap(param.fused_act_type);
                    node.conv_ops.append_eltwise(node.conv_post_op.scale, 
                                                    node.conv_post_op.posts_op, 
                                                    node.conv_post_op.alpha,
                                                    node.conv_post_op.beta);
                    node.conv_attr.set_post_ops(node.conv_ops);
                    node.conv_pdesc = convolution_forward::primitive_desc(convolution_desc, node.conv_attr, BrixLab::graph_eng);
                }else{
                    node.conv_pdesc = convolution_forward::primitive_desc(convolution_desc, BrixLab::graph_eng);
                }
            }else{
                auto convolution_desc = convolution_forward::desc(prop_kind::forward_inference, 
                                        algorithm::convolution_direct, node.src_bottom_md, 
                                        node.src_weights_md, node.layer_top_md, 
                                        node.conv_strides, node.conv_paddingL, node.conv_paddingR);
                if(param.fused_ops){
                    node.conv_post_op = get_posts_opsMap(param.fused_act_type);
                    node.conv_ops.append_eltwise(node.conv_post_op.scale, 
                                                    node.conv_post_op.posts_op, 
                                                    node.conv_post_op.alpha,
                                                    node.conv_post_op.beta);
                    node.conv_attr.set_post_ops(node.conv_ops);
                    node.conv_pdesc = convolution_forward::primitive_desc(convolution_desc, node.conv_attr, BrixLab::graph_eng);
                }else{
                    node.conv_pdesc = convolution_forward::primitive_desc(convolution_desc, BrixLab::graph_eng);
                }
            }
            
        }else if(dilate){
            memory::dims conv_dilates = {node.dilateX, node.dilateX};
            if(node.hasBias){
                auto convolution_desc = convolution_forward::desc(prop_kind::forward_inference,
                                          algorithm::convolution_direct,node.src_bottom_md, node.src_weights_md,
                                          node.src_bias_md, node.layer_top_md, node.conv_strides, conv_dilates,
                                          node.conv_paddingL, node.conv_paddingR);
                if(param.fused_ops){
                    node.conv_post_op = get_posts_opsMap(param.fused_act_type);
                    node.conv_ops.append_eltwise(node.conv_post_op.scale, 
                                                    node.conv_post_op.posts_op, 
                                                    node.conv_post_op.alpha,
                                                    node.conv_post_op.beta);
                    node.conv_attr.set_post_ops(node.conv_ops);
                    node.conv_pdesc = convolution_forward::primitive_desc(convolution_desc, node.conv_attr, BrixLab::graph_eng);
                }else{
                    node.conv_pdesc = convolution_forward::primitive_desc(convolution_desc, BrixLab::graph_eng);
                }
            }else{
                auto convolution_desc = convolution_forward::desc(prop_kind::forward_inference,
                                          algorithm::convolution_direct,node.src_bottom_md, node.src_weights_md,
                                          node.layer_top_md, node.conv_strides, conv_dilates,
                                          node.conv_paddingL, node.conv_paddingR);
                if(param.fused_ops){
                    node.conv_post_op = get_posts_opsMap(param.fused_act_type);
                    node.conv_ops.append_eltwise(node.conv_post_op.scale, 
                                                    node.conv_post_op.posts_op, 
                                                    node.conv_post_op.alpha,
                                                    node.conv_post_op.beta);
                    node.conv_attr.set_post_ops(node.conv_ops);
                    node.conv_pdesc = convolution_forward::primitive_desc(convolution_desc, node.conv_attr, BrixLab::graph_eng);
                }else{
                    node.conv_pdesc = convolution_forward::primitive_desc(convolution_desc, BrixLab::graph_eng);
                }
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
        node->op_args           ={{DNNL_ARG_SRC, node->src_bottom_memory},
                                    {DNNL_ARG_MEAN, node->batchnorm_mean_memory},
                                    {DNNL_ARG_VARIANCE, node->batchnorm_variance_memory},
                                    {DNNL_ARG_SCALE_SHIFT, node->batchnorm_scale_shift_memory},
                                    {DNNL_ARG_DST, node->src_bottom_memory}};
        batch_normalization_forward(node->batchnorm_pdesc).execute(BrixLab::graph_stream, node->op_args);
    }
        
    template<typename DType>
    layerNode<DType> OP_batchnorm_layer_setup(const layerWeightsParam<DType> &param){
        layerNode<DType> node(OP_type::BATCHNORM);
        LOG_CHECK(param.in_shapes.size() == 1, "CHECK_INPUTS");
        LOG_CHECK(param.out_shapes.size() == 1, "CHECK_OUTPUTS");
        int inHeight    = param.in_shapes[0].Height;
        int inChannel   = param.in_shapes[0].Channel;
        int inWidth     = param.in_shapes[0].Width;
        int inBatch     = param.in_shapes[0].Batch;
        node.bottom_shape   = {inBatch, inChannel, inHeight, inWidth};
        node.batchnorm_scale_shift_shape = {2, inChannel};
        node.in_shapes.resize(1);
        node.out_shapes.resize(1);
        node.in_shapes[0]   = param.in_shapes[0];
        node.out_shapes[0]  = param.out_shapes[0];

        QUANITIZED_TYPE quantized_type = param.quantized_type;
        memory::data_type dnnDataType = dt::f32;
        memory::data_type dnnDataBiasType = dt::f32;
        if(quantized_type == BrixLab::QUANITIZED_TYPE::UINT8_QUANTIZED){
            dnnDataType = dt::u8;
            dnnDataBiasType = dt::s32;
        }
        // src
        node.src_bottom_md                  = memory::desc(node.bottom_shape, dnnDataType, tag::nchw);
        // scale_shift_weights
        node.batchnorm_scale_shift_md       = memory::desc(node.batchnorm_scale_shift_shape, dnnDataType, tag::nc);
        node.batchnorm_scale_shift_memory   = memory(node.batchnorm_scale_shift_md, BrixLab::graph_eng);
        write_to_dnnl_memory(param.b_shift_scale, node.batchnorm_scale_shift_memory);
        // descriptor: mean, variance.
        node.batchnorm_mean_memory = memory(node.batchnorm_pdesc.mean_desc(), BrixLab::graph_eng);
        node.batchnorm_variance_memory = memory(node.batchnorm_pdesc.variance_desc(), BrixLab::graph_eng);
        write_to_dnnl_memory(param.b_means, node.batchnorm_mean_memory);
        write_to_dnnl_memory(param.b_variance, node.batchnorm_variance_memory);

        // Create operation descriptor.
        dnnl::batch_normalization_forward::desc batchnorm_desc = batch_normalization_forward::desc(
                                                                        prop_kind::forward_inference, node.src_bottom_md, 1.e-10f,
                                                                        normalization_flags::use_scale_shift
                                                                        | normalization_flags::use_global_stats);

        // Create primitive descriptor.
        node.batchnorm_pdesc = batch_normalization_forward::primitive_desc(batchnorm_desc, BrixLab::graph_eng);
        // top mamory
        node.layer_top_memory = node.src_bottom_memory;
        node.inference_forward = OP_batchnorm_inference_forward;
        return node;
    }
    INSTANCE_LAYEROP(batchnorm);

    template<typename DType>   
    void OP_pooling_inference_forward(layerNode<DType> *node, graphSet<DType> &g_net){
        node->src_bottom_memory = g_net.input;
        node->op_args           = {{DNNL_ARG_SRC, node->src_bottom_memory},
                                    {DNNL_ARG_DST, node->layer_top_memory}};
        if(!node->p_dialiated){
            pooling_forward(node->pooling_pdesc_without_d).execute(BrixLab::graph_stream, node->op_args);
        }else if(node->p_dialiated){
            pooling_v2_forward(node->pooling_pdesc).execute(BrixLab::graph_stream, node->op_args);
        }
    }
    template<typename DType>    
    layerNode<DType> OP_pooling_layer_setup(const layerWeightsParam<DType> &param){
        layerNode<DType> node(OP_type::POOLING);
        LOG_CHECK(param.in_shapes.size()==1, "CHECK inputs");
        LOG_CHECK(param.out_shapes.size()==1, "CHECK outputs");
        int inHeight        = param.in_shapes[0].Height;
        int inChannel       = param.in_shapes[0].Channel;
        int inWidth         = param.in_shapes[0].Width;
        int inBatch         = param.in_shapes[0].Batch;
        int k_w             = param.p_kernelsX;
        int k_h             = param.p_kernelsY;
        int k_sX            = param.p_stridesX;
        int k_sY            = param.p_stridesY;
        int k_padYT         = 0;
        int k_padYB         = 0;
        int k_padXL         = 0;
        int k_padXR         = 0;
        int dilatedX        = param.p_dilatedX;
        int dilatedY        = param.p_dilatedY;
        int DkernelX        = 1 + (k_w - 1) * (dilatedX + 1);
        int DkernelY        = 1 + (k_h - 1) * (dilatedY + 1);
        bool dilated        = false;
        PoolingType p_type  = param.pooling_type;
        node.pooling_type   = get_op_mapped_pooling_type(p_type);
        node.bottom_shape   = {inBatch, inChannel, inHeight, inWidth};
        BrixLab::PaddingType pad_type       = param.pooling_padType;
        QUANITIZED_TYPE quantized_type      = param.quantized_type;
        memory::data_type dnnDataType       = dt::f32;
        memory::data_type dnnDataBiasType   = dt::f32;
        if(quantized_type == BrixLab::QUANITIZED_TYPE::UINT8_QUANTIZED){
            dnnDataType = dt::u8;
            dnnDataBiasType = dt::s32;
        }
        LOG_CHECK(dilatedX >= 0, "CHECK dilatedX");
        LOG_CHECK(dilatedY >= 0, "CHECK dilatedY");
        if(dilatedY == 0  && dilatedX == 0){
            dilated             = false;
            node.p_dialiated    = false;
        }else{
            dilated             = true;
            node.p_dialiated    = true;
        }
        int outHeight = (inHeight - DkernelY + k_padYB + k_padYT) / k_s + 1;
        int outWidth = (inWidth - DkernelX + k_padXL + k_padXR) / k_s + 1;;
        if(pad_type = PaddingType::PaddingSAME){
            outHeight       = std::ceil((inHeight) / k_sY); // oh = ceil(ih / stride)
            outWidth        = std::ceil((inWidth) / k_sX); // ow = ceil(iw / stride)
            int pad_width   = ARGSMAX(0, (outWidth - 1) * k_sX + DkernelX - inWidth);
            int pad_height  = ARGSMAX(0, (outHeight - 1) * k_sY + DkernelY - inHeight);
            k_padYT         = std::floor(pad_height / 2);
            k_padXL         = std::floor(pad_width / 2);
            k_padYB         = pad_height - k_padXL;
            k_padXR         = pad_width - k_padXR;
        }

        node.in_shapes.resize(1);
        node.out_shapes.resize(1);
        int paramOutHeight  = param.out_shapes[0].Height;
        int paramOutWidth   = param.out_shapes[0].Width;
        LOG_CHECK(outHeight == paramOutHeight, "CHECK PADDING CONV OUTHEIGHT");
        LOG_CHECK(outWidth  == paramOutWidth, "CHECK PADDING CONV OUTWIDTH");
        node.in_shapes[0]   = param.in_shapes[0];
        node.out_shapes[0]  = param.out_shapes[0];
        
        node.pooling_kernel = {k_h, k_w};
        node.pooling_strides = {k_sY, k_sX};
        node.pooling_paddingL = {k_padYT, k_padXL};
        node.pooling_paddingR = {k_padYB, k_padXR};
        node.pooling_dialiate = {dilatedX, dilatedY};
        
        node.top_shape = {inBatch, inChannel, outHight, outWidth};
        node.layer_top_md = memory::desc(node.top_shape, dnnDataType, tag::nchw);
        node.layer_top_memory = memory(node.layer_top_md, BrixLab::graph_eng);
        node.src_bottom_md = memory::desc(node.bottom_shape, dnnDataType, tag::nchw);
        if(dilated){
            dnnl::pooling_v2_forward::desc pooling_desc = pooling_v2_forward::desc(prop_kind::forward_inference,
                                            node.pooling_type, node.src_bottom_md, 
                                            node.layer_top_md,
                                            node.pooling_strides, node.pooling_kernel,
                                            node.pooling_dialiate, node.pooling_paddingL, 
                                            node.pooling_paddingR);
            node.pooling_pdesc = pooling_v2_forward::primitive_desc(pooling_desc, BrixLab::graph_eng);
        }else if(!dilated){
            dnnl::pooling_forward::desc pooling_desc_without_d = pooling_forward::desc(prop_kind::forward_inference, 
                                                node.pooling_type, node.src_bottom_md, 
                                                node.layer_top_md, node.pooling_strides, 
                                                node.pooling_kernel,
                                                node.pooling_paddingL,node.pooling_paddingR);
            node.pooling_pdesc_without_d = pooling_forward::primitive_desc(pooling_desc_without_d, BrixLab::graph_eng);
        }
        node.inference_forward = OP_pooling_inference_forward;
        return node;
    }
    INSTANCE_LAYEROP(pooling);

    template<typename DType>     
    void OP_concat_inference_forward(layerNode<DType> *node, graphSet<DType> &g_net){
        for(int ii = 0; ii < node->concat_num; ii++){
            int index                        = node->inputs[ii];
            node->concat_bottom_md[ii]       = g_net[index]->layer_top_md;
            node->concat_bottom_memory[ii]   = g_net[index]->layer_top_memory;
        }
        if(!node->inputset){
            for(int ii = 0; ii < node->concat_num; ii++){
                node->op_args.insert({DNNL_ARG_MULTIPLE_SRC + ii, node->concat_bottom_memory[ii]});
            }
            node->op_args.insert({{DNNL_ARG_DST, node->layer_top_memory}});
            node->inputset = true;
        }else{
            for(int ii = 0; ii < node->concat_num; ii++){
                node->op_args[DNNL_ARG_MULTIPLE_SRC + ii] = node->concat_bottom_memory[ii];
            }
            node->op_args[DNNL_ARG_DST] = node->layer_top_memory;
        }
        concat(node->concat_pdesc).execute(BrixLab::graph_stream, node->op_args);
    }
    template<typename DType>
    layerNode<DType> OP_concat_layer_setup(const layerWeightsParam<DType> &param){
        layerNode<DType> node(OP_type::CONCAT);
        check_inputs_shape(param.in_shapes);
        int inHeight        = param.in_shapes[0].Height;
        int inChannel       = param.in_shapes[0].Channel;
        int inWidth         = param.in_shapes[0].Width;
        int inBatch         = param.in_shapes[0].Batch;
        node.concat_num     = param.inIndexs.size();
        node.concat_axis    = param.concat_axis;

        QUANITIZED_TYPE quantized_type = param.quantized_type;
        memory::data_type dnnDataType = dt::f32;
        if(quantized_type == BrixLab::QUANITIZED_TYPE::UINT8_QUANTIZED){
            dnnDataType = dt::u8;
        }
        node.bottom_shape   = {inBatch, inChannel, inHeight, inWidth};
        node.src_bottom_md  = memory::desc(node.bottom_shape, dnnDataType, tag::nchw);
        for(int ii = 0; ii < node.concat_num; ii++){
            node.inputs.push_back(param.inIndexs[ii]);
            node.concat_bottom_md.push_back(node.src_bottom_md);
        }
        for(unsigned int ii = 0; ii <param.outIndexs.size(); ii++){
            node.outputs.push_back(param.outIndexs[ii]);
        }
        node.out_shapes.resize(1);
        node.out_shapes[0]  = param.out_shapes[0];
        node.in_shapes.resize(node.concat_num);
        for(unsigned int ii = 0; ii < node.concat_num; ii++){
            node.in_shapes[ii] = param.in_shapes[ii];
        }
        node.concat_bottom_memory.resize(node.concat_num);
        node.op_type = param.op_type;
        node.concat_pdesc = concat::primitive_desc(node.concat_axis, node.concat_bottom_md, BrixLab::graph_eng);
        node.layer_top_md = node.concat_pdesc.dst_desc();
        node.layer_top_memory = memory(node.layer_top_md, BrixLab::graph_eng);
        node.inference_forward = OP_concat_inference_forward;
        return node;
    }
    INSTANCE_LAYEROP(concat);

    template<typename DType>    
    void OP_sum_inference_forward(layerNode<DType> *node, graphSet<DType> &g_net){
        for(int ii = 0; ii < node->sum_num; ii++){
            int index           = node->inputs[ii];
            node->sum_bottom_memory[ii] = g_net[index]->layer_top_memory;
        }
        if(!node->inputset){
            for(int ii = 0; ii < node->sum_num; ii++){
                node->op_args.insert({DNNL_ARG_MULTIPLE_SRC + ii, node->sum_bottom_memory[ii]});
            }
            node->op_args.insert({DNNL_ARG_DST, node->layer_top_memory});
        }else{
            for(int ii = 0; ii < node->sum_num; ii++){
                node->op_args[DNNL_ARG_MULTIPLE_SRC + ii] = node->sum_bottom_memory[ii];
            }
            node->op_args[DNNL_ARG_DST]= node->layer_top_memory;
        }
        sum(node->sum_pdesc).execute(BrixLab::graph_stream, node->op_args);
    }

    template<typename DType>    
    layerNode<DType> OP_sum_layer_setup(const layerWeightsParam<DType> &param){
        layerNode<DType> node(OP_type::ELTWISE);
        check_inputs_shape(param.in_shapes);
        node.sum_num = param.inIndexs.size();
        for(int ii = 0; ii < node.sum_num; ii++){
            node.inputs.push_back(param.inIndexs[ii]);
        }
        int inHeight    = param.in_shapes[0].Height;
        int inChannel   = param.in_shapes[0].Channel;
        int inWidth     = param.in_shapes[0].Width;
        int inBatch     = param.in_shapes[0].Batch;
        memory::dims S_Shape = {inBatch, inChannel, inWidth, inHeight};
        QUANITIZED_TYPE quantized_type = param.quantized_type;
        memory::data_type dnnDataType = dt::f32;
        if(quantized_type == BrixLab::QUANITIZED_TYPE::UINT8_QUANTIZED){
            dnnDataType = dt::u8;
        }
        node.src_bottom_md = memory::desc(S_Shape, dnnDataType, tag::nchw);
        for(int ii = 0; ii < node.sum_num; ii++){
            node.sum_scale[ii] = 1.f;
            node.sum_bottom_md.push_back(node.src_bottom_md);
        }
         node.out_shapes.resize(1);
        node.out_shapes[0]  = param.out_shapes[0];
        node.in_shapes.resize(node.sum_num);
        for(unsigned int ii = 0; ii < node.sum_num; ii++){
            node.in_shapes[ii] = param.in_shapes[ii];
        }
        node.sum_bottom_memory.resize(node.sum_num);
        node.sum_pdesc = sum::primitive_desc(node.sum_scale, node.sum_bottom_md, BrixLab::graph_eng);
        node.top_shape = {inBatch, inChannel, inWidth, inHeight};
        node.layer_top_md = memory::desc(node.top_shape, dt::f32, tag::nchw);
        node.layer_top_memory = memory(node.sum_pdesc.dst_desc(), BrixLab::graph_eng);
        
        for(int ii = 0; ii < node.sum_num; ii++){
            node.op_args.insert({DNNL_ARG_MULTIPLE_SRC + ii, node.sum_bottom_memory[ii]});
        }
        node.inference_forward = OP_sum_inference_forward;
        return node;
    }
    INSTANCE_LAYEROP(sum);
    
    template<typename DType>
    void OP_resample_inference_forward(layerNode<DType> *node, graphSet<DType> &g_net){
        node->src_bottom_memory = g_net.input;
        node->op_args= {{DNNL_ARG_SRC, node->src_bottom_memory},
                        {DNNL_ARG_DST, node->layer_top_memory}};
        resampling_forward(node->resample_pdesc).execute(BrixLab::graph_stream, node->op_args);
    }
    template<typename DType>
    layerNode<DType> OP_resample_layer_setup(const layerWeightsParam<DType> &param){
        layerNode<DType> node(OP_type::RESAMPLING);
        LOG_CHECK(param.in_shapes.size()==1, "CHECK_INPUTS");
        LOG_CHECK(param.out_shapes.size()==1, "CHECK_OUTPUTS");
        int inHeight    = param.in_shapes[0].Height;
        int inChannel   = param.in_shapes[0].Channel;
        int inWidth     = param.in_shapes[0].Width;
        int inBatch     = param.in_shapes[0].Batch;
        node.bottom_shape = {inBatch, inChannel, inHeight, inWidth};
        node.op_type = param.op_type;
        QUANITIZED_TYPE quantized_type = param.quantized_type;
        memory::data_type dnnDataType = dt::f32;
        memory::data_type dnnDataBiasType = dt::f32;
        if(quantized_type == BrixLab::QUANITIZED_TYPE::UINT8_QUANTIZED){
            dnnDataType = dt::u8;
            dnnDataBiasType = dt::s32;
        }
        node.src_bottom_md = memory::desc(node.bottom_shape, dnnDataType, tag::nchw);
        int outHeight = int(inHeight * param.adjust_height_scale);
        int outWidth = int(inWidth * param.adjust_width_scale);
        int paramOutHeight  = param.out_shapes[0].Height;
        int paramOutWidth   = param.out_shapes[0].Width;

        LOG_CHECK(outHeight == paramOutHeight, "CHECK RESAMPLE OUTHEIGHT");
        LOG_CHECK(outWidth  == paramOutWidth, "CHECK RESAMPLE OUTWIDTH");

        node.in_shapes.resize(1);
        node.out_shapes.resize(1);
        node.in_shapes[0]   = param.in_shapes[0];
        node.out_shapes[0]  = param.out_shapes[0];
        node.top_shape = {inBatch, inChannel, outHeight, outWidth};
        node.layer_top_md = memory::desc(node.top_shape, dnnDataType, tag::nchw);
        node.layer_top_memory = memory(node.layer_top_md, BrixLab::graph_eng);

        dnnl::resampling_forward::desc resample_desc = resampling_forward::desc(prop_kind::forward_inference,
            algorithm::resampling_linear, node.src_bottom_md, node.layer_top_md);
        node.resample_pdesc = resampling_forward::primitive_desc(resample_desc, BrixLab::graph_eng);

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
        }
        node->op_args = {{DNNL_ARG_SRC, node->src_bottom_memory},
                        {DNNL_ARG_WEIGHTS, node->src_weights_memory},
                        {DNNL_ARG_BIAS, node->src_bias_memory},
                        {DNNL_ARG_DST, node->layer_top_memory}};
        deconvolution_forward(node->deconv_pdesc).execute(BrixLab::graph_stream, node->op_args);
        LOG(DEBUG_INFO, "[OP_deconvolution_inference_forward] done!\n");
    }    

    template<typename DType>
    layerNode<DType> OP_deconvolution_layer_setup(const layerWeightsParam<DType> &param){
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
        LOG_CHECK(param.in_shapes.size()==1, "CHECK_INPUTS");
        LOG_CHECK(param.out_shapes.size()==1, "CHECK_OUTPUTS");
        int inHeight    = param.in_shapes[0].Height;
        int inChannel   = param.in_shapes[0].Channel;
        int inWidth     = param.in_shapes[0].Width;
        int inBatch     = param.in_shapes[0].Batch;
        
        node.bottom_shape       = {inBatch, inChannel, inHeight, inWidth};
        node.weights_shape      = {k_c, inChannel, k_w, k_h};
        node.deconv_strides     = {k_sX, k_sY};
        node.deconv_paddingL    = {k_padYT, k_padXL};
        node.deconv_paddingR    = {k_padYB, k_padXR};
        node.dilateX            = param.dilateX;
        node.dilateY            = param.dilateY;
        node.hasBias            = param.hasBias;
        if(node.hasBias)
            node.bias_shape = {k_c};
        // src bottom data
        node.src_bottom_md = memory::desc({node.bottom_shape}, dnnDataType, tag::any);
        // weights & bias
        node.src_weights_md = memory::desc({node.weights_shape}, dnnDataType, tag::any);
        node.src_weights_memory = memory({{node.weights_shape}, dnnDataType, tag::oihw}, BrixLab::graph_eng);
        write_to_dnnl_memory(param.transposed_weights, node.src_weights_memory);
        if(node.hasBias){
            node.src_bias_md = memory::desc({node.bias_shape}, dnnDataBiasType, tag::any);
            node.src_bias_memory = memory({{node.bias_shape}, dnnDataBiasType, tag::x}, BrixLab::graph_eng);
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

        if(param.padMode == PaddingType::PaddingSAME){

        }
        int paramOutHeight  = param.out_shapes[0].Height;
        int paramOutWidth   = param.out_shapes[0].Width;

        LOG_CHECK(outHeight == paramOutHeight, "CHECK PADDING TRANSPOSED_CONV OUTHEIGHT");
        LOG_CHECK(outWidth  == paramOutWidth, "CHECK PADDING TRANSPOSED_CONV OUTWIDTH");
        node.top_shape = {inBatch, k_c, outHeight, outWidth};

        node.in_shapes.resize(1);
        node.out_shapes.resize(1);
        node.in_shapes[0]   = param.in_shapes[0];
        node.out_shapes[0]  = param.out_shapes[0];

        node.layer_top_md = memory::desc({node.top_shape}, dt::f32, tag::nchw);
        int dilate = ARGSMAX(node.dilateX, node.dilateY);
        if(dilate > 0){
            memory::dims deconv_deliatd = {node.dilateX, node.dilateY};
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
        LOG(DEBUG_INFO,"deconvolution OP set_up!");
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
        node->op_args = {{DNNL_ARG_SRC, node->src_bottom_memory},
                {DNNL_ARG_WEIGHTS, node->src_weights_memory},
                {DNNL_ARG_BIAS, node->src_bias_memory},
                {DNNL_ARG_DST, node->layer_top_memory}};
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
        LOG_CHECK(param.in_shapes.size()==1, "CHECK_INPUTS");
        LOG_CHECK(param.out_shapes.size()==1, "CHECK_OUTPUTS");
        int inHeight    = param.in_shapes[0].Height;
        int inChannel   = param.in_shapes[0].Channel;
        int inWidth     = param.in_shapes[0].Width;
        int inBatch     = param.in_shapes[0].Batch;

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
        
        if(param.fused_ops){
            node.fc_post_op = get_posts_opsMap(param.fused_act_type);
            node.fc_ops.append_eltwise(node.fc_post_op.scale, node.fc_post_op.posts_op, 
                                            node.fc_post_op.alpha, node.fc_post_op.beta);
            node.fc_attr.set_post_ops(node.fc_ops);
            node.inner_pdesc = inner_product_forward::primitive_desc(
                                                inner_desc, node.fc_attr, BrixLab::graph_eng);
        }else{
            node.inner_pdesc = inner_product_forward::primitive_desc(inner_desc, BrixLab::graph_eng);
        }
        node.layer_top_memory = memory(node.inner_pdesc.dst_desc(), BrixLab::graph_eng);

        node.inference_forward = OP_innerproduct_inference_forward; 
        return node;
    }
    INSTANCE_LAYEROP(innerproduct);

    template<typename DType>
    void OP_activation_inference_forward(layerNode<DType> *node, graphSet<DType> &g_net){
        node->src_bottom_memory = g_net.input;
        node->op_args = {
            {DNNL_ARG_SRC, node->src_bottom_memory},
            {DNNL_ARG_DST, node->layer_top_memory}
        };
        eltwise_forward(node->eltwise_pdesc).execute(BrixLab::graph_stream, node->op_args);
    }
    template<typename DType>
    layerNode<DType> OP_activation_layer_setup(const layerWeightsParam<DType> &param){
        layerNode<DType> node(OP_type::ACTIVITION);
        LOG_CHECK(param.in_shapes.size()==1, "CHECK_INPUTS");
        LOG_CHECK(param.out_shapes.size()==1, "CHECK_OUTPUTS");
        int inHeight    = param.in_shapes[0].Height;
        int inChannel   = param.in_shapes[0].Channel;
        int inWidth     = param.in_shapes[0].Width;
        int inBatch     = param.in_shapes[0].Batch;
        //bottom_src memory
        QUANITIZED_TYPE quantized_type = param.quantized_type;
        memory::data_type dnnDataType = dt::f32;
        memory::data_type dnnDataBiasType = dt::f32;
        if(quantized_type == BrixLab::QUANITIZED_TYPE::UINT8_QUANTIZED){
            dnnDataType = dt::u8;
            dnnDataBiasType = dt::s32;
        }
        node.bottom_shape = {inBatch, inChannel, inHeight, inWidth};
        node.src_bottom_md = memory::desc(node.bottom_shape, dnnDataType, tag::nchw);

        node.in_shapes.resize(1);
        node.out_shapes.resize(1);
        node.in_shapes[0]   = param.in_shapes[0];
        node.out_shapes[0]  = param.out_shapes[0];
        
        // top &memory
        node.top_shape = {inBatch, inChannel, inHeight, inWidth};
        node.layer_top_md = memory::desc(node.top_shape, dnnDataType, tag::nchw);
        node.layer_top_memory = memory(node.layer_top_md, BrixLab::graph_eng);

        //op
        node.activate_type = get_op_mapped_activition_type(param.activate_type);
        node.alpha = param.alpha;
        node.beta = param.beta;
        dnnl::eltwise_forward::desc eltwise_desc = eltwise_forward::desc(prop_kind::forward_inference,
                                                        node.activate_type, node.src_bottom_md, node.alpha, node.beta);
        node.eltwise_pdesc = eltwise_forward::primitive_desc(eltwise_desc, BrixLab::graph_eng);

        return node;
    }
    INSTANCE_LAYEROP(activation);

    template<typename DType>
    void OP_binary_inference_forward(layerNode<DType> *node, graphSet<DType> &g_net){
        node->binary_memory[0]  = g_net[node->inputs[0]]->layer_top_memory;
        node->binary_memory[1]  = g_net[node->inputs[1]]->layer_top_memory;
        node->op_args           = {{DNNL_ARG_SRC_0, node->binary_memory[0]},
                                   {DNNL_ARG_SRC_1, node->binary_memory[0]},
                                   {DNNL_ARG_DST, node->layer_top_memory}};
        binary(node->binary_pdesc).execute(BrixLab::graph_stream, node->op_args);
    }
    template<typename DType>
    layerNode<DType> OP_binary_layer_setup(const layerWeightsParam<DType> &param){
        layerNode<DType> node(OP_type::BINARY_OP);
        LOG_CHECK(param.in_shapes.size()==2, "CHECK_INPUTS");
        LOG_CHECK(param.out_shapes.size()==1, "CHECK_OUTPUTS");
        int inHeight    = param.in_shapes[0].Height;
        int inChannel   = param.in_shapes[0].Channel;
        int inWidth     = param.in_shapes[0].Width;
        int inBatch     = param.in_shapes[0].Batch;
        //bottom_src memory
        QUANITIZED_TYPE quantized_type = param.quantized_type;
        memory::data_type dnnDataType = dt::f32;
        memory::data_type dnnDataBiasType = dt::f32;
        if(quantized_type == BrixLab::QUANITIZED_TYPE::UINT8_QUANTIZED){
            dnnDataType = dt::u8;
            dnnDataBiasType = dt::s32;
        }
        node.bottom_shape = {inBatch, inChannel, inHeight, inWidth};
        node.src_bottom_md = memory::desc(node.bottom_shape, dnnDataType, tag::nchw);
        LOG_CHECK(param.inIndexs.size()==2, "CHECK 2 INPUTS");
        for(unsigned int ii = 0; ii < param.inIndexs.size(); ii++){
            node.inputs.push_back(param.inIndexs[ii]);
            node.binary_md.push_back(node.src_bottom_md);
        }
        node.binary_memory.resize(2);
        node.in_shapes.resize(2);
        node.out_shapes.resize(1);
        node.in_shapes[0]   = param.in_shapes[0];
        node.in_shapes[1]   = param.in_shapes[1];
        node.out_shapes[0]  = param.out_shapes[0];
        // top &memory
        node.top_shape = {inBatch, inChannel, inHeight, inWidth};
        node.layer_top_md = memory::desc(node.top_shape, dnnDataType, tag::nchw);
        node.layer_top_memory = memory(node.layer_top_md, BrixLab::graph_eng);

        //op
        node.binary_type = get_op_mapped_binary_type(param.binary_type);
        dnnl::binary::desc binary_desc = binary::desc(node.binary_type, node.binary_md[0], node.binary_md[1], node.layer_top_md);
        node.binary_pdesc = binary::primitive_desc(binary_desc, BrixLab::graph_eng);

        return node;
    }
    INSTANCE_LAYEROP(binary);

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
        //auto& buffers = _tflite_model->buffers;

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
                layerWeightsParam<DType> New_OP;
                auto creator = liteOpConvertMapKit<DType>::get()->search(opCode);
                LOG_CHECK(creator, "CHECK OP FROM TFlite") << "NOT_SUPPORTED_OP: [ " << tflite::EnumNameBuiltinOperator(opCode) << " ]";

                // tflite op to onednn op
                New_OP.node_name    = tensors[ops[j]->outputs[0]]->name;
                New_OP.op_type      = creator->opType(quantizedModel);
                
                // set default input output index
                New_OP.inIndexs.resize(ops[j]->inputs.size());
                New_OP.outIndexs.resize(ops[j]->outputs.size());
                for (unsigned int i = 0; i < ops[j]->inputs.size(); i++) {
                    New_OP.inIndexs[i] = ops[j]->inputs[i];
                }
                for (unsigned int i = 0; i < ops[j]->outputs.size(); i++) {
                    New_OP.outIndexs[i] = ops[j]->outputs[i];
                }
                // Run actual conversion
                creator->run(&New_OP, ops[j], tensors, tfliteModelBuffer, tfliteOpSet, quantizedModel);
                g_net.layer_ops.emplace_back(New_OP);
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
        }else if(func_name == "OP_activation_layer_setup"){
            return OP_activation_layer_setup;
        }else if(func_name == "OP_innerproduct_layer_setup"){
            return OP_innerproduct_layer_setup;
        }else if(func_name == "OP_resample_layer_setup"){
            return OP_resample_layer_setup;
        }else if(func_name == "OP_sum_layer_setup"){
            return OP_sum_layer_setup;
        }else if(func_name == "OP_concat_layer_setup"){
            return OP_concat_layer_setup;
        }else if(func_name == "OP_pooling_layer_setup"){
            return OP_pooling_layer_setup;
        }else if(func_name == "OP_batchnorm_layer_setup"){
            return OP_batchnorm_layer_setup;
        }else{
            return nullptr;
        }
    }

} // namespace BrixLab

