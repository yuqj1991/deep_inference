#include "unit_test.hpp"
#include <numeric>
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"
#define STB_IMAGE_RESIZE_IMPLEMENTATION
#include "stb_image_resize.h"

#include <sys/time.h>
namespace BrixLab
{
    void Test_Deconvulution(float *data){
        BrixLab::strParam<float> deconv_param;
        deconv_param.k_w = 2;
        deconv_param.k_h = 2;
        deconv_param.stridesY = 2;
        deconv_param.stridesX = 2;
        deconv_param.padMode = PaddingVALID;
        deconv_param.k_c = 128;
        deconv_param.in_shapes.resize(1);
        deconv_param.in_shapes[0].Batch = 3;
        deconv_param.in_shapes[0].Channel = 64;
        deconv_param.in_shapes[0].Height = 14;
        deconv_param.in_shapes[0].Width = 14;
        deconv_param.hasBias = true;
        deconv_param.dilateX = 0;
        deconv_param.dilateY = 0;
        deconv_param.quantized_type = QUANITIZED_TYPE::FLOAT32_REGULAR;
        int weights_size = deconv_param.k_c * deconv_param.in_shapes[0].Channel * deconv_param.k_h *deconv_param.k_w;
        deconv_param.transposed_weights = (float* )xcalloc(weights_size, sizeof(float));
        deconv_param.transposed_bias = (float* )xcalloc(deconv_param.k_c, sizeof(float));
        for(int ii = 0; ii < deconv_param.k_c; ii++){
            deconv_param.transposed_bias[ii] = std::tanh(ii* 2.f);
        }
        for(int ii = 0; ii < weights_size; ii++){
            deconv_param.transposed_weights[ii] = std::sin(2 * ii * 2.f);
        }
        deconv_param.op_type = OP_type::DECONVOLUTION;

        dnnl::memory::dims inShape = {deconv_param.in_shapes[0].Batch, deconv_param.in_shapes[0].Channel, 
                                        deconv_param.in_shapes[0].Height, deconv_param.in_shapes[0].Width};
        auto temp_input = dnnl::memory({{inShape}, dt::f32, tag::nchw}, BrixLab::graph_eng);
        write_to_dnnl_memory(data, temp_input);
        graphSetLink<float> g_net(0, 0);
        
        std::string OP_deconvolution = get_mapped_op_string(deconv_param.op_type) + "_layer_setup";

        strLayerNode<float> node = getSetupFunc(OP_deconvolution)(deconv_param);
        graph_insert(g_net, &node);
        LOG(DEBUG_INFO)<<"[Unit test Deconvolution]"<<"Net size: "<< g_net.graph_size;
        assert(g_net.head->node_param.src_weights_memory.get_data_handle() != nullptr);
        auto current_node = g_net.head;
        current_node->node_param.inference_forward(&(current_node->node_param), g_net);
        LOG(DEBUG_INFO)<<"[Unit test Deconvolution]"<<"it passed!";
    }
    memory::dim product(const memory::dims &dims) {
    return std::accumulate(dims.begin(), dims.end(), (memory::dim)1,
            std::multiplies<memory::dim>());
    }

    void Test_convolution() {

        // Tensor dimensions.
        const memory::dim N = 3, // batch size
                IC = 32, // input channels
                IH = 13, // input height
                IW = 13, // input width
                OC = 64, // output channels
                KH = 3, // weights height
                KW = 3, // weights width
                PH_L = 1, // height padding: left
                PH_R = 1, // height padding: right
                PW_L = 1, // width padding: left
                PW_R = 1, // width padding: right
                SH = 4, // height-wise stride
                SW = 4, // width-wise stride
                OH = (IH - KH + PH_L + PH_R) / SH + 1, // output height
                OW = (IW - KW + PW_L + PW_R) / SW + 1; // output width

        // Source (src), weights, bias, and destination (dst) tensors
        // dimensions.
        memory::dims src_dims = {N, IC, IH, IW};
        memory::dims weights_dims = {OC, IC, KH, KW};
        memory::dims bias_dims = {OC};
        memory::dims dst_dims = {N, OC, OH, OW};

        // Strides, padding dimensions.
        memory::dims strides_dims = {SH, SW};
        memory::dims padding_dims_l = {PH_L, PW_L};
        memory::dims padding_dims_r = {PH_R, PW_R};

        // Allocate buffers.
        std::vector<float> src_data(product(src_dims));
        std::vector<float> weights_data(product(weights_dims));
        std::vector<float> bias_data(OC);
        std::vector<float> dst_data(product(dst_dims));

        // Initialize src, weights, and dst tensors.
        std::generate(src_data.begin(), src_data.end(), []() {
            static int i = 0;
            return std::cos(i++ / 10.f);
        });
        std::generate(weights_data.begin(), weights_data.end(), []() {
            static int i = 0;
            return std::sin(i++ * 2.f);
        });
        std::generate(bias_data.begin(), bias_data.end(), []() {
            static int i = 0;
            return std::tanh(i++);
        });

        // Create memory objects for tensor data (src, weights, dst). In this
        // example, NCHW layout is assumed for src and dst, and OIHW for weights.
        auto user_src_mem = memory({src_dims, dt::f32, tag::nhwc}, graph_eng);
        auto user_weights_mem = memory({weights_dims, dt::f32, tag::ohwi}, graph_eng);
        auto user_dst_mem = memory({dst_dims, dt::f32, tag::nchw}, graph_eng);

        // Create memory descriptors with format_tag::any for the primitive. This
        // enables the convolution primitive to choose memory layouts for an
        // optimized primitive implementation, and these layouts may differ from the
        // ones provided by the user.
        auto conv_src_md = memory::desc(src_dims, dt::f32, tag::nchw);
        auto conv_weights_md = memory::desc(weights_dims, dt::f32, tag::oihw);
        auto conv_dst_md = memory::desc(dst_dims, dt::f32, tag::nchw);

        // Create memory descriptor and memory object for input bias.
        auto user_bias_md = memory::desc(bias_dims, dt::f32, tag::a);
        auto user_bias_mem = memory(user_bias_md, graph_eng);

        // Write data to memory object's handle.
        write_to_dnnl_memory(src_data.data(), user_src_mem);
        write_to_dnnl_memory(weights_data.data(), user_weights_mem);
        write_to_dnnl_memory(bias_data.data(), user_bias_mem);

        // Create operation descriptor.
        auto conv_desc = convolution_forward::desc(prop_kind::forward_inference,
                algorithm::convolution_direct, conv_src_md, conv_weights_md,
                user_bias_md, conv_dst_md, strides_dims, padding_dims_l,
                padding_dims_r);

        // Create primitive post-ops (ReLU).
        const float scale = 1.f;
        const float alpha = 0.f;
        const float beta = 0.f;
        post_ops conv_ops;
        conv_ops.append_eltwise(scale, algorithm::eltwise_relu, alpha, beta);
        primitive_attr conv_attr;
        conv_attr.set_post_ops(conv_ops);

        // Create primitive descriptor.
        auto conv_pd
                = convolution_forward::primitive_desc(conv_desc, conv_attr, graph_eng);

        // For now, assume that the src, weights, and dst memory layouts generated
        // by the primitive and the ones provided by the user are identical.
        auto conv_src_mem = user_src_mem;
        auto conv_weights_mem = user_weights_mem;
        auto conv_dst_mem = user_dst_mem;

        // Reorder the data in case the src and weights memory layouts generated by
        // the primitive and the ones provided by the user are different. In this
        // case, we create additional memory objects with internal buffers that will
        // contain the reordered data. The data in dst will be reordered after the
        // convolution computation has finalized.
        if (conv_pd.src_desc() != user_src_mem.get_desc()) {
            conv_src_mem = memory(conv_pd.src_desc(), graph_eng);
            reorder(user_src_mem, conv_src_mem)
                    .execute(graph_stream, user_src_mem, conv_src_mem);
        }

        if (conv_pd.weights_desc() != user_weights_mem.get_desc()) {
            conv_weights_mem = memory(conv_pd.weights_desc(), graph_eng);
            reorder(user_weights_mem, conv_weights_mem)
                    .execute(graph_stream, user_weights_mem, conv_weights_mem);
        }

        if (conv_pd.dst_desc() != user_dst_mem.get_desc()) {
            conv_dst_mem = memory(conv_pd.dst_desc(), graph_eng);
        }

        // Create the primitive.
        auto conv_prim = convolution_forward(conv_pd);

        // Primitive arguments.
        std::unordered_map<int, memory> conv_args;
        conv_args.insert({DNNL_ARG_SRC, conv_src_mem});
        conv_args.insert({DNNL_ARG_WEIGHTS, conv_weights_mem});
        conv_args.insert({DNNL_ARG_BIAS, user_bias_mem});
        conv_args.insert({DNNL_ARG_DST, conv_dst_mem});

        // Primitive execution: convolution with ReLU.
        conv_prim.execute(graph_stream, conv_args);

        // Reorder the data in case the dst memory descriptor generated by the
        // primitive and the one provided by the user are different.
        if (conv_pd.dst_desc() != user_dst_mem.get_desc()) {
            reorder(conv_dst_mem, user_dst_mem)
                    .execute(graph_stream, conv_dst_mem, user_dst_mem);
        } else
            user_dst_mem = conv_dst_mem;

        // Wait for the computation to finalize.
        graph_stream.wait();

        // Read data from memory object's handle.
        read_from_dnnl_memory(dst_data.data(), user_dst_mem);
    }

    void Test_demoNet(float *data){
        // first layer
        dnnl::memory::format_tag tag_type   = tag::nhwc;
        dnnl::memory::data_type d_type      = dt::f32;
        NetT<float> NetList;
        BrixLab::strParam<float>data_param;
        data_param.node_name        = "data_inputs";
        data_param.op_type          = OP_type::DATA_INPUTS;
        NetList.layer_ops.push_back(data_param);
        BrixLab::strParam<float> conv_param;
        conv_param.k_w                      = 3;
        conv_param.k_h                      = 3;
        conv_param.stridesX                 = 2;
        conv_param.stridesY                 = 2;
        conv_param.padMode                  = PaddingVALID;
        conv_param.k_c                      = 32;
        conv_param.in_shapes.resize(1);
        conv_param.in_shapes[0].Batch       = 16;
        conv_param.in_shapes[0].Channel     = 4;
        conv_param.in_shapes[0].Height      = 300;
        conv_param.in_shapes[0].Width       = 300;
        conv_param.in_shapes[0].format      = TENSOR_FORMATE::NHWC;
        conv_param.out_shapes.resize(1);
        conv_param.out_shapes[0].Batch      = 16;
        conv_param.out_shapes[0].Channel    = 32;
        conv_param.out_shapes[0].Height     = 149;
        conv_param.out_shapes[0].Width      = 149;
        conv_param.out_shapes[0].format     = TENSOR_FORMATE::NHWC;
        conv_param.hasBias                  = true;
        conv_param.dilateX                  = 0;
        conv_param.dilateY                  = 0;
        conv_param.groups                   = 1;
        conv_param.node_name                = "conv_str/conv";
        int weights_size                    = conv_param.in_shapes[0].Channel * conv_param.k_c * 
                                                            conv_param.k_w * conv_param.k_h;
        conv_param.conv_weights             = (float* )xcalloc(weights_size, sizeof(float));
        conv_param.conv_bias                = (float* )xcalloc(conv_param.k_c, sizeof(float));
        conv_param.op_type                  = OP_type::CONVOLUTION;
        conv_param.quantized_type           = QUANITIZED_TYPE::FLOAT32_REGULAR;
        dnnl::memory::dims inShape          = {conv_param.in_shapes[0].Batch,
                                                conv_param.in_shapes[0].Channel, 
                                                conv_param.in_shapes[0].Height, 
                                                conv_param.in_shapes[0].Width};
        auto temp_input                     = dnnl::memory({{inShape}, d_type, tag_type}, BrixLab::graph_eng);
        write_to_dnnl_memory(data, temp_input);
        for(int ii = 0; ii < conv_param.k_c; ii++){
            conv_param.conv_bias[ii] = std::tanh(ii * 2.f);
        }
        for(int ii = 0; ii < weights_size; ii++){
            conv_param.conv_weights[ii] = std::sin(2 * ii * 2.f);
        }
        conv_param.inIndexs.push_back(0);
        NetList.layer_ops.push_back(conv_param);

        //second layer
        BrixLab::strParam<float> conv_2_param;
        conv_2_param.k_w                        = 3;
        conv_2_param.k_h                        = 3;
        conv_2_param.stridesY                   = 2;
        conv_2_param.stridesX                   = 2;
        conv_2_param.padMode                    = PaddingVALID;
        conv_2_param.k_c                        = 32;
        conv_2_param.groups                     = 32;
        conv_2_param.in_shapes.resize(1);
        conv_2_param.in_shapes[0]               = conv_param.out_shapes[0];
        conv_2_param.out_shapes.resize(1);
        conv_2_param.out_shapes[0].Batch        = 16;
        conv_2_param.out_shapes[0].Channel      = 32;
        conv_2_param.out_shapes[0].Height       = 74;
        conv_2_param.out_shapes[0].Width        = 74;
        conv_2_param.out_shapes[0].format       = TENSOR_FORMATE::NHWC;
        conv_2_param.hasBias                    = true;
        conv_2_param.dilateX                    = 0;
        conv_2_param.dilateY                    = 0;
        conv_2_param.op_type                    = OP_type::CONVOLUTION;
        conv_2_param.node_name                  = "conv_2/project_1";
        conv_2_param.quantized_type             = QUANITIZED_TYPE::FLOAT32_REGULAR;
        weights_size                            = conv_2_param.k_c * conv_2_param.in_shapes[0].Channel * conv_2_param.k_h 
                                                                    *conv_2_param.k_w;
        conv_2_param.conv_weights               = (float* )xcalloc(weights_size, sizeof(float));
        conv_2_param.conv_bias                  = (float* )xcalloc(conv_2_param.k_c, sizeof(float));
        conv_2_param.inIndexs.push_back(1);
        for(int ii = 0; ii < conv_2_param.k_c; ii++){
            conv_2_param.conv_bias[ii] = std::tanh(ii* 2.f);
        }
        for(int ii = 0; ii < weights_size; ii++){
            conv_2_param.conv_weights[ii] = std::sin(2 * ii * 2.f);
        }       
        NetList.layer_ops.push_back(conv_2_param);

        //third layer
        BrixLab::strParam<float> deconv_param;
        deconv_param.k_w                        = 2;
        deconv_param.k_h                        = 2;
        deconv_param.stridesY                   = 2;
        deconv_param.stridesX                   = 2;
        deconv_param.padMode                    = PaddingVALID;
        deconv_param.k_c                        = 32;
        deconv_param.in_shapes.resize(1);
        deconv_param.in_shapes[0]               = conv_2_param.out_shapes[0];
        deconv_param.out_shapes.resize(1);
        deconv_param.out_shapes[0].Batch        = 16;
        deconv_param.out_shapes[0].Channel      = 32;
        deconv_param.out_shapes[0].Height       = 148;
        deconv_param.out_shapes[0].Width        = 148;
        deconv_param.out_shapes[0].format       = TENSOR_FORMATE::NHWC;
        deconv_param.hasBias                    = true;
        deconv_param.dilateX                    = 0;
        deconv_param.dilateY                    = 0;
        deconv_param.op_type                    = OP_type::DECONVOLUTION;
        deconv_param.node_name                  = "deconv_1/project_1";
        deconv_param.quantized_type             = QUANITIZED_TYPE::FLOAT32_REGULAR;
        weights_size                            = deconv_param.k_c * deconv_param.in_shapes[0].Channel * deconv_param.k_h 
                                                                    *deconv_param.k_w;
        deconv_param.transposed_weights         = (float* )xcalloc(weights_size, sizeof(float));
        deconv_param.transposed_bias            = (float* )xcalloc(deconv_param.k_c, sizeof(float));
        deconv_param.inIndexs.push_back(2);
        for(int ii = 0; ii < deconv_param.k_c; ii++){
            deconv_param.transposed_bias[ii] = std::tanh(ii* 2.f);
        }
        for(int ii = 0; ii < weights_size; ii++){
            deconv_param.transposed_weights[ii] = std::sin(2 * ii * 2.f);
        }
        NetList.layer_ops.push_back(deconv_param); 

        //fourth layer
        BrixLab::strParam<float> conv_3_param;
        conv_3_param.k_w                        = 3;
        conv_3_param.k_h                        = 3;
        conv_3_param.stridesY                   = 2;
        conv_3_param.stridesX                   = 2;
        conv_3_param.padMode                    = PaddingVALID;
        conv_3_param.k_c                        = 64;
        conv_3_param.in_shapes.resize(1);
        conv_3_param.in_shapes[0]               = deconv_param.out_shapes[0];
        conv_3_param.out_shapes.resize(1);
        conv_3_param.out_shapes[0].Batch        = 16;
        conv_3_param.out_shapes[0].Channel      = 64;
        conv_3_param.out_shapes[0].Height       = 73;
        conv_3_param.out_shapes[0].Width        = 73;
        conv_3_param.out_shapes[0].format       = TENSOR_FORMATE::NHWC;
        conv_3_param.hasBias                    = true;
        conv_3_param.dilateX                    = 0;
        conv_3_param.dilateY                    = 0;
        conv_3_param.op_type                    = OP_type::CONVOLUTION;
        conv_3_param.node_name                  = "conv_3/project_1";
        conv_3_param.quantized_type             = QUANITIZED_TYPE::FLOAT32_REGULAR;
        weights_size                            = conv_3_param.k_c * conv_3_param.in_shapes[0].Channel * conv_3_param.k_h 
                                                                    *conv_3_param.k_w;
        conv_3_param.conv_weights               = (float* )xcalloc(weights_size, sizeof(float));
        conv_3_param.conv_bias                  = (float* )xcalloc(conv_3_param.k_c, sizeof(float));
        conv_3_param.inIndexs.push_back(3);
        for(int ii = 0; ii < conv_3_param.k_c; ii++){
            conv_3_param.conv_bias[ii] = std::tanh(ii* 2.f);
        }
        for(int ii = 0; ii < weights_size; ii++){
            conv_3_param.conv_weights[ii] = std::sin(2 * ii * 2.f);
        }       
        NetList.layer_ops.push_back(conv_3_param);



        //fifth layer
        BrixLab::strParam<float> inner_product_param;
        inner_product_param.k_w                        = 1;
        inner_product_param.k_h                        = 1;
        inner_product_param.stridesY                   = 1;
        inner_product_param.stridesX                   = 1;
        inner_product_param.padMode                    = PaddingVALID;
        inner_product_param.k_c                        = 128;
        inner_product_param.in_shapes.resize(1);
        inner_product_param.in_shapes[0]               = conv_3_param.out_shapes[0];
        inner_product_param.out_shapes.resize(1);
        inner_product_param.out_shapes[0].Batch        = 16;
        inner_product_param.out_shapes[0].Channel      = 128;
        inner_product_param.out_shapes[0].Height       = 1;
        inner_product_param.out_shapes[0].Width        = 1;
        inner_product_param.out_shapes[0].format       = TENSOR_FORMATE::NHWC;
        inner_product_param.hasBias                    = true;
        inner_product_param.dilateX                    = 0;
        inner_product_param.dilateY                    = 0;
        inner_product_param.op_type                    = OP_type::INNERPRODUCT;
        inner_product_param.node_name                  = "fully_connected/project_1";
        inner_product_param.quantized_type             = QUANITIZED_TYPE::FLOAT32_REGULAR;
        weights_size                                   = inner_product_param.k_c 
                                                                    * inner_product_param.in_shapes[0].Channel 
                                                                    * inner_product_param.in_shapes[0].Height 
                                                                    * inner_product_param.in_shapes[0].Width;
        inner_product_param.innerWeights               = (float* )xcalloc(weights_size, sizeof(float));
        inner_product_param.innerBias                  = (float* )xcalloc(inner_product_param.k_c, sizeof(float));
        inner_product_param.inIndexs.push_back(4);
        for(int ii = 0; ii < inner_product_param.k_c; ii++){
            inner_product_param.innerBias[ii] = std::tanh(ii* 2.f);
        }
        for(int ii = 0; ii < weights_size; ii++){
            inner_product_param.innerWeights[ii] = std::sin(2 * ii * 2.f);
        }
        NetList.layer_ops.push_back(inner_product_param);

        graphSetLink<float> g_net(0, 0);
        for(unsigned int i = 0; i < NetList.layer_ops.size();i++){
            strParam<float> param   = NetList.layer_ops[i];
            LOG(DEBUG_INFO)<<i<<","<<get_mapped_op_string(param.op_type);
            if(param.op_type == OP_type::DATA_INPUTS){
                strLayerNode<float> da_node(strNodeParam<float>(OP_type::DATA_INPUTS));
                da_node.node_param.node_name        = "data_inputs"; 
                da_node.node_param.top_memory       = dnnl::memory({{inShape}, d_type, tag_type}, BrixLab::graph_eng);
                da_node.node_param.op_type          = DATA_INPUTS;
                write_to_dnnl_memory(data, da_node.node_param.top_memory);
                graph_insert(g_net, &da_node);
            }else{
                std::string OP_name                 = get_mapped_op_string(param.op_type) + "_layer_setup";
                strLayerNode<float> node            = getSetupFunc(OP_name)(param);
                graph_insert(g_net, &node);
            }
        }
        


        LOG(DEBUG_INFO)<<"head op_name: "<<g_net.head->node_param.node_name;

        LOG(DEBUG_INFO)<<"tail op_name: "<<g_net.tail->node_param.node_name<<", cur_index: "<<g_net.current_index;

        LOG(DEBUG_INFO)<<"g_net[1] op_name: "<<g_net[1]->node_param.node_name;
        LOG(DEBUG_INFO)<<"g_net[2] op_name: "<<g_net[2]->node_param.node_name;
        LOG(DEBUG_INFO)<<"g_net[3] op_name: "<<g_net[3]->node_param.node_name;

        strLayerNode<float>* in_node    = g_net.head;
        strLayerNode<float>* front_node = nullptr;
        while(in_node != nullptr){
            OP_type type        = in_node->node_param.op_type;
            std::string OP_name = get_mapped_op_string(type);
            LOG(DEBUG_INFO)<<"next OP_name: "<<OP_name<<", node name: "<<in_node->node_param.node_name;
            front_node          = in_node->front;
            if(front_node != nullptr){
                type            = front_node->node_param.op_type;
                OP_name         = get_mapped_op_string(type);
                LOG(DEBUG_INFO)<<"front OP_name: "<<OP_name<<", node name: "<<front_node->node_param.node_name;
            }

            if(in_node->node_param.op_type == OP_type::DATA_INPUTS){
                LOG(DEBUG_INFO)<<"DATA_INPUTS";
                in_node         = in_node->next;
                continue;
            }
            in_node->node_param.inference_forward(&(in_node->node_param), g_net);
            in_node             = in_node->next;
        }
        graph_stream.wait();

        LOG(DEBUG_INFO)<<"Unit test_convolution passed!";
    }

    void Test_groupConvolution(){
        std::vector<float> src(16*16*16);
        std::vector<float> wei(16*3*3);
        std::vector<float> bias(16);
        std::vector<float> dst(16*16*16);

        memory::dims src_tz = {1, 16, 16, 16 };
        memory::dims wei_tz = {16, 1, 1, 3, 3 };
        memory::dims bias_tz = {16};
        memory::dims dst_tz = src_tz;

        auto u_src_md = memory::desc(src_tz, memory::data_type::f32, memory::format_tag::nchw);
        auto u_wei_md = memory::desc(wei_tz, memory::data_type::f32, memory::format_tag::giohw);
        auto u_dst_md = memory::desc(dst_tz, memory::data_type::f32, memory::format_tag::nchw);

        auto u_src_m = memory(u_src_md, graph_eng, src.data());
        auto u_wei_m = memory(u_wei_md, graph_eng, wei.data());
        auto u_dst_m = memory(u_dst_md, graph_eng, dst.data());

        auto d_src_md = memory::desc(src_tz, memory::data_type::f32, memory::format_tag::any);
        auto d_wei_md = memory::desc(wei_tz, memory::data_type::f32, memory::format_tag::any);
        auto d_dst_md = memory::desc(dst_tz, memory::data_type::f32, memory::format_tag::any);
        auto d_bias_md = memory::desc(bias_tz, memory::data_type::f32, memory::format_tag::any);
        auto deconv_d = convolution_forward::desc(prop_kind::forward_inference,
                dnnl::algorithm::convolution_direct, d_src_md, d_wei_md, d_bias_md,  d_dst_md,
                {1, 1}, {1, 1},{1,1});
        auto deconv_pd = convolution_forward::primitive_desc(deconv_d, graph_eng);

        auto d_src_m = memory(deconv_pd.src_desc(), graph_eng);
        auto d_wei_m = memory(deconv_pd.weights_desc(), graph_eng);
        auto d_dst_m = memory(deconv_pd.dst_desc(), graph_eng);
        auto d_bias_m = memory(deconv_pd.bias_desc(), graph_eng);

        auto dst_shape  = d_dst_m.get_desc().dims();
        print_dnnl_memory_shape(dst_shape, "dst_shape"); 

        std::vector<primitive> net;
        std::vector<std::unordered_map<int, memory>> net_args;
        // note that some of the reorders might be redundant
        // create and execute all of them just for simplicity
        net.push_back(reorder(u_src_m, d_src_m));
        net_args.push_back({{DNNL_ARG_FROM, u_src_m},
                            {DNNL_ARG_TO, d_src_m}});
        reorder(u_wei_m, d_wei_m).execute(graph_stream, u_wei_m, d_wei_m);
        net.push_back(convolution_forward(deconv_pd));
        net_args.push_back({{DNNL_ARG_SRC, d_src_m},
                            {DNNL_ARG_WEIGHTS, d_wei_m},
                            {DNNL_ARG_BIAS, d_bias_m},
                            {DNNL_ARG_DST, d_dst_m}});
        for (size_t i = 0; i < net.size(); ++i)
            net.at(i).execute(graph_stream, net_args.at(i));

        graph_stream.wait();
    }
    double get_current_time(){
        struct timeval tv;
        gettimeofday(&tv, NULL);

        return tv.tv_sec * 1000.0 + tv.tv_usec / 1000.0;
    }
    void post_deeplab_v3(cv::Mat &result, const float *inference, 
                                        const TensorShape &inf_shape, 
                                        const int &in_H, const int &in_W,
                                        const int &src_H, const int &src_W){
        cv::Mat mask = cv::Mat(in_H, in_W, CV_8UC1, cv::Scalar(0));
        unsigned char* maskData = mask.data;
        for (int c = 0; c < inf_shape.Channel; ++c) {
            float cIndex = -1;
            for (int y = 0; y < inf_shape.Height; ++y) {
                for (int x = 0; x < inf_shape.Height; ++x)
                {
                    float max = -50;
                    if (max < inference[c * inf_shape.Height * inf_shape.Width + y * inf_shape.Width + x])
                    {
                        max = inference[c * inf_shape.Height * inf_shape.Width + y * inf_shape.Width + x];
                        cIndex = c;
                    }
                    if (cIndex == 15 && max > 10)
                        maskData[y * inf_shape.Width + x] = 255;
                }
                
            }
        }
        cv::resize(mask, result, cv::Size(src_W, src_H), 0, 0, cv::INTER_CUBIC);
    }

    void Test_tflite(const string& tflite_file, const int& in_H, const int& in_W,
                            const int& in_C, const string& img_file){
        int width = 0, height = 0;

        cv::Mat src_img     = cv::imread(img_file);
        width               = src_img.rows;
        height              = src_img.cols;

        cv::Mat image;
        double start, end;
        start =  get_current_time();
        cv::resize(src_img, image, cv::Size(in_H, in_W), 0, 0, cv::INTER_AREA);
        int cnls = image.type();
        if (cnls == CV_8UC1) {
            cv::cvtColor(image, image, cv::COLOR_GRAY2RGB);
        }
        else if (cnls == CV_8UC3) {
            cv::cvtColor(image, image, cv::COLOR_BGR2RGB);
        }
        else if (cnls == CV_8UC4) {
            cv::cvtColor(image, image, cv::COLOR_BGRA2RGB);
        }
        cv::Mat fimage;
		image.convertTo(fimage, CV_32FC3, 1 / 128.0, -128.0 / 128.0);
        end = get_current_time();
        LOG(DEBUG_INFO)<<"preprocess time: "<<end -start;
		// Copy image into input tensor
		float *input_img    = (float*)xcalloc(in_C * in_H * in_W, sizeof(float));
		memcpy(input_img, fimage.data, sizeof(float) * in_C * in_H * in_W);

        dnnl::memory::dims inShape  = {1, in_C, in_H, in_W};
        auto temp_input             = dnnl::memory({{inShape}, dt::f32, tag::nhwc}, BrixLab::graph_eng);
        write_to_dnnl_memory(input_img, temp_input);
        NetGraph<float> tfGraph_Net(in_H, in_W, 0, tflite_file);
        NetT<float> deeplab_Net     = tfGraph_Net.tfliteConvertGraphList();
        tfGraph_Net.make_graph(deeplab_Net, input_img, tag::nhwc);
        //tfGraph_Net.printf_netGraph();
        start =  get_current_time();
        tfGraph_Net.network_predict();
        strLayerNode<float>* output = tfGraph_Net.getGraphOutput();
        #ifdef USE_DEBUG
            print_dnnl_memory_shape(output->node_param.top_shape, "output_shape");
            LOG(DEBUG_INFO)<<"op_name: "<<output->node_param.node_name;
        #endif
        const int out_size          = product_dnnl_memory_shape(output->node_param.top_shape);
        TensorShape out_shape       = output->node_param.out_shapes[0];
        
        // deeplab_v3_post_ops
        cv::Mat mask;
        float* infernce             = (float*)xcalloc(out_size, sizeof(float));
        read_from_dnnl_memory((void*)infernce, output->node_param.top_memory);
        post_deeplab_v3(mask, infernce, out_shape, in_H, in_W, width, height);
        end = get_current_time();
        std::cout<<"inference & post time: "<<end - start<<"ms"<<std::endl;
        std::string outName         =  "../images/result.png";
        cv::imwrite(outName.c_str(), mask);
    }

    void Test_Reshape_Permute(){
        NetT<float> NetList;
        BrixLab::strParam<float>data_param;
        data_param.node_name        = "data_inputs";
        data_param.op_type          = OP_type::DATA_INPUTS;
        // Input Data
        NetList.layer_ops.push_back(data_param);
        int BSH     = 4;
        int BSW     = 4;
        int CSH     = 0;
        int CEH     = 3;
        int CSW     = 0;
        int CEW     = 3;
        int inB     = 16;
        int inC     = 4;
        int inH     = 8;
        int inW     = 8;
        int outB    = 1;
        int outC    = inC;
        int outH    = inH * BSH - CSH - CEH;
        int outW    = inW * BSW - CSW - CEW;
        int PSH     = 4;
        int PEH     = 7;
        int PSW     = 4;
        int PEW     = 7;
        int outSH   = (outH + PSH + PEH ) / BSH;
        int outSW   = (outW + PSW + PEW ) / BSW;
        std::vector<float> srcSet(inB * inC *inH *inW);
        std::srand((int)time(0));
        std::generate(srcSet.begin(), srcSet.end(), []() {
            return std::rand()%10;
        });
        //Test Batch to Space
        BrixLab::strParam<float> BSreshape_param;
        BSreshape_param.node_name     = "Batch_ToSpaceND";
        BSreshape_param.op_type       = BrixLab::OP_type::SPACE_PERMUTES;
        BSreshape_param.block_shape   = {BSH, BSW};
        BSreshape_param.crop_size     = {CSH, CEH, CSW, CEW};
        BSreshape_param.in_shapes.resize(1);
        BSreshape_param.out_shapes.resize(1);
        BSreshape_param.in_shapes[0].Batch    = inB;
        BSreshape_param.in_shapes[0].Channel  = inC;
        BSreshape_param.in_shapes[0].Height   = inH;
        BSreshape_param.in_shapes[0].Width    = inW;
        BSreshape_param.in_shapes[0].format   = NCHW;
        BSreshape_param.out_shapes[0].Batch   = outB;
        BSreshape_param.out_shapes[0].Channel = outC;
        BSreshape_param.out_shapes[0].Height  = outH;
        BSreshape_param.out_shapes[0].Width   = outW;
        BSreshape_param.out_shapes[0].format  = NCHW;
        BSreshape_param.quantized_type        = QUANITIZED_TYPE::FLOAT32_REGULAR;
        BSreshape_param.perm_type             = BrixLab::DATA_PERMUTES_TYPE::Batch_To_SapceND;
        BSreshape_param.inIndexs.push_back(0);
        NetList.layer_ops.push_back(BSreshape_param);
        BrixLab::strParam<float> SBreshape_param;
        SBreshape_param.node_name     = "Space_ToBatchND";
        SBreshape_param.op_type       = BrixLab::OP_type::SPACE_PERMUTES;
        SBreshape_param.block_shape   = {BSH, BSW};
        SBreshape_param.crop_size     = {PSH, PEH, PSW, PEW};
        SBreshape_param.in_shapes.resize(1);
        SBreshape_param.out_shapes.resize(1);
        SBreshape_param.in_shapes[0].Batch    = outB;
        SBreshape_param.in_shapes[0].Channel  = outC;
        SBreshape_param.in_shapes[0].Height   = outH;
        SBreshape_param.in_shapes[0].Width    = outW;
        SBreshape_param.in_shapes[0].format   = NCHW;
        SBreshape_param.out_shapes[0].Batch   = inB;
        SBreshape_param.out_shapes[0].Channel = inC;
        SBreshape_param.out_shapes[0].Height  = outSH;
        SBreshape_param.out_shapes[0].Width   = outSW;
        SBreshape_param.out_shapes[0].format  = NCHW;
        SBreshape_param.quantized_type        = QUANITIZED_TYPE::FLOAT32_REGULAR;
        SBreshape_param.perm_type             = BrixLab::DATA_PERMUTES_TYPE::Space_To_BatchND;
        SBreshape_param.inIndexs.push_back(1);
        NetList.layer_ops.push_back(SBreshape_param);
        graphSetLink<float> g_net(0, 0);
        for(unsigned int i = 0; i < NetList.layer_ops.size();i++){
            strParam<float> param   = NetList.layer_ops[i];
            LOG(DEBUG_INFO)<<i<<","<<get_mapped_op_string(param.op_type);
            if(param.op_type == OP_type::DATA_INPUTS){
                strLayerNode<float> da_node(strNodeParam<float>(OP_type::DATA_INPUTS));
                da_node.node_param.node_name        = "data_inputs"; 
                da_node.node_param.top_memory       = dnnl::memory({{inB, inC, inH, inW}, dt::f32, tag::nchw}, BrixLab::graph_eng);
                da_node.node_param.op_type          = DATA_INPUTS;
                write_to_dnnl_memory(srcSet.data(), da_node.node_param.top_memory);
                #ifdef USE_DUMP_FILE
                dump_data_to_file_c(srcSet.data(), srcSet.size(), da_node.node_param.node_name.c_str(), da_node.node_param.top_memory.get_desc().dims());
                #endif
                graph_insert(g_net, &da_node);
            }else{
                std::string OP_name                 = get_mapped_op_string(param.op_type) + "_layer_setup";
                strLayerNode<float> node            = getSetupFunc(OP_name)(param);
                graph_insert(g_net, &node);
            }
        }
        strLayerNode<float>* in_node    = g_net.head;
        strLayerNode<float>* front_node = nullptr;
        while(in_node != nullptr){
            OP_type type        = in_node->node_param.op_type;
            std::string OP_name = get_mapped_op_string(type);
            LOG(DEBUG_INFO)<<"next OP_name: "<<OP_name<<", node name: "<<in_node->node_param.node_name;
            front_node          = in_node->front;
            if(front_node != nullptr){
                type            = front_node->node_param.op_type;
                OP_name         = get_mapped_op_string(type);
                LOG(DEBUG_INFO)<<"front OP_name: "<<OP_name<<", node name: "<<front_node->node_param.node_name;
            }
            if(in_node->node_param.op_type == OP_type::DATA_INPUTS){
                LOG(DEBUG_INFO)<<"DATA_INPUTS";
                in_node         = in_node->next;
                continue;
            }
            in_node->node_param.inference_forward(&(in_node->node_param), g_net);
            in_node             = in_node->next;
        }
        graph_stream.wait();

        LOG(DEBUG_INFO)<<"Unit test_data_Transposed passed!";
    }
} // namespace BrixLab
