#include "unit_test.hpp"
#include <numeric>
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"
#define STB_IMAGE_RESIZE_IMPLEMENTATION
#include "stb_image_resize.h"
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
        graphSetLink<float> g_net(0, 0, temp_input);
        
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

    void Test_demoNet(float *data){
        // first layer
        BrixLab::strParam<float> conv_param;
        conv_param.k_w                      = 3;
        conv_param.k_h                      = 3;
        conv_param.stridesX                 = 2;
        conv_param.stridesY                 = 2;
        conv_param.padMode                  = PaddingVALID;
        conv_param.k_c                      = 32;
        conv_param.in_shapes.resize(1);
        conv_param.in_shapes[0].Batch       = 16;
        conv_param.in_shapes[0].Channel     = 3;
        conv_param.in_shapes[0].Height      = 300;
        conv_param.in_shapes[0].Width       = 300;
        conv_param.in_shapes[0].format      = TENSOR_FORMATE::NCHW;
        conv_param.out_shapes.resize(1);
        conv_param.out_shapes[0].Batch      = 16;
        conv_param.out_shapes[0].Channel    = 32;
        conv_param.out_shapes[0].Height     = 149;
        conv_param.out_shapes[0].Width      = 149;
        conv_param.out_shapes[0].format     = TENSOR_FORMATE::NCHW;
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
        dnnl::memory::dims inShape          = {conv_param.in_shapes[0].Batch, conv_param.in_shapes[0].Channel, 
                                                    conv_param.in_shapes[0].Height, conv_param.in_shapes[0].Width};
        auto temp_input                     = dnnl::memory({{inShape}, dt::f32, tag::nchw}, BrixLab::graph_eng);
        for(int ii = 0; ii < conv_param.k_c; ii++){
            conv_param.conv_bias[ii] = std::tanh(ii * 2.f);
        }
        for(int ii = 0; ii < weights_size; ii++){
            conv_param.conv_weights[ii] = std::sin(2 * ii * 2.f);
        }
        conv_param.inIndexs.push_back(0);
        write_to_dnnl_memory(data, temp_input);
        graphSetLink<float> g_net(0, 0, temp_input);
        strLayerNode<float> da_node(strNodeParam<float>(OP_type::DATA_INPUTS));
        da_node.node_param.node_name            = "data_inputs"; 
        graph_insert(g_net, &da_node);

        std::string OP_name         = get_mapped_op_string(conv_param.op_type) + "_layer_setup";
        strLayerNode<float> node    = getSetupFunc(OP_name)(conv_param);
        graph_insert(g_net, &node);

        //second layer
        BrixLab::strParam<float> deconv_param;
        deconv_param.k_w                        = 2;
        deconv_param.k_h                        = 2;
        deconv_param.stridesY                   = 2;
        deconv_param.stridesX                   = 2;
        deconv_param.padMode                    = PaddingVALID;
        deconv_param.k_c                        = 32;
        deconv_param.in_shapes.resize(1);
        deconv_param.in_shapes[0]               = conv_param.out_shapes[0];
        deconv_param.out_shapes.resize(1);
        deconv_param.out_shapes[0].Batch        = 16;
        deconv_param.out_shapes[0].Channel      = 32;
        deconv_param.out_shapes[0].Height       = 298;
        deconv_param.out_shapes[0].Width        = 298;
        deconv_param.out_shapes[0].format       = TENSOR_FORMATE::NCHW;
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
        deconv_param.inIndexs.push_back(1);
        for(int ii = 0; ii < deconv_param.k_c; ii++){
            deconv_param.transposed_bias[ii] = std::tanh(ii* 2.f);
        }
        for(int ii = 0; ii < weights_size; ii++){
            deconv_param.transposed_weights[ii] = std::sin(2 * ii * 2.f);
        }       
        std::string OP_deconvolution    = get_mapped_op_string(deconv_param.op_type) + "_layer_setup";
        node                            = getSetupFunc(OP_deconvolution)(deconv_param);
        graph_insert(g_net, &node);

        //third layer
        BrixLab::strParam<float> conv_2_param;
        conv_2_param.k_w                        = 3;
        conv_2_param.k_h                        = 3;
        conv_2_param.stridesY                   = 2;
        conv_2_param.stridesX                   = 2;
        conv_2_param.padMode                    = PaddingVALID;
        conv_2_param.k_c                        = 64;
        conv_2_param.in_shapes.resize(1);
        conv_2_param.in_shapes[0]               = deconv_param.out_shapes[0];
        conv_2_param.out_shapes.resize(1);
        conv_2_param.out_shapes[0].Batch        = 16;
        conv_2_param.out_shapes[0].Channel      = 64;
        conv_2_param.out_shapes[0].Height       = 148;
        conv_2_param.out_shapes[0].Width        = 148;
        conv_2_param.out_shapes[0].format       = TENSOR_FORMATE::NCHW;
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
        conv_2_param.inIndexs.push_back(2);
        for(int ii = 0; ii < conv_2_param.k_c; ii++){
            conv_2_param.conv_bias[ii] = std::tanh(ii* 2.f);
        }
        for(int ii = 0; ii < weights_size; ii++){
            conv_2_param.conv_weights[ii] = std::sin(2 * ii * 2.f);
        }       
        OP_name             = get_mapped_op_string(conv_2_param.op_type) + "_layer_setup";
        node                = getSetupFunc(OP_name)(conv_2_param);
        graph_insert(g_net, &node);

        //fourth layer
        BrixLab::strParam<float> inner_product_param;
        inner_product_param.k_w                        = 1;
        inner_product_param.k_h                        = 1;
        inner_product_param.stridesY                   = 1;
        inner_product_param.stridesX                   = 1;
        inner_product_param.padMode                    = PaddingVALID;
        inner_product_param.k_c                        = 128;
        inner_product_param.in_shapes.resize(1);
        inner_product_param.in_shapes[0]               = conv_2_param.out_shapes[0];
        inner_product_param.out_shapes.resize(1);
        inner_product_param.out_shapes[0].Batch        = 16;
        inner_product_param.out_shapes[0].Channel      = 128;
        inner_product_param.out_shapes[0].Height       = 1;
        inner_product_param.out_shapes[0].Width        = 1;
        inner_product_param.out_shapes[0].format       = TENSOR_FORMATE::NCHW;
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
        inner_product_param.inIndexs.push_back(3);
        for(int ii = 0; ii < inner_product_param.k_c; ii++){
            inner_product_param.innerBias[ii] = std::tanh(ii* 2.f);
        }
        for(int ii = 0; ii < weights_size; ii++){
            inner_product_param.innerWeights[ii] = std::sin(2 * ii * 2.f);
        }       
        OP_name             = get_mapped_op_string(inner_product_param.op_type) + "_layer_setup";
        node                = getSetupFunc(OP_name)(inner_product_param);
        graph_insert(g_net, &node);


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
                in_node->node_param.layer_top_memory = g_net.input;
                in_node                              = in_node->next;
                continue;
            }
            in_node->node_param.inference_forward(&(in_node->node_param), g_net);
            in_node             = in_node->next;
        }


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
    void post_deeplab_v3(cv::Mat &result, const float *inference, 
                                        const TensorShape &inf_shape, const int &in_H, const int &in_W){
        cv::Mat mask = cv::Mat(inf_shape.Height, inf_shape.Width, CV_8UC1, cv::Scalar(0));
        unsigned char* maskData = mask.data;
        int segmentedPixels = 0;
        if(1){
            for (int y = 0; y < inf_shape.Height; ++y) {
                for (int x = 0; x < inf_shape.Width; ++x) {
                    float max = -50;
                    float cIndex = -1;
                    for (int c = 0; c < 21 ; c++)
                    {
                        if (max < inference[y*inf_shape.Width * 21 + x * 21 + c])
                        {
                            max = inference[y*inf_shape.Width * 21 + x * 21 + c];
                            cIndex = c;
                        }
                    }
                    if (cIndex == 15 && max > 10)
                        maskData[y * inf_shape.Width + x] = 255;
                }
            }
        }else{
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
        }
        cv::resize(mask, result, cv::Size(in_W, in_H), 0, 0, cv::INTER_CUBIC);
    }

    void Test_tflite(const string& tflite_file, const int& in_H, const int& in_W,
                            const int& in_C, const string& img_file){
                                
        int width = 0, height = 0, channel = 0;

        cv::Mat src_img     = cv::imread(img_file);
        width               = src_img.rows;
        height              = src_img.cols;
        channel             = src_img.channels();

        cv::Mat image;
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
		// Copy image into input tensor
		float *input_img    = (float*)xcalloc(in_C * in_H * in_W, sizeof(float));
		memcpy(input_img, fimage.data, sizeof(float) * in_C * in_H * in_W);

        dnnl::memory::dims inShape  = {1, in_H, in_W, in_C};
        auto temp_input             = dnnl::memory({{inShape}, dt::f32, tag::nchw}, BrixLab::graph_eng);
        write_to_dnnl_memory(input_img, temp_input);
        NetGraph<float> tfGraph_Net(in_H, in_W, 0, tflite_file, temp_input);
        NetT<float> deeplab_Net     = tfGraph_Net.tfliteConvertGraphList();
        tfGraph_Net.make_graph(deeplab_Net);
        tfGraph_Net.network_predict();
        strLayerNode<float>* output = tfGraph_Net.getGraphOutput();
        print_dnnl_memory_shape(output->node_param.top_shape, "output_shape");
        const int out_size          = product_dnnl_memory_shape(output->node_param.top_shape);
        
        // deeplab_v3_post_ops
        cv::Mat mask;
        float* infernce             = (float*)xcalloc(out_size, sizeof(float));
        read_from_dnnl_memory((infernce), output->node_param.layer_top_memory);
        post_deeplab_v3(mask, infernce, output->node_param.out_shapes[0], width, height);
        std::string outName         =  "result.png";
        cv::imwrite(outName.c_str(), mask);
    }
} // namespace BrixLab
