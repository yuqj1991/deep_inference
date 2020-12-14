#include "unit_test.hpp"
#include <numeric>
namespace BrixLab
{
    void Test_Deconvulution(float *data){
        BrixLab::layerWeightsParam<float> deconv_param;
        deconv_param.k_w = 2;
        deconv_param.k_h = 2;
        deconv_param.stridesY = 2;
        deconv_param.stridesX = 2;
        deconv_param.padMode = PaddingVALID;
        deconv_param.k_c = 128;

        deconv_param.inBatch = 3;
        deconv_param.inChannel = 64;
        deconv_param.inHeight = 14;
        deconv_param.inWidth = 14;
        deconv_param.hasBias = true;
        deconv_param.dilateX = 0;
        deconv_param.dilateY = 0;
        int weights_size = deconv_param.k_c * deconv_param.inChannel * deconv_param.k_h *deconv_param.k_w;
        deconv_param.transposed_weights = (float* )xcalloc(weights_size, sizeof(float));
        deconv_param.transposed_bias = (float* )xcalloc(deconv_param.k_c, sizeof(float));
        for(int ii = 0; ii < deconv_param.k_c; ii++){
            deconv_param.transposed_bias[ii] = std::tanh(ii* 2.f);
        }
        for(int ii = 0; ii < weights_size; ii++){
            deconv_param.transposed_weights[ii] = std::sin(2 * ii * 2.f);
        }
        deconv_param.op_type = OP_type::DECONVOLUTION;

        dnnl::memory::dims inShape = {deconv_param.inBatch, deconv_param.inChannel, deconv_param.inHeight, deconv_param.inWidth};
        auto temp_input = dnnl::memory({{inShape}, dt::f32, tag::nchw}, BrixLab::graph_eng);
        write_to_dnnl_memory(data, temp_input);
        printf("[Unit test Deconvolution] has writen source data to input memory\n");
        graphSet<float> g_net(0, 0, temp_input);
        
        std::string OP_deconvolution = get_mapped_op_string(deconv_param.op_type) + "_layer_setup";

        layerNode<float> node = getSetupFunc(OP_deconvolution)(deconv_param);
        graph_insert(g_net, &node);
        printf("[Unit test Deconvolution] Net size: %d\n", g_net.graphSize);
        assert(g_net.head->src_weights_memory.get_data_handle() != nullptr);
        auto current_node = g_net.head;
        current_node->inference_forward(current_node, g_net);
        printf("[Unit test Deconvolution] it passed!\n");
    }
    memory::dim product(const memory::dims &dims) {
    return std::accumulate(dims.begin(), dims.end(), (memory::dim)1,
            std::multiplies<memory::dim>());
    }

    void Test_Convulution(float *data){
        BrixLab::layerWeightsParam<float> conv_param;
        conv_param.k_w = 3;
        conv_param.k_h = 3;
        conv_param.stridesX = 2;
        conv_param.stridesY = 2;
        conv_param.padMode = PaddingVALID;
        conv_param.k_c = 128;

        conv_param.inBatch = 3;
        conv_param.inChannel = 64;
        conv_param.inHeight = 14;
        conv_param.inWidth = 14;
        
        conv_param.hasBias = true;
        conv_param.dilateX = 0;
        conv_param.groups = 1;
        int weights_size = conv_param.inChannel * conv_param.k_c * conv_param.k_w * conv_param.k_h;
        conv_param.conv_weights = (float* )xcalloc(weights_size, sizeof(float));
        conv_param.conv_bias = (float* )xcalloc(conv_param.k_c, sizeof(float));
        for(int ii = 0; ii < conv_param.k_c; ii++){
            conv_param.conv_bias[ii] = std::tanh(ii * 2.f);
        }
        for(int ii = 0; ii < weights_size; ii++){
            conv_param.conv_weights[ii] = std::sin(2 * ii * 2.f);
        }
        conv_param.op_type = OP_type::CONVOLUTION;

        dnnl::memory::dims inShape = {conv_param.inBatch, conv_param.inChannel, conv_param.inHeight, conv_param.inWidth};
        auto temp_input = dnnl::memory({{inShape}, dt::f32, tag::nchw}, BrixLab::graph_eng);
        write_to_dnnl_memory(data, temp_input);
        LOG(DEBUG_INFO, "")<<" source data has writen to input memory";
        
        graphSet<float> g_net(0, 0, temp_input);
        
        std::string OP_name = get_mapped_op_string(conv_param.op_type) + "_layer_setup";

        layerNode<float> node = getSetupFunc(OP_name)(conv_param);

        printf("[%s] layer had been setup!\n", __FUNCTION__);

        graph_insert(g_net, &node);

        printf("[%s] has insert new node into the graph,Net size: %d\n", __FUNCTION__, g_net.graphSize);

        auto current_node = g_net.head;
        current_node->inference_forward(current_node, g_net);
        
        printf("[%s] passed!\n", __FUNCTION__);
    }
} // namespace BrixLab
