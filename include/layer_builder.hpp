#ifndef LAYERS_BUILDER_MODELS_
#define LAYERS_BUILDER_MODELS_
#include <algorithm>
#include <cmath>
#include <iostream>
#include <string>
#include <vector>
#include <map>
#include <assert.h>
#include <malloc.h>
#include <fstream>
#include <chrono>
#include <vector>
#include <unordered_map>

#include "oneapi/dnnl.hpp"
#include "utils.hpp"
#include "check_error.hpp"

#include "utils_help.hpp"

#include "flatbuffers/flatbuffers.h"
#include "schema_generated.h"
#include "liteopConvert.hpp"
#include "logkit.hpp"

using namespace dnnl;

namespace BrixLab
{
    using tag = memory::format_tag;
    using dt = memory::data_type;
    
    template<typename DType>
    void OP_convolution_inference_forward(strNodeParam<DType> *node, graphSetLink<DType> &g_net);
    
    template<typename DType>
    strLayerNode<DType> OP_convolution_layer_setup(const strParam<DType> &param);

    template<typename DType>    
    void OP_batchnorm_inference_forward(strNodeParam<DType> *node, graphSetLink<DType> &g_net);
    
    template<typename DType>
    strLayerNode<DType> OP_batchnorm_layer_setup(const strParam<DType> &param);

    template<typename DType>
    void OP_pooling_inference_forward(strNodeParam<DType> *node, graphSetLink<DType> &g_net);
    
    template<typename DType>
    strLayerNode<DType> OP_pooling_layer_setup(const strParam<DType> &param);

    template<typename DType>
    void OP_concat_inference_forward(strNodeParam<DType> *node, graphSetLink<DType> &g_net);
    
    template<typename DType>
    strLayerNode<DType> OP_concat_layer_setup(const strParam<DType> &param);

    template<typename DType>
    void OP_sum_inference_forward(strNodeParam<DType> *node, graphSetLink<DType> &g_net);
    template<typename DType>
    strLayerNode<DType> OP_sum_layer_setup(const strParam<DType> &param);

    template<typename DType>
    void OP_resample_inference_forward(strNodeParam<DType> *node, graphSetLink<DType> &g_net);
    template<typename DType>
    strLayerNode<DType> OP_resample_layer_setup(const strParam<DType> &param);

    template<typename DType>
    void OP_deconvolution_inference_forward(strNodeParam<DType> *node, graphSetLink<DType> &g_net);
    template<typename DType>
    strLayerNode<DType> OP_deconvolution_layer_setup(const strParam<DType> &param);

    template<typename DType>
    void OP_innerproduct_inference_forward(strNodeParam<DType> *node, graphSetLink<DType> &g_net);
    template<typename DType>
    strLayerNode<DType> OP_innerproduct_layer_setup(const strParam<DType> &param);

    template<typename DType>
    void OP_activation_inference_forward(strNodeParam<DType> *node, graphSetLink<DType> &g_net);
    template<typename DType>
    strLayerNode<DType> OP_activation_layer_setup(const strParam<DType> &param);

    template<typename DType>
    void OP_binary_inference_forward(strNodeParam<DType> *node, graphSetLink<DType> &g_net);
    template<typename DType>
    strLayerNode<DType> OP_binary_layer_setup(const strParam<DType> &param);

    
    template<typename DType>
    class NetGraph{
        public:
        int get_graph_size() const;
        int get_GraphinWidth() const;
        int get_GraphinHeight() const;
        strLayerNode<DType> *getGraphOutput();
        NetGraph(const int &inH, const int &inW, const int &size, const std::string &tflite_path, const memory &input);
        ~NetGraph();

        void network_predict();
        void make_netParamfromTflite(const std::string &tflite_file);
        void make_graph(const NetT<DType>& g_net);
        NetT<DType> tfliteConvertGraphList();
        void printf_netGraph();
        private:
        int input_w;
        int input_h;
        graphSetLink<DType> graph_state;
        int graph_size;
        std::unique_ptr<tflite::ModelT> _tflite_model;
        std::string tflite_file;
    };

    typedef strLayerNode<float> (*LayerFloatSetup)(const strParam<float> &param);
    LayerFloatSetup getSetupFunc(const std::string &func_name);

    typedef strLayerNode<uint8_t> (*LayerUint8Setup)(const strParam<uint8_t> &param);
    LayerUint8Setup getSetupUintFunc(const std::string &func_name);


    #define INSTANCE_LAYEROP(opname)    \
        template strLayerNode<float> OP_##opname##_layer_setup(const strParam<float>& param); \
        template void OP_##opname##_inference_forward(strNodeParam<float>* node, graphSetLink<float>& g_net); \
        template strLayerNode<uint8_t> OP_##opname##_layer_setup(const strParam<uint8_t>& param); \
        template void OP_##opname##_inference_forward(strNodeParam<uint8_t>* node, graphSetLink<uint8_t>& g_net);

    #define INSTANEC_CLASSNET(name)     \
        template class name<float>
} // namespace BrixLab

#endif