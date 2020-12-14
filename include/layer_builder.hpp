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
    void OP_convolution_inference_forward(layerNode<DType> *node, graphSet<DType> &g_net);
    
    template<typename DType>
    layerNode<DType> OP_convolution_layer_setup(const layerWeightsParam<DType> &param);

    template<typename DType>    
    void OP_batchnorm_inference_forward(layerNode<DType> *node, graphSet<DType> &g_net);
    
    template<typename DType>
    layerNode<DType> OP_batchnorm_layer_setup(const layerWeightsParam<DType> &param);

    template<typename DType>
    void OP_pooling_inference_forward(layerNode<DType> *node, graphSet<DType> &g_net);
    
    template<typename DType>
    layerNode<DType> OP_pooling_layer_setup(const layerWeightsParam<DType> &param);

    template<typename DType>
    void OP_concat_inference_forward(layerNode<DType> *node, graphSet<DType> &g_net);
    
    template<typename DType>
    layerNode<DType> OP_concat_layer_setup(const layerWeightsParam<DType> &param);

    template<typename DType>
    void OP_sum_inference_forward(layerNode<DType> *node, graphSet<DType> &g_net);
    template<typename DType>
    layerNode<DType> OP_sum_layer_setup(const layerWeightsParam<DType> &param);

    template<typename DType>
    void OP_resample_inference_forward(layerNode<DType> *node, graphSet<DType> &g_net);
    template<typename DType>
    layerNode<DType> OP_resample_layer_setup(const layerWeightsParam<DType> &param);

    template<typename DType>
    void OP_deconvolution_inference_forward(layerNode<DType> *node, graphSet<DType> &g_net);
    template<typename DType>
    layerNode<DType> OP_deconvolution_layer_setup(const layerWeightsParam<DType> &param);

    template<typename DType>
    void OP_innerproduct_inference_forward(layerNode<DType> *node, graphSet<DType> &g_net);
    template<typename DType>
    layerNode<DType> OP_innerproduct_layer_setup(const layerWeightsParam<DType> &param);

    template<typename DType>
    void OP_activation_inference_forward(layerNode<DType> *node, graphSet<DType> &g_net);
    template<typename DType>
    layerNode<DType> OP_activation_layer_setup(const layerWeightsParam<DType> &param);

    
    template<typename DType>
    class NetGraph{
        public:
        int get_Graphsize() const;
        int get_GraphinWidth() const;
        int get_GraphinHeight() const;
        layerNode<DType> *getGraphOutput();
        NetGraph(const int &inH, const int &inW, const int &size, const std::string &tflite_path, const memory &input);
        ~NetGraph();

        void network_predict();
        void make_netParamfromTflite(const std::string &tflite_file);
        void make_graph(const std::vector<layerWeightsParam<DType> > &params, const int &layer_size);
        NetT<DType> tfliteConvertGraphList();
        private:
        int input_w;
        int input_h;
        graphSet<DType> graph_state;
        int graph_size;
        std::unique_ptr<tflite::ModelT> _tflite_model;
        std::string tflite_file;
    };

    typedef layerNode<float> (*LayerSetup)(const layerWeightsParam<float> &param);
    LayerSetup getSetupFunc(const std::string &func_name);

    #define INSTANCE_LAYEROP(opname)    \
        template layerNode<float> OP_##opname##_layer_setup(const layerWeightsParam<float> &param); \
        template layerNode<uint8_t> OP_##opname##_layer_setup(const layerWeightsParam<uint8_t> &param); \
        template void OP_##opname##_inference_forward(layerNode<float> *node, graphSet<float> &g_net);  \
        template void OP_##opname##_inference_forward(layerNode<uint8_t> *node, graphSet<uint8_t> &g_net)

    #define INSTANEC_CLASSNET(name)     \
        template class name<float>;     \
        template class name<uint8_t>
} // namespace BrixLab

#endif