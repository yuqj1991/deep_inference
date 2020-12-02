#ifndef INTEL_MKL_DNN_UNIT_TEST_
#define INTEL_MKL_DNN_UNIT_TEST_
#include "layer_builder.hpp"
#include "check_error.hpp"
#include "oneapi/dnnl/dnnl.hpp"

using namespace BrixLab;
using namespace dnnl;
void Test_Deconvulution(const float *src_data){
    BrixLab::layerWeightsParam<float> deconv_param;
    deconv_param.k_w = 2;
    deconv_param.k_h = 2;
    deconv_param.strides = 2;
    deconv_param.padding = 0;
    deconv_param.k_c = 128;

    deconv_param.transposed_weights = (float* )xcalloc(2*2*128*64, sizeof(float));
    deconv_param.transposed_bias = (float* )xcalloc(64, sizeof(float));
    for(int ii = 0; ii < 64; ii++){
        deconv_param.transposed_bias[ii] = std::tanh(ii++ * 2.f);
        deconv_param.transposed_weights[ii * 2] = std::sin(2 * ii * 2.f);
        deconv_param.transposed_weights[ii * 2 + 1] = std::sin((2 * ii + 1) * 2.f);
    }
    deconv_param.op_type = OP_type::DECONVOLUTION;

    std::string operation = get_mapped_op_string(deconv_param.op_type);

    
    graphState<float> graph_state(1, 0);

    NODE_INTO_GRPAH(operation, deconv_param, graph_state);
    NODE_INTO_GRPAH(operation, graph_state.current, graph_state);
}

#endif