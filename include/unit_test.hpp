#ifndef INTEL_MKL_DNN_UNIT_TEST_
#define INTEL_MKL_DNN_UNIT_TEST_
#include "layer_builder.hpp"
#include "check_error.hpp"
#include "oneapi/dnnl.hpp"
namespace BrixLab{
    void Test_Deconvulution(float *data);
    void Test_Convulution(float *data);
}
#endif