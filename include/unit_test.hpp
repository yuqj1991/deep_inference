#ifndef INTEL_MKL_DNN_UNIT_TEST_
#define INTEL_MKL_DNN_UNIT_TEST_
#include "layer_builder.hpp"
#include "check_error.hpp"
#include "oneapi/dnnl.hpp"
#include "opencv2/core.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/highgui.hpp"
namespace BrixLab{
    void Test_Deconvulution(float *data);

    void Test_convolution();
    void Test_demoNet(float *data);

    void Test_groupConvolution();
    void Test_Reshape_Permute();

    void post_deeplab_v3(cv::Mat &result, const float *inference, 
                            const TensorShape &inf_shape, const int &in_H, const int &in_W,
                                        const int &src_H, const int &src_W);
    
    void Test_tflite(const string& tflite_file, const int& in_H, 
                            const int& in_W, const int& in_C, const string& img_file);
}
#endif