#include "layer_builder.hpp"
#include "utils.hpp"
#include "unit_test.hpp"
#include <iostream>
#include "logkit.hpp"
using namespace BrixLab;
int main(){
    if(1){
        int inBatch     = 16;
        int inChannel   = 3;
        int inHeight    = 300;
        int inWidth     = 300;
        std::vector<float> inData(inBatch*inChannel*inHeight*inWidth);
        float * data = inData.data();
        BrixLab::Test_demoNet(data);
        Test_groupConvolution();
    }else{
        std::string img_file = "../../images/image_1.jpg";
        Test_tflite("../../model/deeplabv3_257_mv_gpu.tflite", 257, 257, 3, img_file);
    }
    return 0;
}