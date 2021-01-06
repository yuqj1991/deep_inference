#include "layer_builder.hpp"
#include "utils.hpp"
#include "unit_test.hpp"
#include <iostream>
#include "logkit.hpp"
using namespace BrixLab;
int main(int argc, const char *argv[]){
    if(0){
        int inBatch     = 16;
        int inChannel   = 4;
        int inHeight    = 300;
        int inWidth     = 300;
        std::vector<float> inData(inBatch*inChannel*inHeight*inWidth);
        std::generate(inData.begin(), inData.end(), []() {
            static int i = 0;
            return std::tanh(i++);
        });
        float * data    = inData.data();
        int Test_Flages = 4;
        printf("\n");
        if(Test_Flages == 1){
            BrixLab::Test_demoNet(data);
        }else if(Test_Flages == 2){
            BrixLab::Test_convolution();
        }else if(Test_Flages == 3){
            BrixLab::Test_groupConvolution();
        }else if(Test_Flages == 4){
            BrixLab::Test_Reshape_Permute();
        }
    }else{
        std::string img_file = argv[2];
        Test_tflite(argv[1], std::stoi(argv[3]), std::stoi(argv[4]), 3, img_file);
    }
    return 0;
}