#include "layer_builder.hpp"
#include "utils.hpp"
#include "unit_test.hpp"
#include <iostream>
#include "logkit.hpp"
using namespace BrixLab;
int main(){
    int inBatch = 3;
    int inChannel = 64;
    int inHeight = 14;
    int inWidth = 14;
    std::vector<float> inData(inBatch*inChannel*inHeight*inWidth);
    float * data = inData.data();
    BrixLab::Test_Convulution(data);
    
    LOG(DEBUG_INFO, "test_LOG") <<" 123 ";
    LOG_CHECK(1>2, "test_CHECK")<< "1 should be less than 2";
    return 0;
}