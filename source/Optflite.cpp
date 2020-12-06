#include "liteopConvert.hpp"

namespace BrixLab
{
    DECLARE_OP_COVERTER(Conv2Dtflite);
    BrixLab::OP_type Conv2Dtflite::opType(bool quantizedModel){
        if(quantizedModel){

        }else{
            return OP_type::CONVOLUTION;
        }
    }
    void Conv2Dtflite::run(layerWeightsParam<float> *dstOp, const std::unique_ptr<tflite::OperatorT>& tfliteOp,
                        const std::vector<std::unique_ptr<tflite::TensorT>>& tfliteTensors,
                        const std::vector<std::unique_ptr<tflite::BufferT>>& tfliteModelBuffer,
                        const std::vector<std::unique_ptr<tflite::OperatorCodeT>>& tfliteOpSet, bool quantizedModel){
        ;
    }


} // namespace BrixLab
