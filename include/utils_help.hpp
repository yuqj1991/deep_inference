#ifndef BRIXLAB_TFLITE_HELPS_
#define BRIXLAB_TFLITE_HELPS_

#include "utils.hpp"
#include "schema_generated.h"

namespace BrixLab
{
    enum DataType {
        DataType_DT_INVALID = 0,
        DataType_DT_FLOAT = 1,
        DataType_DT_DOUBLE = 2,
        DataType_DT_INT32 = 3,
        DataType_DT_UINT8 = 4,
        DataType_DT_INT16 = 5,
        DataType_DT_INT8 = 6,
    };
    static DataType tflite_dataTypeMap(tflite::TensorType type){
        switch (type) {
            case tflite::TensorType_FLOAT32:
                return DataType_DT_FLOAT;
                break;
            case tflite::TensorType_INT32:
                return DataType_DT_INT32;
                break;
            case tflite::TensorType_UINT8:
                return DataType_DT_UINT8;
                break;
            case tflite::TensorType_INT8:
                return DataType_DT_INT8;
                break;
            default:
                return DataType_DT_FLOAT;
                break;
        }
    }

    static bool needExtractInput(uint32_t opCode) {
        #define NONEED(x) if (x == opCode) return false
        NONEED(tflite::BuiltinOperator_CONV_2D);
        NONEED(tflite::BuiltinOperator_DEPTHWISE_CONV_2D);
        NONEED(tflite::BuiltinOperator_SPLIT);
        NONEED(tflite::BuiltinOperator_CONCATENATION);
        NONEED(tflite::BuiltinOperator_CONV_2D);
        NONEED(tflite::BuiltinOperator_RESIZE_BILINEAR);
        NONEED(tflite::BuiltinOperator_RESIZE_NEAREST_NEIGHBOR);
        NONEED(tflite::BuiltinOperator_SOFTMAX);
        return true;
    }

} // namespace BrixLab

#endif
