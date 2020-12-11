#ifndef BRIXLAB_TFLITE_HELPS_
#define BRIXLAB_TFLITE_HELPS_

#include "utils.hpp"
#include "schema_generated.h"
#include "logkit.hpp"
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

    template<typename DType>
    bool convertDataFormatTflite(const DType* src, DType* dst, int KH, int KW, int CI, int CO) {
        LOG_CHECK(KH > 0, "KH>0");
        LOG_CHECK(KW > 0, "KW>0");
        LOG_CHECK(CI > 0, "KI>0");
        LOG_CHECK(CO > 0, "KO>0");
        LOG_CHECK(src != nullptr, "src != nullptr");
        // CO KH KW CI --> CO CI KH KW
        for (int oc = 0; oc < CO; ++oc) {
            for (int ic = 0; ic < CI; ++ic) {
                for (int h = 0; h < KH; ++h) {
                    for (int w = 0; w < KW; ++w) {
                        dst[(oc * CI + ic) * KH * KW + h * KW + w] = src[(oc * KH + h) * KW * CI + w * CI + ic];
                    }
                }
            }
        }

        return true;
    }

    #define ARGSMAX(a, b) ((a) >= (b) ? (a): (b))
    #define ARGSMIN(a, b) ((a) >= (b) ? (b): (a))

} // namespace BrixLab

#endif
