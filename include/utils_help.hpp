#ifndef BRIXLAB_TFLITE_HELPS_
#define BRIXLAB_TFLITE_HELPS_

#include "utils.hpp"
#include "schema_generated.h"
#include "logkit.hpp"

typedef std::unique_ptr<tflite::QuantizationParametersT> tfliteQuanParam;
namespace BrixLab
{
    static DataType get_tensorDataType_tflite(tflite::TensorType type){
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
        LOG_CHECK(KH > 0)/* constant-expression */<<"KH <= 0";
        LOG_CHECK(KW > 0)<<"KW <= 0";
        LOG_CHECK(CI > 0)<<"CI <= 0";
        LOG_CHECK(CO > 0)<<"KO <= 0";
        LOG_CHECK(src != nullptr)<< "src != nullptr";
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

    inline void CalculateActivationRangeUint8(const BrixLab::FusedActivation activation, 
                                                const tfliteQuanParam& outputQuan,
                                                int32_t* act_min, int32_t* act_max) {
        const int32_t qmin      = std::numeric_limits<uint8_t>::min();
        const int32_t qmax      = std::numeric_limits<uint8_t>::max();
        const auto scale        = outputQuan->scale[0];
        const int32_t zeroPoint = static_cast<int32_t>(outputQuan->zero_point[0]);

        auto quantize = [scale, zeroPoint](float f) { return zeroPoint + static_cast<int32_t>(std::round(f / scale)); };

        if (activation == BrixLab::FusedActivation::Fused_kTfLiteActRelu) {
            *act_min = ARGSMAX(qmin, quantize(0.0));
            *act_max = qmax;
        } else if (activation == BrixLab::FusedActivation::Fused_kTfLiteActRelu6) {
            *act_min = ARGSMAX(qmin, quantize(0.0));
            *act_max = ARGSMIN(qmax, quantize(6.0));
        } else if (activation == BrixLab::FusedActivation::Fused_kTfLiteActRelu1) {
            *act_min = ARGSMAX(qmin, quantize(-1.0));
            *act_max = ARGSMIN(qmax, quantize(1.0));
        } else {
            *act_min = qmin;
            *act_max = qmax;
        }
    }

    inline PoolingType get_tflitePooling_Type(tflite::BuiltinOperator opcode){
        PoolingType pool_type;
        switch (opcode)
        {
        case tflite::BuiltinOperator::BuiltinOperator_AVERAGE_POOL_2D:
            pool_type   = BrixLab::PoolingType::PoolingAVAGE;
            break;
        case tflite::BuiltinOperator::BuiltinOperator_MAX_POOL_2D:
            pool_type   = BrixLab::PoolingType::PoolingMAX;
            break;
        default:
            break;
        }
        return pool_type;
    }

    inline Post_OPs_Param get_posts_opsMap(FusedActivation fused_activation){
        Post_OPs_Param fused_op;
        switch (fused_activation)
        {
        case Fused_kTfLiteActRelu1:
            fused_op.posts_op   = dnnl::algorithm::eltwise_relu;
            fused_op.alpha      = 0.f;
            fused_op.beta       = 0.f;
            fused_op.scale      = 1.f;
            break;
        case Fused_kTfLiteActRelu6:
            fused_op.posts_op   = dnnl::algorithm::eltwise_relu;
            fused_op.alpha      = 0.f;
            fused_op.beta       = 6.f;
            fused_op.scale      = 1.f;
            break;

        case Fused_kTfLiteActSigmoid:
            fused_op.posts_op   = dnnl::algorithm::eltwise_logistic;
            fused_op.alpha      = 0.f;
            fused_op.beta       = 0.f;
            fused_op.scale      = 1.f;
            break;
        case Fused_kTfLiteActTanh:
            fused_op.posts_op   = dnnl::algorithm::eltwise_tanh;
            fused_op.alpha      = 0.f;
            fused_op.beta       = 0.f;
            fused_op.scale      = 1.f;
            break;
        default:
            break;
        }

        return fused_op;
    }
} // namespace BrixLab

#endif
