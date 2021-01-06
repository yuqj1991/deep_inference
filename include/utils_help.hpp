#ifndef BRIXLAB_TFLITE_HELPS_
#define BRIXLAB_TFLITE_HELPS_

#include "utils.hpp"
#include "schema_generated.h"
#include "logkit.hpp"
#include <fstream>
#include <iostream>
#include <algorithm>
#include <iomanip>
#include <sstream>
#include <mutex>
#include <stdio.h>
#include <stdlib.h>

typedef std::unique_ptr<tflite::QuantizationParametersT> tfliteQuanParam;
namespace BrixLab
{
    template<typename DType>
    bool convertDataFormatTflite(const DType* src, DType* dst, int KH, int KW, int CI, int CO) {
        if(KH <= 0) return false;
        if(KW <= 0) return false;
        if(CI <= 0) return false;
        if(CO <= 0) return false;
        if(src == nullptr) return false;
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
            fused_op.scale      = -1.f;
            break;
        case Fused_kTfLiteActRelu:
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
            LOG(FATAL_ERROR)<<"Do Not Support other ops: " << fused_activation;
            break;
        }

        return fused_op;
    }


    inline std::string GetCurrentTimeForFileName()
    {
        auto time = std::time(nullptr);
        std::stringstream ss;
        ss << std::put_time(std::localtime(&time), "%F_%T"); // ISO 8601 without timezone information.
        auto s = ss.str();
        std::replace(s.begin(), s.end(), ':', '-');
        return s;
    }

    inline void dump_data_to_file(const float* data, uint32_t size, const std::string name, const dnnl::memory::dims dims){
        static std::mutex lock;
        std::lock_guard<std::mutex> lg(lock);
        FILE *fp;
        static const std::string filename = GetCurrentTimeForFileName()+".dump";
        fp = fopen(filename.data(), "wt+");
        if(fp == nullptr){
            LOG(FATAL_ERROR) << "cannot open file " << filename;
        }
        uint32_t sz = name.size();
        fwrite(&sz, 1, 4, fp);
        fwrite(name.data(), 1, sz, fp);
        sz = dims.size();
        fwrite(&sz, 1, 4, fp);
        for(size_t i=0; i<dims.size(); i++){
            uint32_t dim = dims[i];
            fwrite(&dim, 1, 4, fp);
        }
        size *= sizeof(*data);
        fwrite(&size, 1, 4, fp);
        fwrite(data, 1, size, fp);
        fclose(fp);
    }

    inline void dump_data_to_file_c(const float* data, uint32_t size, const std::string name, const dnnl::memory::dims dims){
        FILE *fp;
        static const std::string filename = GetCurrentTimeForFileName() + ".dump";
        fp = fopen(filename.data(), "a");
        if(fp == nullptr){
            LOG(FATAL_ERROR) << "cannot open file " << filename;
        }
        const int inB   = dims[0];
        const int inC   = dims[1];
        const int inH   = dims[2];
        const int inW   = dims[3];
        fprintf(fp, "%s\n", name.c_str());

        for(int b = 0; b < inB; b++){
            for(int c = 0; c < inC; c++){
                for(int h = 0; h < inH; h++){
                    for(int w = 0; w < inW; w++){
                        fprintf(fp, "%.1f, ", data[b * inC * inH * inW + c * inH * inW + h * inW + w]);
                    }
                    fprintf(fp, "\n");
                }
                fprintf(fp, "\n");
            }
            fprintf(fp, "\n");
        }
        fclose(fp);
    }
} // namespace BrixLab

#endif // BRIXLAB_TFLITE_HELPS_
