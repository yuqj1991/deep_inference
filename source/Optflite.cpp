#include "liteopConvert.hpp"

namespace BrixLab
{
    DECLARE_OP_COVERTER(Conv2Dtflite);
    template<typename DType>
    BrixLab::OP_type Conv2Dtflite<DType>::opType(bool quantizedModel){
        if(quantizedModel){
            return OP_type::CONVOLUTION;
        }else{
            return OP_type::CONVOLUTION;
        }
    }
    template<typename DType>
    void Conv2Dtflite<DType>::run(layerWeightsParam<DType> *dstOp, const std::unique_ptr<tflite::OperatorT>& tfliteOp,
                        const std::vector<std::unique_ptr<tflite::TensorT>>& tfliteTensors,
                        const std::vector<std::unique_ptr<tflite::BufferT>>& tfliteModelBuffer,
                        const std::vector<std::unique_ptr<tflite::OperatorCodeT>>& tfliteOpSet, 
                        bool quantizedModel){
        const int inputSize = tfliteOp->inputs.size();
        LOG_CHECK(inputSize == 2 || inputSize == 3, "varify conv2D input size") << "tflite Conv2D input ERROR!";
        const auto& tfliteConvOption = tfliteOp->builtin_options.AsConv2DOptions();
        // weight index
        const int weightIndex    = tfliteOp->inputs[1];
        const auto& weightTensor = tfliteTensors[weightIndex];
        // co kh kw ci
        const auto& weightShape = weightTensor->shape;
        LOG_CHECK(weightShape.size() == 4, "varify conv2D weights shape size") << "Conv2D weight ERROR!";
        const int co         = weightShape[0];
        const int kh         = weightShape[1];
        const int kw         = weightShape[2];
        const int ci         = weightShape[3];
        const int weightSize = co * kh * kw * ci;

        if (quantizedModel) {
            auto conv2d_param = BrixLab::layerWeightsParam<uint8_t>();
            conv2d_param.quantized_type = true;
            // filterOffset
            if (weightTensor->quantization->zero_point.size() > 0) {
                conv2d_param.weights_zero_point = weightTensor->quantization->zero_point[0];
            } else {
                conv2d_param.weights_zero_point = 0;
            }
            if (weightTensor->quantization->scale.size() > 0) {
                conv2d_param.weights_scale = weightTensor->quantization->scale[0];
            } else {
                conv2d_param.weights_scale = 0.0;
            }

            // input
            const int inputIndex                 = tfliteOp->inputs[0];
            const auto& inputTensor              = tfliteTensors[inputIndex];
            const auto &inputHeight = inputTensor->shape;
            if (inputTensor->quantization->zero_point.size() > 0) {
                conv2d_param.inputs_zero_point = inputTensor->quantization->zero_point[0];
            } else {
                conv2d_param.inputs_zero_point = 0;
            }
            if (inputTensor->quantization->scale.size() > 0) {
                conv2d_param.inputs_scale = inputTensor->quantization->scale[0];
            } else {
                conv2d_param.inputs_scale = 0.0;
            }

            // output
            const int outputIndex                 = tfliteOp->outputs[0];
            const auto& outputTensor              = tfliteTensors[outputIndex];
            if (outputTensor->quantization->scale.size() > 0) {
                conv2d_param.outputs_zero_point = outputTensor->quantization->zero_point[0];
            } else {
                conv2d_param.outputs_zero_point = 0;
            }

            if (outputTensor->quantization->scale.size() > 0) {
                conv2d_param.outputs_scale = outputTensor->quantization->scale[0];
            } else {
                conv2d_param.outputs_scale = 0.0;
            }

            // kernel size
            conv2d_param.k_w     = kw;
            conv2d_param.k_h     = kh;
            conv2d_param.k_c     = co;

            // default
            conv2d_param.groups   = 1;
            conv2d_param.dilateX = tfliteConvOption->dilation_w_factor;
            conv2d_param.dilateY = tfliteConvOption->dilation_h_factor;
            //conv2dParamQuan->depthMultiplier = 1;

            // stride
            conv2d_param.stridesX = tfliteConvOption->stride_w;
            conv2d_param.stridesY = tfliteConvOption->stride_h;
            const auto tflitePadMode = tfliteConvOption->padding;
            if (tflitePadMode == tflite::Padding_SAME) {
                conv2d_param.mpad = BrixLab::PaddingType::PaddingSAME;
            } else if (tflitePadMode == tflite::Padding_VALID) {
                conv2d_param.mpad = BrixLab::PaddingType::PaddingVALID;
            }
            // weight
            LOG_CHECK(weightTensor->type == tflite::TensorType_UINT8, 
                                        "check conv2D weights tensor type") << "Data type ERROR";
            // nhwc->hwcn
            int out_size = kh * kw * ci;
            int in_size  = co;
            auto originalWeightPtr = tfliteModelBuffer[weightTensor->buffer]->data.data();
            conv2d_param.conv_weights = (uint8_t*)xcalloc(in_size * out_size, sizeof(uint8_t));
            convertDataFormatTflite(originalWeightPtr, conv2d_param.conv_weights, kh, kw, ci, co);
            
            conv2d_param.hasBias = (inputSize == 3);
            LOG_CHECK(conv2d_param.hasBias==true, "CHECK the bias flags") << "the bias flags is false";
            const auto& biasTensor = tfliteTensors[tfliteOp->inputs[2]];
            if (inputSize == 3) {
                LOG_CHECK(biasTensor->type == tflite::TensorType_INT32, "CHECK the bias tensor type") << "Bias Type ERROR";
                const auto& biasData                = tfliteModelBuffer[biasTensor->buffer]->data;
                conv2d_param.bias_zero_point = biasTensor->quantization->zero_point[0];
                conv2d_param.bias_scale     = biasTensor->quantization->scale[0];
                LOG_CHECK(biasData.size() / 4 == co, "CHECK the biasDias shape") << "Bias Data ERROR";
                conv2d_param.conv_bias = (int32_t*)biasData.data();
            }
            conv2d_param.fused_act_type = (BrixLab::FusedActivation)tfliteConvOption->fused_activation_function;
            *dstOp = conv2d_param;
        } else {
            auto conv2d_param = BrixLab::layerWeightsParam<float>();
            conv2d_param.conv_weights = (float*)xcalloc(weightSize, sizeof(float));
            // weight
            auto originalWeightPtr = reinterpret_cast<const float*>(tfliteModelBuffer[weightTensor->buffer]->data.data());
            convertDataFormatTflite(originalWeightPtr, conv2d_param.conv_weights, kh, kw, ci, co);
            // bias
            conv2d_param.conv_bias = (float*)xcalloc(co, sizeof(float));
            if (inputSize == 3) {
                const auto& biasTensor = tfliteTensors[tfliteOp->inputs[2]];
                auto biasDataPtr       = reinterpret_cast<const float*>(tfliteModelBuffer[biasTensor->buffer]->data.data());
                ::memcpy(conv2d_param.conv_bias, biasDataPtr, sizeof(float) * co);
            }

            conv2d_param.relu             = false;
            conv2d_param.relu6            = false;
            const auto acticationFun = tfliteConvOption->fused_activation_function;
            if (acticationFun == tflite::ActivationFunctionType_RELU) {
                conv2d_param.relu = true;
            } else if (acticationFun == tflite::ActivationFunctionType_RELU6) {
                conv2d_param.relu6 = true;
            } else if (acticationFun > tflite::ActivationFunctionType_NONE) {
                LOG(FATAL_ERROR, "ONEDNN_TYPE SUPPORTED") << 
                            "ONEDNN Convolution do not Support fused_activation_function: " << acticationFun;
            }

            conv2d_param.groups       = 1;
            conv2d_param.k_c          = co;
            conv2d_param.inChannel    = ci;
            conv2d_param.k_w          = kw;
            conv2d_param.k_h          = kh;
            conv2d_param.dilateX      = tfliteConvOption->dilation_w_factor;
            conv2d_param.dilateY      = tfliteConvOption->dilation_h_factor;
            conv2d_param.stridesX     = tfliteConvOption->stride_w;
            conv2d_param.stridesY     = tfliteConvOption->stride_h;
            conv2d_param.mpad         = BrixLab::PaddingType::PaddingSAME;
            if (tfliteConvOption->padding == tflite::Padding_VALID) {
                conv2d_param.mpad = BrixLab::PaddingType::PaddingVALID;
            }
            *dstOp = conv2d_param;
        }
        
        // set input output index
        dstOp->inIndexs.resize(1);
        dstOp->outIndexs.resize(1);

        dstOp->inIndexs[0]  = tfliteOp->inputs[0];
        dstOp->outIndexs[0] = tfliteOp->outputs[0];
    }

    INSTANEC_OP_CONVERTER(Conv2Dtflite);

    DECLARE_OP_COVERTER(TransposedConv2Dtflite);

    template<typename DType>
    OP_type TransposedConv2Dtflite<DType>::opType(bool quantizedModel){
        if(quantizedModel){
            return BrixLab::OP_type::DECONVOLUTION;
        }else{
            return BrixLab::OP_type::DECONVOLUTION;
        }
    }

    template<typename DType>
    void TransposedConv2Dtflite<DType>::run(layerWeightsParam<DType> *dstOp, const std::unique_ptr<tflite::OperatorT>& tfliteOp,
                                    const std::vector<std::unique_ptr<tflite::TensorT>>& tfliteTensors,
                                    const std::vector<std::unique_ptr<tflite::BufferT>>& tfliteModelBuffer,
                                    const std::vector<std::unique_ptr<tflite::OperatorCodeT>>& tfliteOpSet, 
                                    bool quantizedModel){
        LOG_CHECK(!quantizedModel, "!quantizedModel") << "TransposeConv not support quantized model";
    
        // 3|2 inputs: input tensor, weight, (bias)
        const int inputSize = tfliteOp->inputs.size();
        LOG_CHECK(inputSize == 2 || inputSize == 3, "inputSize == 2 || inputSize == 3") << "tflite Conv2D input ERROR! ";
        /*
        enum Padding : byte { SAME, VALID }
        table TransposeConvOptions {
        padding:Padding;
        stride_w:int;
        stride_h:int;
        }
        */
        const auto& tfliteConvOption = tfliteOp->builtin_options.AsTransposeConvOptions();
        // weight index
        const int weightIndex    = tfliteOp->inputs[1];
        const auto& weightTensor = tfliteTensors[weightIndex];
        // co kh kw ci
        const auto& weightShape = weightTensor->shape;
        LOG_CHECK(weightShape.size() == 4, "weightShape.size() == 4") << "Conv2D weight ERROR!";
        const int co         = weightShape[0];
        const int kh         = weightShape[1];
        const int kw         = weightShape[2];
        const int ci         = weightShape[3];
        const int weightSize = co * kh * kw * ci;
        {
            // weight
            auto originalWeightPtr = reinterpret_cast<const float*>(tfliteModelBuffer[weightTensor->buffer]->data.data());
            auto transposed_param = BrixLab::layerWeightsParam<float>();
            transposed_param.transposed_weights = (float*)xcalloc(weightSize, sizeof(float));
            convertDataFormatTflite(originalWeightPtr, transposed_param.transposed_weights, kh, kw, ci, co);
            // bias
            if (inputSize == 3) {
                transposed_param.transposed_bias = (float*)xcalloc(co, sizeof(co));
                const auto& biasTensor = tfliteTensors[tfliteOp->inputs[2]];
                auto biasDataPtr       = reinterpret_cast<const float*>(tfliteModelBuffer[biasTensor->buffer]->data.data());
                if(biasDataPtr){
                    ::memcpy(biasData.data(), transposed_param.transposed_bias, sizeof(float) * co);
                }
            }
            
            convolution2DFloat->common = std::unique_ptr<MNN::Convolution2DCommonT>(new MNN::Convolution2DCommonT);
            auto& common               = convolution2DFloat->common;

            transposed_param.relu6     = false;
            transposed_param.relu      = false;
            transposed_param.groups    = 1;
            transposed_param.k_c       = co;
            transposed_param.inChannel = ci;
            transposed_param.k_w       = kw;
            transposed_param.k_h       = kh;
            transposed_param.dilateX   = 1;
            transposed_param.dilateY   = 1;
            transposed_param.stridesX  = tfliteConvOption->stride_w;
            transposed_param.stridesY  = tfliteConvOption->stride_h;

            transposed_param.mpad     = BrixLab::PaddingType::PaddingSAME;
            if (tfliteConvOption->padding == tflite::Padding_VALID) {
                transposed_param.mpad = BrixLab::PaddingType::PaddingVALID;
            }

            *dstOp = transposed_param;
        }
        
        // set input output index
        dstOp->inputIndexes.resize(1);
        dstOp->outputIndexes.resize(1);

        dstOp->inputIndexes[0]  = tfliteOp->inputs[0];
        dstOp->outputIndexes[0] = tfliteOp->outputs[0];
    }

    INSTANEC_SINGLE_OP_CONVERTER(TransposedConv2Dtflite);

    DECLARE_OP_COVERTER(FullConnectedTflite);

    template<typename DType>
    OP_type FullConnectedTflite<DType>::opType(bool quantizedModel){
        if(quantizedModel){
            return BrixLab::OP_type::INNERPRODUCT;
        }else{
            return BrixLab::OP_type::INNERPRODUCT;
        }
    }

    template<typename DType>
    void FullConnectedTflite<DType>::run(layerWeightsParam<DType> *dstOp, const std::unique_ptr<tflite::OperatorT>& tfliteOp,
                        const std::vector<std::unique_ptr<tflite::TensorT>>& tfliteTensors,
                        const std::vector<std::unique_ptr<tflite::BufferT>>& tfliteModelBuffer,
                        const std::vector<std::unique_ptr<tflite::OperatorCodeT>>& tfliteOpSet, bool quantizedModel){
        dstOp->main.value = new MNN::ExtraT;
        auto dstP = dstOp->main.AsExtra();
        dstP->engine = "Tflite";
        dstP->type = "FULL_CONNECT";
        const auto& option = tfliteOp->builtin_options.AsFullyConnectedOptions();
        dstP->attr.resize(3);
        dstP->attr[0].reset(new MNN::AttributeT);
        dstP->attr[0]->key = "keep_num_dims";
        dstP->attr[0]->b = option->keep_num_dims;

        dstP->attr[1].reset(new MNN::AttributeT);
        dstP->attr[1]->key = "weights_format";
        dstP->attr[1]->i = option->weights_format;

        dstP->attr[2].reset(new MNN::AttributeT);
        dstP->attr[2]->key = "fused_activation_function";
        dstP->attr[2]->i = option->fused_activation_function;
    }


} // namespace BrixLab
