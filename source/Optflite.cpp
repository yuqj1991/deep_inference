#include "liteopConvert.hpp"
using namespace tflite;
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
        dstOp->fused_ops = false;
        dstOp->in_shapes.resize(1);
        dstOp->out_shapes.resize(1);
        // input shape
        const int inshapeindex  = tfliteOp->inputs[0];
        
        if (quantizedModel) {
            dstOp->quantized_type = BrixLab::QUANITIZED_TYPE::UINT8_QUANTIZED;
            dstOp->op_type = BrixLab::OP_type::CONVOLUTION;
            // filterOffset
            if (weightTensor->quantization->zero_point.size() > 0) {
                dstOp->weights_zero_point = weightTensor->quantization->zero_point[0];
            } else {
                dstOp->weights_zero_point = 0;
            }
            if (weightTensor->quantization->scale.size() > 0) {
                dstOp->weights_scale = weightTensor->quantization->scale[0];
            } else {
                dstOp->weights_scale = 0.0;
            }

            // input
            const int inputIndex                    = tfliteOp->inputs[0];
            const auto& inputTensor                 = tfliteTensors[inputIndex];
            //const auto& inputHeight                 = inputTensor->shape;
            if (inputTensor->quantization->zero_point.size() > 0) {
                dstOp->inputs_zeropoint.push_back(inputTensor->quantization->zero_point[0]);
            } else {
                dstOp->inputs_zeropoint.push_back(0);
            }
            if (inputTensor->quantization->scale.size() > 0) {
                dstOp->inputs_scale.push_back(inputTensor->quantization->scale[0]);
            } else {
                dstOp->inputs_scale.push_back(0.0);
            }

            // output
            const int outputIndex                 = tfliteOp->outputs[0];
            const auto& outputTensor              = tfliteTensors[outputIndex];
            if (outputTensor->quantization->scale.size() > 0) {
                dstOp->outputs_zero_point = outputTensor->quantization->zero_point[0];
            } else {
                dstOp->outputs_zero_point = 0;
            }

            if (outputTensor->quantization->scale.size() > 0) {
                dstOp->outputs_scale = outputTensor->quantization->scale[0];
            } else {
                dstOp->outputs_scale = 0.0;
            }

            // kernel size
            dstOp->k_w     = kw;
            dstOp->k_h     = kh;
            dstOp->k_c     = co;

            // default
            dstOp->groups   = 1;
            dstOp->dilateX = tfliteConvOption->dilation_w_factor - 1;
            dstOp->dilateY = tfliteConvOption->dilation_h_factor - 1;
            //conv2dParamQuan->depthMultiplier = 1;

            // stride
            dstOp->stridesX = tfliteConvOption->stride_w;
            dstOp->stridesY = tfliteConvOption->stride_h;
            const auto tflitePadMode = tfliteConvOption->padding;
            if (tflitePadMode == tflite::Padding_SAME) {
                dstOp->padMode = BrixLab::PaddingType::PaddingSAME;
            } else if (tflitePadMode == tflite::Padding_VALID) {
                dstOp->padMode = BrixLab::PaddingType::PaddingVALID;
            }
            // weight
            LOG_CHECK(weightTensor->type == tflite::TensorType_UINT8, 
                                        "check conv2D weights tensor type") << "Data type ERROR";
            // nhwc->hwcn
            int out_size = kh * kw * ci;
            int in_size  = co;
            auto originalWeightPtr = reinterpret_cast<const DType*>(tfliteModelBuffer[weightTensor->buffer]->data.data());
            dstOp->conv_weights = (DType*)xcalloc(in_size * out_size, sizeof(DType));
            convertDataFormatTflite(originalWeightPtr, dstOp->conv_weights, kh, kw, ci, co);
            
            dstOp->hasBias = (inputSize == 3);
            LOG_CHECK(dstOp->hasBias==true, "CHECK the bias flags") << "the bias flags is false";
            const auto& biasTensor = tfliteTensors[tfliteOp->inputs[2]];
            if (inputSize == 3) {
                LOG_CHECK(biasTensor->type == tflite::TensorType_INT32, "CHECK the bias tensor type") << "Bias Type ERROR";
                const auto& biasData                = tfliteModelBuffer[biasTensor->buffer]->data;
                dstOp->bias_zero_point = biasTensor->quantization->zero_point[0];
                dstOp->bias_scale     = biasTensor->quantization->scale[0];
                LOG_CHECK(biasData.size() == 1, "CHECK the biasDias shape") << "Bias Data ERROR";
                dstOp->quantized_bias = (int32_t*)xcalloc(co, sizeof(int32_t));
                ::memcpy(dstOp->quantized_bias, biasData.data(), sizeof(int32_t) * co);
            }
            dstOp->fused_ops        = true;
            dstOp->fused_act_type   = (BrixLab::FusedActivation)tfliteConvOption->fused_activation_function;
        } else {
            dstOp->quantized_type = BrixLab::QUANITIZED_TYPE::FLOAT32_REGULAR;
            dstOp->op_type = BrixLab::OP_type::CONVOLUTION;
            dstOp->conv_weights = (DType*)xcalloc(weightSize, sizeof(DType));
            // weight
            auto originalWeightPtr = reinterpret_cast<const DType*>(tfliteModelBuffer[weightTensor->buffer]->data.data());
            convertDataFormatTflite(originalWeightPtr, dstOp->conv_weights, kh, kw, ci, co);
            // bias
            dstOp->conv_bias = (DType*)xcalloc(co, sizeof(DType));
            if (inputSize == 3) {
                const auto& biasTensor = tfliteTensors[tfliteOp->inputs[2]];
                auto biasDataPtr       = reinterpret_cast<const DType*>(tfliteModelBuffer[biasTensor->buffer]->data.data());
                ::memcpy(dstOp->conv_bias, biasDataPtr, sizeof(DType) * co);
            }

            dstOp->relu             = false;
            dstOp->relu6            = false;
            const auto acticationFun = tfliteConvOption->fused_activation_function;
            if (acticationFun == tflite::ActivationFunctionType_RELU) {
                dstOp->relu = true;
            } else if (acticationFun == tflite::ActivationFunctionType_RELU6) {
                dstOp->relu6 = true;
            } else if (acticationFun > tflite::ActivationFunctionType_NONE) {
                LOG(FATAL_ERROR, "ONEDNN_TYPE SUPPORTED") << 
                            "ONEDNN Convolution do not Support fused_activation_function: " << acticationFun;
            }

            dstOp->groups       = 1;
            dstOp->k_c          = co;
            dstOp->inChannel    = ci;
            dstOp->k_w          = kw;
            dstOp->k_h          = kh;
            dstOp->dilateX      = tfliteConvOption->dilation_w_factor - 1;
            dstOp->dilateY      = tfliteConvOption->dilation_h_factor - 1;
            dstOp->stridesX     = tfliteConvOption->stride_w;
            dstOp->stridesY     = tfliteConvOption->stride_h;
            dstOp->padMode      = BrixLab::PaddingType::PaddingSAME;
            if (tfliteConvOption->padding == tflite::Padding_VALID) {
                dstOp->padMode  = BrixLab::PaddingType::PaddingVALID;
            }
        }
        
        // set input output index
        dstOp->inIndexs.resize(1);
        dstOp->outIndexs.resize(1);

        dstOp->inIndexs[0]  = tfliteOp->inputs[0];
        dstOp->outIndexs[0] = tfliteOp->outputs[0];
    }

    INSTANEC_OP_CONVERTER(Conv2Dtflite);
    REGISTER_CONVERTER(Conv2Dtflite<float>, float, BuiltinOperator_CONV_2D);
    REGISTER_CONVERTER(Conv2Dtflite<uint8_t>, uint8_t, BuiltinOperator_CONV_2D);

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
        LOG_CHECK(!quantizedModel, "!quantizedModel") << "Transpose_Conv2D not support quantized model";
        // 3|2 inputs: input tensor, weight, (bias)
        /*
        enum Padding : byte { SAME, VALID }
        table TransposeConvOptions {
        padding:Padding;
        stride_w:int;
        stride_h:int;
        }
        */
        const int inputSize = tfliteOp->inputs.size();
        LOG_CHECK(inputSize == 2 || inputSize == 3, "inputSize == 2 || inputSize == 3") << "tflite Tranposed_Conv2D input ERROR! ";
        
        const auto& tfliteConvOption = tfliteOp->builtin_options.AsTransposeConvOptions();
        // weight index
        const int weightIndex    = tfliteOp->inputs[1];
        const auto& weightTensor = tfliteTensors[weightIndex];
        // co kh kw ci
        const auto& weightShape = weightTensor->shape;
        LOG_CHECK(weightShape.size() == 4, "weightShape.size() == 4") << "Transposed_Conv2D weight ERROR!";
        const int co         = weightShape[0];
        const int kh         = weightShape[1];
        const int kw         = weightShape[2];
        const int ci         = weightShape[3];
        const int weightSize = co * kh * kw * ci;
        dstOp->fused_ops     = false;
        // weight
        auto originalWeightPtr = reinterpret_cast<const DType*>(tfliteModelBuffer[weightTensor->buffer]->data.data());
        dstOp->transposed_weights = (DType*)xcalloc(weightSize, sizeof(DType));
        convertDataFormatTflite(originalWeightPtr, dstOp->transposed_weights, kh, kw, ci, co);
        // bias
        if (inputSize == 3) {
            dstOp->transposed_bias = (DType*)xcalloc(co, sizeof(co));
            const auto& biasTensor = tfliteTensors[tfliteOp->inputs[2]];
            auto biasDataPtr       = reinterpret_cast<const DType*>(tfliteModelBuffer[biasTensor->buffer]->data.data());
            if(biasDataPtr){
                ::memcpy(dstOp->transposed_bias, biasDataPtr, sizeof(DType) * co);
            }
        }
        
        dstOp->relu6     = false;
        dstOp->relu      = false;
        dstOp->groups    = 1;
        dstOp->k_c       = co;
        dstOp->inChannel = ci;
        dstOp->k_w       = kw;
        dstOp->k_h       = kh;
        dstOp->dilateX   = 1 - 1;
        dstOp->dilateY   = 1 - 1;
        dstOp->stridesX  = tfliteConvOption->stride_w;
        dstOp->stridesY  = tfliteConvOption->stride_h;

        dstOp->padMode     = BrixLab::PaddingType::PaddingSAME;
        if (tfliteConvOption->padding == tflite::Padding_VALID) {
            dstOp->padMode = BrixLab::PaddingType::PaddingVALID;
        }
        
        // set input output index
        dstOp->inIndexs.resize(1);
        dstOp->outIndexs.resize(1);

        dstOp->inIndexs[0]  = tfliteOp->inputs[0];
        dstOp->outIndexs[0] = tfliteOp->outputs[0];
    }

    INSTANEC_FLOAT_OP_CONVERTER(TransposedConv2Dtflite);
    REGISTER_CONVERTER(TransposedConv2Dtflite<float>, float, BuiltinOperator_TRANSPOSE_CONV);

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
                        const std::vector<std::unique_ptr<tflite::OperatorCodeT>>& tfliteOpSet, 
                        bool quantizedModel){
        const int input_size = tfliteOp->inputs.size();
        LOG_CHECK(input_size == 2 || input_size == 3, "input_size == 2 || input_size == 3") 
                                    << "tflite fully connected inputs error";
        // input data
        const auto in_index     = tfliteOp->inputs[0];
        const auto &in_tensor   = tfliteTensors[in_index];
        const auto in_shape     = in_tensor->shape;
        const int in_size       = in_shape.size();
        LOG_CHECK(in_size == 2 || in_size == 4, "Fully_Connect Input")<<"tflite Fully Connect Input Shape Error";
        TensorShape data_shape;
        data_shape.Batch    = in_shape[0];
        data_shape.Height   = in_shape[1];
        if(in_size == 4){
            data_shape.Width    = in_shape[2];
            data_shape.Channel  = in_shape[3];  
        }
        // weight data
        const auto weights_index    = tfliteOp->inputs[1];
        const auto &weights_tensor  = tfliteTensors[weights_index];
        const auto weights_shape    = weights_tensor->shape;
        LOG_CHECK(weights_shape.size() == 2, "weights_shape.size() == 2")
                    <<"FULLY CONNECTED Weights Shape Error";
        const int co    = weights_shape[0];
        const int ci    = weights_shape[1];
        int weight_size = co * ci;
        //output data
        const auto out_index    = tfliteOp->outputs[0];
        const auto &out_tensor  = tfliteTensors[out_index];
        const auto out_shape    = out_tensor->shape;
        const int out_size      = out_shape.size();
        LOG_CHECK(out_size == 2, "Fully_Connect Output")<<"tflite Fully Connect Output Shape Error";
        TensorShape outdata_shape;
        outdata_shape.Batch     = out_shape[0];
        outdata_shape.Channel   = out_shape[1];
        dstOp->fused_ops        = false;

        if(quantizedModel){
            dstOp->quantized_type = BrixLab::QUANITIZED_TYPE::UINT8_QUANTIZED;
            dstOp->op_type = BrixLab::OP_type::INNERPRODUCT;
            dstOp->inHeight   = data_shape.Height;
            dstOp->inWidth    = data_shape.Width;
            dstOp->inBatch    = data_shape.Batch;
            dstOp->inChannel  = data_shape.Channel;
            dstOp->layer_h    = 1;
            dstOp->layer_w    = 1;
            dstOp->layer_c    = outdata_shape.Channel;
            dstOp->layer_n    = outdata_shape.Batch;
            dstOp->k_c        = co;
            dstOp->k_in       = ci;
            dstOp->innerWeights = (DType *)xcalloc(weight_size, sizeof(DType));
            for(int ii = 0; ii < co; ii++){
                for(int jj = 0; jj < ci; jj++){
                    dstOp->innerWeights[ii *ci + jj] = tfliteModelBuffer[weights_tensor->buffer]->data[ii * ci + jj];
                }
            }
            if(input_size == 3){
                const auto& biasTensor = tfliteTensors[tfliteOp->inputs[2]];
                LOG_CHECK(biasTensor->type == tflite::TensorType::TensorType_INT32, "Fully Connected bias Type")<<"Bias type ERROR";
                dstOp->hasBias = true;
                dstOp->quantized_bias = (int32_t *)xcalloc(co, sizeof(int32_t));
                
                const auto bias_index   = tfliteOp->inputs[2];
                const auto &bias_tensor = tfliteTensors[bias_index];
                const auto bias_shape   = bias_tensor->shape;
                LOG_CHECK(bias_shape.size() == 1, "bias shape") << "Bias Shape ERROR";
                for(int ii = 0; ii < co; ii++){
                    dstOp->quantized_bias[ii] = tfliteModelBuffer[bias_tensor->buffer]->data[ii];
                }
            }
        }else{
            dstOp->quantized_type = BrixLab::QUANITIZED_TYPE::FLOAT32_REGULAR;
            dstOp->op_type = BrixLab::OP_type::INNERPRODUCT;
            dstOp->inHeight   = data_shape.Height;
            dstOp->inWidth    = data_shape.Width;
            dstOp->inBatch    = data_shape.Batch;
            dstOp->inChannel  = data_shape.Channel;
            dstOp->layer_h    = 1;
            dstOp->layer_w    = 1;
            dstOp->layer_c    = outdata_shape.Channel;
            dstOp->layer_n    = outdata_shape.Batch;
            dstOp->k_c        = co;
            dstOp->k_in       = ci;
            dstOp->innerWeights = (DType *)xcalloc(weight_size, sizeof(DType));
            for(int ii = 0; ii < co; ii++){
                for(int jj = 0; jj < ci; jj++){
                    dstOp->innerWeights[ii *ci + jj] = tfliteModelBuffer[weights_tensor->buffer]->data[ii * ci + jj];
                }
            }
            if(input_size == 3){
                const auto& biasTensor = tfliteTensors[tfliteOp->inputs[2]];
                LOG_CHECK(biasTensor->type == tflite::TensorType::TensorType_FLOAT32, "Fully Connected bias Type")<<"Bias type ERROR";
                dstOp->hasBias = true;
                dstOp->innerBias = (DType *)xcalloc(co, sizeof(DType));
                
                const auto bias_index   = tfliteOp->inputs[2];
                const auto &bias_tensor = tfliteTensors[bias_index];
                const auto bias_shape   = bias_tensor->shape;
                LOG_CHECK(bias_shape.size() == 1, "bias shape") << "Bias Shape ERROR";
                for(int ii = 0; ii < co; ii++){
                    dstOp->innerBias[ii] = tfliteModelBuffer[bias_tensor->buffer]->data[ii];
                }
            }
        }
    }

    INSTANEC_OP_CONVERTER(FullConnectedTflite);

    REGISTER_CONVERTER(FullConnectedTflite<float>, float, BuiltinOperator_FULLY_CONNECTED);
    REGISTER_CONVERTER(FullConnectedTflite<uint8_t>, uint8_t, BuiltinOperator_FULLY_CONNECTED);

    DECLARE_OP_COVERTER(DepthConv2DTflite);

    template<typename DType>
    OP_type DepthConv2DTflite<DType>::opType(bool quantizedModel){
        if(quantizedModel){
            return BrixLab::OP_type::CONVOLUTION;
        }else{
            return BrixLab::OP_type::CONVOLUTION;
        }
    }

    template<typename DType>
    void DepthConv2DTflite<DType>::run(layerWeightsParam<DType> *dstOp, const std::unique_ptr<tflite::OperatorT>& tfliteOp,
                        const std::vector<std::unique_ptr<tflite::TensorT>>& tfliteTensors,
                        const std::vector<std::unique_ptr<tflite::BufferT>>& tfliteModelBuffer,
                        const std::vector<std::unique_ptr<tflite::OperatorCodeT>>& tfliteOpSet, 
                        bool quantizedModel){
        // 3|2 inputs: input tensor, weight, (bias)
        const int inputSize = tfliteOp->inputs.size();
        LOG_CHECK(inputSize == 2 || inputSize == 3, "Depthwise_Conv2D input size") << "tflite DepthiwiseConv2D input ERROR! ";
        // weight index
        const int weightIndex    = tfliteOp->inputs[1];
        const auto& weightTensor = tfliteTensors[weightIndex];
        // co kh kw ci
        const auto& weightShape = weightTensor->shape;
        LOG_CHECK(weightShape.size() == 4, "Depthwise_Conv2D weight shape") << "Depthwise_Conv2D weight ERROR!";
        // const int co = weightShape[0];
        const int kh                 = weightShape[1];
        const int kw                 = weightShape[2];
        const int ci                 = weightShape[3];
        int weightSize         = kh * kw * ci;
        const auto& tfliteConvOption = tfliteOp->builtin_options.AsDepthwiseConv2DOptions();
        dstOp->fused_ops             = false;
        if (quantizedModel) {
            dstOp->quantized_type    = BrixLab::QUANITIZED_TYPE::UINT8_QUANTIZED;
            dstOp->op_type           = BrixLab::OP_type::CONVOLUTION;

            // filterOffset
            dstOp->weights_zero_point    = weightTensor->quantization->zero_point[0];
            dstOp->weights_scale         = weightTensor->quantization->scale[0];

            // input
            const int inputIndex                            = tfliteOp->inputs[0];
            const auto& inputTensor                         = tfliteTensors[inputIndex];
            dstOp->inputs_zeropoint.push_back(inputTensor->quantization->zero_point[0]);
            dstOp->inputs_scale.push_back(inputTensor->quantization->scale[0]);

            // output
            const int outputIndex    = tfliteOp->outputs[0];
            const auto& outputTensor = tfliteTensors[outputIndex];
            dstOp->outputs_zero_point                    = outputTensor->quantization->zero_point[0];
            dstOp->outputs_scale                         = outputTensor->quantization->scale[0];

            // kernel size
            dstOp->k_w       = kw;
            dstOp->k_h       = kh;
            dstOp->k_in       = ci;

            // default
            dstOp->dilateX = tfliteConvOption->dilation_w_factor - 1;
            dstOp->dilateY = tfliteConvOption->dilation_h_factor - 1;

            int depthMultiplier = tfliteConvOption->depth_multiplier;
            dstOp->k_c       = depthMultiplier * dstOp->k_in;
            dstOp->groups    = depthMultiplier * dstOp->k_in;
            weightSize                      *= depthMultiplier; 
            // stride
            dstOp->stridesX  = tfliteConvOption->stride_w;
            dstOp->stridesY  = tfliteConvOption->stride_h;

            const auto tflitePadMode = tfliteConvOption->padding;
            if (tflitePadMode == tflite::Padding_SAME) {
                dstOp->padMode = BrixLab::PaddingType::PaddingVALID;
            } else if (tflitePadMode == tflite::Padding_VALID) {
                dstOp->padMode = BrixLab::PaddingType::PaddingSAME;
            }

            // weight
            LOG_CHECK(weightTensor->type == tflite::TensorType_UINT8, "weights tensor Type UINT8") << "weights Data type ERROR";
            dstOp->conv_weights = (DType *)xcalloc(weightSize, sizeof(DType));
            auto originalWeightPtr = reinterpret_cast<const DType*>(tfliteModelBuffer[weightTensor->buffer]->data.data());
            convertDataFormatTflite(originalWeightPtr, dstOp->conv_weights, kh, kw, ci, depthMultiplier);
            dstOp->hasBias = (inputSize == 3);
            // have bias
            if (inputSize == 3) {
                const auto& biasTensor = tfliteTensors[tfliteOp->inputs[2]];
                LOG_CHECK(biasTensor->type == tflite::TensorType_INT32, "Bias Type INT32") << "Bias Type ERROR";
                const auto& biasData = tfliteModelBuffer[biasTensor->buffer]->data;
                dstOp->bias_zero_point               = biasTensor->quantization->zero_point[0];
                dstOp->bias_scale                    = biasTensor->quantization->scale[0];
                auto shape = biasTensor->shape;
                const int cii = biasData.size() / 4;
                LOG_CHECK( cii == ci, "CHECK biasData size") << "Bias Data ERROR";
                dstOp->quantized_bias = (int32_t*) biasData.data();
            }
            dstOp->fused_ops      = true;
            dstOp->fused_act_type = static_cast<BrixLab::FusedActivation>(tfliteConvOption->fused_activation_function);
        } else {
            dstOp->quantized_type    = BrixLab::QUANITIZED_TYPE::FLOAT32_REGULAR;
            dstOp->op_type           = BrixLab::OP_type::CONVOLUTION;
            dstOp->conv_weights      = (DType *)xcalloc(weightSize, sizeof(DType));
            auto originalWeightPtr                  = reinterpret_cast<const DType*>(tfliteModelBuffer[weightTensor->buffer]->data.data());
            int depthMultiplier                     = tfliteConvOption->depth_multiplier;
            weightSize                              *= depthMultiplier; 
            if(originalWeightPtr){
                convertDataFormatTflite(originalWeightPtr, dstOp->conv_weights, kh, kw, ci, depthMultiplier);
                // bias
                if (inputSize == 3) {
                    const auto& biasTensor = tfliteTensors[tfliteOp->inputs[2]];
                    auto originalBiasPtr   = reinterpret_cast<const DType*>(tfliteModelBuffer[biasTensor->buffer]->data.data());
                    dstOp->conv_bias = (DType *)xcalloc(ci * depthMultiplier, sizeof(DType));
                    ::memcpy(dstOp->conv_bias, originalBiasPtr, sizeof(DType) * ci * depthMultiplier);
                }
            }
            dstOp->relu6 = false;
            dstOp->relu  = false;
            auto acticationFun = tfliteConvOption->fused_activation_function;
            if (acticationFun == tflite::ActivationFunctionType_RELU) {
                dstOp->relu = true;
            } else if (acticationFun == tflite::ActivationFunctionType_RELU6) {
                dstOp->relu6 = true;
            } else if (acticationFun > tflite::ActivationFunctionType_NONE) {
                LOG(FATAL_ERROR, "activation error") << "ONEDNN Convolution do not Support fused_activation_function: " << acticationFun;
            }

            // kernel size    
            dstOp->k_w       = kw;
            dstOp->k_h       = kh;
            dstOp->k_in      = ci;
            dstOp->k_c       = ci *depthMultiplier;           
            dstOp->groups    = ci *depthMultiplier;

            // stride
            dstOp->stridesX  = tfliteConvOption->stride_w;
            dstOp->stridesY  = tfliteConvOption->stride_h;
            // default
            dstOp->dilateX = tfliteConvOption->dilation_w_factor - 1;
            dstOp->dilateY = tfliteConvOption->dilation_h_factor - 1;
            dstOp->padMode = BrixLab::PaddingType::PaddingVALID;
            if (tfliteConvOption->padding == tflite::Padding_SAME) {
                dstOp->padMode = BrixLab::PaddingType::PaddingSAME;
            }
        }
        
        // set input output index
        {
            auto originalWeightPtr = reinterpret_cast<const DType*>(tfliteModelBuffer[weightTensor->buffer]->data.data());
            if(originalWeightPtr){
                dstOp->inIndexs.resize(1);
                dstOp->outIndexs.resize(1);
                dstOp->inIndexs[0]  = tfliteOp->inputs[0];
                dstOp->outIndexs[0] = tfliteOp->outputs[0];
            }else{
                dstOp->inIndexs.resize(inputSize);
                dstOp->outIndexs.resize(1);
                dstOp->outIndexs[0] = tfliteOp->outputs[0];
                for(int i = 0; i < inputSize; ++i){
                    dstOp->inIndexs[i] = tfliteOp->inputs[i];
                }
            }
        }
    }

    INSTANEC_OP_CONVERTER(DepthConv2DTflite);
    REGISTER_CONVERTER(DepthConv2DTflite<float>, float, BuiltinOperator_DEPTHWISE_CONV_2D);
    REGISTER_CONVERTER(DepthConv2DTflite<uint8_t>, uint8_t, BuiltinOperator_DEPTHWISE_CONV_2D);

    DECLARE_OP_COVERTER(PoolingTflite);
    template<typename DType>
    OP_type PoolingTflite<DType>::opType(bool quantizedModel){
        if(quantizedModel){
            return BrixLab::OP_type::POOLING;
        }else{
            return BrixLab::OP_type::POOLING;
        }
    }

    template<typename DType>
    void PoolingTflite<DType>::run(layerWeightsParam<DType> *dstOp,
                        const std::unique_ptr<tflite::OperatorT>& tfliteOp,
                        const std::vector<std::unique_ptr<tflite::TensorT>>& tfliteTensors,
                        const std::vector<std::unique_ptr<tflite::BufferT>>& tfliteModelBuffer,
                        const std::vector<std::unique_ptr<tflite::OperatorCodeT>>& tfliteOpSet, 
                        bool quantizedModel){
        const auto& tflitePoolOption = tfliteOp->builtin_options.AsPool2DOptions();

        if(quantizedModel) {
            dstOp->quantized_type    = BrixLab::QUANITIZED_TYPE::UINT8_QUANTIZED;
            dstOp->op_type           = BrixLab::OP_type::POOLING;
            dstOp->p_kernelsX        = tflitePoolOption->filter_width;
            dstOp->p_kernelsY        = tflitePoolOption->filter_height;
            
            dstOp->p_stridesX        = tflitePoolOption->stride_w;
            dstOp->p_stridesY        = tflitePoolOption->stride_h;

            // output
            const int outputIndex    = tfliteOp->outputs[0];
            const auto& outputTensor = tfliteTensors[outputIndex];

            CalculateActivationRangeUint8((BrixLab::FusedActivation)tflitePoolOption->fused_activation_function,
                                            outputTensor->quantization, &dstOp->outActivationMin,
                                            &dstOp->outActivationMax);

            if (tflitePoolOption->padding == tflite::Padding_SAME) {
                dstOp->pooling_padType   = BrixLab::PaddingType::PaddingSAME;
            } else if (tflitePoolOption->padding == tflite::Padding_VALID) {
                dstOp->pooling_padType   = BrixLab::PaddingType::PaddingVALID;
            }
            const auto opIndex = tfliteOp->opcode_index;
            auto opType        = tfliteOpSet[opIndex]->builtin_code;
            dstOp->pooling_type = get_tflitePooling_Type(opType);
        } else {
            LOG_CHECK(tflitePoolOption->fused_activation_function == tflite::ActivationFunctionType_NONE, "NO Activation Type");
            dstOp->quantized_type    = BrixLab::QUANITIZED_TYPE::FLOAT32_REGULAR;
            dstOp->op_type           = BrixLab::OP_type::POOLING;
            
            dstOp->p_kernelsX        = tflitePoolOption->filter_width;
            dstOp->p_kernelsY        = tflitePoolOption->filter_height;
            dstOp->p_stridesY        = tflitePoolOption->stride_h;
            dstOp->p_stridesX        = tflitePoolOption->stride_w;
            if (tflitePoolOption->padding == tflite::Padding_SAME) {
                dstOp->pooling_padType   = BrixLab::PaddingType::PaddingSAME;
            } else if (tflitePoolOption->padding == tflite::Padding_VALID) {
                dstOp->pooling_padType   = BrixLab::PaddingType::PaddingVALID;
            }
            const auto opIndex          = tfliteOp->opcode_index;
            auto opType                 = tfliteOpSet[opIndex]->builtin_code;
            dstOp->pooling_type  = get_tflitePooling_Type(opType);
            //poolParam->isGlobal = false;
        }

        LOG_CHECK(tfliteOp->inputs.size() == 1, "CHECK POOLING INPUTS SIZE") << "Tflite pooling input ERROR";
        
        // set input output index
        dstOp->inIndexs.resize(1);
        dstOp->outIndexs.resize(1);
        dstOp->inIndexs[0]  = tfliteOp->inputs[0];
        dstOp->outIndexs[0] = tfliteOp->outputs[0];
    }

    INSTANEC_OP_CONVERTER(PoolingTflite);
    REGISTER_CONVERTER(PoolingTflite<float>, float, BuiltinOperator_MAX_POOL_2D);
    REGISTER_CONVERTER(PoolingTflite<uint8_t>, uint8_t, BuiltinOperator_MAX_POOL_2D);

    REGISTER_CONVERTER(PoolingTflite<float>, float, BuiltinOperator_AVERAGE_POOL_2D);
    REGISTER_CONVERTER(PoolingTflite<uint8_t>, uint8_t, BuiltinOperator_AVERAGE_POOL_2D);

    DECLARE_OP_COVERTER(AddTflite);

    template<typename DType>
    OP_type AddTflite<DType>::opType(bool quantizedModel){
        if(quantizedModel){
            return BrixLab::OP_type::BINARY_OP;
        }else{
            return BrixLab::OP_type::BINARY_OP;
        }
    }

    template<typename DType>
    void AddTflite<DType>::run(layerWeightsParam<DType> *dstOp, const std::unique_ptr<tflite::OperatorT>& tfliteOp,
                        const std::vector<std::unique_ptr<tflite::TensorT>>& tfliteTensors,
                        const std::vector<std::unique_ptr<tflite::BufferT>>& tfliteModelBuffer,
                        const std::vector<std::unique_ptr<tflite::OperatorCodeT>>& tfliteOpSet, bool quantizedModel){
        const auto& addOption = tfliteOp->builtin_options.AsAddOptions();
        dstOp->fused_ops      = false;
        if (quantizedModel) {
            dstOp->quantized_type    = BrixLab::QUANITIZED_TYPE::UINT8_QUANTIZED;
            dstOp->op_type           = BrixLab::OP_type::BINARY_OP;
            
            LOG_CHECK(tfliteOp->inputs.size() == 2, "OP ADD INPUT SIZE") << "tflite Reshape input ERROR";

            // input0
            const int input1Index                       = tfliteOp->inputs[0];
            const auto& input1Tensor                    = tfliteTensors[input1Index];
            dstOp->inputs_zeropoint.push_back(input1Tensor->quantization->zero_point[0]);
            dstOp->inputs_scale.push_back(input1Tensor->quantization->scale[0]);

            // input1
            const int input2Index                       = tfliteOp->inputs[1];
            const auto& input2Tensor                    = tfliteTensors[input2Index];
            dstOp->inputs_zeropoint.push_back(input2Tensor->quantization->zero_point[0]);
            dstOp->inputs_scale.push_back(input2Tensor->quantization->scale[0]);

            // output
            const int outputIndex                       = tfliteOp->outputs[0];
            const auto& outputTensor                    = tfliteTensors[outputIndex];
            dstOp->outputs_zero_point                = outputTensor->quantization->zero_point[0];
            dstOp->outputs_zero_point                = outputTensor->quantization->scale[0];

            dstOp->fused_ops                         = true;
            dstOp->fused_act_type                    = static_cast<BrixLab::FusedActivation>(addOption->fused_activation_function);

        } else {
            LOG_CHECK(addOption->fused_activation_function == tflite::ActivationFunctionType_NONE, "CHECK the Fused activation Func")
                                                        << "BinaryOP Should not has fused_activation_function";
            dstOp->quantized_type    = BrixLab::QUANITIZED_TYPE::FLOAT32_REGULAR;
            dstOp->op_type           = BrixLab::OP_type::BINARY_OP;
        }
    }
    INSTANEC_OP_CONVERTER(AddTflite);
    REGISTER_CONVERTER(AddTflite<float>, float, BuiltinOperator_ADD);
    REGISTER_CONVERTER(AddTflite<uint8_t>, uint8_t, BuiltinOperator_ADD);

    DECLARE_OP_COVERTER(ConcatTflite);
    template<typename DType>
    OP_type ConcatTflite<DType>::opType(bool quantizedModel){
        if(quantizedModel){
            return BrixLab::OP_type::CONCAT;
        }else{
            return BrixLab::OP_type::CONCAT;
        }
    }

    template<typename DType>
    void ConcatTflite<DType>::run(layerWeightsParam<DType> *dstOp, const std::unique_ptr<tflite::OperatorT>& tfliteOp,
                        const std::vector<std::unique_ptr<tflite::TensorT>>& tfliteTensors,
                        const std::vector<std::unique_ptr<tflite::BufferT>>& tfliteModelBuffer,
                        const std::vector<std::unique_ptr<tflite::OperatorCodeT>>& tfliteOpSet, bool quantizedModel){
        const auto& tfliteConcatOption = tfliteOp->builtin_options.AsConcatenationOptions();
        dstOp->fused_ops               = false;
        if (quantizedModel) {
            dstOp->quantized_type         = BrixLab::QUANITIZED_TYPE::UINT8_QUANTIZED;
            dstOp->op_type                = BrixLab::OP_type::CONCAT;
            dstOp->concat_axis            = tfliteConcatOption->axis;

            for (unsigned int i = 0; i < tfliteOp->inputs.size(); i++) {
                const int inputIndex     = tfliteOp->inputs[i];
                const auto& inputTensor  = tfliteTensors[inputIndex];
                dstOp->inputs_zeropoint.push_back(inputTensor->quantization->zero_point[0]);
                dstOp->inputs_scale.push_back(inputTensor->quantization->scale[0]);
            }

            const int outputIndex                   = tfliteOp->outputs[0];
            const auto& outputTensor                = tfliteTensors[outputIndex];
            dstOp->outputs_zero_point         = outputTensor->quantization->zero_point[0];
            dstOp->outputs_scale              = outputTensor->quantization->scale[0];
            dstOp->fused_ops                  = true;
            dstOp->fused_act_type             = static_cast<BrixLab::FusedActivation>(tfliteConcatOption->fused_activation_function);
        } else {
            LOG_CHECK(tfliteConcatOption->fused_activation_function == tflite::ActivationFunctionType_NONE, "NO Fused activation Function")
                                                    <<"No FusedActivation Function";
            dstOp->quantized_type         = BrixLab::QUANITIZED_TYPE::FLOAT32_REGULAR;
            dstOp->op_type                = BrixLab::OP_type::CONCAT;
            dstOp->concat_axis            = tfliteConcatOption->axis;
        }
    }

    INSTANEC_OP_CONVERTER(ConcatTflite);
    REGISTER_CONVERTER(ConcatTflite<float>, float, BuiltinOperator_CONCATENATION);
    REGISTER_CONVERTER(ConcatTflite<uint8_t>, uint8_t, BuiltinOperator_CONCATENATION);

    DECLARE_OP_COVERTER(ResizeBilinearTflite);

    template<typename DType>
    OP_type ResizeBilinearTflite<DType>::opType(bool quantizedModel){
        LOG_CHECK(!quantizedModel, "CHECK quantizedmodel");
        return BrixLab::OP_type::RESAMPLING;
    }

    template<typename DType>
    void ResizeBilinearTflite<DType>::run(layerWeightsParam<DType> *dstOp, const std::unique_ptr<tflite::OperatorT>& tfliteOp,
                        const std::vector<std::unique_ptr<tflite::TensorT>>& tfliteTensors,
                        const std::vector<std::unique_ptr<tflite::BufferT>>& tfliteModelBuffer,
                        const std::vector<std::unique_ptr<tflite::OperatorCodeT>>& tfliteOpSet, bool quantizedModel){
        LOG_CHECK(!quantizedModel, "resizedBiliar must quantized!");
        dstOp->quantized_type    = BrixLab::QUANITIZED_TYPE::UINT8_QUANTIZED;
        const auto &scaleTensor  = tfliteTensors[tfliteOp->inputs[1]];
        auto code = tfliteOpSet[tfliteOp->opcode_index]->builtin_code;
        if (BuiltinOperator_RESIZE_NEAREST_NEIGHBOR == code) {
            const auto& nearest = tfliteOp->builtin_options.AsResizeNearestNeighborOptions();
            dstOp->resized_type          = BrixLab::ResizingType::ResizingNearest;
            dstOp->resized_alignCorners  = nearest->align_corners;
        } else if (BuiltinOperator_RESIZE_BILINEAR == code) {
            const auto& resizeOption = tfliteOp->builtin_options.AsResizeBilinearOptions();
            dstOp->resized_type          = BrixLab::ResizingType::ResizingBilinear;
            dstOp->resized_alignCorners  = resizeOption->align_corners;
        } else {
            LOG(FATAL_ERROR, "no support other ops");
        }
        auto scaleDataPtr        = reinterpret_cast<const int *>(tfliteModelBuffer[scaleTensor->buffer]->data.data());

        dstOp->resized_height = scaleDataPtr[1];
        dstOp->resized_width  = scaleDataPtr[0];

        dstOp->adjust_height_scale   = 1.0;
        dstOp->adjust_width_scale    = 1.0;
        // set input output index
        dstOp->inIndexs.resize(1);
        dstOp->outIndexs.resize(1);
        dstOp->inIndexs[0]  = tfliteOp->inputs[0];
        dstOp->outIndexs[0] = tfliteOp->outputs[0];
    }
    INSTANEC_UINT8_OP_CONVERTER(ResizeBilinearTflite);
    REGISTER_CONVERTER(ResizeBilinearTflite<uint8_t>, uint8_t, BuiltinOperator_RESIZE_BILINEAR);
    REGISTER_CONVERTER(ResizeBilinearTflite<uint8_t>, uint8_t, BuiltinOperator_RESIZE_NEAREST_NEIGHBOR);
    
    DECLARE_OP_COVERTER(ReluTflite);

    template<typename DType>
    OP_type ReluTflite<DType>::opType(bool quantizedModel){
        if(quantizedModel){
            return BrixLab::OP_type::ACTIVITION;
        }else{
            return BrixLab::OP_type::ACTIVITION;
        }
    }

    template<typename DType>
    void ReluTflite<DType>::run(layerWeightsParam<DType> *dstOp, const std::unique_ptr<tflite::OperatorT>& tfliteOp,
                        const std::vector<std::unique_ptr<tflite::TensorT>>& tfliteTensors,
                        const std::vector<std::unique_ptr<tflite::BufferT>>& tfliteModelBuffer,
                        const std::vector<std::unique_ptr<tflite::OperatorCodeT>>& tfliteOpSet, bool quantizedModel){
        auto code = tfliteOpSet[tfliteOp->opcode_index]->builtin_code;
        if (BuiltinOperator_RELU == code) {
            dstOp->activate_type        = BrixLab::activitionType::ReLU;
            dstOp->alpha                = 0.f;
            dstOp->beta                 = 1.f;
        } else if (BuiltinOperator_RELU6 == code) {
            dstOp->activate_type        = BrixLab::activitionType::ReLU;
            dstOp->alpha                = 0.f;
            dstOp->beta                 = 6.f;
        }else if (BuiltinOperator_HARD_SWISH == code) {
            dstOp->activate_type        = BrixLab::activitionType::SWISH;
            dstOp->alpha                = 1.f;//待定
            dstOp->beta                 = 0.f;
        } else {
            LOG(FATAL_ERROR, "no support other Relu ops");
        }
    }
    INSTANEC_OP_CONVERTER(ReluTflite);
    REGISTER_CONVERTER(ReluTflite<float>, float, BuiltinOperator_RELU);
    REGISTER_CONVERTER(ReluTflite<uint8_t>, uint8_t, BuiltinOperator_RELU);
    REGISTER_CONVERTER(ReluTflite<float>, float, BuiltinOperator_RELU6);
    REGISTER_CONVERTER(ReluTflite<uint8_t>, uint8_t, BuiltinOperator_RELU6);
    REGISTER_CONVERTER(ReluTflite<float>, float, BuiltinOperator_HARD_SWISH);
    REGISTER_CONVERTER(ReluTflite<uint8_t>, uint8_t, BuiltinOperator_HARD_SWISH);

    DECLARE_OP_COVERTER(SoftmaxTflite);

    template<typename DType>
    OP_type SoftmaxTflite<DType>::opType(bool quantizedModel){
        if(quantizedModel){
            return BrixLab::OP_type::SOFTMAX;
        }else {
            return BrixLab::OP_type::SOFTMAX;
        }
    }

    template<typename DType>
    void SoftmaxTflite<DType>::run(layerWeightsParam<DType> *dstOp, const std::unique_ptr<tflite::OperatorT>& tfliteOp,
                        const std::vector<std::unique_ptr<tflite::TensorT>>& tfliteTensors,
                        const std::vector<std::unique_ptr<tflite::BufferT>>& tfliteModelBuffer,
                        const std::vector<std::unique_ptr<tflite::OperatorCodeT>>& tfliteOpSet, bool quantizedModel){
        LOG_CHECK(tfliteOp->inputs.size() == 1, "input size") << "Tflite Softmax input ERROR!";
        const auto& tfliteSoftmaxOption = tfliteOp->builtin_options.AsSoftmaxOptions();

        if (quantizedModel) {
            dstOp->quantized_type    = BrixLab::QUANITIZED_TYPE::UINT8_QUANTIZED;
            dstOp->op_type           = BrixLab::OP_type::SOFTMAX;
            dstOp->softmax_beta      = tfliteSoftmaxOption->beta;
            // input
            const int inputIndex                = tfliteOp->inputs[0];
            const auto& inputTensor             = tfliteTensors[inputIndex];
            dstOp->softmax_inputscale           = inputTensor->quantization->scale[0];
        } else {
            dstOp->quantized_type    = BrixLab::QUANITIZED_TYPE::FLOAT32_REGULAR;
            dstOp->op_type           = BrixLab::OP_type::SOFTMAX;
            dstOp->softmax_axis      = 1;
        }
        // set input output index
        dstOp->inIndexs.resize(1);
        dstOp->outIndexs.resize(1);
        dstOp->inIndexs[0]  = tfliteOp->inputs[0];
        dstOp->outIndexs[0] = tfliteOp->outputs[0];
    }
    INSTANEC_OP_CONVERTER(SoftmaxTflite);
    REGISTER_CONVERTER(SoftmaxTflite<float>, float, BuiltinOperator_SOFTMAX);
    REGISTER_CONVERTER(SoftmaxTflite<uint8_t>, uint8_t, BuiltinOperator_SOFTMAX);

    DECLARE_OP_COVERTER(ReductionTflite);
    template<typename DType>
    OP_type ReductionTflite<DType>::opType(bool quantizedModel){
        if(quantizedModel){
            return BrixLab::OP_type::REDUCTION;
        }else{
            return BrixLab::OP_type::REDUCTION;
        }
    }

    template<typename DType>
    void ReductionTflite<DType>::run(layerWeightsParam<DType> *dstOp, const std::unique_ptr<tflite::OperatorT>& tfliteOp,
                        const std::vector<std::unique_ptr<tflite::TensorT>>& tfliteTensors,
                        const std::vector<std::unique_ptr<tflite::BufferT>>& tfliteModelBuffer,
                        const std::vector<std::unique_ptr<tflite::OperatorCodeT>>& tfliteOpSet, bool quantizedModel){
        auto opt                        = tfliteOp->builtin_options.AsReducerOptions();
        dstOp->reduce_keep_dims   = opt->keep_dims;
        #ifdef TF_CONVERT_ORIGIN
        const int input1Idx                       = tfliteOp->inputs[1];
        const auto& input1Tensor                  = tfliteTensors[input1Idx];
        if(input1Tensor.is_variable == false){
            auto buffer1Idx=input1Tensor.buffer;
            auto buffer1=tfliteModelBuffer[buffer1Idx];
            auto shape=input1Tensor.shape;
            param->dim.resize(shape.size());
            for(decltype(shape.size()) x=0;x<shape.size();x++){
            param->dim[x]=shape[x];
            }
        }
        #endif
        dstOp->op_type      = BrixLab::OP_type::REDUCTION;
        switch(tfliteOpSet[tfliteOp->opcode_index]->builtin_code){
            case tflite::BuiltinOperator_SUM:{
                dstOp->reduction_type = BrixLab::ReductionType::ReductionType_SUM;
                break;
            }
            case tflite::BuiltinOperator_REDUCE_MAX:{
                dstOp->reduction_type = BrixLab::ReductionType::ReductionType_MAXIMUM;
                break;
            }
            case tflite::BuiltinOperator_REDUCE_MIN:{
                dstOp->reduction_type = BrixLab::ReductionType::ReductionType_MINIMUM;
                break;
            }
            case tflite::BuiltinOperator_REDUCE_PROD:{
                dstOp->reduction_type = BrixLab::ReductionType::ReductionType_PROD;
                break;
            }
            case tflite::BuiltinOperator_MEAN:{
                dstOp->reduction_type = BrixLab::ReductionType::ReductionType_MEAN;
                break;
            }
            default:{
                LOG(FATAL_ERROR, "no support reduction type") << "onednn Converter Not Supported!!! Reduction Op: "
                        << tfliteOpSet[tfliteOp->opcode_index]->custom_code;
            }
        }
    }

    INSTANEC_OP_CONVERTER(ReductionTflite);

    REGISTER_CONVERTER(ReductionTflite<float>, float, BuiltinOperator_REDUCE_ANY);
    REGISTER_CONVERTER(ReductionTflite<uint8_t>, uint8_t, BuiltinOperator_REDUCE_ANY);
    REGISTER_CONVERTER(ReductionTflite<float>, float, BuiltinOperator_MEAN);
    REGISTER_CONVERTER(ReductionTflite<uint8_t>, uint8_t, BuiltinOperator_MEAN);
    REGISTER_CONVERTER(ReductionTflite<float>, float, BuiltinOperator_REDUCE_MIN);
    REGISTER_CONVERTER(ReductionTflite<uint8_t>, uint8_t, BuiltinOperator_REDUCE_MIN);
    REGISTER_CONVERTER(ReductionTflite<float>, float, BuiltinOperator_REDUCE_PROD);
    REGISTER_CONVERTER(ReductionTflite<uint8_t>, uint8_t, BuiltinOperator_REDUCE_PROD);
    REGISTER_CONVERTER(ReductionTflite<float>, float, BuiltinOperator_REDUCE_MAX);
    REGISTER_CONVERTER(ReductionTflite<uint8_t>, uint8_t, BuiltinOperator_REDUCE_MAX);
    REGISTER_CONVERTER(ReductionTflite<float>, float, BuiltinOperator_SUM);
    REGISTER_CONVERTER(ReductionTflite<uint8_t>, uint8_t, BuiltinOperator_SUM);

    DECLARE_OP_COVERTER(BinaryTflite);
    template<typename DType>
    OP_type BinaryTflite<DType>::opType(bool quantizedModel){
        if(quantizedModel){
            return BrixLab::OP_type::BINARY_OP;
        }else{
            return BrixLab::OP_type::BINARY_OP;
        }
    }

    template<typename DType>
    void BinaryTflite<DType>::run(layerWeightsParam<DType> *dstOp, const std::unique_ptr<tflite::OperatorT>& tfliteOp,
                        const std::vector<std::unique_ptr<tflite::TensorT>>& tfliteTensors,
                        const std::vector<std::unique_ptr<tflite::BufferT>>& tfliteModelBuffer,
                        const std::vector<std::unique_ptr<tflite::OperatorCodeT>>& tfliteOpSet, bool quantizedModel){
        dstOp->op_type  = BrixLab::OP_type::BINARY_OP;
        switch (tfliteOpSet[tfliteOp->opcode_index]->builtin_code) {
            case tflite::BuiltinOperator_ADD: {
                dstOp->binary_type = BrixLab::BinaryOpOperationType::BinaryOpOperation_ADD;
                break;
            }
            case tflite::BuiltinOperator_SUB: {
                dstOp->binary_type = BrixLab::BinaryOpOperationType::BinaryOpOperation_SUB;
                break;
            }
            case BuiltinOperator_MUL:
            case BuiltinOperator_LOGICAL_AND: {
                dstOp->binary_type = BrixLab::BinaryOpOperationType::BinaryOpOperation_MUL;
                break;
            }
            case tflite::BuiltinOperator_DIV: {
                dstOp->binary_type = BrixLab::BinaryOpOperationType::BinaryOpOperation_DIV;
                break;
            }
            case tflite::BuiltinOperator_MAXIMUM: {
                dstOp->binary_type = BrixLab::BinaryOpOperationType::BinaryOpOperation_MAXIMUM;
                break;
            }
            case tflite::BuiltinOperator_MINIMUM: {
                dstOp->binary_type = BrixLab::BinaryOpOperationType::BinaryOpOperation_MINIMUM;
                break;
            }
            default: {
                LOG(FATAL_ERROR, "BinaryOp not supported") << "onednn Converter Not Supported!!! BinaryOp:"
                                    << tfliteOpSet[tfliteOp->opcode_index]->custom_code;
            }
        }
    }
    INSTANEC_OP_CONVERTER(BinaryTflite);

    //REGISTER_CONVERTER(BinaryTflite<float>, float, BuiltinOperator_ADD);
    //REGISTER_CONVERTER(BinaryTflite<uint8_t>, uint8_t, BuiltinOperator_ADD);

    REGISTER_CONVERTER(BinaryTflite<float>, float, BuiltinOperator_SUB);
    REGISTER_CONVERTER(BinaryTflite<uint8_t>, uint8_t, BuiltinOperator_SUB);

    REGISTER_CONVERTER(BinaryTflite<float>, float, BuiltinOperator_MUL);
    REGISTER_CONVERTER(BinaryTflite<uint8_t>, uint8_t, BuiltinOperator_MUL);

    REGISTER_CONVERTER(BinaryTflite<float>, float, BuiltinOperator_DIV);
    REGISTER_CONVERTER(BinaryTflite<uint8_t>, uint8_t, BuiltinOperator_DIV);

    REGISTER_CONVERTER(BinaryTflite<float>, float, BuiltinOperator_MAXIMUM);
    REGISTER_CONVERTER(BinaryTflite<uint8_t>, uint8_t, BuiltinOperator_MAXIMUM);

    REGISTER_CONVERTER(BinaryTflite<float>, float, BuiltinOperator_MINIMUM);
    REGISTER_CONVERTER(BinaryTflite<uint8_t>, uint8_t, BuiltinOperator_MINIMUM);

    REGISTER_CONVERTER(BinaryTflite<float>, float, BuiltinOperator_LOGICAL_AND);
    REGISTER_CONVERTER(BinaryTflite<uint8_t>, uint8_t, BuiltinOperator_LOGICAL_AND);


} // namespace BrixLab
