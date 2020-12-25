#ifndef INTEL_MKL_DNN_UTILS_
#define INTEL_MKL_DNN_UTILS_
#include <sys/stat.h>
#include <sys/types.h>

#include <algorithm>
#include <cmath>
#include <iostream>
#include <string>
#include <vector>
#include <malloc.h>
#include <string.h>
#include "check_error.hpp"
#include "oneapi/dnnl.hpp"
#include "logkit.hpp"
/**
 * 从tflite文件，读取模型结构，还有权重参数， 
 * 重点是对tflite模型，解析获得节点数据，并且转化成以strLayerNode为节点的graph链表。
 * 然后再推理的时候，根据每一个节点类型依次推断。
 * 主要需要的工作包括：
 * 1）tflite模型解析，生成graph链表，(包括每一个链表节点的数据解析)
 * 2）链表中每个节点的初始化(初始化dnnl所需的内存空间和权重)
 * 3）正向推理
 * 4）完整graph的推理
*/
namespace BrixLab
{
    static dnnl::engine graph_eng(dnnl::engine::kind::cpu, 0);
    static dnnl::stream graph_stream(graph_eng);
    enum TENSOR_FORMATE{
        NHWC,
        NCHW,
    };
    typedef struct _TensorShape{
        int Batch;
        int Channel;
        int Height;
        int Width;
        TENSOR_FORMATE format;
        _TensorShape operator = (const _TensorShape& right){
            this->Batch     = right.Batch;
            this->Channel   = right.Channel;
            this->Height    = right.Height;
            this->Width     = right.Width;
            this->format    = right.format;
            return *this;
        }
        bool operator == (const _TensorShape& right){
            bool euqal = false;
            if(Batch == right.Batch && Channel == right.Channel &&
                Height == right.Height && Width == right.Width){
                euqal  = true;
            }
            return euqal;
        }
    }TensorShape;
    // 目前支持float32、uint8、Int32运算
    enum DataType {
        DataType_DT_INVALID = 0,
        DataType_DT_FLOAT = 1,
        DataType_DT_DOUBLE = 2,
        DataType_DT_INT32 = 3,
        DataType_DT_UINT8 = 4,
        DataType_DT_INT16 = 5,
        DataType_DT_INT8 = 6,
    };
    enum OP_type{
        DATA_INPUTS,
        CONVOLUTION,
        POOLING,
        BATCHNORM,
        ELTWISE,
        DETECTION,
        CONCAT,
        DECONVOLUTION,
        RESAMPLING,
        ACTIVITION,
        INNERPRODUCT,
        SOFTMAX,
        REDUCTION,
        BINARY_OP,
        //OP_TYPE_CONST,
    };
    enum activitionType{
        ReLU,
        Sigmoid,
        Tanh,
        ABS,
        BOUNDED_RELU,
        CLIP,
        SWISH,
        LINEAR_SCALE_BIAS,
        LOG,
    };
    enum FusedActivation {
        Fused_kTfLiteActNone = 0,
        Fused_kTfLiteActRelu = 1,
        Fused_kTfLiteActRelu1 = 2,
        Fused_kTfLiteActRelu6 = 3,
        Fused_kTfLiteActTanh = 4,
        Fused_kTfLiteActSignBit = 5,
        Fused_kTfLiteActSigmoid = 6,
        Fused_MIN = Fused_kTfLiteActNone,
        Fused_MAX = Fused_kTfLiteActSigmoid
    };

    enum EltwiseType{
        ElementSum,
        ElementProduct,
    };
    enum PoolingType{
        PoolingMAX,
        PoolingAVAGE,
    };
    enum QUANITIZED_TYPE{
        UINT8_QUANTIZED,
        FLOAT32_REGULAR,
    };
    enum PaddingType{
        PaddingSAME,
        PaddingVALID,
    };
    enum ResizingType{
        ResizingNearest,
        ResizingBilinear,
    };
    enum ReductionType {
        ReductionType_SUM = 0,
        ReductionType_ASUM = 1,
        ReductionType_SUMSQ = 2,
        ReductionType_MEAN = 3,
        ReductionType_MAXIMUM = 4,
        ReductionType_MINIMUM = 5,
        ReductionType_PROD = 6,
        ReductionType_ANY = 7,
        ReductionType_ALL = 8,
        ReductionType_MIN = ReductionType_SUM,
        ReductionType_MAX = ReductionType_ALL
    };
    enum BinaryOpOperationType {
        BinaryOpOperation_ADD = 0,
        BinaryOpOperation_SUB = 1,
        BinaryOpOperation_MUL = 2,
        BinaryOpOperation_DIV = 3,
        BinaryOpOperation_MAX_TEMP = 4,
        BinaryOpOperation_MIN_TEMP = 5,
        BinaryOpOperation_POW = 6,
        BinaryOpOperation_REALDIV = 7,
        BinaryOpOperation_MINIMUM = 8,
        BinaryOpOperation_MAXIMUM = 9,
        BinaryOpOperation_GREATER = 10,
        BinaryOpOperation_GREATER_EQUAL = 11,
        BinaryOpOperation_LESS = 12,
        BinaryOpOperation_FLOORDIV = 13,
        BinaryOpOperation_SquaredDifference = 14,
        BinaryOpOperation_EQUAL = 15,
        BinaryOpOperation_LESS_EQUAL = 16,
        BinaryOpOperation_FLOORMOD = 17,
        BinaryOpOperation_MOD = 19,
        BinaryOpOperation_ATAN2 = 20,
        BinaryOpOperation_LOGICALOR = 21,
        BinaryOpOperation_NOTEQUAL = 22,
        BinaryOpOperation_MIN = BinaryOpOperation_ADD,
        BinaryOpOperation_MAX = BinaryOpOperation_NOTEQUAL
    };

    struct Post_OPs_Param
    {
        dnnl::algorithm posts_op;
        float alpha;
        float beta;
        float scale;
    };
    
    template<typename DType>
    struct strParam{
        std::string node_name;
        OP_type op_type;
        DataType data_type;
        std::vector<TensorShape> in_shapes;
        std::vector<TensorShape> out_shapes;
        std::vector<int> inIndexs;
        std::vector<int> outIndexs;
        //DType *extradata;
        //quantized_uint8 needed for weights & inputs
        int weights_zero_point;
        float weights_scale;
        int bias_zero_point;
        float bias_scale;
        std::vector<int> inputs_zeropoint;
        std::vector<float> inputs_scale;
        int outputs_zero_point;
        float outputs_scale;
        QUANITIZED_TYPE quantized_type;
        int32_t *quantized_bias;
        // (de)convolution params
        bool relu;
        bool relu6;
        int k_w;
        int k_h;
        PaddingType padMode;
        int stridesX, stridesY;
        int k_c;
        int k_in;
        int dilateX, dilateY;
        bool hasBias;
        int groups;
        //convolution weights
        DType *conv_weights;
        DType *conv_bias;
        bool fused_ops;
        FusedActivation fused_act_type;
        //deconvolution weights
        DType *transposed_weights;
        DType *transposed_bias;

        //batchNorm params
        DType *b_means;
        DType *b_variance;
        DType *b_shift_scale;

        // activition params
        activitionType activate_type;
        bool in_palce;                      

        //pooling params
        int32_t outActivationMin;
        int32_t outActivationMax;
        PoolingType pooling_type;
        PaddingType pooling_padType;
        int p_kernelsX, p_kernelsY;
        int p_stridesX, p_stridesY;
        int p_dilatedX, p_dilatedY;
        //innerproducts params
        float alpha, beta;
        DType *innerWeights;
        DType *innerBias;
        //concat params
        int concat_axis;
        //Eltwise_SUM params
        int *sum_index;
        int sum_num;
        EltwiseType eleType;
        //resample(resizedBilinar) layer
        float adjust_width_scale;
        float adjust_height_scale;
        ResizingType resized_type;
        bool resized_alignCorners;
        int resized_height;
        int resized_width;

        //softmax layer
        float softmax_beta;
        float softmax_inputscale;
        float softmax_axis;

        //reduction layer
        bool reduce_keep_dims;
        ReductionType reduction_type;

        //binary op_layer
        BinaryOpOperationType binary_type;
    };

    template<typename DType>
    struct NetT{
        std::vector<strParam<DType> >layer_ops;
        std::vector<std::string> tensorName;
        std::vector<std::string> output_name;
        int32_t tensorNumber;
    };

    template<typename DType>
    struct graphSetLink;

    template<typename DType>
    struct strNodeParam{
        // layer info
        std::string node_name;
        int fatureSize;
        int weightSize;
        int biasSize;
        std::vector<TensorShape> in_shapes;
        std::vector<TensorShape> out_shapes;
        OP_type op_type;
        std::vector<int>inputs;
        std::vector<int>outputs;            
        
        dnnl::memory::dims top_shape;
        dnnl::memory layer_top_memory;
        dnnl::memory::desc layer_top_md;

        dnnl::memory::dims bottom_shape;
        dnnl::memory::desc src_bottom_md;
        dnnl::memory src_bottom_memory;

        dnnl::memory::dims weights_shape;
        dnnl::memory::desc src_weights_md;
        dnnl::memory src_weights_memory;

        dnnl::memory::dims bias_shape;
        dnnl::memory::desc src_bias_md;
        dnnl::memory src_bias_memory;
        // (de)convolution layer & param
        int groups;
        int dilateX, dilateY;
        bool hasBias;
        dnnl::memory::dims conv_strides;
        dnnl::memory::dims conv_paddingL;
        dnnl::memory::dims conv_paddingR;
        dnnl::convolution_forward::primitive_desc conv_pdesc;
        dnnl::post_ops conv_ops;
        dnnl:: primitive_attr conv_attr;
        Post_OPs_Param conv_post_op;
        // deconvolution(transposed convolution) layer & param
        dnnl::memory::dims deconv_strides;
        dnnl::memory::dims deconv_paddingL, deconv_paddingR;
        dnnl::deconvolution_forward::primitive_desc deconv_pdesc;
        dnnl::post_ops deconv_ops;
        dnnl::primitive_attr deconv_attr;
        
        // batchnorm layer
        dnnl::memory::dims batchnorm_scale_shift_shape;
        dnnl::memory batchnorm_scale_shift_memory;
        dnnl::memory::desc batchnorm_scale_shift_md;
        dnnl::batch_normalization_forward::primitive_desc batchnorm_pdesc;
        dnnl::memory batchnorm_mean_memory;
        dnnl::memory batchnorm_variance_memory;

        //pooling layer
        bool p_dialiated;
        dnnl::algorithm pooling_type;
        dnnl::memory::dims pooling_kernel;
        dnnl::memory::dims pooling_strides;
        dnnl::memory::dims pooling_paddingL;
        dnnl::memory::dims pooling_paddingR;
        dnnl::memory::dims pooling_dialiate;
        // pooling with dialated
        dnnl::pooling_v2_forward::primitive_desc pooling_pdesc;
        // pooling without dialated
        dnnl::pooling_forward::primitive_desc pooling_pdesc_without_d;

        // concat layer
        bool inputset;
        int concat_num;
        int concat_axis;
        std::vector<dnnl::memory::desc> concat_bottom_md;
        std::vector<dnnl::memory> concat_bottom_memory;
        dnnl::concat::primitive_desc concat_pdesc;
        // multi inputsum layer
        int sum_num;
        std::vector<float> sum_scale;
        std::vector<dnnl::memory> sum_bottom_memory;
        std::vector<dnnl::memory::desc> sum_bottom_md;
        dnnl::sum::primitive_desc sum_pdesc;

        // activation(elementwise) layer
        dnnl::algorithm activate_type;
        dnnl::eltwise_forward::primitive_desc eltwise_pdesc;
        float alpha, beta;

        //binary op layer
        dnnl::algorithm binary_type;
        std::vector<dnnl::memory::desc> binary_md;
        std::vector<dnnl::memory> binary_memory;
        dnnl::binary::primitive_desc binary_pdesc;

        // resampleing layer
        float adjust_scale;
        dnnl::resampling_forward::primitive_desc resample_pdesc;

        //inner-product layer
        dnnl::post_ops fc_ops;
        dnnl:: primitive_attr fc_attr;
        Post_OPs_Param fc_post_op;
        dnnl::inner_product_forward::primitive_desc inner_pdesc;

        // common layer
        std::unordered_map<int, dnnl::memory> op_args;
        void (*inference_forward)(strNodeParam<DType> *, graphSetLink<DType>&);
        strNodeParam<DType>(OP_type type):op_type(type){}
        strNodeParam<DType> operator = (strNodeParam<DType> node){
            this->op_type       = node.op_type;
            this->node_name     = node.node_name;
            if(node.in_shapes.size() > 0){
                this->in_shapes.resize(node.in_shapes.size());
                for(unsigned int i = 0; i < node.in_shapes.size(); i++){
                    this->in_shapes[i]  = node.in_shapes[i];
                }
            }
            if(node.out_shapes.size()){
                this->out_shapes.resize(node.out_shapes.size());
                for(unsigned int i = 0; i < node.out_shapes.size(); i++){
                    this->out_shapes[i] = node.out_shapes[i];
                }
            }
            if(node.inputs.size() > 0){
                this->inputs.resize(node.inputs.size());
                for(unsigned int i = 0; i < node.inputs.size(); i++){
                    this->inputs[i]     = node.inputs[i];
                }
            }
            if(node.outputs.size() > 0){
                this->outputs.resize(node.outputs.size());
                for(unsigned int i = 0; i < node.outputs.size(); i++){
                    this->outputs[i]    = node.outputs[i];
                }
            }
            if(node.top_shape.size() > 0){
                this->top_shape.resize(node.top_shape.size());
                for(unsigned int i = 0; i < node.top_shape.size(); i++){
                    this->top_shape[i]  = node.top_shape[i];
                }
            }
            this->layer_top_memory  = node.layer_top_memory;
            this->layer_top_md      = node.layer_top_md;

            if(node.bottom_shape.size() > 0){
                this->bottom_shape.resize(node.bottom_shape.size());
                for(unsigned int i = 0; i < node.bottom_shape.size(); i++){
                    this->bottom_shape[i]  = node.bottom_shape[i];
                }
            }
            this->src_bottom_md     = node.src_bottom_md;
            this->src_bottom_memory = node.src_bottom_memory;

            if(node.weights_shape.size() > 0){
                this->weights_shape.resize(node.weights_shape.size());
                for(unsigned int i = 0; i < node.weights_shape.size(); i++){
                    this->weights_shape[i]  = node.weights_shape[i];
                }
            }
            this->src_weights_md     = node.src_weights_md;
            this->src_weights_memory = node.src_weights_memory;

            if(node.bias_shape.size() > 0){
                this->bias_shape.resize(node.bias_shape.size());
                for(unsigned int i = 0; i < node.bias_shape.size(); i++){
                    this->bias_shape[i]  = node.bias_shape[i];
                }
            }
            
            this->src_bias_md       = node.src_bias_md;
            this->src_bias_memory   = node.src_bias_memory;
            this->groups            = node.groups;
            this->dilateX           = node.dilateX;
            this->dilateY           = node.dilateY;
            this->hasBias           = node.hasBias;
            this->conv_strides      = node.conv_strides;
            this->conv_paddingL     = node.conv_paddingL;
            this->conv_paddingR     = node.conv_paddingR;
            this->conv_pdesc        = node.conv_pdesc;
            this->conv_ops          = node.conv_ops;
            this->conv_post_op      = node.conv_post_op;
            this->conv_attr         = node.conv_attr;

            this->deconv_strides    = node.deconv_strides;
            this->deconv_paddingL   = node.deconv_paddingL;
            this->deconv_paddingR   = node.deconv_paddingR;
            this->deconv_pdesc      = node.deconv_pdesc;
            this->deconv_ops        = node.deconv_ops;
            this->deconv_attr       = node.deconv_attr;
            this->batchnorm_scale_shift_shape   = node.batchnorm_scale_shift_shape;
            this->batchnorm_scale_shift_md      = node.batchnorm_scale_shift_md;
            this->batchnorm_scale_shift_memory  = node.batchnorm_scale_shift_memory;
            this->batchnorm_pdesc               = node.batchnorm_pdesc;
            this->batchnorm_mean_memory         = node.batchnorm_mean_memory;
            this->batchnorm_variance_memory     = node.batchnorm_variance_memory;

            this->pooling_dialiate              = node.pooling_dialiate;
            this->pooling_kernel                = node.pooling_kernel;
            this->pooling_strides               = node.pooling_strides;
            this->pooling_paddingL              = node.pooling_paddingL;
            this->pooling_paddingR              = node.pooling_paddingR;
            this->pooling_type                  = node.pooling_type;
            this->pooling_pdesc                 = node.pooling_pdesc;
            this->pooling_pdesc_without_d       = node.pooling_pdesc_without_d;

            this->inputset                      = node.inputset;
            this->concat_num                    = node.concat_num;
            this->concat_axis                   = node.concat_axis;
            this->concat_pdesc                  = node.concat_pdesc;
            if(node.concat_bottom_md.size()>0){
                this->concat_bottom_md.resize(node.concat_bottom_md.size());
                for(unsigned int i = 0; i < node.concat_bottom_md.size(); i++){
                    this->concat_bottom_md[i]   = node.concat_bottom_md[i];
                }
            }
            if(node.concat_bottom_memory.size()>0){
                this->concat_bottom_memory.resize(node.concat_bottom_memory.size());
                for(unsigned int i = 0; i < node.concat_bottom_memory.size(); i++){
                    this->concat_bottom_memory[i]   = node.concat_bottom_memory[i];
                }
            }
            this->sum_num                   = node.sum_num;
            this->sum_scale                 = node.sum_scale;
            this->sum_bottom_memory.resize(node.sum_bottom_memory.size());
            for(unsigned int i = 0; i < node.sum_bottom_memory.size(); i++){
                this->sum_bottom_memory[i]   = node.sum_bottom_memory[i];
            }
            if(node.sum_bottom_md.size()>0){
                this->sum_bottom_md.resize(node.sum_bottom_md.size());
                for(unsigned int i = 0; i < node.sum_bottom_md.size(); i++){
                    this->sum_bottom_md[i]   = node.sum_bottom_md[i];
                }
            }
            this->activate_type         = node.activate_type;
            this->eltwise_pdesc         = node.eltwise_pdesc;
        
            this->alpha                 = node.alpha; 
            this->beta                  = node.beta;
            this->binary_type           = node.binary_type;
            this->binary_pdesc          = node.binary_pdesc;

            if(node.binary_memory.size()>0){
                this->binary_memory.resize(node.binary_memory.size());
                for(unsigned int i = 0; i < node.binary_memory.size(); i++){
                    this->binary_memory[i]   = node.binary_memory[i];
                }
            }

            if(node.binary_md.size()>0){
                this->binary_md.resize(node.binary_md.size());
                for(unsigned int i = 0; i < node.binary_md.size(); i++){
                    this->binary_md[i]   = node.binary_md[i];
                }
            }
            this->adjust_scale      = node.adjust_scale;
            this->resample_pdesc    = node.resample_pdesc;

            this->fc_ops            = node.fc_ops;
            this->fc_attr           = node.fc_attr;
            this->fc_post_op        = node.fc_post_op;

            this->inner_pdesc       = node.inner_pdesc;
            this->op_args           = node.op_args;
            this->inference_forward = node.inference_forward;
            return *this;
        }
    };

    template<typename DType>
    struct strLayerNode{
        strNodeParam<DType> node_param;
        strLayerNode<DType> *next;
        strLayerNode<DType> *front;
        strLayerNode<DType>(const strNodeParam<DType>& param): node_param(param), next(nullptr), front(nullptr){
        }
    };
    
    template<typename DType>
    struct graphSetLink{
        dnnl::memory input;
        strLayerNode<DType> *head;
        strLayerNode<DType> *current;
        strLayerNode<DType> *tail;
        int current_index;
        int graph_size;

        graphSetLink(const int &size, const int &index, const dnnl::memory &temp_memory):
                head(nullptr), current(nullptr),current_index(index), graph_size(size) {
            input = temp_memory;
        }
        strLayerNode<DType> *operator[](const int &index){
            strLayerNode<DType> *temp = nullptr;
            if(index > (graph_size - 1) || index < 0){
                LOG(FATAL_ERROR)<<"the index: "<<index<<", graph_size: "<<graph_size;
                return nullptr;
            }else if(index == 0){
                return head;
            }else if(index == current_index){
                return current;
            }else if(index >= 0 && index <= current_index){
                if((index - 0) <= (current_index - index)){
                    temp        = head;
                    int count   = 0;
                    while(temp != nullptr){
                        if(count == index){
                            break;
                        }
                        temp    = temp->next;
                        count++;
                    }
                }else{
                    int count   = current_index;
                    temp        = current;
                    while(temp != nullptr){
                        if(count == index)
                            break;
                        temp = temp->front;
                        count--;
                    }
                }
            }else if((index > current_index) && (index <= (graph_size - 1))){
                int count   = 0;
                if(current == nullptr && current_index == 0){
                    count       = 0;
                    temp        = head;
                }else{
                    count       = current_index;
                    temp        = current;
                }
                while(temp != nullptr){
                    if(count == index)
                        break;
                    temp = temp->next;
                    count++;
                }
            }
            current_index   = index;
            current         = temp;
            return temp;
        }
    };

    inline void *xmalloc(size_t size) {
        void *ptr = malloc(size);
        if(!ptr) {
            malloc_error();
        }
        return ptr;
    }

    inline void *xcalloc(size_t nmemb, size_t size) {
        void *ptr = calloc(nmemb,size);
        if(!ptr) {
            calloc_error();
        }
        memset(ptr, 0, nmemb * size);
        return ptr;
    }

    // Read from memory, write to handle
    void read_from_dnnl_memory(void *handle, dnnl::memory &mem);
    void write_to_dnnl_memory(void *handle, dnnl::memory &mem);
    dnnl::algorithm get_op_mapped_pooling_type(PoolingType type);
    dnnl::algorithm get_op_mapped_activition_type(activitionType type);
    dnnl::algorithm get_op_mapped_binary_type(BinaryOpOperationType type);
    void checK_equal_dims(const dnnl::memory::dims &A_Shape, const dnnl::memory::dims &B_Shape);
    void check_inputs_shape(const std::vector<TensorShape> &inputs);
    template<typename DType> void graph_insert(graphSetLink<DType> &g_state, strLayerNode<DType> *node);
    std::string get_quantized_type(QUANITIZED_TYPE type);
    std::string get_binary_type(BinaryOpOperationType type);
    std::string get_mapped_op_string(OP_type type);
    template<typename DType> int get_net_index_by_name(const std::vector<strParam<DType> > &layer_ops, const std::string &node_name);
    void print_dnnl_memory_shape(const dnnl::memory::dims &shape, const string &shape_name);
    int product_dnnl_memory_shape(const dnnl::memory::dims &shape);
} // namespace BrixLab

#endif // #ifndef


