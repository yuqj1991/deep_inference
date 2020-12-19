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
/**
 * 从tflite文件，读取模型结构，还有权重参数， 
 * 重点是对tflite模型，解析获得节点数据，并且转化成以layerNode为节点的graph链表。
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
        NHCW,
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
                Height == right.Channel && Width == right.Width){
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
    struct layerWeightsParam{
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
        std::vector<layerWeightsParam<DType> >layer_ops;
        std::vector<std::string> tensorName;
        std::vector<std::string> output_name;
        int32_t tensorNumber;
    };

    template<typename DType>
    struct graphSet;

    template<typename DType>
    struct layerNode{
        // layer info
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
        layerNode<DType> *next;
        layerNode<DType> *front;
        void (*inference_forward)(layerNode<DType> *, graphSet<DType>&);
        layerNode<DType>(OP_type type):op_type(type){}
    };

    template<typename DType>
    struct graphSet{
        dnnl::memory input;
        layerNode<DType> *head;
        layerNode<DType> *current;
        layerNode<DType> *tail;
        int current_index;
        int graphSize;
        
        graphSet(const int &size, const int &index, const dnnl::memory &temp_memory):
                head(nullptr), current(nullptr),current_index(index), graphSize(size) {
            input = temp_memory;
        }
        layerNode<DType> *operator[](const int &index){
            layerNode<DType> *temp = nullptr;
            if(index >= graphSize || index < 0){
                printf("the index %d is override the graphSize!", index, graphSize);
                return nullptr;
            }else if(index == 0){
                return head;
            }else if(index == current_index){
                return current;
            }else if(index >= 0 && index <= current_index){
                if((index - 0) <= (current_index - index)){
                    temp = head;
                    int count = 0;
                    while(temp != nullptr){
                        if(count == index){
                            break;
                        }
                        temp = temp->next;
                        count++;
                    }
                }else{
                    int count = current_index;
                    temp = current;
                    while(temp != nullptr){
                        if(count == index)
                            break;
                        temp = temp->front;
                        count--;
                    }
                }
            }else if((index > current_index) && (index < (graphSize - 1))){
                int count = current_index;
                temp = current;
                while(temp != nullptr){
                        if(count == index)
                            break;
                        temp = temp->next;
                        count++;
                }
            }
            return temp;
        }
    };

    inline std::string get_mapped_op_string(OP_type type){
        std::string op_name;
        switch(type){
            case OP_type::CONVOLUTION:{
                op_name = std::string("OP_convolution");
                break;
            }
            case OP_type::DECONVOLUTION:{
                op_name = std::string("OP_deconvolution");
                break;
            }
            case OP_type::BATCHNORM:{
                op_name = std::string("OP_batchnorm");
                break;
            }
            case OP_type::ELTWISE:{
                op_name = std::string("OP_eltwise");
                break;
            }
            case OP_type::POOLING:{
                op_name = std::string("OP_pooling");
                break;
            }
            case OP_type::DATA_INPUTS:{
                op_name = std::string("OP_inputs");
                break;
            }
            case OP_type::DETECTION:{
                op_name = std::string("OP_detection");
                break;
            }
            case OP_type::CONCAT:{
                op_name = std::string("OP_concat");
                break;
            }
            case OP_type::RESAMPLING:{
                op_name = std::string("OP_resample");
                break;
            }
            case OP_type::ACTIVITION:{
                op_name = std::string("OP_activition");
                break;
            }
            case OP_type::INNERPRODUCT:{
                op_name = std::string("OP_innerproduct");
                break;
            }
            case OP_type::SOFTMAX:{
                op_name = std::string("OP_softmax");
                break;
            }
            case OP_type::REDUCTION:{
                op_name = std::string("OP_reduction");
                break;
            }
            case OP_type::BINARY_OP:{
                op_name = std::string("OP_binary");
                break;
            }
        }
        return op_name;
    }
    inline void *xmalloc(size_t size) {
        void *ptr=malloc(size);
        if(!ptr) {
            malloc_error();
        }
        return ptr;
    }

    inline void *xcalloc(size_t nmemb, size_t size) {
        void *ptr=calloc(nmemb,size);
        if(!ptr) {
            calloc_error();
        }
        memset(ptr, 0, nmemb * size);
        return ptr;
    }

    // Read from memory, write to handle
    inline void read_from_dnnl_memory(void *handle, dnnl::memory &mem) {
        dnnl::engine eng = mem.get_engine();
        size_t size = mem.get_desc().get_size();

    #if DNNL_WITH_SYCL
        bool is_cpu_sycl = (DNNL_CPU_RUNTIME == DNNL_RUNTIME_SYCL
                && eng.get_kind() == dnnl::engine::kind::cpu);
        bool is_gpu_sycl = (DNNL_GPU_RUNTIME == DNNL_RUNTIME_SYCL
                && eng.get_kind() == dnnl::engine::kind::gpu);
        if (is_cpu_sycl || is_gpu_sycl) {
            auto mkind = dnnl::sycl_interop::get_memory_kind(mem);
            if (mkind == dnnl::sycl_interop::memory_kind::buffer) {
                auto buffer = dnnl::sycl_interop::get_buffer<uint8_t>(mem);
                auto src = buffer.get_access<cl::sycl::access::mode::read>();
                uint8_t *src_ptr = src.get_pointer();
                for (size_t i = 0; i < size; ++i)
                    ((uint8_t *)handle)[i] = src_ptr[i];
            } else {
                assert(mkind == dnnl::sycl_interop::memory_kind::usm);
                uint8_t *src_ptr = (uint8_t *)mem.get_data_handle();
                if (is_cpu_sycl) {
                    for (size_t i = 0; i < size; ++i)
                        ((uint8_t *)handle)[i] = src_ptr[i];
                } else {
                    auto sycl_queue
                            = dnnl::sycl_interop::get_queue(dnnl::stream(eng));
                    sycl_queue.memcpy(src_ptr, handle, size).wait();
                }
            }
            return;
        }
    #endif
    #if DNNL_GPU_RUNTIME == DNNL_RUNTIME_OCL
        if (eng.get_kind() == dnnl::engine::kind::gpu) {
            dnnl::stream s(eng);
            cl_command_queue q = dnnl::ocl_interop::get_command_queue(s);
            cl_mem m = dnnl::ocl_interop::get_mem_object(mem);

            cl_int ret = clEnqueueReadBuffer(
                    q, m, CL_TRUE, 0, size, handle, 0, NULL, NULL);
            if (ret != CL_SUCCESS)
                throw std::runtime_error("clEnqueueReadBuffer failed.");
            return;
        }
    #endif

        if (eng.get_kind() == dnnl::engine::kind::cpu) {
            uint8_t *src = static_cast<uint8_t *>(mem.get_data_handle());
            for (size_t i = 0; i < size; ++i)
                ((uint8_t *)handle)[i] = src[i];
            return;
        }

        assert(!"not expected");
    }

    // Read from handle, write to memory
    inline void write_to_dnnl_memory(void *handle, dnnl::memory &mem) {
        dnnl::engine eng = mem.get_engine();
        size_t size = mem.get_desc().get_size();
    #if DNNL_WITH_SYCL
        bool is_cpu_sycl = (DNNL_CPU_RUNTIME == DNNL_RUNTIME_SYCL
                && eng.get_kind() == dnnl::engine::kind::cpu);
        bool is_gpu_sycl = (DNNL_GPU_RUNTIME == DNNL_RUNTIME_SYCL
                && eng.get_kind() == dnnl::engine::kind::gpu);
        if (is_cpu_sycl || is_gpu_sycl) {
            auto mkind = dnnl::sycl_interop::get_memory_kind(mem);
            if (mkind == dnnl::sycl_interop::memory_kind::buffer) {
                auto buffer = dnnl::sycl_interop::get_buffer<uint8_t>(mem);
                auto dst = buffer.get_access<cl::sycl::access::mode::write>();
                uint8_t *dst_ptr = dst.get_pointer();
                for (size_t i = 0; i < size; ++i)
                    dst_ptr[i] = ((uint8_t *)handle)[i];
            } else {
                assert(mkind == dnnl::sycl_interop::memory_kind::usm);
                uint8_t *dst_ptr = (uint8_t *)mem.get_data_handle();
                if (is_cpu_sycl) {
                    for (size_t i = 0; i < size; ++i)
                        dst_ptr[i] = ((uint8_t *)handle)[i];
                } else {
                    auto sycl_queue
                            = dnnl::sycl_interop::get_queue(dnnl::stream(eng));
                    sycl_queue.memcpy(dst_ptr, handle, size).wait();
                }
            }
            return;
        }
    #endif
    #if DNNL_GPU_RUNTIME == DNNL_RUNTIME_OCL
        if (eng.get_kind() == dnnl::engine::kind::gpu) {
            dnnl::stream s(eng);
            cl_command_queue q = dnnl::ocl_interop::get_command_queue(s);
            cl_mem m = dnnl::ocl_interop::get_mem_object(mem);

            cl_int ret = clEnqueueWriteBuffer(
                    q, m, CL_TRUE, 0, size, handle, 0, NULL, NULL);
            if (ret != CL_SUCCESS)
                throw std::runtime_error("clEnqueueWriteBuffer failed.");
            return;
        }
    #endif

        if (eng.get_kind() == dnnl::engine::kind::cpu) {
            uint8_t *dst = static_cast<uint8_t *>(mem.get_data_handle());
            for (size_t i = 0; i < size; ++i)
                dst[i] = ((uint8_t *)handle)[i];
            return;
        }

        assert(!"not expected");
    }
    dnnl::algorithm get_op_mapped_pooling_type(PoolingType type);
    dnnl::algorithm get_op_mapped_activition_type(activitionType type);
    dnnl::algorithm get_op_mapped_binary_type(BinaryOpOperationType type);
    void checK_equal_dims(const dnnl::memory::dims &A_Shape, const dnnl::memory::dims &B_Shape);
    void check_inputs_shape(const std::vector<TensorShape> &inputs);
    template<typename DType> void graph_insert(graphSet<DType> &g_state, layerNode<DType> *node);
} // namespace BrixLab

#endif // #ifndef


