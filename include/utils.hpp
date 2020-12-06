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
    enum DATA_FORMATE{
        NHWC,
        NHCW,
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
    enum EltwiseType{
        ElementSum,
        ElementProduct,
    };
    enum PoolingType{
        PoolingMAX,
        PoolingAVAGE,
    };
    template<typename DType>
    struct layerWeightsParam{
        std::string node_name;
        DATA_FORMATE formate;
        OP_type op_type;
        int inBatch, inChannel, inHeight, inWidth;
        // (de)convolution params
        int k_w;
        int k_h;
        int padding;
        int strides;
        int k_c;
        int dialited_rate;
        bool hasBias;
        int groups;
        DType *conv_weights;
        DType *conv_bias;
        //deconvolution layer
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
        int p_kw;
        int p_kh;
        int p_strides;
        int p_padding;
        PoolingType p_type;
        int p_diliated;

        //innerproducts params
        int inner_out;
        float alpha, beta;
        DType *inner_weights;
        bool has_inner_product_bias;
        DType *inner_bias;

        //concat params
        int *concat_index;
        int concat_num;
        int concat_axis;

        //Eltwise_SUM params
        int *sum_index;
        int sum_num;
        int layer_h;
        int layer_w;
        EltwiseType eleType;
        //resample layer
        float adjust_scale;
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
        int layer_h;
        int layer_w;
        int layer_c;
        int layer_n;
        OP_type op_type;            
        dnnl::memory::dims bottom_shape;
        dnnl::memory::dims top_shape;
        dnnl::memory::dims weights_shape;
        dnnl::memory::dims bias_shape;
        dnnl::memory layer_top_memory;
        dnnl::memory::desc layer_top_md;
        dnnl::memory src_bottom_memory;
        dnnl::memory::desc src_bottom_md;
        dnnl::memory src_weights_memory;
        dnnl::memory::desc src_weights_md;
        dnnl::memory src_bias_memory;
        dnnl::memory::desc src_bias_md;
        // convolution layer & param
        int groups;
        int dialited_rate;
        bool hasBias;
        dnnl::memory::dims conv_strides;
        dnnl::memory::dims conv_padding;

        dnnl::convolution_forward::primitive_desc convolution_pdesc;
        
        // batchnorm layer
        dnnl::memory::dims batchnorm_scale_shift_shape;
        dnnl::memory batchnorm_scale_shift_memory;
        dnnl::memory::desc batchnorm_scale_shift_md;
        dnnl::batch_normalization_forward::primitive_desc batchnorm_pdesc;
        dnnl::memory batchnorm_mean_memory;
        dnnl::memory batchnorm_variance_memory;

        //pooling layer
        int p_dialiated;
        dnnl::algorithm pooling_type;
        dnnl::memory::dims pooling_kernel;
        dnnl::memory::dims pooling_strides;
        dnnl::memory::dims pooling_padding;
        dnnl::memory::dims pooling_dialiate;
        // pooling with dialated

        dnnl::pooling_v2_forward::primitive_desc pooling_pdesc;
        // pooling without dialated
        
        dnnl::pooling_forward::primitive_desc pooling_pdesc_without_d;

        // concat layer
        int *concat_index;
        int concat_num;
        int concat_axis;
        std::vector<dnnl::memory::desc> concat_bottom_md;
        std::vector<dnnl::memory> concat_bottom_memory;
        dnnl::concat::primitive_desc concat_pdesc;
        // eltwise_sum layer
        int *sum_index;
        int sum_num;
        std::vector<DType> sum_scale;
        std::vector<dnnl::memory> sum_bottom_memory;
        std::vector<dnnl::memory::desc> sum_bottom_md;
        dnnl::sum::primitive_desc sum_pdesc;

        // activation layer
        dnnl::algorithm activate_type;
        dnnl::eltwise_forward::primitive_desc eltwise_pdesc;
        float alpha, beta;

        // resampleing layer
        float adjust_scale;
        dnnl::resampling_forward::primitive_desc resample_pdesc;

        // deconvolution(transposed convolution) layer & param
        int dedialited_rate;
        bool hasdeBias;
        dnnl::memory::dims deconv_strides;
        dnnl::memory::dims deconv_padding;
        dnnl::deconvolution_forward::primitive_desc deconv_pdesc;

        //inner-product layer
        dnnl::inner_product_forward::primitive_desc inner_pdesc;

        // common layer
        std::unordered_map<int, dnnl::memory> op_args;
        layerNode<DType> *next;
        layerNode<DType> *front;
        void (*inference_forward)(layerNode<DType> &, graphSet<DType>&);
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
        
        graphSet(const int &size, const int &index, const dnnl::memory &temp_memory): current_index(index), 
                                        graphSize(size), head(nullptr), current(nullptr){
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
    template<typename DType> void graph_insert(graphSet<DType> &g_state, layerNode<DType> *node);
} // namespace BrixLab

#endif // #ifndef


