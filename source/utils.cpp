#include "check_error.hpp"
#include "unit_test.hpp"
#include "utils.hpp"
namespace BrixLab{
    void checK_equal_dims(const dnnl::memory::dims &A_Shape,  
                                const dnnl::memory::dims &B_Shape){
        unsigned int A_dims = A_Shape.size();
        unsigned int B_dims = B_Shape.size();
        LOG_CHECK(A_dims = B_dims)<<"CHECK DIMS equal";
        for(unsigned ii = 0; ii < A_dims; ii++){
            LOG_CHECK(A_Shape[ii] == B_Shape[ii])<<"CHECK DIMS equal";
        }
    }
    void check_inputs_shape(const std::vector<TensorShape> &inputs){
        TensorShape first = inputs[0];
        LOG_CHECK(inputs.size() > 1)<<"CHECK INPUTS > 1";
        for(unsigned int i = 1; i < inputs.size(); i++){
            LOG_CHECK(first == inputs[i])<<"CHECK EQUAL";
        }
    }
    dnnl::algorithm get_op_mapped_pooling_type(PoolingType type){
        dnnl::algorithm algo;
        switch (type)
        {
        case PoolingType::PoolingAVAGE:
            algo = dnnl::algorithm::pooling_avg;
            break;
        case PoolingType::PoolingMAX:
            algo = dnnl::algorithm::pooling_max;
            break;
        default:
            break;
        }
        return algo;
    }
    dnnl::algorithm get_op_mapped_activition_type(activitionType type){
        dnnl::algorithm algo;
        switch (type)
        {
            case activitionType::Tanh:
                algo = dnnl::algorithm::eltwise_tanh;
                break;
            case activitionType::ReLU:
                algo = dnnl::algorithm::eltwise_relu;
                break;
            case activitionType::LINEAR_SCALE_BIAS:
                algo = dnnl::algorithm::eltwise_linear;
                break;
            case activitionType::Sigmoid:
                algo = dnnl::algorithm::eltwise_logistic;
                break;
            case activitionType::SWISH:
                algo = dnnl::algorithm::eltwise_swish;
                break;
            default:
                break;
        }
        return algo;
    }

    template<typename DType>
    void graph_insert(graphSetLink<DType> &g_state, strLayerNode<DType> *node){
        if(!g_state.graph_size){
            g_state.head                = new strLayerNode<DType>(strNodeParam<DType>(node->node_param.op_type));
            g_state.head->node_param    = node->node_param;
            g_state.head->front         = nullptr;
            g_state.tail                = g_state.head;
            g_state.graph_size++;
        }else{
            g_state.tail->next              = new strLayerNode<DType>(strNodeParam<DType>(node->node_param.op_type));
            g_state.tail->next->node_param  = node->node_param;
            g_state.tail->next->front       = g_state.tail;
            g_state.tail                    = g_state.tail->next;
            g_state.graph_size++;
        }
    }
    template void graph_insert(graphSetLink<float> &g_state, strLayerNode<float> *node);

    dnnl::algorithm get_op_mapped_binary_type(BinaryOpOperationType type){
        dnnl::algorithm algo;
        switch (type)
        {
            case BinaryOpOperation_ADD:
                algo = dnnl::algorithm::binary_add;
                break;
            case BinaryOpOperation_SUB:
                algo = dnnl::algorithm::binary_sub;
                break;
            case BinaryOpOperation_MUL:
                algo = dnnl::algorithm::binary_mul;
                break;
            case BinaryOpOperation_DIV:
                algo = dnnl::algorithm::binary_div;
                break;
            case BinaryOpOperation_MAXIMUM:
                algo = dnnl::algorithm::binary_max;
                break;
            case BinaryOpOperation_MINIMUM:
                algo = dnnl::algorithm::binary_min;
                break;
            default:
                LOG(FATAL_ERROR)<<"not support binary op"<<get_binary_type(type);
                break;
        }
        return algo;
    }
    std::string get_quantized_type(QUANITIZED_TYPE type){
        std::string quan_name;
        if(type == QUANITIZED_TYPE::FLOAT32_REGULAR){
            quan_name   = "FLOAT32_REGULAR";
        }else{
            quan_name   = "UINT8_QUANTIZED";
        }
        return quan_name;
    }

    std::string get_binary_type(BinaryOpOperationType type){
        std::string algo;
        switch (type)
        {
            case BinaryOpOperation_ADD:
                algo = "dnnl::algorithm::binary_add";
                break;
            case BinaryOpOperation_SUB:
                algo = "dnnl::algorithm::binary_sub";
                break;
            case BinaryOpOperation_MUL:
                algo = "dnnl::algorithm::binary_mul";
                break;
            case BinaryOpOperation_DIV:
                algo = "dnnl::algorithm::binary_div";
                break;
            case BinaryOpOperation_MAXIMUM:
                algo = "dnnl::algorithm::binary_max";
                break;
            case BinaryOpOperation_MINIMUM:
                algo = "dnnl::algorithm::binary_min";
                break;
            default:
                LOG(FATAL_ERROR)<<"not support binary op: "<<type;
                break;
        }
        return algo;
    }

    void read_from_dnnl_memory(void *handle, dnnl::memory &mem) {
        dnnl::engine eng    = mem.get_engine();
         size_t size         = mem.get_desc().get_size();
        LOG(DEBUG_INFO)<<"size: "<<size;

        #if DNNL_WITH_SYCL
        bool is_cpu_sycl = (DNNL_CPU_RUNTIME == DNNL_RUNTIME_SYCL
                && eng.get_kind() == dnnl::engine::kind::cpu);
        bool is_gpu_sycl = (DNNL_GPU_RUNTIME == DNNL_RUNTIME_SYCL
                && eng.get_kind() == dnnl::engine::kind::gpu);
        if (is_cpu_sycl || is_gpu_sycl) {
            auto mkind = dnnl::sycl_interop::get_memory_kind(mem);
            if (mkind == dnnl::sycl_interop::memory_kind::buffer) {
                auto buffer     = dnnl::sycl_interop::get_buffer<uint8_t>(mem);
                auto src        = buffer.get_access<cl::sycl::access::mode::read>();
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
                    auto sycl_queue = dnnl::sycl_interop::get_queue(dnnl::stream(eng));
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
    void write_to_dnnl_memory(void *handle, dnnl::memory &mem) {
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

    std::string get_mapped_op_string(OP_type type){
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
            case OP_type::SPACE_PERMUTES:{
                op_name = std::string("OP_spaceTransposed");
                break;
            }
            default:
                LOG(FATAL_ERROR)<<"NO SUPPORT OPS: "<< type;
                break;
        }
        return op_name;
    }

    void print_dnnl_memory_shape(const dnnl::memory::dims &shape, const string &shape_name){
        #ifdef USE_DEBUG
        printf("%s: ", shape_name.c_str());
        for(unsigned int i = 0; i < shape.size(); i++){
            if(i < shape.size() - 1)
                printf("%ld,", shape[i]);
            else
            {
                printf("%ld", shape[i]);
            }
            
        }
        printf("\n");
        #endif
    }

    template<typename DType> 
    int get_net_index_by_name(const std::vector<strParam<DType> > &layer_ops, const std::string &node_name){
        int count   = -1;
        for(unsigned int i = 0; i < layer_ops.size(); i++){
            strParam<DType> New_OP = layer_ops[i];
            if(New_OP.node_name ==  node_name){
                count   = i;
                break;
            }
        }
        return count;
    }
    template int get_net_index_by_name(const std::vector<strParam<float> > &layer_ops, const std::string &node_name);
    int product_dnnl_memory_shape(const dnnl::memory::dims &shape){
        int size    = shape[0];
        for(unsigned int i = 1; i < shape.size(); i++)
            size    *= shape[i];
        return size;
    }

    template<typename DType>
    void PermuteMemory(const DType* src, DType* dst, const dnnl::memory::dims& src_dims, 
                                                    const dnnl::memory::dims& dst_dims,
                                                    const int num_axes,
                                                    const int* permute_order,
                                                    const int* old_steps,
                                                    const int* new_steps){
        const int src_size                  = product_dnnl_memory_shape(src_dims);
        const int dst_size                  = product_dnnl_memory_shape(dst_dims);
        const int src_dims_size             = src_dims.size();
        const int dst_dims_size             = dst_dims.size();
        LOG_CHECK(src_size == dst_size)<<"CHECK MEMORY EQUAL ERROR";
        LOG_CHECK(src_dims_size == dst_dims_size)<<"CHECK MEMORY EQUAL INPUTS ERROR";
        for(int i = 0; i < src_size; i++){
            int old_idx     = 0;
            int idx         = i;
            for (int j = 0; j < num_axes; ++j) {
                int order   = permute_order[j];
                old_idx     += (idx / new_steps[j]) * old_steps[order];
                idx         %= new_steps[j];
            }
            dst[i]          = src[old_idx];
        }
        
    }
    template void PermuteMemory(const float* src, float* dst, const dnnl::memory::dims& src_dims, 
                                                            const dnnl::memory::dims& dst_dims,
                                                            const int num_axes,
                                                            const int* permute_order,
                                                            const int* old_steps,
                                                            const int* new_steps);
    template void PermuteMemory(const unsigned char* src, unsigned char* dst, const dnnl::memory::dims& src_dims, 
                                                            const dnnl::memory::dims& dst_dims,
                                                            const int num_axes,
                                                            const int* permute_order,
                                                            const int* old_steps,
                                                            const int* new_steps);

    int dims_count(const dnnl::memory::dims& shape, const int& start, const int& end){
        int counts      = 1;
        const int size  = shape.size() - 1;
        LOG_CHECK((start <= end) &&(start >= 0) &&(end <= size))<<"CHECK INDEX ERROR";
        for(int i = start; i <= end; i++){
            counts  *= shape[i];
        }
        return counts;
    }

    template<typename DType> void CropMemory(const DType* src, DType* dst, const dnnl::memory::dims& src_dims,
                                                                            const dnnl::memory::dims& dst_dims,
                                                                            const int& SH,
                                                                            const int& EH,
                                                                            const int& SW,
                                                                            const int& EW){
        const int inB   = src_dims[0];
        const int inC   = src_dims[1];
        const int inH   = src_dims[2];
        const int inW   = src_dims[3];
        const int outB  = dst_dims[0];
        const int outC  = dst_dims[1];
        const int outH  = dst_dims[2];
        const int outW  = dst_dims[3];
        LOG_CHECK(inC == outC) << "CHECK CHANNEL ERROR";
        LOG_CHECK(inB == outB) << "CHECK BATCH ERROR";
        for(int b = 0; b < inB; b++){
            for(int c = 0; c< inC; c++){
                for(int h = SH; h < EH; h++){
                    for(int w = SW; w < EW; w++){
                        int idx     = b * inC * inH * inW + c * inH * inW + h * inW + w;
                        int odx     = b * outC * outH * outW + c * outH * outW + (h - SH) * outW + (w-SW);
                        dst[odx]    = src[idx];
                    }
                }
            }
        }
    }
    template void CropMemory(const float* src, float* dst, const dnnl::memory::dims& src_dims,
                                                                            const dnnl::memory::dims& dst_dims,
                                                                            const int& SH,
                                                                            const int& EH,
                                                                            const int& SW,
                                                                            const int& EW);
    template void CropMemory(const unsigned char* src, unsigned char* dst, const dnnl::memory::dims& src_dims,
                                                                            const dnnl::memory::dims& dst_dims,
                                                                            const int& SH,
                                                                            const int& EH,
                                                                            const int& SW,
                                                                            const int& EW);
    template<typename DType>
    void fillMemory(const DType* src, DType* dst, const dnnl::memory::dims& src_dims,
                                                                            const dnnl::memory::dims& dst_dims,
                                                                            const int& SH,
                                                                            const int& EH,
                                                                            const int& SW,
                                                                            const int& EW){
        const int inB   = src_dims[0];
        const int inC   = src_dims[1];
        const int inH   = src_dims[2];
        const int inW   = src_dims[3];
        const int outB  = dst_dims[0];
        const int outC  = dst_dims[1];
        const int outH  = dst_dims[2];
        const int outW  = dst_dims[3];
        LOG_CHECK(inC == outC) << "CHECK CHANNEL ERROR";
        LOG_CHECK(inB == outB) << "CHECK BATCH ERROR";
        for(int b = 0; b < inB; b++){
            for(int c = 0; c< inC; c++){
                for(int h = SH; h < EH; h++){
                    for(int w = SW; w < EW; w++){
                        int idx     = b * inC * inH * inW + c * inH * inW + (h - SH) * inW + (w-SW);
                        int odx     = b * outC * outH * outW + c * outH * outW + h * outW + w;
                        dst[odx]    = src[idx];
                    }
                }
            }
        }
    }
    template void fillMemory(const float* src, float* dst, const dnnl::memory::dims& src_dims,
                                                                            const dnnl::memory::dims& dst_dims,
                                                                            const int& SH,
                                                                            const int& EH,
                                                                            const int& SW,
                                                                            const int& EW);
    template void fillMemory(const unsigned char* src, unsigned char* dst, const dnnl::memory::dims& src_dims,
                                                                            const dnnl::memory::dims& dst_dims,
                                                                            const int& SH,
                                                                            const int& EH,
                                                                            const int& SW,
                                                                            const int& EW);
    template<typename DType>
    void ReshapeExpandChannelMemory(const DType* src, DType* dst, const dnnl::memory::dims& dim_shape){
        const int inB   = dim_shape[0];
        const int inC   = dim_shape[1];
        const int inH   = dim_shape[2];
        const int inW   = dim_shape[3];
        for(int b = 0; b < inB; b++){
            for(int c = 0; c< inC; c++){
                for(int h = 0; h < inH; h++){
                    for(int w = 0; w < inW; w++){
                        int idx     = b * inC * inH * inW + c * inH * inW + h * inW + w;
                        dst[idx]    = src[c];
                    }
                }
            }
        }
    }
    template void ReshapeExpandChannelMemory(const float* src, float* dst, const dnnl::memory::dims& dim_shape);
    template void ReshapeExpandChannelMemory(const unsigned char* src, unsigned char* dst, const dnnl::memory::dims& dim_shape);
}//BrixLab