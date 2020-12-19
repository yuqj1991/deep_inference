#include "check_error.hpp"
#include "unit_test.hpp"
#include "utils.hpp"
namespace BrixLab{
    void checK_equal_dims(const dnnl::memory::dims &A_Shape,  
                                const dnnl::memory::dims &B_Shape){
        unsigned int A_dims = A_Shape.size();
        unsigned int B_dims = B_Shape.size();
        LOG_CHECK(A_dims = B_dims, "CHECK DIMS");
        for(unsigned ii = 0; ii < A_dims; ii++){
            LOG_CHECK(A_Shape[ii] == B_Shape[ii], "CHECK DIMS");
        }
    }
    void check_inputs_shape(const std::vector<TensorShape> &inputs){
        TensorShape first = inputs[0];
        LOG_CHECK(inputs.size(0) > 1, "CHECK INPUTS");
        for(unsigned int i = 1; i < inputs.size(); i++){
            LOG_CHECK(first == inputs[i], "CHECK EQUAL");
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
    void graph_insert(graphSet<DType> &g_state, layerNode<DType> *node){
        OP_type op_type = node->op_type;
        if(!g_state.graphSize){
            g_state.head = node;
            g_state.head->front = nullptr;
            g_state.tail = g_state.head;
            g_state.graphSize++;
        }else{
            g_state.tail->next = node;
            g_state.tail->next->front = g_state.tail;
            g_state.tail = g_state.tail->next;
            g_state.graphSize++;
        }
    }
    template void graph_insert(graphSet<float> &g_state, layerNode<float> *node);

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
                LOG(FATAL_ERROR, "not support binary op");
                break;
        }
        return algo;
    }

    
}//BrixLab