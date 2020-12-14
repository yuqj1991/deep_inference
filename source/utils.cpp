#include "check_error.hpp"
#include "unit_test.hpp"
#include "utils.hpp"
#include <assert.h>
namespace BrixLab{
    void checK_equal_dims(const dnnl::memory::dims &A_Shape,  
                                const dnnl::memory::dims &B_Shape){
        unsigned int A_dims = A_Shape.size();
        unsigned int B_dims = B_Shape.size();
        assert(A_dims = B_dims);
        for(unsigned ii = 0; ii < A_dims; ii++){
            assert(A_Shape[ii] == B_Shape[ii]);
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

    
}//BrixLab