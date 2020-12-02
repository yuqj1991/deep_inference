#include "layer_builder.hpp"

namespace BrixLab
{
    template<typename DType>
    NetGraph<DType>::NetGraph(const int &inH, const int &inW, const int &size){
        input_w = inW;
        input_h = inH;
        graph_size = size;
    }

    template<typename DType>
    int NetGraph<DType>::get_Graphsize() const{
        return graph_size;
    }

    template<typename DType>
    int NetGraph<DType>::get_GraphinWidth() const{
        return input_w;
    }

    template<typename DType>
    int NetGraph<DType>::get_GraphinHeight() const{
        return input_h;
    }

    template<typename DType>
    void NetGraph<DType>::network_predict(){
        int size = graph_state.graphSize;
        int layer_count = 0;
        layerNode<DType> *layer_node = graph_state.head;
        while(layer_node != nullptr){
            OP_type type= layer_node->op_type;
            std::string OP_name = get_mapped_op_string(type);
            layer_node->inference_forward((*layer_node), graph_state);
            graph_state.input = layer_node->feature;
            layer_node = layer_node->next;
            layer_count++;
        }
    }

    template<typename DType>
    layerNode<DType>* NetGraph<DType>::getGraphOutput(){
        if(graph_state.current_index <= graph_size){
            
        }
        return graph_state.current;
    }

    template<typename DType>
    void NetGraph<DType>::make_graph(const std::string modelFile){

    }

} // namespace BrixLab

