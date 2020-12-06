#ifndef BRIXLAB_CHECK_ERROR_
#define BRIXLAB_CHECK_ERROR_
#include <assert.h>
#include "oneapi/dnnl.hpp"
namespace BrixLab{
    void checK_equal_dims(const dnnl::memory::dims &A_Shape, 
                                const dnnl::memory::dims &B_Shape);
    inline void malloc_error(){
        fprintf(stderr, "xMalloc error - possibly out of CPU RAM \n");
        exit(EXIT_FAILURE);
    }

    inline void calloc_error(){
        fprintf(stderr, "Calloc error - possibly out of CPU RAM \n");
        exit(EXIT_FAILURE);
    }
}//namespace BrixLab

#endif