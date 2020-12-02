#ifndef BRIXLAB_CHECK_ERROR_
#define BRIXLAB_CHECK_ERROR_
#include <assert.h>
#include "oneapi/dnnl/dnnl.hpp"
namespace BrixLab{
    void checK_equal_dims(dnnl::memory::dims A_Shape, 
                                dnnl::memory::dims B_Shape){
        int A_dims = A_Shape.size();
        int B_dims = B_Shape.size();
        assert(A_dims = B_dims);
        for(unsigned ii = 0; ii < A_dims; ii++){
            assert(A_Shape[ii] == B_Shape[ii]);
        }
    }
    inline void malloc_error()
    {
        fprintf(stderr, "xMalloc error - possibly out of CPU RAM \n");
        exit(EXIT_FAILURE);
    }

    inline void calloc_error()
    {
        fprintf(stderr, "Calloc error - possibly out of CPU RAM \n");
        exit(EXIT_FAILURE);
    }
}//namespace BrixLab

#endif