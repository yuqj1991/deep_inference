#ifndef BRIXLAB_LITECONVERTX_HELPS_
#define BRIXLAB_LITECONVERTX_HELPS_
#include <map>
// tflite fbs header
#include "schema_generated.h"
#include "logkit.hpp"
#include "utils.hpp"
#include "utils_help.hpp"

namespace BrixLab{

    template<typename DType> class liteOpConvertMapKit;
    template<typename DType>
    class liteOpConverter {
    public:
        virtual void run(layerWeightsParam<DType> *dstOp, const std::unique_ptr<tflite::OperatorT>& tfliteOp,
                        const std::vector<std::unique_ptr<tflite::TensorT>>& tfliteTensors,
                        const std::vector<std::unique_ptr<tflite::BufferT>>& tfliteModelBuffer,
                        const std::vector<std::unique_ptr<tflite::OperatorCodeT>>& tfliteOpSet, bool quantizedModel) = 0;
        virtual BrixLab::OP_type opType(bool quantizedModel)                                                               = 0;
        liteOpConverter() {
        }
        virtual ~liteOpConverter() {
        }

        friend class liteOpConvertMapKit<DType>;
    };

    template<typename DType>
    class liteOpConvertMapKit {
    public:
        static liteOpConvertMapKit<DType>* get();
        liteOpConverter<DType>* search(const tflite::BuiltinOperator opIndex);
         void insert(liteOpConverter<DType>* t, const tflite::BuiltinOperator opIndex);
    private:
        liteOpConvertMapKit() 
        {
        }
        ~liteOpConvertMapKit();
        static liteOpConvertMapKit<DType>* _uniqueSuit;
        std::map<tflite::BuiltinOperator, liteOpConverter<DType>* > _liteOpConverters;
    };

    

    template <class T, typename DType>
    class liteOpConverterRegister {
    public:
        liteOpConverterRegister(const tflite::BuiltinOperator opIndex) {
            T* converter                  = new T;
            liteOpConvertMapKit<DType>* liteSuit = liteOpConvertMapKit<DType>::get();
            liteSuit->insert(converter, opIndex);
        }

        ~liteOpConverterRegister() {
        }
    };

    #define DECLARE_OP_COVERTER(name)                                                                                       \
        template<typename DType>                                                                                            \
        class name : public liteOpConverter<DType> {                                                                        \
        public:                                                                                                             \
            virtual void run(layerWeightsParam<DType> *dstOp, const std::unique_ptr<tflite::OperatorT>& tfliteOp,           \
                            const std::vector<std::unique_ptr<tflite::TensorT>>& tfliteTensors,                             \
                            const std::vector<std::unique_ptr<tflite::BufferT>>& tfliteModelBuffer,                         \
                            const std::vector<std::unique_ptr<tflite::OperatorCodeT>>& tfliteOpSet, bool quantizedModel);   \
            name() {                                                                                                        \
            }                                                                                                               \
            virtual ~name() {                                                                                               \
            }                                                                                                               \
            virtual BrixLab::OP_type opType(bool quantizedModel);                                                           \
        }

    #define REGISTER_CONVERTER(name, DType, opType) static liteOpConverterRegister<name, DType> _Convert##DType##opType(opType)

    #define INSTANEC_OP_CONVERTER(OP_classname) \
                template class OP_classname<float>; \
                template class OP_classname<uint8_t> 
    
    #define INSTANEC_FLOAT_OP_CONVERTER(OP_classname) \
                template class OP_classname<float>
    
    #define INSTANEC_UINT8_OP_CONVERTER(OP_classname) \
                template class OP_classname<uint8_t>


}// BrixLab


#endif