#ifndef BRIXLAB_LITECONVERTX_HELPS_
#define BRIXLAB_LITECONVERTX_HELPS_
#include <map>
// tflite fbs header
#include "schema_generated.h"
#include "logkit.hpp"
#include "utils.hpp"

namespace BrixLab{
    class liteOpConverter {
    public:
        virtual void run(layerWeightsParam<float> *dstOp, const std::unique_ptr<tflite::OperatorT>& tfliteOp,
                        const std::vector<std::unique_ptr<tflite::TensorT>>& tfliteTensors,
                        const std::vector<std::unique_ptr<tflite::BufferT>>& tfliteModelBuffer,
                        const std::vector<std::unique_ptr<tflite::OperatorCodeT>>& tfliteOpSet, bool quantizedModel) = 0;
        virtual BrixLab::OP_type opType(bool quantizedModel)                                                               = 0;
        liteOpConverter() {
        }
        virtual ~liteOpConverter() {
        }

        friend class liteOpConvertMapKit;
    };

    class liteOpConvertMapKit {
    public:
        static liteOpConvertMapKit* get();
        liteOpConverter* search(const tflite::BuiltinOperator opIndex);

    private:
        void insert(liteOpConverter* t, const tflite::BuiltinOperator opIndex);
        liteOpConvertMapKit() 
        {
        }
        ~liteOpConvertMapKit();
        static liteOpConvertMapKit* _uniqueSuit;
        std::map<tflite::BuiltinOperator, liteOpConverter*> _liteOpConverters;
    };

    template <class T>
    class liteOpConverterRegister {
    public:
        liteOpConverterRegister(const tflite::BuiltinOperator opIndex) {
            T* converter                  = new T;
            liteOpConvertMapKit* liteSuit = liteOpConvertMapKit::get();
            liteSuit->insert(converter, opIndex);
        }

        ~liteOpConverterRegister() {
        }
    };

    #define DECLARE_OP_COVERTER(name)                                                                                      \
        class name : public liteOpConverter {                                                                              \
        public:                                                                                                            \
            virtual void run(layerWeightsParam<float> *dstOp, const std::unique_ptr<tflite::OperatorT>& tfliteOp,                          \
                            const std::vector<std::unique_ptr<tflite::TensorT>>& tfliteTensors,                           \
                            const std::vector<std::unique_ptr<tflite::BufferT>>& tfliteModelBuffer,                       \
                            const std::vector<std::unique_ptr<tflite::OperatorCodeT>>& tfliteOpSet, bool quantizedModel); \
            name() {                                                                                                       \
            }                                                                                                              \
            virtual ~name() {                                                                                              \
            }                                                                                                              \
            virtual BrixLab::OP_type opType(bool quantizedModel);                                                          \
        }

    #define REGISTER_CONVERTER(name, opType) static liteOpConverterRegister<name> _Convert##opType(opType)


}// BrixLab


#endif