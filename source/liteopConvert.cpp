#include "liteopConvert.hpp"

using namespace BrixLab;

template<typename DType>
liteOpConvertMapKit<DType>* liteOpConvertMapKit<DType>::_uniqueSuit = nullptr;

template<typename DType>
liteOpConverter<DType>* liteOpConvertMapKit<DType>::search(const tflite::BuiltinOperator opIndex) {
    auto iter = _liteOpConverters.find(opIndex);
    if (iter == _liteOpConverters.end()) {
        return nullptr;
    }
    return iter->second;
}

template<typename DType>
liteOpConvertMapKit<DType>* liteOpConvertMapKit<DType>::get() {
    if (_uniqueSuit == nullptr) {
        _uniqueSuit = new liteOpConvertMapKit;
    }
    return _uniqueSuit;
}

template<typename DType>
liteOpConvertMapKit<DType>::~liteOpConvertMapKit() {
    for (auto& it : _liteOpConverters) {
        delete it.second;
    }
    _liteOpConverters.clear();
}

template<typename DType>
void liteOpConvertMapKit<DType>::insert(liteOpConverter<DType>* t, tflite::BuiltinOperator opIndex) {
    _liteOpConverters.insert(std::make_pair(opIndex, t));
}

template class liteOpConvertMapKit<float>;
template class liteOpConvertMapKit<uint8_t>;

