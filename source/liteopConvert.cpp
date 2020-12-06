#include "liteopConvert.hpp"

using namespace BrixLab;

liteOpConvertMapKit* liteOpConvertMapKit::_uniqueSuit = nullptr;

liteOpConverter* liteOpConvertMapKit::search(const tflite::BuiltinOperator opIndex) {
    auto iter = _liteOpConverters.find(opIndex);
    if (iter == _liteOpConverters.end()) {
        return nullptr;
    }
    return iter->second;
}

liteOpConvertMapKit* liteOpConvertMapKit::get() {
    if (_uniqueSuit == nullptr) {
        _uniqueSuit = new liteOpConvertMapKit;
    }
    return _uniqueSuit;
}

liteOpConvertMapKit::~liteOpConvertMapKit() {
    for (auto& it : _liteOpConverters) {
        delete it.second;
    }
    _liteOpConverters.clear();
}

void liteOpConvertMapKit::insert(liteOpConverter* t, tflite::BuiltinOperator opIndex) {
    _liteOpConverters.insert(std::make_pair(opIndex, t));
}

