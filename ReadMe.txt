2020.12.29更新: 程序可以跑通，但是跑出来的结果和tflite跑出来的结果，相差挺大的。推理时间92ms，cpu_runtime:openMP。使用方法：./demo_inference
结果相差比较大的原因：1）tflite使用的是data order顺序是nhwc，但是我在onednn里面已经做了reorder的操作（包括权重tflite中是ohwi->onednn中的oihw）。
但是计算的结果还是不正确。2）中间层是使用了inPlace？

2020.12.17更新： 增加in_shapes & out_shapes, 方便查验tflite和onednn的shape计算是否一样
2020.12.17更新： 理清tflite中涉及padding和dilated rate参数的卷积方式，计算输出shape，修改onednn端卷积代码
2020.12.16更新： 修改onednn端的一些bug，下步开始测试
2020.12.14更新： 完成tflite转换代码
2020.12.13更新：目前完成了卷积(常规卷积、转置卷积、可分离卷积、全链接)、池化的转换操作代码，还剩下add、concat、softmax、resize、reduction、sum等操作未完成。
后面继续写代码，之后测试每个操作子的代码
