#ifndef POOLING2D_H
#define POOLING2D_H

#include "../session.h"
/*
 * Pooling layer for 2D inputs.
 * Here support pooling as follow:
 * CUDNN_POOLING_MAX
 * CUDNN_POOLING_AVERAGE_COUNT_INCLUDE_PADDING
 * CUDNN_POOLING_AVERAGE_COUNT_EXCLUDE_PADDING
 */

class Pooling2d : public IOperator
{
public:
    Pooling2d(int window, int stride, Padding_t pad = valid, cudnnPoolingMode_t mode = CUDNN_POOLING_MAX);
    ~Pooling2d();
    void AddInput(ITensor *);
    ITensor *LayerInit();
    void Forward(bool);
    float *Backward(float *grads_down, bool del);

private:
    int padA_[2];
    int strideA_[2];
    int windowDimA_[2];
    int nbDims_;
    float alpha;
    float beta;
    float *grads_input_;
    cudnnPoolingDescriptor_t desc_;
    Tensor4d *p_input_;
    Tensor4d *p_output_;
    Padding_t padding_mode_;
    cudnnPoolingMode_t mode_;
}; 

#endif
