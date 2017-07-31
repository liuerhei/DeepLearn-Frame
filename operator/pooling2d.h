#ifndef POOLING2D_H
#define POOLING2D_H

#include "../session.h"
/*
 * Pooling layer for 3D inputs.
 * Here support pooling as follow:
 * CUDNN_POOLING_MAX
 * CUDNN_POOLING_AVERAGE_COUNT_INCLUDE_PADDING
 * CUDNN_POOLING_AVERAGE_COUNT_EXCLUDE_PADDING
 */

class Pooling2d : public IOperator
{
public:
    Pooling2d(int window, int stride, Padding_t pad = valid,
              cudnnPoolingMode_t mode = CUDNN_POOLING_MAX);
    ~Pooling2d();
    ITensor *add_input(ITensor *input, bool del);
    //Tensor4d *add_input(ITensor *input, bool del);
    void Forward(bool del);
    void set_input_shape(int C, int H, int W);
private:
    int padA_[2];
    int strideA_[2];
    int windowDimA_[2];
    int nbDims_;
    float alpha;
    float beta;
    cudnnPoolingDescriptor_t desc_;
    ITensor *p_input_;
    ITensor *p_output_;
    Padding_t padding_mode_;
    cudnnPoolingMode_t mode_;
}; 

#endif
