#ifndef CONV2D_H
#define CONV2D_H

#include "../wheel.h"
#include "../tensor/tensor4d.h"
#include "../tensor/filter4d.h"
#include "ioperator.h"
#include "../session.h"

class Conv2d : public IOperator
{
public:
    Conv2d(int k, int s, int i, Padding_t mode = valid);
    //Conv2d(int k, int s, int t);
    ~Conv2d();
    ITensor* add_input(ITensor* input, bool del);
    void Forward(bool del);
    void Backward(cudnnTensorDescriptor_t a, cudnnTensorDescriptor_t b, float *c, bool d);
    void set_input_shape(int n, int c, int h, int w);
    void set_weights(float data);
private:
    int K_; // output channel
    int S_; // kernel height
    int T_; // kernel width
    int padA_[2];
    int filterStrideA_[2];
    int dilationA_[2];
    float alpha;
    float beta;
    float *grads_filter_;
    float *grads_data_;
    size_t size_in_bytes;
    Padding_t padding_mode_;
    cudnnConvolutionDescriptor_t desc_;
    cudnnConvolutionFwdAlgo_t algo_;
    Tensor4d *p_input_;
    Tensor4d *p_output_;
    Filter4d *p_filter_;
};

#endif
