#ifndef CONV2D_H
#define CONV2D_H

#include "ioperator.h"
#include "../wheel.h"
#include "../tensor/tensor4d.h"
#include "../tensor/filter4d.h"
#include "../session.h"

class Conv2d : public IOperator
{
public:
    Conv2d(int k, int s, int i, Padding_t mode = valid);
    ~Conv2d();
    void AddInput(ITensor* input);
    ITensor *LayerInit();
    void Forward(bool del = false);
    float *Backward(float *c, bool del = false);
    void UpdateWeights();
    void ToFile(const char *fileprefix);

    /*
     * This function is used to init the filter weights by reading data from files
     */
    void SetWeights(float data);
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
    float *grads_bias_;
    size_t size_in_bytes;
    Padding_t padding_mode_;
    cudnnConvolutionDescriptor_t desc_;
    cudnnConvolutionFwdAlgo_t algo_;
    cudnnConvolutionBwdDataAlgo_t dalgo_;
    cudnnConvolutionBwdFilterAlgo_t falgo_;
    Tensor4d *p_input_;
    Tensor4d *p_output_;
    Filter4d *p_filter_;
    Tensor4d *bias_;
};

#endif
