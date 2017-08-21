#ifndef FC2D_H
#define FC2D_H

#include "ioperator.h"
#include "../wheel.h"
#include "../tensor/tensor4d.h"
#include "../tensor/filter4d.h"
#include "../session.h"

class Fc2d : public IOperator
{
public:
    Fc2d(int k);
    ~Fc2d();
    void AddInput(ITensor *input);
    ITensor *LayerInit();
    void Forward(bool del = false);
    float *Backward(float *c, bool del = false);
    void UpdateWeights();
    void SetWeights(float data);
private:
    int K_;
    size_t size_in_bytes;
    float alpha;
    float beta;
    float *grads_weights_;
    float *grads_bias_;
    float *grads_data_;
    int padA_[2];
    int filterStrideA_[2];
    int filationA_[2];
    cudnnConvolutionDescriptor_t desc_;
    cudnnConvolutionFwdAlgo_t algo_;
    cudnnConvolutionBwdFilterAlgo_t bwdalgo_;
    Tensor4d *p_input_;
    Tensor4d *p_output_;
    Filter4d *p_weights_;
    Tensor4d *bias_;
};

#endif
