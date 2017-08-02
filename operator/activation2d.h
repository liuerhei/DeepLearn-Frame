#ifndef ACTIVATE2D_H
#define ACTIVATE2D_H

#include "ioperator.h"
#include "../wheel.h"
#include "../tensor/tensor4d.h"
#include "../session.h"
/*
 * Activation layer for 2d layer
 * Here support activation as fellow:
 *
 * CUDNN_ACTIVATION_SIGMOID
 * CUDNN_ACTIVATION_RELU
 * CUDNN_ACTIVATION_TANH
 * CUDNN_ACTIVATION_CLIPPED_RELU
 * CUDNN_ACTIVATION_ELU
 */

class Activation2d : public IOperator
{
public:
    Activation2d(cudnnActivationMode_t mode = CUDNN_ACTIVATION_RELU);
    ~Activation2d();
    void AddInput(ITensor *);
    ITensor *LayerInit();
    void Forward(bool del = false);
    //float *Backward(float *, bool);
    //void UpdateWeights();
private:
    cudnnActivationMode_t mode_;
    cudnnActivationDescriptor_t desc_;
    Tensor4d *p_input_;
    Tensor4d *p_output_;
    //float *grads_filter_;
    //float *grads_data_;
    float alpha;
    float beta;
};

#endif
