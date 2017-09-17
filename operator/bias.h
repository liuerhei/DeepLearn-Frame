#ifndef BIAS2D_H
#define BIAS2D_H

#include <iostream>
#include "../wheel.h"
#include "../tensor/tensor4d.h"
#include "../session.h"

class Bias2d : public IOperator
{
public:
    Bias2d(int c, int h, int w);
    ~Bias2d();
    void AddInput(ITensor*);
    ITensor *LayerInit();
    void Forward(bool del = false);
    float *Backward(float *c, bool del = false);
    void UpdateWeights(float learning_rate = 0.01);
private:
    int C_;
    int H_;
    int W_;
    Tensor4d *p_input_;
    Tensor4d *p_output_;
    Tensor4d *p_bias_;
    float *grads_bias_;
}
