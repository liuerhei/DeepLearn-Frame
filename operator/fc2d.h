#ifndef FC2D_H
#define FC2D_H

#include "../tensor/tensor4d.h"
#include "../session.h"

class Fc2d : public IOperator
{
public:
    Fc2d(int k);
    ~Fc2d();
    void AddInput(ITensor *input);
    ITensor *LayerInit();
    void Forward(bool del = false);
    float *Backward(float *down_grads, bool del = false);
    void UpdateWeights(float learning_rate = 0.01);
    void SetWeights(float data);
private:
    int K_;
    size_t size_in_bytes;
    size_t length_;
    float alpha;
    float beta;
    Tensor4d *p_input_;
    Tensor4d *p_weights_;
    Tensor4d *p_output_;
    Tensor4d *p_bias_;
    float *grads_weights_;
    float *grads_data_;
    float *grads_bias_;
    float *onevec;
};
#endif
