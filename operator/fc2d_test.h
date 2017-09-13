#ifndef FC2D_TEST_H
#define FC2D_TEST_H

#include "cublas_v2.h"
#include "../tensor/tensor4d.h"
#include "../session.h"

class Fc2d_test : public IOperator
{
public:
    Fc2d_test(int k);
    ~Fc2d_test();
    void AddInput(ITensor *input);
    ITensor *LayerInit();
    void Forward(bool del = false);
    float *Backward(float *down_grads, bool del = false);
    void UpdateWeights();
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
    float *grads_weights_;
    float *grads_data_;
    cublasHandle_t cublasHandle_;
};
#endif
