#ifndef BATChNORMALIZATION2D_H
#define BATCHNORMALIZATION2D_H

#include "../tensor/tensor4d.h"
#include "../session.h"


class BatchNormalization2d : public IOperator
{
public:
    BatchNormalization2d(cudnnBatchNormMode_t mode = CUDNN_BATCHNORM_PER_ACTIVATION);
    ~BatchNormalization2d();
    void AddInput(ITensor *);
    ITensor *LayerInit();
    void Inference(bool del = false);
    void Forward(bool del = false);
    float *Backward(float *grads_down, bool del = false);

private:
    cudnnBatchNormMode_t   mode_;
    Tensor4d *p_input_;
    Tensor4d *p_output_;
    Tensor4d *p_bnscale_;
    Tensor4d *p_bnbias_;
    Tensor4d *runmean_;
    Tensor4d *runvariance_;
    Tensor4d *savemean_;
    Tensor4d *savevariance_;
    Tensor4d *estimatedMean_;
    Tensor4d *estimatedVariance_;
    double   epsilon_;
    double   exponentialAverageFactor_;
    float    alpha;
    float    beta;

};
#endif
