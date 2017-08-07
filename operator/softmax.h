#ifndef SOFTMAX_H
#define SOFTMAX_H

#include "../session.h"
#include "../tensor/tensor4d.h"

/*
 * Here support Softmax as follow:
 * CUDNN_SOFTMAX_MODE_INSTANCE
 * CUDNN_SOFTMAX_MODE_CHANNEL
 */
class Softmax : public IOperator
{
public:
      Softmax(cudnnSoftmaxMode_t mode = CUDNN_SOFTMAX_MODE_INSTANCE);
      ~Softmax();
      void AddInput(ITensor *);
      ITensor *LayerInit();
      void Forward(bool);
      float *Backward(float *grads_down, bool del);

private:
      cudnnSoftmaxMode_t mode_;
      cudnnSoftmaxAlgorithm_t algo_;
      float alpha;
      float beta;
      float *grads_input_;
      Tensor4d *p_input_;
      Tensor4d *p_output_;
};
#endif
