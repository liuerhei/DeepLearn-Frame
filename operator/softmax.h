#ifndef SOFTMAX_H
#define SOFTMAX_H

#include "../session.h"
#include "../tensor/tensor4d.h"

/*
 * Here support Softmax mode as follow:
 * CUDNN_SOFTMAX_MODE_INSTANCE
 * CUDNN_SOFTMAX_MODE_CHANNEL
 * 
 * Softmax algorithm as follow:
 * CUDNN_SOFTMAX_ACCURATE
 * CUDNN_SOFTMAX_LOG
 * CUDNN_SOFTMAX_FAST
 */
class Softmax : public IOperator
{
public:
      Softmax(cudnnSoftmaxMode_t mode = CUDNN_SOFTMAX_MODE_CHANNEL,
              cudnnSoftmaxAlgorithm_t algo = CUDNN_SOFTMAX_ACCURATE);
      ~Softmax();
      void AddInput(ITensor *input);
      ITensor *LayerInit();
      void Forward(bool del = false);
      float *Backward(float *grads_down, bool del = false);

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
