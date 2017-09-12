#include "softmax.h"

Softmax::Softmax(cudnnSoftmaxMode_t mode) : mode_(mode)
{
    this->alpha        = 1.0f;
    this->beta         = 0.0f;
    this->p_input_     = nullptr;
    this->p_output_    = nullptr;
    this->grads_input_ = nullptr;
}

Softmax::~Softmax()
{
    delete(p_input_);
    delete(p_output_);
    cudaFree(grads_input_);
    std::cout << "Softmax Layer Delete\n";
}

void Softmax::AddInput(ITensor *input)
{
    this->p_input_ = dynamic_cast<Tensor4d*>(input);
}

ITensor *Softmax::LayerInit()
{
    if (this->p_output_ == nullptr)
    {
        this->p_output_ = new Tensor4d(p_input_->N(), p_input_->C(), p_input_->H(), p_input_->W());
    }
    algo_ = CUDNN_SOFTMAX_ACCURATE;
    return p_output_;
}

void Softmax::Forward(bool del = false)
{
    p_input_->PrintK(10);
    checkCudnn(cudnnSoftmaxForward(
        Session::instance().cudnn_handle(), algo_, mode_, &alpha, p_input_->Desc(), p_input_->GpuPointer(), 
        &beta, p_output_->Desc(), p_output_->GpuPointer()
    ));
    p_output_->PrintK(10);
}

float *Softmax::Backward(float *grads_down, bool del)
{
    //p_output_->PrintK(10);
    if (this->grads_input_ == nullptr)
    {
       checkCudaError(cudaMalloc(&this->grads_input_, sizeof(float) * p_input_->Size()));
       //std::cout << "allocate grads_input_ " << this->grads_input_ << "\n";
    }
    checkCudnn(cudnnSoftmaxBackward(
       Session::instance().cudnn_handle(), algo_, mode_, &alpha, p_output_->Desc(), p_output_->GpuPointer(), 
       p_output_->Desc(), grads_down, &beta, p_input_->Desc(), grads_input_
    ));
    //std::cout << "after compution " << this->grads_input_ << "\n";
    //std::cout << "p_output_ " << this->p_output_->GpuPointer() << "\n";
    //float *a = (float*)malloc(sizeof(float) * p_input_->Size());
    //checkCudaError(cudaMemcpyAsync(a, grads_input_, sizeof(float) * p_input_->Size(), cudaMemcpyDeviceToHost));
    //for(int i = 0; i < p_input_->Size(); ++i)
    //   std::cout << a[i] << ' ';

     return grads_input_;
}
