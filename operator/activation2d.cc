#include "activation2d.h"

Activation2d::Activation2d(cudnnActivationMode_t mode) : mode_(mode)
{
    alpha         = 1.0f;
    beta          = 0.0f;
    p_input_      = nullptr;
    p_output_     = nullptr;
    grads_input_   = nullptr;
}

Activation2d::~Activation2d()
{
    checkCudnn(cudnnDestroyActivationDescriptor(desc_));
    delete p_input_;
    delete p_output_;
    //free(grads_data_);
    //free(grads_filter_);
    std::cout << "Activation Delete\n";
}

void Activation2d::AddInput(ITensor *input)
{
    this->p_input_ = dynamic_cast<Tensor4d*>(input);
}

ITensor *Activation2d::LayerInit()
{
    checkCudnn(cudnnCreateActivationDescriptor(&desc_));
    checkCudnn(cudnnSetActivationDescriptor(desc_, mode_, CUDNN_PROPAGATE_NAN, 0));
    if (this->p_output_ == nullptr)
    {
        p_output_ = new Tensor4d(p_input_->N(), p_input_->C(),
                                p_input_->H(), p_input_->W());
    }
    return p_output_;
}

void Activation2d::Forward(bool del)
{
    checkCudnn(cudnnActivationForward(
        Session::instance().cudnn_handle(), desc_, &alpha, p_input_->Desc(), p_input_->GpuPointer(),
        &beta, p_output_->Desc(), p_output_->GpuPointer()));
    //p_output_->PrintAll();
}

float *Activation2d::Backward(float *grads_down, bool del)
{
    if (this->grads_input_ == nullptr)
    {
        checkCudaError(cudaMalloc(&this->grads_input_, sizeof(float) * p_input_->Size()));
    }
    checkCudnn(cudnnActivationBackward(
        Session::instance().cudnn_handle(), desc_, &alpha, p_output_->Desc(), p_output_->GpuPointer(), 
        p_output_->Desc(), grads_down, p_input_->Desc(), p_input_->GpuPointer(), &beta, p_input_->Desc(), grads_input_
    ));
    float *a = (float *)malloc(sizeof(float) * p_input_->Size());
    checkCudaError(cudaMemcpy(a, grads_input_, sizeof(float) * p_input_->Size(), cudaMemcpyDeviceToHost));
    std::cout << "activation data gradients\n";
    //for (int i = 0; i < p_input_->Size(); ++i)
    for (int i = 0; i < 20; ++i)
        std::cout << a[i] << ' ';
    std::cout << "\n";
    free(a);
    return grads_input_;
}


