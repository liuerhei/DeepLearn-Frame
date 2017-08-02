#include "pooling2d.h"
#include "../tensor/tensor4d.h"

Pooling2d::Pooling2d(int window, int stride, Padding_t pad, cudnnPoolingMode_t mode)
                    : mode_(mode), padding_mode_(pad)
{
    this->nbDims_        = 2;
    this->alpha          = 1.0f;
    this->beta           = 0.0f;
    this->strideA_[0]    = this->strideA_[1]    = stride;
    this->windowDimA_[0] = this->windowDimA_[1] = window;
    this->p_output_      = nullptr;
    this->p_input_       = nullptr;
    this->grads_input_   = nullptr;
}

Pooling2d::~Pooling2d()
{
    cudnnDestroyPoolingDescriptor(desc_);
    delete(p_input_);
    delete(p_output_);
    std::cout << "Pooling Layer Delete\n";
}

void Pooling2d::AddInput(ITensor *input)
{
    this->p_input_ = dynamic_cast<Tensor4d*>(input);
}

ITensor *Pooling2d::LayerInit()
{
    int c = p_input_->C();
    int h = p_input_->H();
    int w = p_input_->W();
    if(padding_mode_ == valid)
    {
        padA_[0] = padA_[1] = 0;
        H_out = (h - windowDimA_[0] + 1) / strideA_[0] + (h - windowDimA_[0] + 1) % strideA_[0];
        W_out = (w - windowDimA_[1] + 1) / strideA_[1] + (w - windowDimA_[1] + 1) % strideA_[1];
    } else
    {
        H_out = h / strideA_[0] + h % strideA_[0];
        W_out = w / strideA_[1] + w % strideA_[1];
        padA_[0] = ((H_out - 1) * strideA_[0] + windowDimA_[0] - h) / 2 + ((H_out - 1) * strideA_[0] + windowDimA_[0] - h) % 2;
        padA_[1] = ((W_out - 1) * strideA_[1] + windowDimA_[0] - w) / 2 + ((W_out - 1) * strideA_[1] + windowDimA_[0] - w) % 2;
    }
    checkCudnn(cudnnCreatePoolingDescriptor(&desc_));
    checkCudnn(cudnnSetPoolingNdDescriptor(
        desc_, mode_, CUDNN_PROPAGATE_NAN, nbDims_, windowDimA_, padA_, strideA_
    ));
    C_out = c;
    
    if (this->p_output_ == nullptr)
    {
        this->p_output_ = new Tensor4d(p_input_->N(), C_out, H_out, W_out);
    }
    return p_output_;
}

void Pooling2d::Forward(bool del = false)
{
    checkCudnn(cudnnPoolingForward(
        Session::instance().cudnn_handle(), desc_, &alpha, this->p_input_->Desc(),
        this->p_input_->GpuPointer(), &beta, this->p_output_->Desc(), this->p_output_->GpuPointer() 
    ));
    this->p_output_->PrintAll();
}

float *Pooling2d::Backward(float *grads_down, bool del)
{
    if (grads_input_ == nullptr)
    {
        checkCudaError(cudaMalloc(&grads_input_, sizeof(float) * p_input_->Size()));
    }
    checkCudnn(cudnnPoolingBackward(
        Session::instance().cudnn_handle(), desc_, &alpha, p_output_->Desc(), p_output_->GpuPointer(), 
        p_output_->Desc(), grads_down, p_input_->Desc(), p_input_->GpuPointer(),
        &beta, p_input_->Desc(), grads_input_
    ));
    
    float *a = (float *)malloc(sizeof(float) * p_input_->Size());
    checkCudaError(cudaMemcpy(a, grads_input_, sizeof(float) * p_input_->Size(), cudaMemcpyDeviceToHost));
    for (int i = 0; i < p_input_->Size(); ++i)
        std::cout << a[i] << ' ';
    std::cout << "\n";
    free(a);
    return grads_input_;
}


