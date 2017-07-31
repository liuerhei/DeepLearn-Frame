#include "pooling2d.h"
#include "../tensor/tensor4d.h"

Pooling2d::Pooling2d(int window, int stride, Padding_t pad, cudnnPoolingMode_t mode)
                    : mode_(mode), padding_mode_(pad)
{
    this->nbDims_ = 2;
    alpha = 1.0f;
    beta = 0.0f;
    strideA_[0] = strideA_[1] = stride;
    windowDimA_[0] = windowDimA_[1] = window;
    p_output_ = nullptr;
}

Pooling2d::~Pooling2d()
{
    cudnnDestroyPoolingDescriptor(desc_);
    std::cout << "Pooling Layer Delete\n";
}

ITensor *Pooling2d::add_input(ITensor *input, bool del = false)
{
    this->p_input_ = dynamic_cast<Tensor4d*>(input);
    Tensor4d *now = dynamic_cast<Tensor4d*>(input);
    this->set_input_shape(now->C(),now->H(), now->W());
    if (del || p_output_ == nullptr)
    {
        p_output_ = new Tensor4d(now->N(), C_out, H_out, W_out);
        p_output_->print_shape();
    }
    return p_output_;
}

void Pooling2d::Forward(bool del = false)
{
    Tensor4d *in = dynamic_cast<Tensor4d*>(p_input_);
    Tensor4d *out = dynamic_cast<Tensor4d*>(p_output_);
    checkCudnn(cudnnPoolingForward(
        Session::instance().cudnn_handle(), desc_, &alpha, in->desc(),
        in->gpu_pointer(), &beta, out->desc(), out->gpu_pointer() 
    ));
    out->print_all();
}

void Pooling2d::set_input_shape(int C, int H, int W)
{
    if(padding_mode_ == valid)
    {
        padA_[0] = padA_[1] = 0;
        H_out = (H - windowDimA_[0] + 1) / strideA_[0] + (H - windowDimA_[0] + 1) % strideA_[0];
        W_out = (W - windowDimA_[1] + 1) / strideA_[1] + (W - windowDimA_[1] + 1) % strideA_[1];
    } else
    {
        H_out = H / strideA_[0] + H % strideA_[0];
        W_out = W / strideA_[1] + W % strideA_[1];
        padA_[0] = ((H_out - 1) * strideA_[0] + windowDimA_[0] - H) / 2 + ((H_out - 1) * strideA_[0] + windowDimA_[0] - H) % 2;
        padA_[1] = ((W_out - 1) * strideA_[1] + windowDimA_[0] - H) / 2 + ((W_out - 1) * strideA_[1] + windowDimA_[0] - H) % 2;
    }
    checkCudnn(cudnnCreatePoolingDescriptor(&desc_));
    checkCudnn(cudnnSetPoolingNdDescriptor(
        desc_, mode_, CUDNN_PROPAGATE_NAN, nbDims_, windowDimA_, padA_, strideA_
    ));
    C_out = C;

}
