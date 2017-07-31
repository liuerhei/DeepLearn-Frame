#include "conv2d.h"

//Conv2d::Conv2d(int k, int s, int t ,padding_t mode)
//    : K_(k), S_(s), T_(t), padding_mode_(mode)
//{
//    alpha = 1.0f;
//    bate = 0.0f;
//}
Conv2d::Conv2d(int k, int s, int t, Padding_t mode)
    : K_(k), S_(s), T_(t), padding_mode_(mode)
{
    alpha = 1.0f;
    beta = 0.0f;
    p_input_ = nullptr;
    p_output_ = nullptr;
}

Conv2d::~Conv2d()
{
    checkCudnn(cudnnDestroyConvolutionDescriptor(desc_));
    delete p_filter_;
    std::cout << "Conv2dLayer Delete\n";
}

ITensor *Conv2d::add_input(ITensor *input, bool del = false)
{
    this->p_input_ = dynamic_cast<Tensor4d*>(input);
    this->set_input_shape(p_input_->N(), p_input_->C(), p_input_->H(), p_input_->W());
    if(del || p_output_ == nullptr)
    {
        p_output_ = new Tensor4d(p_input_->N(), C_out, H_out, W_out);
        p_output_->print_shape();
        Tensor4d *out = dynamic_cast<Tensor4d*>(p_output_);

        checkCudnn(cudnnGetConvolutionForwardAlgorithm(
            Session::instance().cudnn_handle(), p_input_->desc(), p_filter_->desc(), desc_,
            out->desc(), CUDNN_CONVOLUTION_FWD_PREFER_FASTEST, 0, &algo_
        ));
        checkCudnn(cudnnGetConvolutionForwardWorkspaceSize(
            Session::instance().cudnn_handle(), p_input_->desc(), p_filter_->desc(), desc_,
            out->desc(), algo_, &size_in_bytes
        ));
        Session::instance().update_workspace_size(size_in_bytes);
    }
    return p_output_;
}

void Conv2d::Forward(bool del = false)
{
    Tensor4d *out = dynamic_cast<Tensor4d*>(p_output_);
    checkCudnn(cudnnConvolutionForward(
        Session::instance().cudnn_handle(), &alpha, p_input_->desc(), p_input_->gpu_pointer(),
        p_filter_->desc(), p_filter_->gpu_pointer(), desc_, algo_, 
        Session::instance().workspace(), Session::instance().workspace_size(),
        &beta, out->desc(), out->gpu_pointer() 
    ));
    out->print_all();
}

void Conv2d::Backward(cudnnTensorDescriptor_t uptensordesc, cudnnTensorDescriptor_t downtensordesc, float *grad, bool del)
{
     checkCudaError(cudaMalloc(&grads_filter_, sizeof(float) * p_filter_->size()));
     checkCudaError(cudaMalloc(&grads_data_,   sizeof(float) * N_out * C_out * H_out * W_out));
     checkCudnn(cudnnConvolutionBackwardFilter(
          Session::instance().cudnn_handle(), &alpha, p_input_->desc(), p_input_->gpu_pointer(),
          uptensordesc, grad, desc_, CUDNN_CONVOLUTION_BWD_FILTER_ALGO_0,
          Session::instance().workspace(), Session::instance().workspace_size(),
          &beta, p_filter_->desc(), grads_filter_
     ));
     checkCudnn(cudnnConvolutionBackwardData(
          Session::instance().cudnn_handle(), &alpha, p_filter_->desc(), p_filter_->gpu_pointer(),
          uptensordesc, grad, desc_, CUDNN_CONVOLUTION_BWD_DATA_ALGO_0,
          Session::instance().workspace(), Session::instance().workspace_size(),
          &beta, downtensordesc, grads_data_
     ));
}
void Conv2d::set_input_shape(int n, int c, int h, int w)
{
    p_filter_ = new Filter4d(K_, c, S_, T_);
    p_filter_->print_shape();
    p_filter_->set_value(1);
    filterStrideA_[0] = 1;
    filterStrideA_[1] = 1;
    dilationA_[0] = 1;
    dilationA_[1] = 1;
    if(padding_mode_ == valid)
    {
        padA_[0] = 0;
        padA_[1] = 0;
        H_out = (h - S_ + 1) / filterStrideA_[0] + (h - S_ + 1) % filterStrideA_[0];
        W_out = (w - T_ + 1) / filterStrideA_[0] + (h - T_ + 1) % filterStrideA_[1];
    }
    else{
        H_out = h / filterStrideA_[0] + h % filterStrideA_[0];
        W_out = w / filterStrideA_[1] + w % filterStrideA_[1];
    }
    checkCudnn(cudnnCreateConvolutionDescriptor(&desc_));
    checkCudnn(cudnnSetConvolutionNdDescriptor(
        desc_, 2, padA_, filterStrideA_, dilationA_,
        CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT
    ));
    C_out = K_;
    N_out = n;
}

void Conv2d::set_weights(float data)
{
    p_filter_->set_value(data);
    // p_filter_->randomize();
}
