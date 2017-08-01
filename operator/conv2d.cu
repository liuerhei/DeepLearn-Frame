#include "conv2d.h"

//Conv2d::Conv2d(int k, int s, int t ,padding_t mode)
//    : K_(k), S_(s), T_(t), padding_mode_(mode)
//{
//    alpha = 1.0f;
//    bate = 0.0f;
//}


__global__ void update(float *data, float *grad, int size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return ;
    data[idx] += grad[idx];
    __syncthreads();
}

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
    delete p_input_;
    delete p_output_;
    free(grads_data_);
    std::cout << "Conv2dLayer Delete\n";
}

void Conv2d::add_input(ITensor *input)
{
    this->p_input_ = dynamic_cast<Tensor4d*>(input);
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
    //out->print_all();
}

float *Conv2d::Backward(float *down_grads, bool del = false)
{
     checkCudaError(cudaMalloc(&grads_filter_, sizeof(float) * p_filter_->size()));
     std::cout << N_out << ' ' << C_out << ' ' << H_out << ' ' << W_out << "\n";
     checkCudaError(cudaMalloc(&grads_data_,   sizeof(float) * N_out * C_out * H_out * W_out));
     /*
      * Here need to figure out the upTensorDescriptor and the downTensorDescriptor
      * For backwardfilter
      * the x is the current layer input, y is the current layer output, but dy is the loss from upper layer--down_grads
      */

     checkCudnn(cudnnConvolutionBackwardFilter(
          Session::instance().cudnn_handle(), &alpha, p_input_->desc(), p_input_->gpu_pointer(),
          p_output_->desc(), down_grads, desc_, CUDNN_CONVOLUTION_BWD_FILTER_ALGO_0,
          Session::instance().workspace(), Session::instance().workspace_size(),
          &beta, p_filter_->desc(), grads_filter_
     ));
     checkCudnn(cudnnConvolutionBackwardData(
          Session::instance().cudnn_handle(), &alpha, p_filter_->desc(), p_filter_->gpu_pointer(),
          p_output_->desc(), down_grads, desc_, CUDNN_CONVOLUTION_BWD_DATA_ALGO_0,
          Session::instance().workspace(), Session::instance().workspace_size(),
          &beta, p_input_->desc(), grads_data_
     ));
     return grads_data_;
}

/*
void Conv2d::update_weights()
{
    std::cout << "I am in the update_weights\n";
    int size = p_filter_->size();
    std::cout << size << "\n";
    p_filter_->sync_to_cpu();  // Error here
    std::cout << "success\n";
    float *pointer = p_filter_->cpu_pointer();
    std::cout << pointer << "\n";
    float *a = (float*)malloc(sizeof(float) * size);
    checkCudaError(cudaMemcpy(a, this->grads_filter_, sizeof(float) * size, cudaMemcpyDeviceToHost));
    std::cout << "copy success\n";
    for(int i = 0; i < size; ++i)
        pointer[i] += a[i];
    p_filter_->sync_to_gpu();
    free(a);
    // TODO need to free grads_filter_
    //p_filter_->print_all();
}
*/

void Conv2d::update_weights()
{
     std::cout << "I am in the update_weights\n";

     std::cout << p_filter_->cpu_pointer() << ' ' << p_filter_->gpu_pointer() << "\n";
     int size = p_filter_->size();
     update<<<(size + 256) / 256, 256>>>(p_filter_->gpu_pointer(), grads_filter_, size);
     p_filter_->print_all();
}

ITensor *Conv2d::set_input_shape()
{
    p_filter_ = new Filter4d(K_, p_input_->C(), S_, T_);
    std::cout << "-------->Alloc new filter here\n";
    int h = p_input_->H();
    int w = p_input_->W();
    int n = p_input_->N();
    p_filter_->print_shape();
    p_filter_->set_value(1);
    p_filter_->print_all();

    std::cout << p_filter_->cpu_pointer() << ' ' << p_filter_->gpu_pointer() << "\n";
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

    if(p_output_ == nullptr)
    {
        p_output_ = new Tensor4d(p_input_->N(), C_out, H_out, W_out);
        //p_output_->print_shape();
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
    // TODO need to rebuild
}

void Conv2d::set_weights(float data)
{
    p_filter_->set_value(data);
    // p_filter_->randomize();
}
