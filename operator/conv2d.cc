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
    alpha         = 1.0f;
    beta          = 0.0f;
    p_input_      = nullptr;
    p_output_     = nullptr;
    p_filter_     = nullptr;
    grads_filter_ = nullptr;
    grads_data_   = nullptr;
    grads_bias_   = nullptr;
    bias_         = nullptr;
}

Conv2d::~Conv2d()
{
    checkCudnn(cudnnDestroyConvolutionDescriptor(desc_));
    delete p_filter_;
    delete p_input_;
    delete p_output_;
    delete bias_;
    free(grads_data_);
    free(grads_filter_);
    /*
     * TODO
     * when the function will be called ?
     * And whether pointers should be deleted ?
     */
    std::cout << "Conv2dLayer Delete\n";
}

void Conv2d::AddInput(ITensor *input)
{
    this->p_input_ = dynamic_cast<Tensor4d*>(input);
    // When backward complete, the input should be deleted
}

ITensor *Conv2d::LayerInit()
{
    if (this->p_filter_ == nullptr)
    {
        this->p_filter_ = new Filter4d(K_, p_input_->C(), S_, T_);
        std::cout << "Add new Filter here\n";
        p_filter_->PrintShape();
        SetWeights(1);
        //p_filter_->PrintAll();
    }
    // Init the space and weights of filter.
    
    int h = p_input_->H();
    int w = p_input_->W();
    int n = p_input_->N();
    //std::cout << p_filter_->CpuPointer() << ' ' << p_filter_->GpuPointer() << "\n";
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
    // compute the size of output
    
    if (this->p_output_ == nullptr)
    {
        p_output_ = new Tensor4d(N_out, C_out, H_out, W_out);
        Tensor4d *out = dynamic_cast<Tensor4d*>(p_output_);
    
        checkCudnn(cudnnGetConvolutionForwardAlgorithm(
            Session::instance().cudnn_handle(), p_input_->Desc(), p_filter_->Desc(), desc_,
            out->Desc(), CUDNN_CONVOLUTION_FWD_PREFER_FASTEST, 0, &algo_
        ));
        checkCudnn(cudnnGetConvolutionForwardWorkspaceSize(
            Session::instance().cudnn_handle(), p_input_->Desc(), p_filter_->Desc(), desc_,
            out->Desc(), algo_, &size_in_bytes
        ));
        Session::instance().update_workspace_size(size_in_bytes);
    }

    if (this->bias_ == nullptr)
    {
        bias_ = new Tensor4d(N_out, C_out, H_out, W_out);
        bias_->SetValue(1);
    }
    return p_output_;
}

void Conv2d::Forward(bool del = false)
{
    Tensor4d *out = dynamic_cast<Tensor4d*>(p_output_);
    checkCudnn(cudnnConvolutionForward(
        Session::instance().cudnn_handle(), &alpha, p_input_->Desc(), p_input_->GpuPointer(),
        p_filter_->Desc(), p_filter_->GpuPointer(), desc_, algo_, 
        Session::instance().workspace(), Session::instance().workspace_size(),
        &beta, out->Desc(), out->GpuPointer() 
    ));
    // out->PrintAll();
    //checkCudnn(cudnnAddTensor(
    //    Session::instance().cudnn_handle(), &alpha, bias_->Desc(), bias_->GpuPointer(), &beta, out->Desc(), out->GpuPointer()
    //));
    //out->PrintAll();
}

float *Conv2d::Backward(float *down_grads, bool del = false)
{
     if (grads_filter_ == nullptr && grads_data_ == nullptr)
     {
        checkCudaError(cudaMalloc(&grads_filter_, sizeof(float) * p_filter_->Size()));
        //checkCudaError(cudaMalloc(&grads_data_,   sizeof(float) * N_out * C_out * H_out * W_out));
        checkCudaError(cudaMalloc(&grads_data_,   sizeof(float) * p_input_->Size()));
     }
     // TODO
     // Here maybe have BUG
     // Because the size of each layer make sence, so the space can allocate once.
     checkCudnn(cudnnConvolutionBackwardFilter(
          Session::instance().cudnn_handle(), &alpha, p_input_->Desc(), p_input_->GpuPointer(),
          p_output_->Desc(), down_grads, desc_, CUDNN_CONVOLUTION_BWD_FILTER_ALGO_0,
          Session::instance().workspace(), Session::instance().workspace_size(),
          &beta, p_filter_->Desc(), grads_filter_
     ));
     checkCudnn(cudnnConvolutionBackwardData(
          Session::instance().cudnn_handle(), &alpha, p_filter_->Desc(), p_filter_->GpuPointer(),
          p_output_->Desc(), down_grads, desc_, CUDNN_CONVOLUTION_BWD_DATA_ALGO_0,
          Session::instance().workspace(), Session::instance().workspace_size(),
          &beta, p_input_->Desc(), grads_data_
     ));
     /*
      * TODO
      * when should the input and output tensor delete after backward complete ?
      */

     /*
     float *a = (float *)malloc(sizeof(float) * p_filter_->Size());
     checkCudaError(cudaMemcpy(a, grads_filter_, sizeof(float) * p_filter_->Size(), cudaMemcpyDeviceToHost));
     std::cout << "conv filter gradients\n";
     for(int i = 0; i < p_filter_->Size(); ++i)
        std::cout << a[i] << ' ';
     //   a[i] = i;
     //checkCudaError(cudaMemcpy(grads_filter_, a, sizeof(float) * p_filter_->Size(), cudaMemcpyHostToDevice));
     // This is a test, it seems that the gradients always is 0
     std::cout << "\n";

     float *b = (float *)malloc(sizeof(float) * p_input_->Size());
     checkCudaError(cudaMemcpy(a, grads_data_,  sizeof(float) * p_input_->Size(),   cudaMemcpyDeviceToHost));
     std::cout << "conv data gradients\n";
     for(int i = 0; i < p_input_->Size(); ++i)
        std::cout << b[i] << ' ';
     std::cout << "\n";
     //free(a);
     //free(b);
     */

     return grads_data_;
}

void Conv2d::UpdateWeights()
// TODO
// Need to rebuild by using CUDA
{
    int size = p_filter_->Size();
    p_filter_->SyncToCpu();  
    float *pointer = p_filter_->CpuPointer();
    float *a = (float*)malloc(sizeof(float) * size);
    checkCudaError(cudaMemcpy(a, this->grads_filter_, sizeof(float) * size, cudaMemcpyDeviceToHost));
    for(int i = 0; i < size; ++i)
    {
        //std::cout << pointer[i] << ' ' << a[i % 9] << "\n";
        std::cout << pointer[i] << ' ' << a[i] << "\n";
        pointer[i] += a[i];
    }
    p_filter_->SyncToGpu();
    // TODO need to free grads_filter_
    //p_filter_->PrintAll();
    free(a);
}

void Conv2d::SetWeights(float data)
{
    p_filter_->SetValue(data);
    // p_filter_->randomize();
}
