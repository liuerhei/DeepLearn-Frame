#include "conv2d.h"


//__global__ void DUpdate(float *data, float *grad, int size, int RST, float learn)
//{
//    int idx = threadIdx.x + blockDim.x * blockIdx.x;
//    if (idx >= size) return;
//    data[idx] += grad[idx % RST] * learn;
//    __syncthreads();
//}
//
//__global__ void AddBias(float *data, float *bias, int size, int k)
//{
//    int idx = threadIdx.x + blockDim.x * blockIdx.x;
//    if (idx >= size) return;
//    data[idx] += bias[idx % k];
//    __syncthreads();
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
    free(grads_bias_);
    std::cout << "Conv layer delete\n";
    std::cout << "Conv2dLayer Delete\n";
}

void Conv2d::AddInput(ITensor *input)
{
    this->p_input_ = dynamic_cast<Tensor4d*>(input);
    //log_info("Conv Input");
    //p_input_->PrintShape();
}

ITensor *Conv2d::LayerInit()
{
    if (this->p_filter_ == nullptr)
    {
        this->p_filter_ = new Filter4d(K_, p_input_->C(), S_, T_);
        //p_filter_->SetValue(1);
        p_filter_->Randomize();
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
    }

    Tensor4d *out = dynamic_cast<Tensor4d*>(p_output_);
    checkCudnn(cudnnGetConvolutionForwardAlgorithm(
        Session::instance().cudnn_handle(), p_input_->Desc(), p_filter_->Desc(), desc_,
        out->Desc(), CUDNN_CONVOLUTION_FWD_PREFER_FASTEST, 0, &algo_
    ));
    checkCudnn(cudnnGetConvolutionForwardWorkspaceSize(
        Session::instance().cudnn_handle(), p_input_->Desc(), p_filter_->Desc(), desc_,
        out->Desc(), algo_, &size_in_bytes
    ));
    // compute the filter backward workspace size
    size_t fsize_bytes, dsize_bytes;
    checkCudnn(cudnnGetConvolutionBackwardFilterAlgorithm(
        Session::instance().cudnn_handle(), p_input_->Desc(), p_output_->Desc(), desc_,
        p_filter_->Desc(), CUDNN_CONVOLUTION_BWD_FILTER_PREFER_FASTEST, 0, &falgo_
    ));
    checkCudnn(cudnnGetConvolutionBackwardFilterWorkspaceSize(
        Session::instance().cudnn_handle(), p_input_->Desc(), p_output_->Desc(), desc_,
        p_filter_->Desc(), falgo_, &fsize_bytes
    ));
    // compute the data backward workspace size
    checkCudnn(cudnnGetConvolutionBackwardDataAlgorithm(
        Session::instance().cudnn_handle(), p_filter_->Desc(), p_output_->Desc(), desc_,
        p_input_->Desc(), CUDNN_CONVOLUTION_BWD_DATA_PREFER_FASTEST, 0, &dalgo_
    ));
    checkCudnn(cudnnGetConvolutionBackwardDataWorkspaceSize(
        Session::instance().cudnn_handle(), p_filter_->Desc(), p_output_->Desc(), desc_,
        p_input_->Desc(), dalgo_, &dsize_bytes
    ));
    size_in_bytes = (size_in_bytes > fsize_bytes) ? size_in_bytes : fsize_bytes;
    size_in_bytes = (size_in_bytes > dsize_bytes) ? size_in_bytes : dsize_bytes;

    Session::instance().update_workspace_size(size_in_bytes);
    // compute the workspace size of convolution forward and backward

    if (this->bias_ == nullptr)
    {
        //bias_ = new Tensor4d(1, C_out, H_out, W_out);
        bias_ = new Tensor4d(1, C_out, 1, 1);
        // in lenet, the bias tensor shape is 1, channel, 1, 1
        bias_->Randomize();
        //bias_->SetValue(0.01);
    }
    //p_output_->PrintShape();
    return p_output_;
}

void Conv2d::Forward(bool del)
{
    Tensor4d *out = dynamic_cast<Tensor4d*>(p_output_);
    checkCudnn(cudnnConvolutionForward(
        Session::instance().cudnn_handle(), &alpha, p_input_->Desc(), p_input_->GpuPointer(),
        p_filter_->Desc(), p_filter_->GpuPointer(), desc_, algo_, 
        Session::instance().workspace(), Session::instance().workspace_size(),
        &beta, out->Desc(), out->GpuPointer() 
    ));
    //std::cout << "Conv layer input****************************\n";
    //p_input_->PrintK(10);
    //std::cout << "Conv layer output****************************\n";
    //out->PrintK(10);
    //std::cout << "conv layer bias******************************\n";
    //bias_->PrintK(10);
    //AddBias<<<(out->Size() + 255) / 256, 256>>>(out->GpuPointer(), bias_->GpuPointer(), out->Size(), p_output_->H() * p_output_->W()); 
    checkCudnn(cudnnAddTensor(
        Session::instance().cudnn_handle(), &alpha, bias_->Desc(), bias_->GpuPointer(), 
        &alpha, out->Desc(), out->GpuPointer()
    ));
    //log_info("Conv Input");
    //p_input_->PrintAll();
    //log_info("Conv Output");
    //out->PrintK(10);
    //std::cout << "Conv layer add bias & out ****************************\n";
    //out->PrintK(10);
}

float *Conv2d::Backward(float *down_grads, bool del)
{
     if (grads_filter_ == nullptr && grads_data_ == nullptr)
     {
        checkCudaError(cudaMalloc(&grads_filter_, sizeof(float) * p_filter_->Size()));
        checkCudaError(cudaMalloc(&grads_data_,   sizeof(float) * p_input_->Size()));
        checkCudaError(cudaMalloc(&grads_bias_,   sizeof(float) * bias_->Size()));
     }
     // TODO
     // Here maybe have BUG
     // Because the size of each layer make sence, so the space can allocate once.
     //checkCudnn(cudnnConvolutionBackwardBias(
     //     Session::instance().cudnn_handle(), &alpha, p_output_->Desc(),
     //     down_grads, &beta, bias_->Desc(), grads_bias_
     //));
     checkCudnn(cudnnConvolutionBackwardBias(
          Session::instance().cudnn_handle(), &alpha, p_output_->Desc(), down_grads,
          &beta, bias_->Desc(), grads_bias_
     ));
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

     //float *a = (float *)malloc(sizeof(float) * 10); 
     //checkCudaError(cudaMemcpy(a, grads_bias_, sizeof(float) * 10, cudaMemcpyDeviceToHost));
     //std::cout << "conv bias gradients\n";
     //for(int i = 0; i < 10; ++i)
     //   std::cout << a[i] << ' ';
     //std::cout << "\n";
     //free(a);

     //float *b = (float *)malloc(sizeof(float) * 10);
     //checkCudaError(cudaMemcpy(b, grads_filter_,  sizeof(float) * 10,   cudaMemcpyDeviceToHost));
     //std::cout << "conv filter gradients\n";
     //for(int i = 0; i < 10; ++i)
     //   std::cout << b[i] << ' ';
     //std::cout << "\n";
     //free(b);

     //float *c = (float *)malloc(sizeof(float) * 10);
     //checkCudaError(cudaMemcpy(b, grads_data_,  sizeof(float) * 10, cudaMemcpyDeviceToHost));
     //std::cout << "conv data gradients\n";
     //for(int i = 0; i < 10; ++i)
     //   std::cout << b[i] << ' ';
     //std::cout << "\n";
     //free(c);
     return grads_data_;
}

void Conv2d::UpdateWeights(float learning_rate)
{
    int size = p_filter_->Size();
    int K = p_filter_->K();
    float rate = -learning_rate;
    //std::cout << " Before weights\n";
    //p_filter_->PrintK(10);
    //DUpdate<<<(size + 255) / 256, 256>>>(p_filter_->GpuPointer(), grads_filter_, size, size / K, rate);
    //std::cout << "bias **********\n";
    //bias_->PrintAll();
    checkCudaError(cublasSaxpy(
        Session::instance().cublas_handle(), p_filter_->Size(), 
        &rate, grads_filter_, 1, p_filter_->GpuPointer(), 1
    ));
    checkCudaError(cublasSaxpy(
        Session::instance().cublas_handle(), bias_->Size(), 
        &rate, grads_bias_, 1, bias_->GpuPointer(), 1
    ));
    //log_info(" After weights");
    //p_filter_->PrintK(10);
    //log_info("bias **********");
    //bias_->PrintAll();
}

void Conv2d::SetWeights(float data)
{
    //p_filter_->SetValue(data);
    // p_filter_->randomize();
}

void Conv2d::ToFile(const char *fileprefix)
{
    std::stringstream ssf;
    ssf << fileprefix << ".bin";

    FILE *fp = fopen(ssf.str().c_str(), "w+");
    if(!fp)
    {
        log_error("FILE cannot open");
        exit(0);
    }
    //this->p_filter_->SyncToCpu();
    fwrite(this->p_filter_->CpuPointer(), sizeof(float), this->p_filter_->Size(), fp);
    fclose(fp);
}

void Conv2d::FromFile(const char *fileprefix)
{
    std::stringstream ssf, ssbf;
    ssf << fileprefix << ".bin";
    ssbf << fileprefix << ".bias.bin";

    FILE *fp = fopen(ssf.str().c_str(), "r");
    if(!fp)
    {
        log_error("FILE connot open");
        exit(0);
    }
    fread(this->p_filter_->CpuPointer(), sizeof(float), this->p_filter_->Size(), fp);
    this->p_filter_->SyncToGpu();
    fclose(fp);

    fp = fopen(ssbf.str().c_str(), "r");
    if(!fp)
    {
        log_error("FILE connot open");
        exit(0);
    }
    fread(this->bias_->CpuPointer(), sizeof(float), this->bias_->Size(), fp);
    this->bias_->SyncToGpu();
    fclose(fp);
}
