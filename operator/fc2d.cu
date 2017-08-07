#include "fc2d.h"

__global__ void Update(float *data, float *grad, int size, int RST)
{
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    if (idx >= size) return;
    data[idx] += grad[idx % RST];
    __syncthreads();
}

Fc2d::Fc2d(int k) : K_(k)
{
    alpha          = 1.0f;
    beta           = 0.0f;
    p_input_       = nullptr;
    p_output_      = nullptr;
    p_weights_     = nullptr;
    grads_weights_ = nullptr;
    grads_data_    = nullptr;
    grads_bias_    = nullptr;
    bias_          = nullptr;
}

Fc2d::~Fc2d()
{
    checkCudnn(cudnnDestroyConvolutionDescriptor(desc_));
    delete p_weights_;
    delete bias_;
    free(grads_weights_);
    free(grads_data_);
    free(grads_bias_);
}

void Fc2d::AddInput(ITensor *input)
{
    this->p_input_ = dynamic_cast<Tensor4d*>(input);
}

ITensor *Fc2d::LayerInit()
{
    if (this->p_weights_ == nullptr)
    {
        this->p_weights_ = new Filter4d(K_, p_input_->C(), p_input_->H(), p_input_->W());
        std::cout << "Init weights here\n";
        p_weights_->PrintShape();
        SetWeights(0.01f);
    }

    filterStrideA_[0] = 1;
    filterStrideA_[1] = 1;
    filationA_[0] = 1;
    filationA_[1] = 1;
    padA_[0] = 0;
    padA_[1] = 0;
    checkCudnn(cudnnCreateConvolutionDescriptor(&desc_));
    checkCudnn(cudnnSetConvolutionNdDescriptor(
        desc_, 2, padA_, filterStrideA_, filationA_,
        //CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT
        CUDNN_CONVOLUTION, CUDNN_DATA_FLOAT
    ));

    N_out = p_input_->N();
    C_out = K_;
    H_out = 1;
    W_out = 1;

    if (this->p_output_ == nullptr)
    {
        p_output_ = new Tensor4d(N_out, C_out, H_out, W_out);
        p_input_->PrintShape();
        p_weights_->PrintShape();
        p_output_->PrintShape();
        std::cout << p_input_->Desc() << ' ' << p_weights_->Desc() << ' ' << desc_ << ' ' << p_output_->Desc() << "\n";
        checkCudnn(cudnnGetConvolutionForwardAlgorithm(
                Session::instance().cudnn_handle(), p_input_->Desc(), p_weights_->Desc(), desc_,
                p_output_->Desc(), CUDNN_CONVOLUTION_FWD_PREFER_FASTEST, 0, &algo_));
        std::cout << algo_ << "\n";
        checkCudnn(cudnnGetConvolutionForwardWorkspaceSize(
                Session::instance().cudnn_handle(), p_input_->Desc(), p_weights_->Desc(), desc_,
                p_output_->Desc(), algo_, &size_in_bytes));
    }
    Session::instance().update_workspace_size(size_in_bytes);

    //if (this->bias_ == nullptr)
    //{
    //    bias_ = new Tensor4d(N_out, C_out, H_out, W_out);
    //    bias_->SetValue(1);
    //}
    return p_output_;
}

void Fc2d::Forward(bool del)
{
    Tensor4d *out = dynamic_cast<Tensor4d*>(p_output_);
    std::cout << p_input_->Desc() << ' ' << p_weights_->Desc() << ' ' << desc_ << ' ' << ' ' << out->Desc()<< "\n";
    //p_weights_->PrintAll();
    checkCudnn(cudnnConvolutionForward(
        Session::instance().cudnn_handle(), &alpha, p_input_->Desc(), p_input_->GpuPointer(),
        p_weights_->Desc(), p_weights_->GpuPointer(), desc_, algo_,
        Session::instance().workspace(), Session::instance().workspace_size(),
        &beta, out->Desc(), out->GpuPointer()
    ));
    out->PrintShape();
    out->PrintAll();
}

float *Fc2d::Backward(float *down_grads, bool del)
{
     if (grads_weights_ == nullptr && grads_data_ == nullptr)
     {
        checkCudaError(cudaMalloc(&grads_weights_, sizeof(float) * p_weights_->Size()));
        checkCudaError(cudaMalloc(&grads_data_,    sizeof(float) * p_input_->Size()));
     }
     checkCudnn(cudnnConvolutionBackwardFilter(
        Session::instance().cudnn_handle(), &alpha, p_input_->Desc(), p_input_->GpuPointer(),
        p_output_->Desc(), down_grads, desc_, CUDNN_CONVOLUTION_BWD_FILTER_ALGO_0,
        Session::instance().workspace(), Session::instance().workspace_size(),&beta, p_weights_->Desc(), grads_weights_
     ));
     checkCudnn(cudnnConvolutionBackwardData(
        Session::instance().cudnn_handle(), &alpha, p_weights_->Desc(), p_weights_->GpuPointer(),
        p_output_->Desc(), down_grads, desc_, CUDNN_CONVOLUTION_BWD_DATA_ALGO_0,
        Session::instance().workspace(), Session::instance().workspace_size(),
        &beta, p_input_->Desc(), grads_data_
     ));
     return grads_data_;
}

void Fc2d::UpdateWeights()
{
     int size = p_weights_->Size();
     int K = p_weights_->K();
     Update<<<(size + 255) / 256, 256>>>(p_weights_->GpuPointer(), grads_weights_, size, size / K);
     //p_weights_->SyncToCpu();
     //p_weights_->PrintAll();
}

void Fc2d::SetWeights(float data)
{
     p_weights_->SetValue(data);
}
