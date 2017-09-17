#include"fc2d.h"

__global__ void WUpdate(float *data, float *grad, int size, int RST, float learning_rate)
{
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    if (idx >= size) return;
    data[idx] += grad[idx % RST] * learning_rate;
    __syncthreads();
}

__global__ void FileOnes(float *data, int size)
{
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    if (idx >= size) return;
    data[idx] = 1.0f;
    __syncthreads();
}
__global__ void BiasForward(float *data, const float *bias, int size, int c)
{
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    if (idx >= size) return;
    data[idx] += bias[idx % c];
    __syncthreads();
}

Fc2d::Fc2d(int k) : K_(k)
{
    alpha          = 1.0f;
    beta           = 0.0f;
    p_bias_        = nullptr;
    p_input_       = nullptr;
    p_output_      = nullptr;
    p_weights_     = nullptr;
    grads_bias_    = nullptr;
    grads_data_    = nullptr;
    grads_weights_ = nullptr;
    onevec         = nullptr;
}

Fc2d::~Fc2d()
{
    delete p_weights_;
    free(grads_weights_);
    free(grads_data_);
    free(grads_bias_);
    free(onevec);
}

void Fc2d::AddInput(ITensor *input)
{
    this->p_input_ = dynamic_cast<Tensor4d*>(input);
}

ITensor *Fc2d::LayerInit()
{
    length_ = p_input_->C() * p_input_->H() * p_input_->W();
    if(this->p_weights_ == nullptr)
    {
        this->p_weights_ = new Tensor4d(K_, p_input_->C(), p_input_->H(), p_input_->W());
        p_weights_->Randomize();
    }
    if(this->p_output_ == nullptr)
    {
        p_output_ = new Tensor4d(p_input_->N(), K_, 1, 1);
    }
    if(this->p_bias_ == nullptr)
    {
        p_bias_ = new Tensor4d(1, K_, 1, 1);
        p_bias_->Randomize();
    }
    checkCudaError(cudaMalloc(&onevec, sizeof(float) * p_input_->N()));
    FileOnes<<<(p_input_->N() + 255)/256, 256>>>(onevec, p_input_->N());
    return p_output_;
}

void Fc2d::Forward(bool del)
{
    Tensor4d *out = p_output_;
    checkCudaError(cublasSgemm(Session::instance().cublas_handle(), CUBLAS_OP_T, CUBLAS_OP_N,
                               K_, p_input_->N(), length_,
                               &alpha,
                               p_weights_->GpuPointer(), length_,
                               p_input_->GpuPointer(),   length_,
                               &beta,
                               out->GpuPointer(),        K_
    ));
    log_info("Forward");
    p_output_->PrintAll();
    log_info("bias");
    p_bias_->PrintAll();
    BiasForward<<<(p_input_->Size() + 255)/256, 256>>>(p_output_->GpuPointer(), p_bias_->GpuPointer(), p_input_->Size(), K_);
    log_info("Add bias");
    p_output_->PrintAll();
}

float *Fc2d::Backward(float *down_grads, bool del)
{
    if (grads_weights_ == nullptr && grads_data_ == nullptr)
    {
        checkCudaError(cudaMalloc(&grads_weights_, sizeof(float) * p_weights_->Size()));
        checkCudaError(cudaMalloc(&grads_data_,    sizeof(float) * p_input_->Size()));
        checkCudaError(cudaMalloc(&grads_bias_,    sizeof(float) * p_bias_->Size()));
    }
    checkCudaError(cublasSgemm(Session::instance().cublas_handle(), CUBLAS_OP_N, CUBLAS_OP_T,
                               length_, K_, p_input_->N(),
                               &alpha,
                               p_input_->GpuPointer(), length_, 
                               down_grads,             K_,
                               &beta,
                               grads_weights_,         length_
    ));
    checkCudaError(cublasSgemv(Session::instance().cublas_handle(), CUBLAS_OP_N,
                               K_, p_input_->N(), 
                               &alpha,
                               down_grads,  K_,
                               onevec,      1,
                               &alpha,
                               grads_bias_, 1
    ));
    float *a = (float*)malloc(sizeof(float) * 10);
    if(a != nullptr)
    {
        checkCudaError(cudaMemcpy(a, grads_bias_, sizeof(float) * 10, cudaMemcpyDeviceToHost));
        std::cout << "fc bias gradients\n";
        for(int i = 0; i < 10; i++)
             std::cout << a[i] << ' ';
        std::cout << "\n";
        free(a);
    }

    checkCudaError(cublasSgemm(Session::instance().cublas_handle(), CUBLAS_OP_N, CUBLAS_OP_N,
                               length_, p_input_->N(), K_,
                               &alpha,
                               p_weights_->GpuPointer(), length_,
                               down_grads,               K_,
                               &beta,
                               grads_data_,              length_
    ));
    
    //float *b = (float*)malloc(sizeof(float) * 100);
    //if(b != nullptr)
    //{
    //    checkCudaError(cudaMemcpy(b, grads_data_, sizeof(float) * 100, cudaMemcpyDeviceToHost));
    //    std::cout << "fc data gradients\n";
    //    for(int i = 0; i < 100; i++)
    //        std::cout << b[i] << ' ';
    //    std::cout << "\n";
    //    free(b);
    //}
    return grads_data_;
}

void Fc2d::UpdateWeights(float learning_rate)
{
     int size = p_weights_->Size();
     int N = p_weights_->N();
     WUpdate<<<(size + 255) / 256, 256>>>(p_weights_->GpuPointer(), grads_weights_, size, size / N, learning_rate);
}

void Fc2d::SetWeights(float data)
{
     p_weights_->SetValue(data);
}
