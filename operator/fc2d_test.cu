#include"fc2d_test.h"

__global__ void WUpdate(float *data, float *grad, int size, int RST)
{
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    if (idx >= size) return;
    data[idx] += grad[idx % RST];
    __syncthreads();
}

Fc2d_test::Fc2d_test(int k) : K_(k)
{
    alpha          = 1.0f;
    beta           = 0.0f;
    p_input_       = nullptr;
    p_output_      = nullptr;
    p_weights_     = nullptr;
    grads_data_    = nullptr;
    grads_weights_ = nullptr;
}

Fc2d_test::~Fc2d_test()
{
    checkCudaError(cublasDestroy(cublasHandle_));
    delete p_weights_;
    free(grads_weights_);
    free(grads_data_);
}

void Fc2d_test::AddInput(ITensor *input)
{
    this->p_input_ = dynamic_cast<Tensor4d*>(input);
}

ITensor *Fc2d_test::LayerInit()
{
    length_ = p_input_->C() * p_input_->H() * p_input_->W();
    if(this->p_weights_ == nullptr)
    {
        /*
         * 这里需要借鉴lenet的代码，因为对于全连接网络的参数，只需要是(batchsize, output_channel*output_height,*output_width)
         * lenet中是使用vector来实现的。
         * 此处代码需重新规划，但是难度不大，可以使用cublas实现。
         */
        this->p_weights_ = new Tensor4d(K_, p_input_->C(), p_input_->H(), p_input_->W());
        //SetWeights(0.01f);
        p_weights_->Randomize();
        p_weights_->PrintK(10);
    }
    checkCudaError(cublasCreate(&cublasHandle_));
    if(this->p_output_ == nullptr)
    {
        p_output_ = new Tensor4d(p_input_->N(), K_, 1, 1);
    }
    std::cout << "Input Tensor shape: ";
    p_input_->PrintShape();
    std::cout << "Weights shape: ";
    p_weights_->PrintShape();
    std::cout << "Output Tensor shape: ";
    p_output_->PrintShape();
    return p_output_;
}

void Fc2d_test::Forward(bool del)
{
    Tensor4d *out = p_output_;
    checkCudaError(cublasSgemm(cublasHandle_, CUBLAS_OP_T, CUBLAS_OP_N,
                               K_, p_input_->N(), length_,
                               &alpha,
                               p_weights_->GpuPointer(), length_,
                               p_input_->GpuPointer(),   length_,
                               &beta,
                               out->GpuPointer(),        K_
    ));
    out->PrintK(20);
}

float *Fc2d_test::Backward(float *down_grads, bool del)
{
    if (grads_weights_ == nullptr && grads_data_ == nullptr)
    {
        checkCudaError(cudaMalloc(&grads_weights_, sizeof(float) * p_weights_->Size()));
        checkCudaError(cudaMalloc(&grads_data_,    sizeof(float) * p_input_->Size()));
    }
    checkCudaError(cublasSgemm(cublasHandle_, CUBLAS_OP_N, CUBLAS_OP_T,
                               length_, K_, p_input_->N(),
                               &alpha,
                               p_input_->GpuPointer(), length_, 
                               down_grads,             K_,
                               &beta,
                               grads_weights_,         length_
    ));

    float *a = (float*)malloc(sizeof(float) * 10);
    if(a != nullptr)
    {
        checkCudaError(cudaMemcpy(a, grads_weights_, sizeof(float) * 10, cudaMemcpyDeviceToHost));
        std::cout << "fc weights gradients\n";
        for(int i = 0; i < 10; i++)
            std::cout << a[i] << ' ';
        std::cout << "\n";
    }

    checkCudaError(cublasSgemm(cublasHandle_, CUBLAS_OP_N, CUBLAS_OP_N,
                               length_, p_input_->N(), K_,
                               &alpha,
                               p_weights_->GpuPointer(), length_,
                               down_grads,               K_,
                               &beta,
                               grads_data_,              length_
    ));
    std::cout << "The fc grads_data shape is：" << p_input_->N() << '*' << length_ << '=' << p_input_->Size() << "\n";
    
    a = (float *)malloc(sizeof(float) * 10);
    if(a != nullptr)
    {
        checkCudaError(cudaMemcpy(a, grads_data_, sizeof(float) * 10, cudaMemcpyDeviceToHost));
        std::cout << "fc data gradients\n";
        for(int i = 0; i < 10; i++)
            std::cout << a[i] << ' ';
        std::cout << "\n";
        free(a);
    }
    return grads_data_;
}

void Fc2d_test::UpdateWeights()
{
     int size = p_weights_->Size();
     int K = p_weights_->N();
     WUpdate<<<(size + 255) / 256, 256>>>(p_weights_->GpuPointer(), grads_weights_, size, size / K);
}

void Fc2d_test::SetWeights(float data)
{
     p_weights_->SetValue(data);
}
