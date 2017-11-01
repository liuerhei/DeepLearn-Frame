#include"fc2d.h"

//__global__ void WUpdate(float *data, float *grad, int size, int RST, float learning_rate)
//{
//    int idx = threadIdx.x + blockDim.x * blockIdx.x;
//    if (idx >= size) return;
//    data[idx] += grad[idx % RST] * learning_rate;
//    __syncthreads();
//}

__global__ void FileOnes(float *data, int size)
{
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    if (idx > size) return;
    data[idx] = 1.0f;
    __syncthreads();
}
//__global__ void BiasForward(float *data, const float *bias, int size, int c)
//{
//    int idx = threadIdx.x + blockDim.x * blockIdx.x;
//    if (idx >= size) return;
//    data[idx] += bias[idx % c];
//    __syncthreads();
//}

Fc2d::Fc2d(int k) : K_(k)
{
    alpha          = 1.0f;
    beta           = 0.0f;
    p_bias_        = nullptr;
    p_input_       = nullptr;
    p_output_      = nullptr;
    p_weights_     = nullptr;
    //grads_bias_    = nullptr;
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
    /*
     * There need to change the parameters, using gpu pointer instead of cpu pointer
     * To reduce the connection between cpu and gpu.
     */
}

ITensor *Fc2d::LayerInit()
{
    length_ = p_input_->C() * p_input_->H() * p_input_->W();
    if(this->p_weights_ == nullptr)
    {
        this->p_weights_ = new Tensor4d(K_, p_input_->C(), p_input_->H(), p_input_->W());
        p_weights_->Randomize();
        //p_weights_->SetValue(0.01);
    }
    if(this->p_output_ == nullptr)
    {
        p_output_ = new Tensor4d(p_input_->N(), K_, 1, 1);
    }
    if(this->p_bias_ == nullptr)
    {
        p_bias_ = new Tensor4d(1, K_, 1, 1);
        //p_bias_->SetValue(0.01);
        p_bias_->Randomize();
    }
    checkCudaError(cudaMalloc(&onevec, sizeof(float) * p_input_->N()));
    FileOnes<<<(p_input_->N() + 255)/256, 256>>>(onevec, p_input_->N());
    //std::cout << "FC Output pointer " << p_output_ << "\n";
    return p_output_;
}

void Fc2d::Forward(bool del)
{
    Tensor4d *out = p_output_;
    //p_input_->PrintAll();
    checkCudaError(cublasSgemm(Session::instance().cublas_handle(), CUBLAS_OP_T, CUBLAS_OP_N,
                               K_, p_input_->N(),        length_,
                               &alpha,
                               p_weights_->GpuPointer(), length_,
                               p_input_->GpuPointer(),   length_,
                               &beta,
                               out->GpuPointer(),        K_
    ));
    checkCudaError(cublasSgemm(Session::instance().cublas_handle(), CUBLAS_OP_N, CUBLAS_OP_N,
                               K_, p_input_->N(),     1,
                               &alpha,
                               p_bias_->GpuPointer(), K_,
                               onevec,                p_input_->N(),
                               &alpha,
                               out->GpuPointer(),     K_
    ));
    //log_info("Fc Output");
    //out->PrintK(10);
    //BiasForward<<<(p_input_->Size() + 255)/256, 256>>>(p_output_->GpuPointer(), p_bias_->GpuPointer(), p_input_->Size(), K_);
    //log_info("fc forward");
    //p_output_->PrintK(10);
}

float *Fc2d::Backward(float *down_grads, bool del)
{
    //float *a = (float*)malloc(sizeof(float) * p_output_->Size());
    //if(a != nullptr)
    //{
    //    checkCudaError(cudaMemcpy(a, down_grads, sizeof(float) * p_output_->Size(), cudaMemcpyDeviceToHost));
    //    log_info("fc receive gradients");
    //    for(int i = 0; i < p_output_->Size(); i++)
    //         std::cout << a[i] << ' ';
    //    std::cout << "\n";
    //    free(a);
    //}
    if (grads_weights_ == nullptr && grads_data_ == nullptr && grads_bias_ == nullptr)
    {
        checkCudaError(cudaMalloc(&grads_weights_, sizeof(float) * p_weights_->Size()));
        checkCudaError(cudaMalloc(&grads_data_,    sizeof(float) * p_input_->Size()));
        checkCudaError(cudaMalloc(&grads_bias_,    sizeof(float) * p_bias_->Size()));
    }
    checkCudaError(cublasSgemm(Session::instance().cublas_handle(), CUBLAS_OP_N, CUBLAS_OP_T,
                               length_,                K_,          p_input_->N(),
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
                               &beta,
                               grads_bias_, 1
    ));
    checkCudaError(cublasSgemm(Session::instance().cublas_handle(), CUBLAS_OP_N, CUBLAS_OP_N,
                               length_,                  p_input_->N(),   K_,
                               &alpha,
                               p_weights_->GpuPointer(), length_,
                               down_grads,               K_,
                               &beta,
                               grads_data_,              length_
    ));

    //float *a = (float*)malloc(sizeof(float) * p_bias_->Size());
    //if(a != nullptr)
    //{
    //    checkCudaError(cudaMemcpy(a, grads_bias_, sizeof(float) * p_bias_->Size(), cudaMemcpyDeviceToHost));
    //    log_info("fc bias gradients");
    //    for(int i = 0; i < p_bias_->Size(); i++)
    //         std::cout << a[i] << ' ';
    //    std::cout << "\n";
    //    free(a);
    //}

    //float *b = (float*)malloc(sizeof(float) * 10);
    //if(b != nullptr)
    //{
    //    checkCudaError(cudaMemcpy(b, grads_data_, sizeof(float) * 10, cudaMemcpyDeviceToHost));
    //    log_info("fc data gradients");
    //    for(int i = 0; i < 10; i++)
    //        std::cout << b[i] << ' ';
    //    std::cout << "\n";
    //    free(b);
    //}

    //log_info(length_ * K_);
    //float *c = (float*)malloc(sizeof(float) * 10); 
    //if(c != nullptr)
    //{
    //    checkCudaError(cudaMemcpy(c, grads_weights_, sizeof(float) * 100, cudaMemcpyDeviceToHost));
    //    log_info("fc weights gradients");
    //    for(int i = 0; i < 100; i++)
    //        std::cout << c[i] << ' ';
    //    std::cout << "\n";
    //    free(c);
    //}
    return grads_data_;
}

void Fc2d::UpdateWeights(float learning_rate)
{
     //int size = p_weights_->Size();
     //int N = p_weights_->N();
     float rate = -learning_rate;
     //log_info("fc weights before update");
     //p_weights_->PrintK(100);
     //log_info("fc bias beffore update");
     //p_bias_->PrintK(100);
     //WUpdate<<<(size + 255) / 256, 256>>>(p_weights_->GpuPointer(), grads_weights_, size, size / N, rate);
     //log_info(rate);
     checkCudaError(cublasSaxpy(
        Session::instance().cublas_handle(), p_weights_->Size(),
        &rate, grads_weights_, 1, p_weights_->GpuPointer(), 1
     ));
     checkCudaError(cublasSaxpy(
        Session::instance().cublas_handle(), p_bias_->Size(),
        &rate, grads_bias_, 1, p_bias_->GpuPointer(), 1
     ));
     //log_info("fc weights after update");
     //p_weights_->PrintK(10);
     //p_weights_->PrintAll();
     //log_info("fc bias after update");
     //p_bias_->PrintK(10);
     //p_bias_->PrintAll();
}

void Fc2d::SetWeights(float data)
{
     p_weights_->SetValue(data);
}

void Fc2d::FromFile(const char *fileprefix)
{
     std::stringstream ssf, ssbf;
     ssf << fileprefix << ".bin";
     ssbf << fileprefix << ".bias.bin";
     
     FILE *fp = fopen(ssf.str().c_str(), "r");
     if(!fp)
     {
        log_info("cannot open");
        exit(0);
     }
     fread(p_weights_->CpuPointer(), sizeof(float), p_weights_->Size(), fp);
     p_weights_->SyncToGpu();
     fclose(fp);

     fp = fopen(ssbf.str().c_str(), "r");
     if(!fp)
     {
        log_info("cannot open");
        exit(0);
     }
     fread(p_bias_->CpuPointer(), sizeof(float), p_bias_->Size(), fp);
     p_bias_->SyncToGpu();
     fclose(fp);
}
