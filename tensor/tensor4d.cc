#include "tensor4d.h"

Tensor4d::Tensor4d(int n, int c, int h, int w)
            : N_(n), C_(c), H_(h), W_(w)
{
    this->size_ = n * c * h * w;
    if(this->size_ == 0)
    {
        // throw std::runtime_error("tensor size should be greater than zero\n");
        throw "tensor size should be greater than zero\n";
    }
    h_data_ = (float*)malloc(this->size_ * sizeof(float));
    checkCudaError(cudaMalloc(&d_data_, this->size_ * sizeof(float)));
    checkCudnn(cudnnCreateTensorDescriptor(&desc_));
    checkCudnn(cudnnSetTensor4dDescriptor(desc_, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, n, c, h, w));
    //std::cout << "Tensor Init success\n";
}

Tensor4d::~Tensor4d()
{
    //std::cout << "Tensor Delete\n";
    checkCudaError(cudaFree(d_data_));
    checkCudnn(cudnnDestroyTensorDescriptor(desc_));
    free(h_data_);
}

Tensor4d::Tensor4d(const Tensor4d &m) 
    : size_(m.size()), N_(m.N()), C_(m.C()), H_(m.H()), W_(m.W())
{
    this->h_data_ = (float*)malloc(this->size_ * sizeof(float));
    checkCudaError(cudaMalloc(&d_data_, this->size_ * sizeof(float)));
    checkCudaError(cudaMemcpy(d_data_, m.gpu_pointer(), this->size_ * sizeof(float), cudaMemcpyDeviceToDevice));
    checkCudnn(cudnnCreateTensorDescriptor(&desc_));
    checkCudnn(cudnnSetTensor4dDescriptor(desc_, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, N_, C_, H_, W_));
}

Tensor4d& Tensor4d::operator=(const Tensor4d &m)
{
    this->size_ = m.size();
    this->N_    = m.N();
    this->C_    = m.C();
    this->H_ = m.H();
    this->W_ = m.W();

    checkCudaError(cudaFree(d_data_));
    checkCudnn(cudnnDestroyTensorDescriptor(desc_));
    free(h_data_);

    this->h_data_ = (float*)malloc(this->size_ * sizeof(float));
    checkCudaError(cudaMalloc(&d_data_, this->size_ * sizeof(float)));
    checkCudaError(cudaMemcpy(d_data_, m.gpu_pointer(), this->size_ * sizeof(float), cudaMemcpyDeviceToDevice));
    checkCudnn(cudnnCreateTensorDescriptor(&desc_));
    checkCudnn(cudnnSetTensor4dDescriptor(desc_, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, N_, C_, H_, W_));
    return *this;
}

bool Tensor4d::operator==(const Tensor4d &m)
{

}

void Tensor4d::randomize()
{
    for(int i = 0; i < this->size_; i++)
    {
        h_data_[i] = (rand() - RAND_MAX / 2) / (10.0 * RAND_MAX);
    }
    this->sync_to_gpu();
}

void Tensor4d::set_value(float val)
{
    for(int i = 0; i < this->size_; ++i)
        h_data_[i] = val;
    this->sync_to_gpu();
}

void Tensor4d::print_k(int count) const
{
    this->sync_to_cpu();
    for(int i = 0; i < H_; ++i)
    {
        for(int j = 0; j < W_; ++j)
              std::cout << h_data_[i * W_ + j] << "\t";
        std::cout << "\n";
    }
}

void Tensor4d::print_all() const
{
    this->print_k(this->size_);
}

void Tensor4d::print_shape() const
{
    std::cout << "shape is " << this->N_ << ' ' << this->C_ << ' ' << this->H_ <<' ' << this->W_ << '\n';
}

float* Tensor4d::gpu_pointer() const
{
    return this->d_data_;
}

float* Tensor4d::cpu_pointer() const
{
    return this->h_data_;
}

float *Tensor4d::gpu_pointer()
{
    return this->d_data_;
}

int Tensor4d::N() const
{
    return this->N_;
}

int Tensor4d::C() const
{
    return this->C_;
}

int Tensor4d::H() const
{
    return this->H_;
}

int Tensor4d::W() const
{
    return this->W_;
}

int Tensor4d::size() const
{
    return this->size_;
}

void Tensor4d::sync_to_cpu() const
{
    checkCudaError(cudaMemcpy(this->h_data_, this->d_data_, this->size_ * sizeof(float), cudaMemcpyDeviceToHost));
}

void Tensor4d::sync_to_gpu() const
{
    checkCudaError(cudaMemcpy(this->d_data_, this->h_data_, this->size_ * sizeof(float), cudaMemcpyHostToDevice));
}

cudnnTensorDescriptor_t Tensor4d::desc() const
{
    return this->desc_;
}
