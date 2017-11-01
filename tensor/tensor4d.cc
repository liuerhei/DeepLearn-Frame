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
    : size_(m.Size()), N_(m.N()), C_(m.C()), H_(m.H()), W_(m.W())
{
    this->h_data_ = (float*)malloc(this->size_ * sizeof(float));
    checkCudaError(cudaMalloc(&d_data_, this->size_ * sizeof(float)));
    checkCudaError(cudaMemcpyAsync(d_data_, m.GpuPointer(), this->size_ * sizeof(float), cudaMemcpyDeviceToDevice));
    checkCudnn(cudnnCreateTensorDescriptor(&desc_));
    checkCudnn(cudnnSetTensor4dDescriptor(desc_, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, N_, C_, H_, W_));
}

Tensor4d& Tensor4d::operator=(const Tensor4d &m)
{
    this->size_ = m.Size();
    this->N_    = m.N();
    this->C_    = m.C();
    this->H_ = m.H();
    this->W_ = m.W();

    checkCudaError(cudaFree(d_data_));
    checkCudnn(cudnnDestroyTensorDescriptor(desc_));
    free(h_data_);

    this->h_data_ = (float*)malloc(this->size_ * sizeof(float));
    checkCudaError(cudaMalloc(&d_data_, this->size_ * sizeof(float)));
    checkCudaError(cudaMemcpyAsync(d_data_, m.GpuPointer(), this->size_ * sizeof(float), cudaMemcpyDeviceToDevice));
    checkCudnn(cudnnCreateTensorDescriptor(&desc_));
    checkCudnn(cudnnSetTensor4dDescriptor(desc_, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, N_, C_, H_, W_));
    return *this;
}

bool Tensor4d::operator==(const Tensor4d &m)
{

}

void Tensor4d::Randomize(float diff)
{
    for(int i = 0; i < this->size_; i++)
    {
        h_data_[i] = (rand() - RAND_MAX / 2) / (10.0 * RAND_MAX) + diff;
        //h_data_[i] = (rand() - RAND_MAX / 2) / (sqrt(2.0 / size_) * RAND_MAX) + diff;
        //h_data_[i] = rand() / (100.0 * RAND_MAX) + diff;
        //h_data_[i] = rand() / (10.0 * RAND_MAX) + diff;
    }
    this->SyncToGpu();
}

void Tensor4d::SetValue(float val)
{
    for(int i = 0; i < this->size_; ++i)
        h_data_[i] = val;
    this->SyncToGpu();
}

void Tensor4d::SetValue(float *data, size_t size/*, bool onehot*/)
{
    size = size > size_ ? size_ : size;
    for(size_t i = 0; i < size; ++i)
        h_data_[i] = data[i];
    this->SyncToGpu();
    //if(!onehot)
    //{
    //    size = size > size_ ? size_ : size;
    //    for(size_t i = 0; i < size; ++i)
    //        h_data_[i] = data[i];
    //}
    //size = size > size_ ? size_ : size;
    //for(size_t i = 0; i < size; ++i)
    //{
    //    for(size_t j = 0; j < 10; ++j)
    //    {
    //        h_data_[i * 10 + j] = 0;
    //    }
    //    h_data_[(int)data[i]] = 1;
    //}
    //this->SyncToGpu();
}

void Tensor4d::PrintK(int count) const
{
    this->SyncToCpu();
    count = (count > size_) ? size_ : count;
    for(int i = 0; i < count; ++i)
    {
        std::cout << std::setw(9) << h_data_[i] << "\t";
    }
    std::cout << "\n";
}

void Tensor4d::PrintAll() const
{
    this->PrintK(this->size_);
    //this->SyncToCpu();
    //for(int i = 0; i < N_; ++i)
    //    for(int j = 0; j < C_; ++j)
    //        for(int i = 0; i < H_; ++i)
    //        {
    //            for(int j = 0; j < W_; ++j)
    //                std::cout << std::setw(9) << h_data_[i * W_ + j] << "\t";
    //            std::cout << "\n";
    //        }
}


void Tensor4d::PrintShape() const
{
    std::cout << "shape is " << this->N_ << ' ' << this->C_ << ' ' << this->H_ <<' ' << this->W_ << '\n';
}

float* Tensor4d::GpuPointer() const
{
    return this->d_data_;
}

float* Tensor4d::CpuPointer() const
{
    return this->h_data_;
}

//float *Tensor4d::GpuPointer()
//{
//    return this->d_data_;
//}

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

int Tensor4d::Size() const
{
    return this->size_;
}

void Tensor4d::SyncToCpu() const
{
    checkCudaError(cudaMemcpyAsync(this->h_data_, this->d_data_, this->size_ * sizeof(float), cudaMemcpyDeviceToHost));
}

void Tensor4d::SyncToGpu() const
{
    checkCudaError(cudaMemcpyAsync(this->d_data_, this->h_data_, this->size_ * sizeof(float), cudaMemcpyHostToDevice));
}

cudnnTensorDescriptor_t Tensor4d::Desc() const
{
    return this->desc_;
}
