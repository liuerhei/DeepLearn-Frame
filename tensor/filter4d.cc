#include "filter4d.h"

Filter4d::Filter4d(int k, int c, int r, int s)
            : K_(k), C_(c), R_(r), S_(s)
{
    this->size_ = k * c * r * s;
    if(this->size_ == 0)
    {
        // throw std::runtime_error("Filter size should greater than 0\n");
        throw "Filter size should greater than 0\n";
    }
    h_data_ = (float*)malloc(this->size_ * sizeof(float));
    checkCudaError(cudaMalloc(&d_data_, this->size_ * sizeof(float)));
    checkCudnn(cudnnCreateFilterDescriptor(&desc_));
    checkCudnn(cudnnSetFilter4dDescriptor(desc_, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, K_, C_, R_, S_));
}

Filter4d::~Filter4d()
{
    std::cout << "Filter Delete\n";
    checkCudaError(cudaFree(d_data_));
    checkCudnn(cudnnDestroyFilterDescriptor(desc_));
    free(h_data_);
}

Filter4d::Filter4d(const Filter4d &m) 
    : size_(m.size()), K_(m.K()), C_(m.C()), R_(m.R()), S_(m.S())
{
    this->h_data_ = (float*)malloc(this->size_ * sizeof(float));
    checkCudaError(cudaMalloc(&d_data_, this->size_ * sizeof(float)));
    checkCudaError(cudaMemcpy(d_data_, m.gpu_pointer(), this->size_ * sizeof(float), cudaMemcpyDeviceToDevice));
    checkCudnn(cudnnCreateFilterDescriptor(&desc_));
    checkCudnn(cudnnSetFilter4dDescriptor(desc_, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, K_, C_, R_, S_));
}

Filter4d& Filter4d::operator=(const Filter4d &m)
{
    this->size_ = m.size();
    this->K_    = m.K();
    this->C_    = m.C();
    this->R_ = m.R();
    this->S_ = m.S();

    checkCudaError(cudaFree(d_data_));
    checkCudnn(cudnnDestroyFilterDescriptor(desc_));
    free(h_data_);

    this->h_data_ = (float*)malloc(this->size_ * sizeof(float));
    checkCudaError(cudaMalloc(&d_data_, this->size_ * sizeof(float)));
    checkCudaError(cudaMemcpy(d_data_, m.gpu_pointer(), this->size_ * sizeof(float), cudaMemcpyDeviceToDevice));
    checkCudnn(cudnnCreateFilterDescriptor(&desc_));
    checkCudnn(cudnnSetFilter4dDescriptor(desc_, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, K_, C_, R_, S_));
    return *this;
}

bool Filter4d::operator==(const Filter4d &m)
{

}

void Filter4d::randomize()
{
    for(int i = 0; i < this->size_; i++)
    {
        d_data_[i] = (rand() - RAND_MAX / 2) / (10.0 * RAND_MAX);
    }
    this->sync_to_gpu();
}

void Filter4d::set_value(float val)
{
    for(int i = 0; i < this->size_; ++i)
        h_data_[i] = val;
    this->sync_to_gpu();
}

void Filter4d::print_k(int count) const
{
    this->sync_to_cpu();
    count = count < this->size_ ? count : this->size_;
    for(int i = 0; i < count; ++i)
        std::cout << h_data_[i] << "\t";
    std::cout << "\n";
}

void Filter4d::print_all() const
{
    this->print_k(this->size_);
}

void Filter4d::print_shape() const
{
    std::cout << "shape is " << this->K_ << ' ' << this->C_ << ' ' << this->R_ <<' ' << this->S_ << '\n';
}

const float* Filter4d::gpu_pointer() const
{
    return this->d_data_;
}

const float* Filter4d::cpu_pointer() const
{
    return this->h_data_;
}

float *Filter4d::gpu_pointer()
{
    return this->d_data_;
}

int Filter4d::K() const
{
    return this->K_;
}

int Filter4d::C() const
{
    return this->C_;
}

int Filter4d::R() const
{
    return this->R_;
}

int Filter4d::S() const
{
    return this->S_;
}

int Filter4d::size() const
{
    return this->size_;
}

void Filter4d::sync_to_cpu() const
{
    checkCudaError(cudaMemcpy(h_data_, d_data_, size_ * sizeof(float), cudaMemcpyDeviceToHost));
}

void Filter4d::sync_to_gpu() const
{
    checkCudaError(cudaMemcpy(d_data_, h_data_, size_ * sizeof(float), cudaMemcpyHostToDevice));
}

cudnnFilterDescriptor_t Filter4d::desc() const
{
    return this->desc_;
}
