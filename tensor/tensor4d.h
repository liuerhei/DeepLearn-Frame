#ifndef TENSOR4D_H
#define TENSOR4D_H

#include <iostream>
#include "../wheel.h"
#include "itensor.h"
#include "cudnn.h"

class Tensor4d : public ITensor
{
public:
    Tensor4d(int n, int c, int h, int w);
    ~Tensor4d();
    Tensor4d(const Tensor4d&m);
    Tensor4d& operator=(const Tensor4d& m);
    bool operator==(const Tensor4d& m);

    void randomize();
    void set_value(float val);
    void print_k(int k) const;
    void print_all() const;
    void print_shape() const;
    const float* gpu_pointer() const;
    const float* cpu_pointer() const;
    float* gpu_pointer();
    void sync_to_cpu() const;
    void sync_to_gpu() const;

    int N() const;
    int C() const;
    int H() const;
    int W() const;
    int size() const;
    cudnnTensorDescriptor_t desc() const;

private:
    float* h_data_;
    float* d_data_;
    int N_;
    int C_;
    int H_;
    int W_;
    int size_;
    cudnnTensorDescriptor_t desc_;
};

#endif
