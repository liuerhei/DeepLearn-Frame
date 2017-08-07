#ifndef TENSOR4D_H
#define TENSOR4D_H

#include <iostream>
#include <iomanip>
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

    void Randomize();
    void SetValue(float val);
    void PrintK(int k) const;
    void PrintAll() const;
    void PrintShape() const;
    float* GpuPointer() const;
    float* CpuPointer() const;
    float* GpuPointer();
    void SyncToCpu() const;
    void SyncToGpu() const;

    int N() const;
    int C() const;
    int H() const;
    int W() const;
    int Size() const;
    cudnnTensorDescriptor_t Desc() const;

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
