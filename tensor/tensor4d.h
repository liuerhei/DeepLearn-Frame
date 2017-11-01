#ifndef TENSOR4D_H
#define TENSOR4D_H

#include <iostream>
#include <iomanip>
#include "../wheel.h"
#include "itensor.h"
#include "cudnn.h"
#include "math.h"

class Tensor4d : public ITensor
{
public:
    Tensor4d(int n, int c, int h, int w);
    Tensor4d(const Tensor4d&m);
    ~Tensor4d();
    Tensor4d& operator=(const Tensor4d& m);
    bool operator==(const Tensor4d& m);

    void Randomize(float diff = 0.0f);
    void SetValue(float val);
    //void SetValue(float *data, size_t size, bool onehot = false);
    void SetValue(float *data, size_t size);
    void PrintK(int k) const;
    void PrintAll() const;
    void PrintShape() const;
    void SyncToCpu() const;
    void SyncToGpu() const;

    float* GpuPointer() const;
    float* CpuPointer() const;
    //float* GpuPointer();

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
