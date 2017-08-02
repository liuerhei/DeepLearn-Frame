#ifndef FILTER4D_H
#define FILTER4D_H

#include "../wheel.h"
#include "itensor.h"
#include "cudnn.h"

class Filter4d : public ITensor
{
public:
    Filter4d(int k, int c, int r, int s);
    ~Filter4d();

    Filter4d(const Filter4d& m);
    Filter4d& operator=(const Filter4d& m);
    bool operator==(const Filter4d& m);

    void Randomize();
    void SetValue(float val);
    void PrintK(int count) const;
    void PrintAll() const;
    void PrintShape() const;
    float* GpuPointer() const;
    float* CpuPointer() const;
    float* GpuPointer();
    void SyncToCpu() const;
    void SyncToGpu() const;

    int K() const;
    int C() const;
    int R() const;
    int S() const;
    int Size() const;
    cudnnFilterDescriptor_t Desc() const;
    
private:
    float *h_data_;
    float *d_data_;
    int K_;
    int C_;
    int R_;
    int S_;
    int size_;
    cudnnFilterDescriptor_t desc_;
};

#endif
