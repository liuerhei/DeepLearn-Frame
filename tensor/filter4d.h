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

    void randomize();
    void set_value(float val);
    void print_k(int count) const;
    void print_all() const;
    void print_shape() const;
    const float* gpu_pointer() const;
    const float* cpu_pointer() const;
    float* gpu_pointer();
    void sync_to_cpu() const;
    void sync_to_gpu() const;

    int K() const;
    int C() const;
    int R() const;
    int S() const;
    int size() const;
    cudnnFilterDescriptor_t desc() const;
    
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
