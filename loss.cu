#include "loss.h"

__global__ void loss1(float *d_res, float *d_lab, float *d_loss, int size)
{
     int index = threadIdx.x + blockIdx.x * blockDim.x;
     if(index >= size) return;
     float tmp = d_res[index] - d_lab[index];
     d_loss[index] = tmp * tmp;
}
Loss::Loss()
{
     loss_pointer_ = nullptr;
}

Loss::~Loss()
{}

//void Loss1(ITensor const *Result, ITensor const *Label, ITensor *Loss)
void Loss::Loss1(Tensor4d const *result, Tensor4d const *label)
{
    //Tensor4d *result = dynamic_cast<Tensor4d*>(Result);
    //Tensor4d *label  = dynamic_cast<Tensor4d*>(Label);
    //Tensor4d *loss   = dynamic_cast<Tensor4d*>(Loss);
    checkCudaError(cudaMalloc(&loss_pointer_, result->Size()));
    loss1<<<(result->Size() + 255) / 256, 256>>>(result->GpuPointer(), label->GpuPointer(), loss_pointer_, result->Size());
}

float *Loss::LossData()
{
    return loss_pointer_;
}
