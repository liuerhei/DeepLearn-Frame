#include "tensor/tensor4d.h"
#include "session.h"
#include "operator/conv2d.h"
#include <iostream>

int main(void)
{
    checkCudaError(cudaSetDevice(0));

    int N = 1, C = 1, H = 10, W = 10;
    Tensor4d *a = new Tensor4d(N, C, H, W);
    a->Randomize();
    Conv2d *conv = new Conv2d(4, 5, 5);
    
    conv->AddInput(a);
    ITensor *b = conv->LayerInit();
    conv->Forward();

    //Tensor4d *c = dynamic_cast<Tensor4d*>(b);
    //Tensor4d *c = new Tensor4d(1, 1, 16, 16);
    //c->Randomize(0.5);
    //conv->Backward(c->GpuPointer());
    //conv->UpdateWeights(-0.01);
    
    return 0;
}
