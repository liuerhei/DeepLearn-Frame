#include "tensor/tensor4d.h"
#include "session.h"
#include "operator/conv2d.h"
#include <iostream>

int main(void)
{
    checkCudaError(cudaSetDevice(0));

    int N = 1, C = 1, H = 20, W = 20;
    Tensor4d *a = new Tensor4d(N, C, H, W);
    a->Randomize();
    Conv2d *conv = new Conv2d(4, 5, 5);
    
    conv->AddInput(a);
    conv->LayerInit();
    conv->Forward();
    
    return 0;
}
