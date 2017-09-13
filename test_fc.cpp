#include "tensor/tensor4d.h"
#include "session.h"
#include "operator/fc2d_test.h"
#include "operator/conv2d.h"
#include "loss.h"
#include <iostream>

int main(void)
{
    checkCudaError(cudaSetDevice(0));

    int N=1, C=1, H=10, W=10;
    
    Tensor4d *a = new Tensor4d(N, C,  H, W);
    Tensor4d *b = new Tensor4d(N, 1024, 1, 1);
    
    ITensor *input   = nullptr;
    ITensor *fc_out = nullptr;

    Fc2d_test *fc = new Fc2d_test(1024);
    a->Randomize();
    input = a;

    fc->AddInput(input);
    fc_out = fc->LayerInit();
    fc->Forward(false);
    fc->Backward(dynamic_cast<Tensor4d*>(fc_out)->GpuPointer(), false);

    return 0;
}
