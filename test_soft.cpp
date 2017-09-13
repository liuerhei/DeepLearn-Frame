#include "tensor/tensor4d.h"
#include "session.h"
#include "operator/softmax.h"
#include <iostream>

int main(void)
{
    checkCudaError(cudaSetDevice(0));
    int N=1, C=1, H=1, W=10;
    Tensor4d *a = new Tensor4d(N, C, H, W);
    ITensor *input  = nullptr;
    ITensor *output = nullptr;
    Softmax *soft = new Softmax();
    
    for(int i = 1; i < 100000; i *= 10)
    {
        a->Randomize(i);
        input = a;
        soft->AddInput(input);
        soft->LayerInit();
        soft->Forward(false);
    }
    return 0;
}
