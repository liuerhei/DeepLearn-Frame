#include "tensor/tensor4d.h"
#include "session.h"
#include "operator/fc2d.h"
#include "operator/conv2d.h"
#include "loss.h"
#include <iostream>

int main(void)
{
    checkCudaError(cudaSetDevice(0));

    int N=1, C=1, H=10, W=10;
    
    Tensor4d *a = new Tensor4d(N, C, H, W);
    Tensor4d *b = new Tensor4d(N, 10, 1, 1);
    float data[100];
    
    ITensor *input   = nullptr;
    ITensor *fc_out  = nullptr;

    Fc2d*fc = new Fc2d(10);
    for (int i = 0; i < 100; ++i)
    {
        data[i] = i / 10;
    }

    a->SetValue(data, 1000);
    a->PrintAll();
    log_ok("*****\n");
    input = a;

    fc->AddInput(input);
    fc_out = fc->LayerInit();
    fc->Forward(false);
    Tensor4d *c = dynamic_cast<Tensor4d*>(fc_out);
    c->PrintAll();
    log_ok("**********************************");
    fc->Backward(c->GpuPointer(), false);
    fc->UpdateWeights(0.001);

    return 0;
}
