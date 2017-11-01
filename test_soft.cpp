#include "tensor/tensor4d.h"
#include "session.h"
#include "operator/softmax.h"
#include <iostream>

int main(void)
{
    checkCudaError(cudaSetDevice(0));
    int N=1, C=10, H=1, W=1;
    Tensor4d *a = new Tensor4d(N, C, H, W);
    Tensor4d *b = new Tensor4d(N, C, H, W);
    ITensor *input  = nullptr;
    ITensor *output = nullptr;
    Softmax *soft = new Softmax();
    float data[10] = {0.0340188, -0.0105617, 0.0283099, 0.029844, 0.0411647, -0.0302449, -0.0164777, 0.026823, -0.0222225, 0.005397};
    float data1[10] = {-0.897459, 0.0980705, 0.101958, 0.102114, 0.103277, 0.0961591, 0.0974921, 0.101806, 0.0969336, 0.0996482};
    a->SetValue(data, 10);
    b->SetValue(data1, 10);
    for(double i = 1; i < 2; i *= 10)
    {
        //a->Randomize(i);
        input = a;
        soft->AddInput(input);
        soft->LayerInit();
        soft->Forward(false);
        soft->Backward(b->GpuPointer());
    }
    return 0;
}
