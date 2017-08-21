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
    
    Tensor4d *a = new Tensor4d(N, C,  H, W);
    Tensor4d *b = new Tensor4d(N, 1024, 1, 1);
    
    ITensor *input   = nullptr;
    ITensor *fc1_out = nullptr;
    //ITensor *conv1_out = nullptr;

    Fc2d *fc1 = new Fc2d(1024);
    Loss *loss = new Loss();
    //Conv2d *conv1 = new Conv2d(1, 10, 10);
    std::cout << "#############################################\n";
    a->SetValue(0.01);
    b->Randomize();
    input = a;

    std::cout << "===>Fc1 add input\n";
    fc1->AddInput(input);
    std::cout << "===>Fc1 set weights\n";
    fc1_out = fc1->LayerInit();
    std::cout << "===>Fc1 Forward\n";
    fc1->Forward(false);
    std::cout << "Forward success\n";

    loss->Loss1(dynamic_cast<Tensor4d*>(fc1_out), b);
    fc1->Backward(dynamic_cast<Tensor4d*>(fc1_out)->GpuPointer(), false);
    fc1->UpdateWeights();
    std::cout << "Backward success\n";
    
    //std::cout << "===>Conv1 add input\n";
    //conv1->AddInput(input);
    //std::cout << "===>Conv1 set weights\n";
    //conv1_out = conv1->LayerInit();
    //std::cout << "===>Conv1 Forward\n";
    //conv1->Forward(false);
    //std::cout << "Forward success\n";
    //
    //conv1->Backward(dynamic_cast<Tensor4d*>(conv1_out)->GpuPointer(), false);
    //conv1->UpdateWeights();
    //std::cout << "Backward success\n";
    return 0;
}
