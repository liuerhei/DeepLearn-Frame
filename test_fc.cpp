#include "tensor/tensor4d.h"
#include "session.h"
#include "operator/fc2d.h"
#include "operator/conv2d.h"
#include <iostream>

int main(void)
{
    checkCudaError(cudaSetDevice(0));

    int N=1, C=1, H=10, W=10;
    
    Tensor4d *a = new Tensor4d(N, C, H, W);
    
    ITensor *input   = nullptr;
    //ITensor *fc1_out = nullptr;
    ITensor *conv1_out = nullptr;

    //Fc2d *fc1 = new Fc2d(10);
    Conv2d *conv1 = new Conv2d(1, 10, 10);
    std::cout << "#############################################\n";
    a->SetValue(0.01);
    input = a;

    //std::cout << "===>Fc1 add input\n";
    //fc1->AddInput(input);
    //std::cout << "===>Fc1 set weights\n";
    //fc1_out = fc1->LayerInit();
    //std::cout << "===>Fc1 Forward\n";
    //fc1->Forward(false);

    std::cout << "===>Conv1 add input\n";
    conv1->AddInput(input);
    std::cout << "===>Conv1 set weights\n";
    conv1_out = conv1->LayerInit();
    std::cout << "===>Conv1 Forward\n";
    conv1->Forward(false);
    return 0;
}
