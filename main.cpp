#include "tensor/tensor4d.h"
#include "session.h"
#include "operator/conv2d.h"
#include "operator/pooling2d.h"
#include "math.h"
#include <iostream>


int main(void)
{
    
    checkCudaError(cudaSetDevice(1));

    int N=1, C=1, H=10, W=10;
    int K = 3, R = 3, S = 3;
    ITensor *output = nullptr;
    float *d_conv1_data;
    float *d_conv2_data;

    Tensor4d *a = new Tensor4d(N, C, H, W);
    Tensor4d *b = new Tensor4d(1, 1, 4, 4);
    Tensor4d *c;
    b->SetValue(1);
    ITensor *input     = nullptr;
    ITensor *conv1_out = nullptr;
    ITensor *pool1_out = nullptr;
    ITensor *conv2_out = nullptr;
    ITensor *pool2_out = nullptr;
    //a->randomize();
    Conv2d *conv1 = new Conv2d(K, R, S);
    Pooling2d *pool1 = new Pooling2d(2, 1);
    Conv2d *conv2 = new Conv2d(1, R, S);
    Pooling2d *pool2 = new Pooling2d(2, 1);

    std::cout << "#################################################\n";
    std::cout << "This is the 1 step\n";
    a->Randomize();
    a->PrintAll();
    input = a;
    std::cout << "=====>Conv1 add input\n";
    conv1->AddInput(input);
    std::cout << "=====>Conv1 set weights\n";
    conv1_out = conv1->LayerInit();
    //conv1->set_weights(1);
    std::cout << "=====>Conv1 Forward\n";
    conv1->Forward(false);
    
    std::cout << "=====>Pool1 add input\n";
    pool1->AddInput(conv1_out);
    std::cout << "=====>Pool1 set weights\n";
    pool1_out = pool1->LayerInit();
    std::cout << "=====>Pool1 Forward\n";
    pool1->Forward(false);

    std::cout << "=====>Conv2 add input\n";
    conv2->AddInput(pool1_out);
    std::cout << "=====>Conv2 set weights\n";
    conv2_out = conv2->LayerInit();
    //conv2->set_weights(1);
    std::cout << "=====>Conv2 Forward\n";
    conv2->Forward(false);

    std::cout << "=====>Pool2 add input\n";
    pool2->AddInput(conv2_out);
    std::cout << "=====>Pool2 set weights\n";
    pool2_out = pool2->LayerInit();
    std::cout << "=====>Pool2 Forward\n";
    pool2->Forward(false);
    std::cout << "=====>Have Done\n";

    std::cout << "=====>Now begin to compute the gradients\n";
    c = dynamic_cast<Tensor4d*>(pool2_out);
    c->PrintAll();
    float *c_pointer = c->CpuPointer();
    float *b_pointer = b->CpuPointer();
    float *grads1, *grads2;
    std::cout << "##############################\n";
    for(int i = 0; i < 16; ++i)
    {
        std::cout << c_pointer[i] << ' ' << b_pointer[i] << "\n";
        c_pointer[i] = pow((c_pointer[i] - b_pointer[i]), 2);
    }
    c->SyncToGpu();
    c->PrintAll();
    //这里要考虑池化对输出的影响，形状不一致
    grads1 = conv2->Backward(c->GpuPointer(), false);
    //for(int i = 0; i < 16; ++i)
    //    std::cout << grads1[i] << ' ';
    //std::cout << "\n";
    std::cout << "=====>Backward success\n";
    conv2->UpdateWeights();
    std::cout << "=====>updata success\n";
    grads2 = conv1->Backward(grads1, false);
    conv1->UpdateWeights();
    std::cout << "=====>Finish\n\n";
    for(int i = 2; i < 5; ++i)
    {
        std::cout << "#################################################\n";
        std::cout << "This is the " << i << " step\n";
        a->Randomize();
        input = a;
        std::cout << "=====>Conv1 add input\n";
        conv1->AddInput(input);
        std::cout << "=====>Conv1 Forward\n";
        conv1->Forward(false);
        
        std::cout << "=====>Pool1 add input\n";
        pool1->AddInput(conv1_out);
        std::cout << "=====>Pool1 Forward\n";
        pool1->Forward(false);

        std::cout << "=====>Conv2 add input\n";
        conv2->AddInput(pool1_out);
        //conv2_out = conv2->set_input_shape();
        std::cout << "=====>Conv2 Forward\n";
        conv2->Forward(false);

        std::cout << "=====>Pool2 add input\n";
        pool2->AddInput(conv2_out);
        std::cout << "=====>Pool2 Forward\n";
        pool2->Forward(false);
        std::cout << "=====>Have Done\n";

        std::cout << "=====>Now begin to compute the gradients\n";
        c = dynamic_cast<Tensor4d*>(pool2_out);
        float *c_pointer = c->CpuPointer();
        float *b_pointer = b->CpuPointer();
        float *grads1, *grads2;
        for(int i = 0; i < 16; ++i)
            c_pointer[i] = pow((c_pointer[i] - b_pointer[i]), 2);
        c->SyncToGpu();
        //这里要考虑池化对输出的影响，形状不一致
        grads1 = conv2->Backward(c->GpuPointer(), false);
        std::cout << "=====>Backward success\n";
        conv2->UpdateWeights();
        std::cout << "=====>updata success\n";
        grads2 = conv1->Backward(grads1, false);
        conv1->UpdateWeights();
        std::cout << "=====>Finish\n\n";
    }
    return 0;
}

