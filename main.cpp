#include "tensor/tensor4d.h"
#include "session.h"
#include "operator/conv2d.h"
#include "operator/pooling2d.h"
#include <iostream>


int main(void)
{
    checkCudaError(cudaSetDevice(0));

    int N=1, C=1, H=10, W=10;
    int K = 4, R = 3, S = 3;
    ITensor *output = nullptr;

    //Session::instance().add(new Conv2d(K, R, S));
    //Session::instance().add(new Pooling2d(2, 1));

    //Tensor4d *a = new Tensor4d(N, C, H, W);
    //a->set_value(1);
    //Session::instance().set_input(a);
    //Session::instance().run();
    Tensor4d *a = new Tensor4d(N, C, H, W);
    ITensor *input     = nullptr;
    ITensor *conv1_out = nullptr;
    ITensor *pool1_out = nullptr;
    ITensor *conv2_out = nullptr;
    ITensor *pool2_out = nullptr;
    a->set_value(1);
    Conv2d *conv1 = new Conv2d(K, R, S);
    Pooling2d *pool1 = new Pooling2d(2, 1);
    Conv2d *conv2 = new Conv2d(1, R, S);
    Pooling2d *pool2 = new Pooling2d(2, 1);
    
    input = a;
    std::cout << "Conv1 add input\n";
    conv1_out = conv1->add_input(input, false);
    std::cout << "Conv1 set weights\n";
    conv1->set_weights(1);
    std::cout << "Conv1 Forward\n";
    conv1->Forward(false);
    
    std::cout << "Pool1 add input\n";
    pool1_out = pool1->add_input(conv1_out, false);
    std::cout << "Pool1 Forward\n";
    pool1->Forward(false);

    std::cout << "Conv2 add input\n";
    conv2_out = conv2->add_input(pool1_out, false);
    std::cout << "Conv2 set weights\n";
    conv2->set_weights(1);
    std::cout << "Conv2 Forward\n";
    conv2->Forward(false);

    std::cout << "Pool2 add input\n";
    pool2_out = pool2->add_input(conv2_out, false);
    std::cout << "Pool2 Forward\n";
    pool2->Forward(false);
    std::cout << "Have Done\n";

    return 0;
}

