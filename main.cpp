#include "tensor/tensor4d.h"
#include "session.h"
#include "operator/conv2d.h"
#include <iostream>


int main(void)
{
    checkCudaError(cudaSetDevice(0));

    int N=1, C=1, H=100, W=100;
    int K = 32, R = 3, S = 3;
    ITensor *output = nullptr;

    //Tensor4d *a = new Tensor4d(N, C, H, W);
    //a->set_value(1.0f);
    //Conv2d *conv1 = new Conv2d(K, R, S);
    //conv1->set_input_shape(K, R, S);
    //conv1->set_weights(1.0f);
    //output = conv1->add_input(a);
    //conv1->forward();
    Session::instance().add(new Conv2d(K, R, S, valid));
    std::cout << "Layer success init\n";

    //std::cout << Session::instance().size() << "\n";

    Tensor4d *a = new Tensor4d(N, C, H, W);
    a->set_value(1);
    
    std::cout << "input tensor seccess init\n";
    Session::instance().set_input(a);
    std::cout << "Set input success\n";
    Session::instance().run();

    return 0;
}
