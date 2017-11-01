#include <iostream>
#include "wheel.h"
#include "session.h"
#include "tensor/tensor4d.h"
#include "readubyte.h"
#include "operator/conv2d.h"
#include "operator/pooling2d.h"
#include "operator/activation2d.h"
#include "operator/softmax.h"
#include "operator/fc2d.h"
#include "operator/batchnormalization2d.h"

int main(void)
{
    Tensor4d *a = new Tensor4d(1, 1, 28, 28);
    Tensor4d *b = new Tensor4d(1, 1, 1,  1);
    Tensor4d *Final = nullptr;
    
    Session::instance().AddInput(a);
    Session::instance().AddLayer(new Conv2d(20, 5, 5));
    Session::instance().AddLayer(new Pooling2d(2, 2));
    Session::instance().AddLayer(new Conv2d(50, 5, 5));
    Session::instance().AddLayer(new Pooling2d(2, 2));
    Session::instance().AddLayer(new Fc2d(500));
    Session::instance().AddLayer(new Activation2d());
    Session::instance().AddLayer(new Fc2d(10));

    Session::instance().Build();

    a->SetValue(0.5);
    Session::instance().Forward();
    Final = dynamic_cast<Tensor4d*>(Session::instance().Output());
    //Final->PrintAll();

    Tensor4d *loss = new Tensor4d(1, 10, 1, 1);
    loss->SetValue(0.01);
    Session::instance().Backward(loss->GpuPointer());
    Session::instance().UpdateWeights(0.01);
    return 0;
}

