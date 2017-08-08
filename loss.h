#ifndef LOSS_H
#define LOSS_H

#include <iostream>
#include "tensor/tensor4d.h"

class Loss
{
public:
     Loss();
     ~Loss();
     //void Loss1(ITensor const *result, ITensor const *label, ITensor loss);
     void Loss1(Tensor4d const *result, Tensor4d const *label);
     float *LossData();
private:
     float *loss_pointer_;
};
#endif
