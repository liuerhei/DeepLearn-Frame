#ifndef IOPERATOR_H
#define IOPERATOR_H

#include "../wheel.h"
#include "../tensor/itensor.h"

enum Padding_t
{
    valid,
    same
};

class IOperator
{
public:
    virtual ~IOperator();
    virtual void Forward(bool);
    //virtual ITensor* add_input(ITensor *, bool del);
protected:
    int N_out, C_out, H_out, W_out;
};

#endif // IOPERATOR_H
