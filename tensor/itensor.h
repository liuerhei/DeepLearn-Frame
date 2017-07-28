#ifndef TENSOR_H
#define TENSOR_H

class ITensor
{
public:
    virtual ~ITensor();
    virtual void print_shape() const;
};

#endif // ITENSOR_H
