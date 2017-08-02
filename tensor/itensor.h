#ifndef TENSOR_H
#define TENSOR_H

class ITensor
{
public:
    virtual ~ITensor();
    virtual void PrintShape() const;
};

#endif // ITENSOR_H
