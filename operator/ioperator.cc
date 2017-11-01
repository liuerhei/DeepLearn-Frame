#include "ioperator.h"

IOperator::~IOperator()
{

}

float *IOperator::Backward(float *a, bool del)
{       
     return a;
}

void IOperator::UpdateWeights(float learning_rate)
{

}

void IOperator::FromFile(const char *fileprefix)
{

}
