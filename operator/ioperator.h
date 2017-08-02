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
    /*
     * This function is used to add input tensor to the layer
     * And init the p_input_ of every layer
     */
    virtual void AddInput(ITensor*) = 0;

    /*
     * This function is used to init the shape and weights of filter of conv and
     * pooling.
     * When building a model, you call this function just once when you building 
     * the model first
     * Return the output of current layer.
     */
    virtual ITensor *LayerInit() = 0;

    /*
     * This function is used to compute the output of current layer.
     * And the paramater decides whether to delete the unuseful variables
     */
    virtual void Forward(bool) = 0;

    /*
     * This function is used to compute the gradients of current layer, 
     * and the loss of down layer which is named data 
     * by using cudnnConvolutionBackwardData().
     */
    virtual float *Backward(float*, bool);

    /*
     * update filter weights by calling this function
     * Now the function does not synchronize with the cpu.
     */
    virtual void UpdateWeights();
protected:
    /*
     * This is used to save the shape of output tensor.
     */
    int N_out, C_out, H_out, W_out;
};

#endif // IOPERATOR_H
