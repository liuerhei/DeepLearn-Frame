#ifndef SESSION_H
#define SESSION_H

#include "wheel.h"
#include "cublas_v2.h"
#include "operator/ioperator.h"
#include "tensor/itensor.h"
#include "tensor/tensor4d.h"
#include <deque>

class Session
{
public:
    static Session& instance(void);

    void *workspace() const;
    size_t workspace_size() const;
    void allocate_workspace();
    void update_workspace_size(size_t size);
    cudnnHandle_t cudnn_handle() const;
    cublasHandle_t cublas_handle() const;
    void AddLayer(IOperator *op);
    void AddInput(Tensor4d *input);
    void Build();
    void Forward();
    void Backward(float *loss);
    void UpdateWeights(float learning_rate = 0.01);
    ITensor* Output();


private:
    Session()
    {
        workspace_size_ = 0;
        have_workspace_ = false;
        workspace_      = nullptr;
        p_input_        = nullptr;
        p_output_       = nullptr;
        checkCudnn(cudnnCreate(&cudnnHandle_));
        checkCudaError(cublasCreate(&cublasHandle_));
        if(!model_.empty())
             model_.clear();
        if(!output_.empty())
             output_.clear();
    }

    ~Session();

    bool have_workspace_;
    cudnnHandle_t cudnnHandle_;
    cublasHandle_t cublasHandle_;
    void *workspace_;
    size_t workspace_size_;
    static Session instance_;
    std::deque<IOperator*> model_;
    std::deque<ITensor*> output_;
    ITensor *p_input_;
    ITensor *p_output_;
};

#endif

