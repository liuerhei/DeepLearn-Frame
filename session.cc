#include "session.h"

Session& Session::instance()
{
    return instance_;
}

void Session::update_workspace_size(size_t size)
{
    if(!have_workspace_)
    {
        checkCudaError(cudaMalloc(&workspace_, workspace_size_));
        have_workspace_ = true;
    }
    std::cout << "The last workspace size is " << workspace_size_ << "\n";
    std::cout << "The current workspace size is " << size << "\n";
    if(workspace_size_ < size)
    {
        workspace_size_ = size;
        if(have_workspace_)
        {
            checkCudaError(cudaFree(workspace_));
            checkCudaError(cudaMalloc(&workspace_, workspace_size_));
            std::cout << "The workspace has updated\n";
        }
    }
}

void Session::allocate_workspace()
{
    if(!have_workspace_)
    {
        checkCudaError(cudaMalloc(&workspace_, workspace_size_));
        have_workspace_ = true;
    }
}

cudnnHandle_t Session::cudnn_handle() const
{
    return this->cudnnHandle_;
}

cublasHandle_t Session::cublas_handle() const
{
    return this->cublasHandle_;
}

void *Session::workspace() const
{
    return this->workspace_;
}

size_t Session::workspace_size() const
{
    return this->workspace_size_;
}

void Session::AddInput(Tensor4d *input)
{
    p_input_ = input;
}

void Session::AddLayer(IOperator *op)
{
    model_.push_back(op);
}

void Session::Build()
{
    ITensor *input  = p_input_;
    ITensor *output = nullptr;
    for (IOperator* op : model_)
    {
        op->AddInput(input);
        output = op->LayerInit();
        output_.push_back(output);
        input  = output;
    }
    p_output_ = output;
}

void Session::Forward()
{
    ITensor *input  = p_input_;
    ITensor *output = nullptr;
    for (int i = 0; i < model_.size(); ++i)
    {
        model_.at(i)->AddInput(input);
        model_.at(i)->Forward(false);
        input = output_.at(i);
    }
}

void Session::Backward(float *loss)
{
    float *grads_loss = loss;
    std::deque<IOperator*>::reverse_iterator iter;
    for (iter = model_.rbegin(); iter != model_.rend(); ++iter)
    {
        grads_loss = (*iter)->Backward(grads_loss, false);
    }
}

void Session::UpdateWeights(float learning_rate)
{
    std::deque<IOperator*>::iterator iter;
    for (iter = model_.begin(); iter != model_.end(); ++iter)
    {
        (*iter)->UpdateWeights(learning_rate);
    }
}

ITensor *Session::Output()
{
    return p_output_;
}

Session::~Session()
{
    checkCudnn(cudnnDestroy(cudnnHandle_));
    checkCudaError(cublasDestroy(cublasHandle_));
    checkCudaError(cudaFree(workspace_));
}

Session Session::instance_;
