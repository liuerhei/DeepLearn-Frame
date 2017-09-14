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
    return this->handle_;
}

void *Session::workspace() const
{
    return this->workspace_;
}

size_t Session::workspace_size() const
{
    return this->workspace_size_;
}

void Session::AddLayer(IOperator *op)
{
    model_.push_front(op);
}

int Session::size()
{
    return model_.size();
}

ITensor* Session::Output() 
{
    return p_output_;
}

void Session::AddInput(Tensor4d *input)
{
    p_input_ = input;
}

void Session::Build()
{
    ITensor *input  = p_input_;
    ITensor *output = nullptr;
    std::deque<IOperator*>::reverse_iterator iter;
    for (iter = model_.rbegin(); iter != model_.rend(); ++iter)
    {
        (*iter)->AddInput(input);
        output = (*iter)->LayerInit();
        input  = output;
    }
    p_output_ = output;
}

void Session::Forward()
// 这里需要重构代码，应该是先建立网络模型，之后独立进行forward和backward
// The model_ is a deque, so the rbegin is the first add layer
{
    ITensor *input  = p_input_;
    ITensor *output = nullptr;
    std::deque<IOperator*>::reverse_iterator iter;
    for (iter = model_.rbegin(); iter != model_.rend(); ++iter)
    {
        (*iter)->AddInput(input);
        output = (*iter)->LayerInit();
        // when use build and delete this code, there will have a bug.
        // TODO
        (*iter)->Forward(false);
        input  = output;
    }
    p_output_ = output;
}

void Session::Backward(float *loss)
{
    float *grads_loss = loss;
    std::deque<IOperator*>::iterator iter;
    for (iter = model_.begin(); iter != model_.end(); ++iter)
    {
        grads_loss = (*iter)->Backward(grads_loss, false);
    }
}

void Session::UpdateWeights(float learning_rate)
{
    std::deque<IOperator*>::reverse_iterator iter;
    for (iter = model_.rbegin(); iter != model_.rend(); ++iter)
    {
        (*iter)->UpdateWeights(learning_rate);
    }
}

Session::~Session()
{
    checkCudnn(cudnnDestroy(handle_));
    checkCudaError(cudaFree(workspace_));
}

Session Session::instance_;
