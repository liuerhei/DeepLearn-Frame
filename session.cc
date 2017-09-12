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

//void Session::add(IOperator *op)
//{
//    model_.push(op);
//}

int Session::size()
{
    return model_.size();
}

//void Session::set_input(Tensor4d *input)
//{
//    p_input_ = input;
//}
//
//void Session::run()
//{
//    ITensor *input = p_input_;
//    ITensor *output = nullptr;
//    while (!model_.empty())
//    {
//        output = model_.front()->add_input(input, false);
//        model_.front()->Forward();
//        delete model_.front();
//        model_.pop();
//        input = output;
//    }
//}

Session::~Session()
{
    checkCudnn(cudnnDestroy(handle_));
    checkCudaError(cudaFree(workspace_));
}

Session Session::instance_;
