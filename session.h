#ifndef SESSION_H
#define SESSION_H

#include "wheel.h"
#include "operator/ioperator.h"
#include "tensor/itensor.h"
#include "tensor/tensor4d.h"
#include <queue>

class Session
{
public:
    static Session& instance(void);

    void *workspace() const;
    size_t workspace_size() const;
    void allocate_workspace();
    void update_workspace_size(size_t size);
    cudnnHandle_t cudnn_handle() const;
//    void add(IOperator *op);
//    void run();
    int size();
//    void set_input(Tensor4d *input);

private:
    Session()
    {
        workspace_size_ = 0;
        workspace_ = nullptr;
        have_workspace_ = false;
        checkCudnn(cudnnCreate(&handle_));
        while(!model_.empty())
            model_.pop();
        p_input_ = nullptr;
        p_output_= nullptr;
    }

    ~Session();

    bool have_workspace_;
    cudnnHandle_t handle_;
    void *workspace_;
    size_t workspace_size_;
    static Session instance_;
    std::queue<IOperator*> model_;
    ITensor *p_input_;
    ITensor *p_output_;
};

#endif

