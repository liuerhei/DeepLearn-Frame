#include "softmax.h"

Softmax::Softmax(cudnnSoftmaxMode_t mode, cudnnSoftmaxAlgorithm_t algo) : mode_(mode), algo_(algo)
{
    this->alpha        = 1.0f;
    this->beta         = 0.0f;
    this->p_input_     = nullptr;
    this->p_output_    = nullptr;
    this->grads_input_ = nullptr;
}

Softmax::~Softmax()
{
    delete(p_input_);
    delete(p_output_);
    cudaFree(grads_input_);
    std::cout << "Softmax Layer Delete\n";
}

void Softmax::AddInput(ITensor *input)
{
    this->p_input_ = dynamic_cast<Tensor4d*>(input);
}

ITensor *Softmax::LayerInit()
{
    if (this->p_output_ == nullptr)
    {
        p_output_ = new Tensor4d(p_input_->N(), p_input_->C(), p_input_->H(), p_input_->W());
    }
    return p_output_;
}

void Softmax::Forward(bool del)
{
    Tensor4d *out = p_output_;
    checkCudnn(cudnnSoftmaxForward(
        Session::instance().cudnn_handle(), algo_, mode_, 
        &alpha, p_input_->Desc(), p_input_->GpuPointer(), 
        &beta, out->Desc(), out->GpuPointer()
    ));
    //std::cout << "Softmax layer input*****************\n";
    //p_input_->PrintK(10);
    //log_info("Softmax layer output");
    //out->PrintK(10);
}

float *Softmax::Backward(float *grads_down, bool del)
{
    //float *b = (float*)malloc(sizeof(float) * p_input_->Size());
    //checkCudaError(cudaMemcpyAsync(b, grads_down, sizeof(float) * p_input_->Size(), cudaMemcpyDeviceToHost));
    //log_info("softmax input loss");
    //for(int i = 0; i < p_input_->Size(); ++i)
    //   std::cout << b[i] << ' ';
    //std::cout << "\n";
    //free(b);

    if (this->grads_input_ == nullptr)
    {
       checkCudaError(cudaMalloc(&this->grads_input_, sizeof(float) * p_input_->Size()));
    }
    checkCudnn(cudnnSoftmaxBackward(
       Session::instance().cudnn_handle(), algo_, mode_, &alpha, p_output_->Desc(), p_output_->GpuPointer(), 
       p_output_->Desc(), grads_down, &beta, p_input_->Desc(), grads_input_
    ));

    //float *a = (float*)malloc(sizeof(float) * p_input_->Size());
    //checkCudaError(cudaMemcpyAsync(a, grads_input_, sizeof(float) * p_input_->Size(), cudaMemcpyDeviceToHost));
    //log_info("softmax backward data gradients");
    //for(int i = 0; i < p_input_->Size(); ++i)
    //   std::cout << a[i] << ' ';
    //std::cout << "\n";
    //free(a);
    return grads_input_;
}
