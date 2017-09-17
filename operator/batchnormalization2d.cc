#include "batchnormalization2d.h"

BatchNormalization2d::BatchNormalization2d(cudnnBatchNormMode_t mode) : mode_(mode)
{
    alpha                     = 1.0f;
    beta                      = 0.0f;
    p_input_                  = nullptr;
    p_output_                 = nullptr;
    p_bnscale_                = nullptr;
    p_bnbias_                 = nullptr;
    savemean_                 = nullptr;
    savevariance_             = nullptr;
    runmean_                  = nullptr;
    runvariance_              = nullptr;
    estimatedMean_            = nullptr;
    estimatedVariance_        = nullptr;
    //epsilon_                  = 1e-3;
    epsilon_                  = CUDNN_BN_MIN_EPSILON;
    exponentialAverageFactor_ = 1e-2;
}

BatchNormalization2d::~BatchNormalization2d()
{
    delete p_input_;
    delete p_output_;
    delete p_bnscale_;
    delete p_bnbias_;
    delete savemean_;
    delete savevariance_;
    delete runmean_;
    delete runvariance_;
    delete estimatedMean_;
    delete estimatedVariance_;
    std::cout << "BatchNormalization Delete\n";
}

void BatchNormalization2d::AddInput(ITensor *input)
{
    this->p_input_ = dynamic_cast<Tensor4d*>(input);
}

ITensor *BatchNormalization2d::LayerInit()
{
    p_output_          = new Tensor4d(p_input_->N(), p_input_->C(), p_input_->H(), p_input_->W());
    p_bnscale_         = new Tensor4d(1, p_input_->C(), p_input_->H(), p_input_->W());
    p_bnbias_          = new Tensor4d(1, p_input_->C(), p_input_->H(), p_input_->W());
    estimatedMean_     = new Tensor4d(1, p_input_->C(), p_input_->H(), p_input_->W());
    estimatedVariance_ = new Tensor4d(1, p_input_->C(), p_input_->H(), p_input_->W());
    runmean_           = new Tensor4d(1, p_input_->C(), 1, 1);
    runvariance_       = new Tensor4d(1, p_input_->C(), 1, 1);
    savemean_          = new Tensor4d(1, p_input_->C(), 1, 1);
    savevariance_      = new Tensor4d(1, p_input_->C(), 1, 1);

    p_bnbias_->SetValue(0);
    p_bnscale_->SetValue(1);
    runmean_->SetValue(0);
    runvariance_->SetValue(0);
    estimatedMean_->Randomize();
    estimatedVariance_->Randomize();
    return p_output_;
}

void BatchNormalization2d::Inference(bool del)
{
    checkCudnn(cudnnBatchNormalizationForwardInference(
        Session::instance().cudnn_handle(), mode_, &alpha, &beta,
        p_input_->Desc(), p_input_->GpuPointer(), p_output_->Desc(), p_output_->GpuPointer(),
        p_bnbias_->Desc(), p_bnscale_->GpuPointer(), p_bnbias_->GpuPointer(), 
        estimatedMean_->GpuPointer(), estimatedVariance_->GpuPointer(), epsilon_
    ));
}

void BatchNormalization2d::Forward(bool del)
{
    log_info("Batchnormalization input");
    p_input_->PrintK(100);
    checkCudnn(cudnnBatchNormalizationForwardTraining(
        Session::instance().cudnn_handle(), mode_, &alpha, &beta,
        p_input_->Desc(), p_input_->GpuPointer(), p_output_->Desc(), p_output_->GpuPointer(), 
        p_bnbias_->Desc(), p_bnscale_->GpuPointer(), p_bnbias_->GpuPointer(),
        exponentialAverageFactor_, runmean_, runvariance_, epsilon_, savemean_, savevariance_
    ));
    /*
     * TODO
     * Here have bug.
     */
    log_info("Batchnormalization mean");
    savemean_->PrintK(100);
    log_info("Batchnormalization variance");
    savevariance_->PrintK(100);
}

float *BatchNormalization2d::Backward(float *grads_down, bool del)
{
    //float *grads_data_ = nullptr;
    //checkCudaError(cudaMalloc(&grads_data_, sizeof(float) * p_input_->Size()));
    //checkCudnn(cudnnBatchNormalizationBackward(
    //    Session::instance().cudnn_handle(), mode_, ))
    return grads_down;
}
