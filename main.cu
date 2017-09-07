#include <iostream>
#include <vector>
#include "wheel.h"
#include "session.h"
#include "tensor/tensor4d.h"
#include "readubyte.h"
#include "operator/conv2d.h"
#include "operator/pooling2d.h"
#include "operator/activation2d.h"
#include "operator/softmax.h"
#include "operator/fc2d.h"
#include "operator/fc2d_test.h"

__global__ void loss(const float *Label, int NumLabel, int Batchsize, float *LossData)
{
    int idx = blockIdx.x  * blockDim.x + threadIdx.x;
    if(idx > Batchsize) return ;
    const int LabelValue = static_cast<int>(Label[idx]);
    LossData[idx * NumLabel + LabelValue] -= 1.0f;
}

int main(void)
{
        /*
    size_t width, height;
    log_ok("Reading MNIST data");
    const char kImage_dir[] = "../MNIST_DATA/train-images-idx3-ubyte";
    const char kLabel_dir[] = "../MNIST_DATA/train-labels-idx1-ubyte";
    const int  kChannel     = 1;
    const int  kBatchsize   = 10;
    size_t train_size = ReadUByteDataset(kImage_dir, kLabel_dir, nullptr, nullptr, width, height);
    std::cout << "Training images size is: " << train_size << "\n";
    std::vector<uint8_t> train_image(train_size * width * height * kChannel), train_label(train_size);
    if(ReadUByteDataset(kImage_dir, kLabel_dir, &train_image[0], &train_label[0], width, height) != train_size)
        return 1;
    std::vector<float> train_image_float(train_image.size()), train_label_float(train_size);

    for(size_t i = 0; i < train_size * kChannel * width * height; ++i)
        train_image_float[i] = (float)train_image[i] / 255.0f;
    for(size_t i = 0; i < train_size; ++i)
        train_label_float[i] = (float)train_label[i];
    log_ok("Data success");

    checkCudaError(cudaSetDevice(1));

    ITensor *input      = nullptr;
    ITensor *conv1_out  = nullptr;
    ITensor *pool1_out  = nullptr;
    ITensor *conv2_out  = nullptr;
    ITensor *pool2_out  = nullptr;
    ITensor *fc_out     = nullptr;
    ITensor *soft_out   = nullptr;

    Conv2d    *conv1    = new Conv2d(32, 5, 5);
    Pooling2d *pool1    = new Pooling2d(2, 2);
    Conv2d    *conv2    = new Conv2d(64, 5, 5);
    Pooling2d *pool2    = new Pooling2d(2, 2);
    Fc2d_test *fc       = new Fc2d_test(10);
    //Conv2d    *fc1      = new Conv2d(10, 4, 4);
    //Conv2d    *fc2      = new Conv2d(10, 1, 1);
    Fc2d      *fc1      = new Fc2d(1024);
    Fc2d      *fc2      = new Fc2d(10);
    Softmax   *soft     = new Softmax();

    Tensor4d *a = new Tensor4d(kBatchsize, kChannel, width, height);
    Tensor4d *b = new Tensor4d(kBatchsize, kChannel, 1,     1);
    a->SetValue(&train_image_float[0], kBatchsize * kChannel * width * height);
    b->SetValue(&train_label_float[0], kBatchsize);
    input = a;

    log_info("Conv1 add input");
    conv1->AddInput(input);
    log_info("Conv1 set weights");
    conv1_out = conv1->LayerInit();
    log_info("Conv1 Forward");
    conv1->Forward(false);
    
    log_info("Pool1 add input");
    pool1->AddInput(conv1_out);
    log_info("Pool1 set weights");
    pool1_out = pool1->LayerInit();
    log_info("Pool1 Forward");
    pool1->Forward(false);

    log_info("Conv2 add input");
    conv2->AddInput(pool1_out);
    log_info("Conv2 set weights");
    conv2_out = conv2->LayerInit();
    log_info("Conv2 Forward");
    conv2->Forward(false);

    log_info("Pool2 add input");
    pool2->AddInput(conv2_out);
    log_info("Pool2 set weights");
    pool2_out = pool2->LayerInit();
    log_info("Pool2 Forward");
    pool2->Forward(false);

    log_info("Fc add input");
    fc->AddInput(pool2_out);
    log_info("Fc set weights");
    fc_out = fc->LayerInit();
    log_info("Fc Forward");
    fc->Forward(false);

    log_info("Softmax add input");
    soft->AddInput(fc_out);
    log_info("Softmax set weights");
    soft_out = soft->LayerInit();
    log_info("Softmac Forward");
    soft->Forward(false);
    log_info("Forward Operation has done!");
    log_info("Now begin to Backward");

    float *grads_soft;
    float *grads_fc;
    //float *grads_fc1,   *grads_fc2;
    float *grads_pool1, *grads_pool2;
    float *grads_conv1, *grads_conv2;

    loss<<<1, 100>>>(b->GpuPointer(), 10, kBatchsize, dynamic_cast<Tensor4d*>(soft_out)->GpuPointer());
    log_info("Compute Complete");
    // The loss function just reduce the current label location by step 1
    grads_soft = soft->Backward(dynamic_cast<Tensor4d*>(soft_out)->GpuPointer(), false);
    log_info("Softmax Backward success");

    //grads_fc2  = fc2->Backward(grads_soft, false);
    //fc2->UpdateWeights();
    //log_info("Fc2 Backward success");
    //
    //grads_fc1  = fc1->Backward(grads_fc2, false);
    //fc1->UpdateWeights();
    //log_info("Fc1 Backward success");

    grads_fc = fc->Backward(grads_soft, false);
    fc->UpdateWeights();
    log_info("Fc Backward success");

    grads_pool2 = pool2->Backward(grads_fc, false);
    log_info("Pooling2 Backward success");

    grads_conv2 = conv2->Backward(grads_pool2, false);
    conv2->UpdateWeights();
    log_info("Conv2 Backward success");

    grads_pool1 = pool1->Backward(grads_conv2, false);
    log_info("Pooling1 Backward success");

    grads_conv1 = conv1->Backward(grads_pool1, false);
    conv1->UpdateWeights();
    log_info("Conv1 Backward success");
    log_info("Backward Operation has done");

    */
    Fc2d_test *fc = new Fc2d_test(10);
    Tensor4d *a = new Tensor4d(2, 1, 10, 10);
    ITensor *input = nullptr;

    a->SetValue(1.0f);
    input = a;
    fc->AddInput(input);
    fc->LayerInit();
    fc->Forward(false);
    return 0;
}
