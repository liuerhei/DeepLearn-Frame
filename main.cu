#include <iostream>
#include <vector>
#include "session.h"
#include "tensor/tensor4d.h"
#include "readubyte.h"
#include "operator/conv2d.h"
#include "operator/pooling2d.h"
#include "operator/activation2d.h"
#include "operator/softmax.h"
#include "operator/fc2d.h"

__global__ void loss(const float *Label, int NumLabel, int Batchsize, float *LossData)
{
    int idx = blockIdx.x  * blockDim.x + threadIdx.x;
    if(idx > Batchsize) return ;
    const int LabelValue = static_cast<int>(Label[idx]);
    LossData[idx * NumLabel + LabelValue] -= 1.0f;
}

int main(void)
{
    size_t width, height;
    std::cout << "Reading MNIST data\n";
    const char kImage_dir[] = "../mnist/train-images-idx3-ubyte";
    const char kLabel_dir[] = "../mnist/train-labels-idx1-ubyte";
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
    std::cout << "Data success\n";

    checkCudaError(cudaSetDevice(0));

    ITensor *input      = nullptr;
    ITensor *conv1_out  = nullptr;
    ITensor *pool1_out  = nullptr;
    ITensor *conv2_out  = nullptr;
    ITensor *pool2_out  = nullptr;
    ITensor *fc1_out    = nullptr;
    ITensor *fc2_out    = nullptr;
    ITensor *soft_out   = nullptr;

    Conv2d    *conv1    = new Conv2d(32, 5, 5);
    Pooling2d *pool1    = new Pooling2d(2, 2);
    Conv2d    *conv2    = new Conv2d(64, 5, 5);
    Pooling2d *pool2    = new Pooling2d(2, 2);
    Fc2d      *fc1      = new Fc2d(1024);
    Fc2d      *fc2      = new Fc2d(10);
    Softmax   *soft     = new Softmax();

    Tensor4d *a = new Tensor4d(kBatchsize, kChannel, width, height);
    Tensor4d *b = new Tensor4d(kBatchsize, kChannel, 1,     1);
    a->SetValue(&train_image_float[0], kBatchsize * kChannel * width * height);
    b->SetValue(&train_label_float[0], kBatchsize);
    input = a;

    std::cout << "=====>Conv1 add input\n";
    conv1->AddInput(input);
    std::cout << "=====>Conv1 set weights\n";
    conv1_out = conv1->LayerInit();
    std::cout << "=====>Conv1 Forward\n";
    conv1->Forward(false);
    
    std::cout << "=====>Pool1 add input\n";
    pool1->AddInput(conv1_out);
    std::cout << "=====>Pool1 set weights\n";
    pool1_out = pool1->LayerInit();
    std::cout << "=====>Pool1 Forward\n";
    pool1->Forward(false);

    std::cout << "=====>Conv2 add input\n";
    conv2->AddInput(pool1_out);
    std::cout << "=====>Conv2 set weights\n";
    conv2_out = conv2->LayerInit();
    std::cout << "=====>Conv2 Forward\n";
    conv2->Forward(false);

    std::cout << "=====>Pool2 add input\n";
    pool2->AddInput(conv2_out);
    std::cout << "=====>Pool2 set weights\n";
    pool2_out = pool2->LayerInit();
    std::cout << "=====>Pool2 Forward\n";
    pool2->Forward(false);


    std::cout << "=====>Fc1 add input\n";
    fc1->AddInput(pool2_out);
    std::cout << "=====>Fc1 set weights\n";
    fc1_out = fc1->LayerInit();
    std::cout << "=====>Fc1 Forward\n";
    fc1->Forward(false);

    std::cout << "=====>Fc2 add input\n";
    fc2->AddInput(fc1_out);
    std::cout << "=====>Fc2 set weights\n";
    fc2_out = fc2->LayerInit();
    std::cout << "=====>Fc2 Forward\n";
    fc2->Forward(false);

    std::cout << "=====>Softmax add input\n";
    soft->AddInput(fc2_out);
    std::cout << "=====>Softmax set weights\n";
    soft_out = soft->LayerInit();
    soft_out->PrintShape();
    std::cout << "=====>Softmac Forward\n";
    soft->Forward(false);
    std::cout << "=====>Forward Operation has done!\n";
    std::cout << "=====>Now begin to Backward\n";

    float *grads_soft;
    float *grads_fc1,   *grads_fc2;
    float *grads_pool1, *grads_pool2;
    float *grads_conv1, *grads_conv2;

    std::cout << "-----------------------------\n";
    loss<<<1, 100>>>(b->GpuPointer(), 10, kBatchsize, dynamic_cast<Tensor4d*>(soft_out)->GpuPointer());
    std::cout << "Compute Complete\n";
    // The loss function just reduce the current label location by step 1
    grads_soft = soft->Backward(dynamic_cast<Tensor4d*>(soft_out)->GpuPointer(), false);
    std::cout << "Softmax Backward success\n";

    grads_fc2  = fc2->Backward(grads_soft, false);
    fc2->UpdateWeights();
    std::cout << "Fc2 Backward success\n";
    
    grads_fc1  = fc1->Backward(grads_fc2, false);
    fc1->UpdateWeights();
    std::cout << "Fc1 Backward success\n";

    grads_pool2 = pool2->Backward(grads_fc1, false);
    std::cout << "Pooling2 Backward success\n";

    grads_conv2 = conv2->Backward(grads_pool2, false);
    conv2->UpdateWeights();
    std::cout << "Conv2 Backward success\n";

    grads_pool1 = pool1->Backward(grads_conv2, false);
    std::cout << "Pooling1 Backward success\n";

    grads_conv1 = conv1->Backward(grads_pool1, false);
    conv1->UpdateWeights();
    std::cout << "Conv1 Backward success\n";
    std::cout << "Backward Operation has done\n";

    return 0;
}
