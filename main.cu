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
    LossData[idx * NumLabel + LabelValue] -= 0.01f;
}

int main(void)
{
    size_t width, height;
    log_ok("Reading MNIST data");
    const char kImage_dir[] = "../MNIST_DATA/train-images-idx3-ubyte";
    const char kLabel_dir[] = "../MNIST_DATA/train-labels-idx1-ubyte";
    const char kImg_dir[] = "../MNIST_DATA/t10k-images-idx3-ubyte";
    const char kLab_dir[] = "../MNIST_DATA/t10k-labels-idx1-ubyte";
    const int  kChannel     = 1;
    const int  kBatchsize   = 64;
    size_t train_size = ReadUByteDataset(kImage_dir, kLabel_dir, nullptr, nullptr, width, height);
    size_t test_size  = ReadUByteDataset(kImg_dir, kLab_dir, nullptr, nullptr, width, height);
    std::cout << "Training images size is: " << train_size << "\n";
    std::cout << "Testing images size is: " << test_size << "\n";
    std::vector<uint8_t> train_image(train_size * width * height * kChannel), train_label(train_size);
    std::vector<uint8_t> test_image(test_size * width * height * kChannel), test_label(test_size);
    if(ReadUByteDataset(kImage_dir, kLabel_dir, &train_image[0], &train_label[0], width, height) != train_size)
        return 1;
    if(ReadUByteDataset(kImg_dir, kLab_dir, &test_image[0], &test_label[0], width, height) != test_size)
        return 1;
    std::vector<float> train_image_float(train_image.size()), train_label_float(train_size);
    std::vector<float> test_image_float(test_image.size()), test_label_float(test_size);

    for(size_t i = 0; i < train_size * kChannel * width * height; ++i)
        train_image_float[i] = (float)train_image[i] / 255.0f;
    for(size_t i = 0; i < train_size; ++i)
        train_label_float[i] = (float)train_label[i];
    for(size_t i = 0; i < test_size * kChannel * width * height; ++i)
        test_image_float[i] = (float)test_image[i] / 255.0f;
    for(size_t i = 0; i < test_size; ++i)
        test_label_float[i] = (float)test_label[i];
    log_ok("Data success");

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
    Fc2d_test *fc1      = new Fc2d_test(1024);
    Fc2d_test *fc2      = new Fc2d_test(10);
    Softmax   *soft     = new Softmax();

    Tensor4d *a = new Tensor4d(kBatchsize, kChannel, width, height);
    Tensor4d *b = new Tensor4d(kBatchsize, kChannel, 1,     1);
    a->SetValue(&train_image_float[0], kBatchsize * kChannel * width * height);
    b->SetValue(&train_label_float[0], kBatchsize);

    Tensor4d *test_a = new Tensor4d(kBatchsize, kChannel, width, height);
    //Tensor4d *test_b = new Tensor4d(kBatchsize, kChannel, width, height);
    test_a->SetValue(&test_image_float[0], kBatchsize * kChannel * width * height);
    //test_b->SetValue(&test_label_float[0], kBatchsize * kChannel * width * height);

    input = a;
    a->PrintK(10);

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

    log_info("Fc1 add input");
    fc1->AddInput(pool2_out);
    log_info("Fc1 set weights");
    fc1_out = fc1->LayerInit();
    log_info("Fc1 Forward");
    fc1->Forward(false);

    log_info("Fc2 add input");
    fc2->AddInput(fc1_out);
    log_info("Fc2 set weights");
    fc2_out = fc2->LayerInit();
    log_info("Fc2 Forward");
    fc2->Forward(false);

    log_info("Softmax add input");
    soft->AddInput(fc2_out);
    log_info("Softmax set weights");
    soft_out = soft->LayerInit();
    log_info("Softmac Forward");
    soft->Forward(false);
    log_info("Forward Operation has done!");

    //-------------------------------------
    //this is the backward compution
    log_info("Now begin to Backward");

    float *grads_soft;
    float *grads_fc1,   *grads_fc2;
    float *grads_pool1, *grads_pool2;
    float *grads_conv1, *grads_conv2;

    /*
     * Here comupte the loss data by use cuda kernels
     */
    float scalVal = 1.0 / (kBatchsize * 10);
    cublasHandle_t cublasHandle;
    checkCudaError(cublasCreate(&cublasHandle));
    loss<<<1, 100>>>(b->GpuPointer(), 10, kBatchsize, dynamic_cast<Tensor4d*>(soft_out)->GpuPointer());
    checkCudaError(cublasSscal(cublasHandle, dynamic_cast<Tensor4d*>(soft_out)->Size(), &scalVal, dynamic_cast<Tensor4d*>(soft_out)->GpuPointer(), 1));
    // reduce loss data by cublasSscal function
    log_info("Compute Complete");
    std::cout << "----------------------------\n";

    grads_soft = soft->Backward(dynamic_cast<Tensor4d*>(soft_out)->GpuPointer());
    log_info("Softmax Backward success");

    grads_fc2 = fc2->Backward(grads_soft);
    fc2->UpdateWeights();
    log_info("Fc2 Backward success");
    
    grads_fc1 = fc1->Backward(grads_fc2);
    fc1->UpdateWeights();
    log_info("Fc1 Backward success");

    grads_pool2 = pool2->Backward(grads_fc1);
    log_info("Pooling2 Backward success");

    grads_conv2 = conv2->Backward(grads_pool2);
    conv2->UpdateWeights();
    log_info("Conv2 Backward success");

    grads_pool1 = pool1->Backward(grads_conv2);
    log_info("Pooling1 Backward success");

    grads_conv1 = conv1->Backward(grads_pool1);
    conv1->UpdateWeights();
    log_info("Conv1 Backward success");
    log_info("Backward Operation has done");
    //------------------------------------------------

    for(int iter = 1; iter < 1000; ++iter)
    {
        int imageID = iter % 6000;
        a->SetValue(&train_image_float[imageID * kBatchsize * kChannel * width * height], kBatchsize * kChannel * width * height);
        b->SetValue(&train_label_float[imageID * kBatchsize], kBatchsize);
        input = a;

        log_info("Conv1 add input");
        conv1->AddInput(input);
        log_info("Conv1 Forward");
        conv1->Forward(false);
        
        log_info("Pool1 add input");
        pool1->AddInput(conv1_out);
        log_info("Pool1 Forward");
        pool1->Forward(false);

        log_info("Conv2 add input");
        conv2->AddInput(pool1_out);
        log_info("Conv2 Forward");
        conv2->Forward(false);

        log_info("Pool2 add input");
        pool2->AddInput(conv2_out);
        log_info("Pool2 Forward");
        pool2->Forward(false);

        log_info("Fc1 add input");
        fc1->AddInput(pool2_out);
        log_info("Fc1 Forward");
        fc1->Forward(false);

        log_info("Fc2 add input");
        fc2->AddInput(fc1_out);
        log_info("Fc2 Forward");
        fc2->Forward(false);

        log_info("Softmax add input");
        soft->AddInput(fc2_out);
        log_info("Softmac Forward");
        soft->Forward(false);
        log_info("Forward Operation has done!");
        log_info("Now begin to Backward");

        loss<<<1, 100>>>(b->GpuPointer(), 10, kBatchsize, dynamic_cast<Tensor4d*>(soft_out)->GpuPointer());
        checkCudaError(cublasSscal(cublasHandle, dynamic_cast<Tensor4d*>(soft_out)->Size(), &scalVal, dynamic_cast<Tensor4d*>(soft_out)->GpuPointer(), 1));
        log_info("Compute Complete");
        // The loss function just reduce the current label location by step 1
        grads_soft = soft->Backward(dynamic_cast<Tensor4d*>(soft_out)->GpuPointer(), false);
        log_info("Softmax Backward success");

        grads_fc2 = fc2->Backward(grads_soft, false);
        fc2->UpdateWeights();
        log_info("Fc2 Backward success");
        
        grads_fc1 = fc1->Backward(grads_fc2, false);
        fc1->UpdateWeights();
        log_info("Fc1 Backward success");

        grads_pool2 = pool2->Backward(grads_fc1, false);
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
    }

    // Test operator
    input = test_a;

    log_info("Conv1 add input");
    conv1->AddInput(input);
    log_info("Conv1 Forward");
    conv1->Forward(false);
    
    log_info("Pool1 add input");
    pool1->AddInput(conv1_out);
    log_info("Pool1 Forward");
    pool1->Forward(false);

    log_info("Conv2 add input");
    conv2->AddInput(pool1_out);
    log_info("Conv2 Forward");
    conv2->Forward(false);

    log_info("Pool2 add input");
    pool2->AddInput(conv2_out);
    log_info("Pool2 Forward");
    pool2->Forward(false);

    log_info("Fc1 add input");
    fc1->AddInput(pool2_out);
    log_info("Fc1 Forward");
    fc1->Forward(false);

    log_info("Fc2 add input");
    fc2->AddInput(fc1_out);
    log_info("Fc2 Forward");
    fc2->Forward(false);

    log_info("Softmax add input");
    soft->AddInput(fc2_out);
    log_info("Softmac Forward");
    soft->Forward(false);
    log_info("Forward Operation has done!");
    
    std::vector<float> class_vec(10 * kBatchsize);
    checkCudaError(cudaMemcpy(&class_vec[0], dynamic_cast<Tensor4d*>(soft_out)->GpuPointer(), 10 * kBatchsize, cudaMemcpyDeviceToHost));
    
    float classification_error = 1.0f;
    int num_errors = 0;
    int chosen = 0;
    for (int i = 0; i < kBatchsize; ++i)
    {
        for (int j = 1; j < 10; ++j)
        {
            if (class_vec[chosen] < class_vec[j]) chosen = j;
        }
        if (chosen != test_label_float[i]) 
            ++num_errors;
        chosen = 0;
    }
    
    classification_error = (float)num_errors / (float)kBatchsize;
    std::cout << "Classification result: " << classification_error * 100.0f << "error\n";
    return 0;
}
