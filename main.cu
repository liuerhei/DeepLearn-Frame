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
    const int kImgsize      = kBatchsize * kChannel * height * width;
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


    Tensor4d *a = new Tensor4d(kBatchsize, kChannel, width, height);
    Tensor4d *b = new Tensor4d(kBatchsize, kChannel, 1,     1);
    a->SetValue(&train_image_float[0], kImgsize);
    b->SetValue(&train_label_float[0], kBatchsize);

    Session::instance().AddInput(a);
    Session::instance().AddLayer(new Conv2d(32, 5, 5));
    Session::instance().AddLayer(new Pooling2d(2, 2));
    Session::instance().AddLayer(new Conv2d(64, 5, 5));
    Session::instance().AddLayer(new Pooling2d(2, 2));
    Session::instance().AddLayer(new Fc2d_test(1024));
    Session::instance().AddLayer(new Fc2d_test(10));
    Session::instance().AddLayer(new Softmax());

    Session::instance().Build();
    Session::instance().Forward();
    //a->SetValue(&train_image_float[1 * kImgsize],   kImgsize);
    //b->SetValue(&train_label_float[1 * kBatchsize], kBatchsize);
    //Session::instance().AddInput(a);
    //Session::instance().Forward();

    for(int iter = 0; iter < 1000; iter++)
    {
        int index = iter % (train_size / kBatchsize);
        a->SetValue(&train_image_float[index * kImgsize],   kImgsize);
        b->SetValue(&train_label_float[index * kBatchsize], kBatchsize);
        Session::instance().AddInput(a);
        Session::instance().Forward();
        std::cout << train_label_float[0] << "\n";
        Tensor4d *Final = dynamic_cast<Tensor4d*>(Session::instance().Output());
        Final->PrintK(10);

        //float scalVal = 1.0 / (kBatchsize * 10);
        //cublasHandle_t cublasHandle;
        //checkCudaError(cublasCreate(&cublasHandle));
        //loss<<<1, 100>>>(b->GpuPointer(), 10, kBatchsize, Final->GpuPointer());
        //checkCudaError(cublasSscal(cublasHandle, Final->Size(), &scalVal, Final->GpuPointer(), 1));
        //float  learning_rate = 0.01;
        //double lr_gamma      = 0.0001;
        //double lr_power      = 0.75; 

        //Session::instance().Backward(Final->GpuPointer());

        ////learning_rate = static_cast<float>(learning_rate * pow((1.0 + lr_gamma * iter), (-lr_power)));
        //Session::instance().UpdateWeights(learning_rate);
    }
/*
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
    */
    return 0;
}
