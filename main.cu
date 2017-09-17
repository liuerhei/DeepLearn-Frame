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
#include "operator/batchnormalization2d.h"

//__global__ void loss(const float *Label, int NumLabel, int Batchsize, float *LossData)
//{
//    int idx = blockIdx.x  * blockDim.x + threadIdx.x;
//    if(idx > Batchsize) return ;
//    const int LabelValue = static_cast<int>(Label[idx]);
//    LossData[idx * NumLabel + LabelValue] -= 0.1;
//}
//
//__global__ void loss(const float *label, float *loss, int size)
//{
//    int idx = threadIdx.x + blockDim.x * blockIdx.x;
//    if (idx >= size) return;
//    float y  = loss[idx];
//    float y_ = label[idx];
//    y = powf(y_, logf(y)) + (1 - y_) * logf(1 - y);
//    loss[idx] = y;
//    __syncthreads();
//}

__global__ void loss(const float *label, float *data, int size)
{
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    if (idx >= size) return;
    float y  = data[idx];
    float y_ = label[idx];
    y = powf(y - y_, 2);
    data[idx] = y;
    __syncthreads();
}

int main(void)
{
    //size_t width, height;
    //log_ok("Reading MNIST data");
    //const char kImage_dir[] = "../MNIST_DATA/train-images-idx3-ubyte";
    //const char kLabel_dir[] = "../MNIST_DATA/train-labels-idx1-ubyte";
    //const char kImg_dir[] = "../MNIST_DATA/t10k-images-idx3-ubyte";
    //const char kLab_dir[] = "../MNIST_DATA/t10k-labels-idx1-ubyte";
    //const int  kChannel     = 1;
    //const int  kBatchsize   = 64;
    //size_t train_size = ReadUByteDataset(kImage_dir, kLabel_dir, nullptr, nullptr, width, height);
    //size_t test_size  = ReadUByteDataset(kImg_dir, kLab_dir, nullptr, nullptr, width, height);
    //const int kImgsize      = kBatchsize * kChannel * height * width;
    //std::cout << "Training images size is: " << train_size << "\n";
    //std::cout << "Testing images size is: " << test_size << "\n";
    //std::vector<uint8_t> train_image(train_size * width * height * kChannel), train_label(train_size);
    //std::vector<uint8_t> test_image(test_size * width * height * kChannel), test_label(test_size);
    //if(ReadUByteDataset(kImage_dir, kLabel_dir, &train_image[0], &train_label[0], width, height) != train_size)
    //    return 1;
    //if(ReadUByteDataset(kImg_dir, kLab_dir, &test_image[0], &test_label[0], width, height) != test_size)
    //    return 1;
    //std::vector<float> train_image_float(train_image.size()), train_label_float(train_size);
    //std::vector<float> test_image_float(test_image.size()), test_label_float(test_size);

    //for(size_t i = 0; i < train_size * kChannel * width * height; ++i)
    //    train_image_float[i] = (float)train_image[i] / 255.0f;
    //for(size_t i = 0; i < train_size; ++i)
    //    train_label_float[i] = (float)train_label[i];
    //for(size_t i = 0; i < test_size * kChannel * width * height; ++i)
    //    test_image_float[i] = (float)test_image[i] / 255.0f;
    //for(size_t i = 0; i < test_size; ++i)
    //    test_label_float[i] = (float)test_label[i];
    //log_ok("Data success");
    /*
     * The above code is to read the mnist data
     */

    checkCudaError(cudaSetDevice(1));
    Tensor4d *a = new Tensor4d(1, 1, 1, 1);
    Tensor4d *b = new Tensor4d(1, 1, 1, 1);

    Session::instance().AddInput(a);
    Session::instance().AddLayer(new Fc2d(10));
    Session::instance().AddLayer(new Fc2d(1));
    Session::instance().Build();
    for(int iter = 0; iter < 1000; iter++)
    {
        a->Randomize();
        b->Randomize(4);
        Session::instance().Forward();
        Tensor4d *c= dynamic_cast<Tensor4d*>(Session::instance().Output());
        if(iter % 10 == 0)
             c->PrintAll();
        //c->PrintAll();
        loss<<<1, 128>>>(b->GpuPointer(), c->GpuPointer(), 1);
        Session::instance().Backward(c->GpuPointer());
        Session::instance().UpdateWeights(0.01);
    }

    /*
    Tensor4d *a = new Tensor4d(kBatchsize, kChannel, width, height);
    Tensor4d *b = new Tensor4d(kBatchsize, kChannel, 1,     1);
    a->SetValue(&train_image_float[0], kImgsize);
    b->SetValue(&train_label_float[0], kBatchsize);

    Session::instance().AddInput(a);
    Session::instance().AddLayer(new Conv2d(20, 5, 5));
    Session::instance().AddLayer(new Pooling2d(2, 2));
    //Session::instance().AddLayer(new Activation2d());
    Session::instance().AddLayer(new Conv2d(50, 5, 5));
    Session::instance().AddLayer(new Pooling2d(2, 2));
    //Session::instance().AddLayer(new Activation2d());
    Session::instance().AddLayer(new Fc2d(500));
    //Session::instance().AddLayer(new BatchNormalization2d());
    //Session::instance().AddLayer(new Activation2d());
    // Here add the activation layer, and the result of softmax is 0.1
    // Need to fix.
    Session::instance().AddLayer(new Fc2d(10));
    //Session::instance().AddLayer(new Activation2d());
    Session::instance().AddLayer(new Softmax());

    Session::instance().Build();
    //Session::instance().Forward();
    
    for(int iter = 0; iter < 1; iter++)
    {
        int index = iter % (train_size / kBatchsize);
        //a->SetValue(&train_image_float[0], kImgsize);
        //b->SetValue(&train_label_float[0], kBatchsize);
        a->SetValue(&train_image_float[index * kImgsize],   kImgsize);
        b->SetValue(&train_label_float[index * kBatchsize], kBatchsize);
        log_info("read data is");
        a->PrintK(100);
        Session::instance().AddInput(a);
        Session::instance().Forward();
        //std::cout << "Result is " << train_label_float[iter * kBatchsize] << "\n";
        Tensor4d *Final = dynamic_cast<Tensor4d*>(Session::instance().Output());
        //Final->PrintK(100);
        float *dloss_data = Final->GpuPointer();

        float scalVal = 1.0f / kBatchsize;
        cublasHandle_t cublasHandle;
        checkCudaError(cublasCreate(&cublasHandle));
        loss<<<1, 100>>>(b->GpuPointer(), 10, kBatchsize, dloss_data); 
        checkCudaError(cublasSscal(cublasHandle, Final->Size(), &scalVal, dloss_data, 1));
        float  learning_rate = 0.01;
        double lr_gamma      = 0.0001;
        double lr_power      = 0.75; 

        Session::instance().Backward(dloss_data);

        learning_rate = static_cast<float>(learning_rate * pow((1.0 + lr_gamma * iter), (-lr_power)));
        Session::instance().UpdateWeights(learning_rate);
    }
    */
/*
    // Test operator
    float classification_error = 1.0f;
    int num_errors = 0;
    std::vector<float> class_vec(10 * kBatchsize);

    for(int i = 0; i < 10000; ++i)
    {
        int index = i % (train_size / kBatchsize);
        a->SetValue(&test_image_float[index * kImgsize],  kImgsize);
        //checkCudaError(cudaMamcpy(class_vec, output))
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
    }
    classification_error = (float)num_errors / (float)kBatchsize;
    std::cout << "Classification result: " << classification_error * 100.0f << "error\n";
*/
    return 0;
}
