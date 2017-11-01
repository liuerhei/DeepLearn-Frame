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

__global__ void max(float *matrix, float *a, int size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int t = 0;
    if(idx > size) return;
    for(int i = 1; i < 10; i++)
        if(matrix[idx * 10 + t] < matrix[idx * 10 + i])
            t = i;
    a[idx] = t;
    __syncthreads();
}

__global__ void loss(const float *Label, int NumLabel, int Batchsize, float *LossData)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx > Batchsize) return ;
    const int LabelValue = static_cast<int>(Label[idx]);
    LossData[idx * NumLabel + LabelValue] -= 1;
}

//__global__ void loss(const float *label, float *loss, int size, float *data)
//{
//    int idx = threadIdx.x + blockDim.x * blockIdx.x;
//    if (idx >= size) return;
//    float y  = loss[idx];
//    if (y < 1e-2) y = 1e-2;
//    if (y > 0.9)  y = 0.9;
//    float y_ = label[idx];
//    loss[idx] = (y_ * log10f(y) + (1 - y_) * log10f(1 - y))* (-1);
//    __syncthreads();
//    atomicAdd(data, loss[idx]);
//    __syncthreads();
//    if (idx == 0) *data = *data / size;
//}

//__global__ void loss(const float *label, float *data, int size)
//{
//    int idx = threadIdx.x + blockDim.x * blockIdx.x;
//    if (idx >= size) return;
//    float y  = data[idx];
//    float y_ = label[idx];
//    y = powf(y - y_, 2);
//    data[idx] = y;
//    __syncthreads();
//}

int main(void)
{
    size_t width, height;
    log_ok("Reading MNIST data");
    const char kImage_dir[] = "../MNIST_DATA/train-images-idx3-ubyte";
    const char kLabel_dir[] = "../MNIST_DATA/train-labels-idx1-ubyte";
    const char kImg_dir[]   = "../MNIST_DATA/t10k-images-idx3-ubyte";
    const char kLab_dir[]   = "../MNIST_DATA/t10k-labels-idx1-ubyte";
    const int  kChannel     = 1;
    const int  kBatchsize   = 64;
    size_t train_size       = ReadUByteDataset(kImage_dir, kLabel_dir, nullptr, nullptr, width, height);
    size_t test_size        = ReadUByteDataset(kImg_dir, kLab_dir, nullptr, nullptr, width, height);
    const int kImgsize      = kBatchsize * kChannel * height * width;
    std::cout << "Training images size is: " << train_size << "\n";
    std::cout << "Testing  images size is: " << test_size << "\n";
    std::vector<uint8_t> train_image(train_size * width * height * kChannel), train_label(train_size);
    std::vector<uint8_t> test_image(test_size * width * height * kChannel), test_label(test_size);
    if(ReadUByteDataset(kImage_dir, kLabel_dir, &train_image[0], &train_label[0], width, height) != train_size)
        return 1;
    if(ReadUByteDataset(kImg_dir, kLab_dir, &test_image[0], &test_label[0], width, height) != test_size)
        return 1;
    //std::vector<float> train_image_float(train_image.size()), train_label_float(train_size);
    //std::vector<float> test_image_float(test_image.size()), test_label_float(test_size);
    //std::vector<float> train_image_float(train_image.size()), train_label_int(train_size);
    //std::vector<float> test_image_float(test_image.size()), test_label_int(test_size);

    std::vector<float> train_image_float(train_size * width * height * kChannel), train_label_int(train_size);
    std::vector<float> test_image_float(test_size * width * height * kChannel), test_label_int(test_size);
    for(size_t i = 0; i < train_size * kChannel * width * height; ++i)
        train_image_float[i] = (float)train_image[i] / 255.0f;
    //for(size_t i = 0; i < train_size; ++i)
    //    train_label_float[i] = (float)train_label[i];
    for(size_t i = 0; i < train_size; ++i)
        train_label_int[i] = (int)train_label[i];
    for(size_t i = 0; i < test_size * kChannel * width * height; ++i)
        test_image_float[i] = (float)test_image[i] / 255.0f;
    //for(size_t i = 0; i < test_size; ++i)
    //    test_label_float[i] = (float)test_label[i];
    for(size_t i = 0; i < test_size; ++i)
        test_label_int[i] = (int)test_label[i];
    log_ok("Data success");
    /*
     * The above code is to read the mnist data
     */

    checkCudaError(cudaSetDevice(0));
    Tensor4d *a = new Tensor4d(kBatchsize, kChannel, width, height);
    Tensor4d *b = new Tensor4d(kBatchsize,        1,     1,      1);
    //Tensor4d *c = new Tensor4d(1,                 1,     1,      1);
    Tensor4d *Final = nullptr;
    Softmax *soft = new Softmax();
    //c->SetValue(0);
    //float data[kBatchsize * 10] = {0};

    Session::instance().AddInput(a);
    Session::instance().AddLayer(new Conv2d(20, 5, 5));
    Session::instance().AddLayer(new Pooling2d(2, 2));
    Session::instance().AddLayer(new Conv2d(50, 5, 5));
    Session::instance().AddLayer(new Pooling2d(2, 2));
    Session::instance().AddLayer(new Fc2d(500));
    Session::instance().AddLayer(new Activation2d());
    Session::instance().AddLayer(new Fc2d(10));
    //Session::instance().AddLayer(new Activation2d());
    //Session::instance().AddLayer(new Softmax());

    Session::instance().Build();

    soft->AddInput(Session::instance().Output());
    ITensor *soft_out = soft->LayerInit();
    //dynamic_cast<Tensor4d*>(soft_out)->PrintShape();

    float jnliu = train_size / kBatchsize + 1;
    for(int iter = 0; iter < jnliu; iter++)
    {
        int index = iter % (train_size / kBatchsize);
        //for(int i = 0; i < kBatchsize; ++i)
        //{
        //    for(int j = 0; j < 10; ++j)
        //    {
        //         data[i * 10 + j] = 0;
        //    }
        //    int v = train_label_int[index * kBatchsize + i];
        //    //int a = train_label_int[i];
        //    data[i * 10 + v] = 1;
        //}
        a->SetValue(&train_image_float[index * kImgsize], kImgsize);
        //a->SetValue(&train_image_float[0], kImgsize);
        b->SetValue(&train_label_int[index * kBatchsize], kBatchsize);
        Session::instance().Forward();
        //log_info("fc2 output");
        //dynamic_cast<Tensor4d*>(Session::instance().Output())->PrintK(10);
        //Tensor4d *Final = dynamic_cast<Tensor4d*>(Session::instance().Output());
        //soft->AddInput(Session::instance().Output());
        soft->Forward();
        Final = dynamic_cast<Tensor4d*>(soft_out);
        //log_info("softmax output");
        //Final->PrintK(10);

        float scalVal = 1.0f / kBatchsize;
        loss<<<1, kBatchsize>>>(b->GpuPointer(), 10, kBatchsize, Final->GpuPointer());
        //loss<<<1, kBatchsize * 10>>>(b->GpuPointer(), Final->GpuPointer(), 1000, c->GpuPointer());
        //c->PrintAll();
        //c->SetValue(0);

        //checkCudaError(cublasSscal(cublasHandle, Final->Size(), &scalVal, Final->GpuPointer(), 1));
        checkCudaError(cublasSscal(Session::instance().cublas_handle(), Final->Size(), &scalVal, Final->GpuPointer(), 1));
        //Final->PrintAll();

        float  learning_rate = 0.01;
        double lr_gamma      = 0.0001;
        double lr_power      = 0.75; 

        Session::instance().Backward(Final->GpuPointer());
        learning_rate = static_cast<float>(learning_rate * pow((1.0 + lr_gamma * iter), (-lr_power)));
        Session::instance().UpdateWeights(learning_rate);
    }
#if 1
    // Test operator
    float classification_error = 1.0f;
    int Total = test_size / kBatchsize + 1;
    int num_errors = 0;
    Tensor4d *res = new Tensor4d(kBatchsize, 1, 1, 1);
    float *p = res->CpuPointer();
    /*
     * //when Batchsize is 1
    float *p = Final->CpuPointer();
    int chosen = 0;
    */

    for(int i = 0; i < Total; ++i)
    {
        int index = i % (test_size / kBatchsize);
        a->SetValue(&test_image_float[index * kImgsize],  kImgsize);
        Session::instance().Forward();
        soft->Forward();
        res->SetValue(0);
        max<<<1, 100>>>(Final->GpuPointer(), res->GpuPointer(), kBatchsize);
        res->SyncToCpu();
        //res->PrintAll();
        
        for (int k = 0; k < kBatchsize; ++k)
             if(p[k] != test_label_int[index*kBatchsize + k])
                     ++num_errors;
        
        /* 
        // when batchsize is 1, the result is right.
        Final->PrintAll();
        chosen = 0;
        for(int j = 1; j < 10; j++)
            if(p[chosen] < p[j]) chosen= j;
        if (chosen != test_label_int[index]) ++num_errors;
        std::cout << "TEST result: " << chosen << "\nLabel      : "<< test_label_int[index] << "\n";
        */
        }
    classification_error = (float)num_errors / 100;
    std::cout << "Classification result: " << classification_error << "% error\n";
    std::cout << "Wrong result: " << num_errors << "img, total size: 10000 img\n";
#endif
    return 0;
}
