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

int main(void)
{
    size_t width, height;
    std::cout << "Reading MNIST data\n";
    const char kImage_dir[] = "../mnist/train-images-idx3-ubyte";
    const char kLabel_dir[] = "../mnist/train-labels-idx1-ubyte";
    size_t train_size = ReadUByteDataset(kImage_dir, kLabel_dir, nullptr, nullptr, width, height);
    std::cout << "Training images size is: " << train_size << "\n";
    std::vector<uint8_t> train_image(train_size * width * height * 1), train_label(train_size);
    if(ReadUByteDataset(kImage_dir, kLabel_dir, &train_image[0], &train_label[0], width, height) != train_size)
        return 1;
    std::vector<float> train_image_float(train_image.size()), train_label_float(train_size);

    for(size_t i = 0; i < train_size * 1 * width * height; ++i)
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

    //Tensor4d *a = new Tensor4d(train_size, 1, width, height);
    Tensor4d *a = new Tensor4d(10, 1, width, height);
    //a->SetValue(&train_image_float[0], train_size * 1 * width * height);
    a->SetValue(&train_image_float[0], 10 * 1 * width * height);
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
    //conv2->set_weights(1);
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

    return 0;
}
