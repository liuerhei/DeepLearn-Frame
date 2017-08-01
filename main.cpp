#include "tensor/tensor4d.h"
#include "session.h"
#include "operator/conv2d.h"
#include "operator/pooling2d.h"
#include "math.h"
#include <iostream>


int main(void)
{
    checkCudaError(cudaSetDevice(1));

    int N=1, C=1, H=100, W=100;
    int K = 32, R = 3, S = 3;
    ITensor *output = nullptr;
    float *d_conv1_data;
    float *d_conv2_data;

    //Session::instance().add(new Conv2d(K, R, S));
    //Session::instance().add(new Pooling2d(2, 1));

    //Tensor4d *a = new Tensor4d(N, C, H, W);
    //a->set_value(1);
    //Session::instance().set_input(a);
    //Session::instance().run();
    Tensor4d *a = new Tensor4d(N, C, H, W);
    Tensor4d *b = new Tensor4d(1, 1, 94, 94);
    Tensor4d *c;
    b->set_value(1);
    ITensor *input     = nullptr;
    ITensor *conv1_out = nullptr;
    ITensor *pool1_out = nullptr;
    ITensor *conv2_out = nullptr;
    ITensor *pool2_out = nullptr;
    //a->randomize();
    Conv2d *conv1 = new Conv2d(K, R, S);
    Pooling2d *pool1 = new Pooling2d(2, 1);
    Conv2d *conv2 = new Conv2d(1, R, S);
    Pooling2d *pool2 = new Pooling2d(2, 1);

    std::cout << "#################################################\n";
    std::cout << "This is the 1 step\n";
    a->randomize();
    input = a;
    std::cout << "=====>Conv1 add input\n";
    conv1->add_input(input);
    std::cout << "=====>Conv1 set weights\n";
    conv1_out = conv1->set_input_shape();
    //conv1->set_weights(1);
    std::cout << "=====>Conv1 Forward\n";
    conv1->Forward(false);
    
    std::cout << "=====>Pool1 add input\n";
    pool1_out = pool1->add_input(conv1_out, false);
    std::cout << "=====>Pool1 Forward\n";
    pool1->Forward(false);

    std::cout << "=====>Conv2 add input\n";
    conv2->add_input(pool1_out);
    std::cout << "=====>Conv2 set weights\n";
    conv2_out = conv2->set_input_shape();
    //conv2->set_weights(1);
    std::cout << "=====>Conv2 Forward\n";
    conv2->Forward(false);

    std::cout << "=====>Pool2 add input\n";
    pool2_out = pool2->add_input(conv2_out, false);
    std::cout << "=====>Pool2 Forward\n";
    pool2->Forward(false);
    std::cout << "=====>Have Done\n";

    std::cout << "=====>Now begin to compute the gradients\n";
    c = dynamic_cast<Tensor4d*>(pool2_out);
    float *c_pointer = c->cpu_pointer();
    float *b_pointer = b->cpu_pointer();
    float *grads1, *grads2;
    for(int i = 0; i < 8836; ++i)
        c_pointer[i] = sqrt(c_pointer[i] - b_pointer[i]);
    c->sync_to_gpu();
    //这里要考虑池化对输出的影响，形状不一致
    grads1 = conv2->Backward(c->gpu_pointer(), false);
    std::cout << "=====>Backward success\n";
    conv2->update_weights();
    std::cout << "=====>updata success\n";
    grads2 = conv1->Backward(grads1, false);
    conv1->update_weights();
    std::cout << "=====>Finish\n\n";
   
    for(int i = 2; i < 1000; ++i)
    {
        std::cout << "#################################################\n";
        std::cout << "This is the " << i << " step\n";
        a->randomize();
        input = a;
        std::cout << "=====>Conv1 add input\n";
        conv1->add_input(input);
        std::cout << "=====>Conv1 set weights\n";
        //conv1_out = conv1->set_input_shape();
        std::cout << "=====>Conv1 Forward\n";
        conv1->Forward(false);
        
        std::cout << "=====>Pool1 add input\n";
        pool1_out = pool1->add_input(conv1_out, false);
        std::cout << "=====>Pool1 Forward\n";
        pool1->Forward(false);

        std::cout << "=====>Conv2 add input\n";
        conv2->add_input(pool1_out);
        std::cout << "=====>Conv2 set weights\n";
        //conv2_out = conv2->set_input_shape();
        std::cout << "=====>Conv2 Forward\n";
        conv2->Forward(false);

        std::cout << "=====>Pool2 add input\n";
        pool2_out = pool2->add_input(conv2_out, false);
        std::cout << "=====>Pool2 Forward\n";
        pool2->Forward(false);
        std::cout << "=====>Have Done\n";

        std::cout << "=====>Now begin to compute the gradients\n";
        c = dynamic_cast<Tensor4d*>(pool2_out);
        float *c_pointer = c->cpu_pointer();
        float *b_pointer = b->cpu_pointer();
        float *grads1, *grads2;
        for(int i = 0; i < 8836; ++i)
            c_pointer[i] = sqrt(c_pointer[i] - b_pointer[i]);
        c->sync_to_gpu();
        //这里要考虑池化对输出的影响，形状不一致
        grads1 = conv2->Backward(c->gpu_pointer(), false);
        std::cout << "=====>Backward success\n";
        conv2->update_weights();
        std::cout << "=====>updata success\n";
        grads2 = conv1->Backward(grads1, false);
        conv1->update_weights();
        std::cout << "=====>Finish\n\n";
    }
    /*
    input = a;
    std::cout << "=====>Conv1 add input\n";
    conv1_out = conv1->add_input(input, false);
    std::cout << "=====>Conv1 set weights\n";
    conv1->set_weights(1);
    std::cout << "=====>Conv1 Forward\n";
    conv1->Forward(false);
    
    std::cout << "=====>Pool1 add input\n";
    pool1_out = pool1->add_input(conv1_out, false);
    std::cout << "=====>Pool1 Forward\n";
    pool1->Forward(false);

    std::cout << "=====>Conv2 add input\n";
    conv2_out = conv2->add_input(pool1_out, false);
    std::cout << "=====>Conv2 set weights\n";
    conv2->set_weights(1);
    std::cout << "=====>Conv2 Forward\n";
    conv2->Forward(false);

    std::cout << "=====>Pool2 add input\n";
    pool2_out = pool2->add_input(conv2_out, false);
    std::cout << "=====>Pool2 Forward\n";
    pool2->Forward(false);
    std::cout << "=====>Have Done\n";

    std::cout << "=====>Now begin to compute the gradients\n";
    c = dynamic_cast<Tensor4d*>(pool2_out);
    float *c_pointer = c->cpu_pointer();
    float *b_pointer = b->cpu_pointer();
    float *grads1, *grads2;
    for(int i = 0; i < 8836; ++i)
        c_pointer[i] = sqrt(c_pointer[i] - b_pointer[i]);
    c->sync_to_gpu();
    //这里要考虑池化对输出的影响，形状不一致
    grads1 = conv2->Backward(c->gpu_pointer(), false);
    std::cout << "=====>Backward success\n";
    conv2->update_weights();
    std::cout << "=====>updata success\n";
    grads2 = conv1->Backward(grads1, false);
    conv1->update_weights();
    std::cout << "=====>Finish\n";
    */

    return 0;
}

