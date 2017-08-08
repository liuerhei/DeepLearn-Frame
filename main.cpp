#include "tensor/tensor4d.h"
#include "session.h"
#include "operator/conv2d.h"
#include "operator/pooling2d.h"
#include "operator/activation2d.h"
#include "operator/softmax.h"
#include "operator/fc2d.h"
#include "math.h"
#include "loss.h"
#include <iostream>


int main(void)
{
    
    checkCudaError(cudaSetDevice(0));

    int N=1, C=1, H=100, W=100;
    int K = 3, R = 3, S = 3;
    ITensor *output = nullptr;
    float *d_conv1_data;
    float *d_conv2_data;

    Tensor4d *a = new Tensor4d(N, C, H, W);
    Tensor4d *b = new Tensor4d(1, 10, 1, 1);
    //b->SetValue(1);
    b->Randomize();
    ITensor *input     = nullptr;
    ITensor *conv1_out = nullptr;
    ITensor *pool1_out = nullptr;
    ITensor *conv2_out = nullptr;
    ITensor *pool2_out = nullptr;
    ITensor *acti_out  = nullptr;
    ITensor *soft_out  = nullptr;
    ITensor *fc1_out    = nullptr;
    ITensor *fc2_out    = nullptr;
    //a->randomize();
    Conv2d *conv1 = new Conv2d(K, R, S);
    Pooling2d *pool1 = new Pooling2d(2, 2);
    Conv2d *conv2 = new Conv2d(1, R, S);
    Pooling2d *pool2 = new Pooling2d(2, 2);
    Activation2d *acti = new Activation2d();
    Softmax *soft = new Softmax();
    Fc2d *fc1 = new Fc2d(100);
    Fc2d *fc2 = new Fc2d(10);
    Loss *loss = new Loss();

    std::cout << "#################################################\n";
    std::cout << "This is the 1 step\n";
    a->Randomize();
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
    
    std::cout << "=====>Pool2 add input\n";
    pool2->AddInput(conv2_out);
    std::cout << "=====>Pool2 set weights\n";
    pool2_out = pool2->LayerInit();
    std::cout << "=====>Pool2 Forward\n";
    pool2->Forward(false);

    std::cout << "=====>Activation add input\n";
    acti->AddInput(pool2_out);
    std::cout << "=====>Activation set weights\n";
    acti_out = acti->LayerInit();
    std::cout << "=====>Activation Forward\n";
    acti->Forward(false);

    std::cout << "=====>Fc1 add input\n";
    fc1->AddInput(acti_out);
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
    std::cout << "=====>Softmac Forward\n";
    soft->Forward(false);

    std::cout << "=====>Have Done\n";

    std::cout << "=====>Now begin to compute the gradients\n";

    float *grads_acti,  *grads_soft;
    float *grads_conv1, *grads_conv2;
    float *grads_pool1, *grads_pool2;
    float *grads_fc1,   *grads_fc2;
    float *h_conv1,     *h_conv2;
    float *h_pool1,     *h_pool2;
    std::cout << "##############################\n";
    loss->Loss1(a, b);
    //for(int i = 0; i < c->Size(); ++i)
    //{
    //    std::cout << c_pointer[i] << ' ' << b_pointer[i] << "\n";
    //    c_pointer[i] = pow((c_pointer[i] - b_pointer[i]), 2);
    //}
    //c->SyncToGpu();

    grads_soft  = soft->Backward(loss->LossData(),false);
    std::cout << "=====>softmax Backward success\n";

    grads_fc2   = fc2->Backward(grads_soft, false);
    std::cout << "=====>fc2 Backward success\n";
    fc2->UpdateWeights();
    std::cout << "=====>fc2 Update success\n";
    
    grads_fc1   = fc1->Backward(grads_fc2,      false);
    std::cout << "=====>fc1 Backward success\n";
    fc1->UpdateWeights();
    std::cout << "=====>fc1 Update success\n";

    grads_acti  = acti->Backward(grads_fc1,     false);
    std::cout << "=====>Activation Backward success\n";

    grads_pool2 = pool2->Backward(grads_acti,   false);
    std::cout << "=====>Pooling2 Backward success\n";

    grads_conv2 = conv2->Backward(grads_pool2,  false);
    std::cout << "=====>Conv2 Backward success\n";
    conv2->UpdateWeights();
    std::cout << "=====>Conv2 Updata success\n";

    grads_pool1 = pool1->Backward(grads_conv2,  false);
    std::cout << "=====>Pooling2 Backward success\n";

    grads_conv1 = conv1->Backward(grads_pool1,  false);
    std::cout << "=====>Conv1 Backward success\n";
    conv1->UpdateWeights();
    std::cout << "=====>Conv1 Updata success\n";
    std::cout << "=====>Finish\n\n";

    /*
    for (int i = 2; i < 3; i++)
    {
        std::cout << "#################################################\n";
        std::cout << "This is the " << i << " step\n";
        a->Randomize();
        input = a;
        std::cout << "=====>Conv1 add input\n";
        conv1->AddInput(input);
        std::cout << "=====>Conv1 Forward\n";
        conv1->Forward(false);
        
        std::cout << "=====>Pool1 add input\n";
        pool1->AddInput(conv1_out);
        std::cout << "=====>Pool1 Forward\n";
        pool1->Forward(false);

        std::cout << "=====>Conv2 add input\n";
        conv2->AddInput(pool1_out);
        std::cout << "=====>Conv2 Forward\n";
        conv2->Forward(false);

        std::cout << "=====>Pool2 add input\n";
        pool2->AddInput(conv2_out);
        std::cout << "=====>Pool2 Forward\n";
        pool2->Forward(false);
        
        std::cout << "=====>Pool2 add input\n";
        pool2->AddInput(conv2_out);
        std::cout << "=====>Pool2 Forward\n";
        pool2->Forward(false);

        std::cout << "=====>Activation add input\n";
        acti->AddInput(pool2_out);
        std::cout << "=====>Activation Forward\n";
        acti->Forward(false);

        std::cout << "=====>Fc1 add input\n";
        fc1->AddInput(acti_out);
        std::cout << "=====>Fc1 Forward\n";
        fc1->Forward(false);

        std::cout << "=====>Fc2 add input\n";
        fc2->AddInput(fc1_out);
        std::cout << "=====>Fc2 Forward\n";
        fc2->Forward(false);
        //std::cout << "=====>Softmax add input\n";
        //soft->AddInput(fc2_out);
        //std::cout << "=====>Softmax set weights\n";
        //soft_out = soft->LayerInit();
        //std::cout << "=====>Softmac Forward\n";
        //soft->Forward(false);
        std::cout << "=====>Have Done\n";
        std::cout << "=====>Now begin to compute the gradients\n";
        c = dynamic_cast<Tensor4d*>(fc2_out);
        std::cout << "##############################\n";
        for(int i = 0; i < c->Size(); ++i)
        {
            std::cout << c_pointer[i] << ' ' << b_pointer[i] << "\n";
            c_pointer[i] = pow((c_pointer[i] - b_pointer[i]), 2);
        }
        c->SyncToGpu();
        
        grads_fc2   = fc2->Backward(c->GpuPointer(), false);
        std::cout << "=====>fc2 Backward success\n";
        fc2->UpdateWeights();
        std::cout << "=====>fc2 Update success\n";
        
        grads_fc1   = fc1->Backward(grads_fc2,      false);
        std::cout << "=====>fc1 Backward success\n";
        fc1->UpdateWeights();
        std::cout << "=====>fc1 Update success\n";

        grads_acti  = acti->Backward(grads_fc1,     false);
        std::cout << "=====>Activation Backward success\n";

        grads_pool2 = pool2->Backward(grads_acti,   false);
        std::cout << "=====>Pooling2 Backward success\n";

        grads_conv2 = conv2->Backward(grads_pool2,  false);
        std::cout << "=====>Conv2 Backward success\n";
        conv2->UpdateWeights();
        std::cout << "=====>Conv2 Updata success\n";

        grads_pool1 = pool1->Backward(grads_conv2,  false);
        std::cout << "=====>Pooling2 Backward success\n";

        grads_conv1 = conv1->Backward(grads_pool1,  false);
        std::cout << "=====>Conv1 Backward success\n";
        conv1->UpdateWeights();
        std::cout << "=====>Conv1 Updata success\n";
        std::cout << "=====>Finish\n\n";
    }
    */

    return 0;
}

