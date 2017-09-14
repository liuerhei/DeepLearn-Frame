# DeepLearn-Frame
This is a 2D depth learning framework by using cuDNN. It will contain the convolution layer, pool layer, activation layer, the whole connection layer, etc., can achieve forward and backward computing
Now the code just has convolution forward, I will still complete it.   --2017-7-28

Now the code support convolution and pooling forward, and the result is right. And also the code support convolution backward.           --2017-07-31

Up to now, the frame support convolution, pooling, activation OP. And also support backward of convolution and pooling.
The next need to do is add bias in convolution.                                                                                          --2017-08-02

There is a problem that we cannot use pooling by stride is 2. The pooling result is correct, but the convolution forward will be error.   --2017-08-06

The above error is due to not allocate workspace for convolution especially when the fully connection computes. By adding allocation to the update_workspace to solve the problem. 
And up to now, the fully connection have done, but the softmax still has some problem.                                                   --2017-08-07
Now the main.cpp is a file to test the result of fc2d. Up to now, when weights is int, the result is right, but when the weights is float, for example 0.01, the result is wrong. Need to find the reason. And the old main.cpp is the main.cpp.back now.                                                        --2017-08-18
Now have a new question, when training 10 images, the softmax output are all 0.1. This need to fix.                                      --2017-08-18

Now the fc2d Backward has a problem, and add the filter backward workspace update. Up to now, the data workspace don't need add.         --2017-08-21

Use cublasSgemm to compute the fully connection layer. Bue there is a problem that when weights is setted 0.01, the result was wrong.    --2017-09-07
Now the fc backward has some problem, the GPU data cannot copy to CPU, and when add fc layer, the backward parameters is wrong. So need to check the backward code.
                                                                                                                                         --2017-09-08
Today, finding the problem of cudaMemcpy, because of the loss function. But the problem has solved automaticly. Need to make sense. And when use cudnn convolution backward, we need to allocated new workspace to satisfied the compution.                                                                      --2017-09-12
The softmax layer, should set the cudnnSoftmaxAlgorithm_t CUDNN_SOFTMAX__ACCURATE, to avoid potential floating point overflows in the softmax evaluation. But the FC layer weights update should deal with.                                                                                                      --2017-09-12
The problem has resoluved by using cublasSscal function to reduce the loss data. There no Learning_rate, so the weights became bigger. But need rebuild code, bucause build the network code too much.                                                                                                           --2017-09-13
Add session to build the framework, but have problem as follow: the final output should return; build framework and forward should devide into peices; And maybe the different layer should re-build.                                                                                                            --2017-09-14
