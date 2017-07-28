#include <iostream>
#include <sstream>

#define FatalError(s) do{                                              \
    std::stringstream _where, _message;                                \
    _where << __FILE__ << ':' << __LINE__;                             \
    _message << std::string(s) + "\n" << __FILE__ << ':' << __LINE__;  \
    std::cerr << _message.str() << "\nAborting...\n";                  \
    cudaDeviceReset();                                                 \
    exit(1);                                                           \
}while(0)

#define checkCudnn(status) do{                                        \
    std::stringstream _error;                                         \
    if(status != CUDNN_STATUS_SUCCESS){                               \
        _error << "CUDNN failure: " << cudnnGetErrorString(status);   \
        FatalError(_error.str());                                     \
    }                                                                 \
}while(0)

#define checkCudaError(status) do {                                    \
    std::stringstream _error;                                          \
    if (status != 0) {                                                 \
      _error << "Cuda failure: " << status;                            \
      FatalError(_error.str());                                        \
    }                                                                  \
} while(0)


