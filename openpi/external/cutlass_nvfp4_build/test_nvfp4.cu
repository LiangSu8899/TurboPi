#include <iostream>
#include <cuda_runtime.h>

// 检查 SM 版本
int main() {
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    std::cout << "GPU: " << prop.name << std::endl;
    std::cout << "SM: " << prop.major << "." << prop.minor << std::endl;
    
    // SM 11.0 = Thor
    // SM 10.0 = B200
    if (prop.major == 11 && prop.minor == 0) {
        std::cout << "✅ Thor detected (SM 11.0)" << std::endl;
    }
    return 0;
}
