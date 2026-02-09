// Shim to bridge ABI difference between TRT-LLM and PyTorch 2.10
namespace c10 {
namespace cuda {
// PyTorch provides this with unsigned int
extern void c10_cuda_check_implementation(int err, const char* filename, const char* funcname, unsigned int line, bool include_device_assertions);

// TRT-LLM expects this with int
extern "C" void _ZN3c104cuda29c10_cuda_check_implementationEiPKcS2_ib(int err, const char* filename, const char* funcname, int line, bool include_device_assertions) {
    c10_cuda_check_implementation(err, filename, funcname, static_cast<unsigned int>(line), include_device_assertions);
}
} // namespace cuda
} // namespace c10
