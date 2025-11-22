#include <hip/hip_runtime.h>
#include <iostream>

__global__ void vectorAdd(float *a, float *b, float *c, int n) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) {
    c[idx] = a[idx] + b[idx];
  }
}

int main() {
  int deviceCount = 0;
  hipGetDeviceCount(&deviceCount);
  std::cout << "Number of HIP devices: " << deviceCount << std::endl;

  if (deviceCount == 0) {
    std::cout << "No HIP devices found!" << std::endl;
    return 1;
  }

  // Get device properties
  hipDeviceProp_t prop;
  hipGetDeviceProperties(&prop, 0);
  std::cout << "Device: " << prop.name << std::endl;
  std::cout << "Compute Capability: " << prop.major << "." << prop.minor
            << std::endl;
  std::cout << "Compute Units: " << prop.multiProcessorCount << std::endl;

  // Simple vector addition test
  const int N = 1024;
  size_t size = N * sizeof(float);

  float *h_a = new float[N];
  float *h_b = new float[N];
  float *h_c = new float[N];

  // Initialize host arrays
  for (int i = 0; i < N; i++) {
    h_a[i] = i;
    h_b[i] = i * 2;
  }

  // Allocate device memory
  float *d_a, *d_b, *d_c;
  hipMalloc(&d_a, size);
  hipMalloc(&d_b, size);
  hipMalloc(&d_c, size);

  // Copy data to device
  hipMemcpy(d_a, h_a, size, hipMemcpyHostToDevice);
  hipMemcpy(d_b, h_b, size, hipMemcpyHostToDevice);

  // Launch kernel
  int threadsPerBlock = 256;
  int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
  hipLaunchKernelGGL(vectorAdd, dim3(blocksPerGrid), dim3(threadsPerBlock), 0,
                     0, d_a, d_b, d_c, N);

  // Copy result back to host
  hipMemcpy(h_c, d_c, size, hipMemcpyDeviceToHost);

  // Verify results
  bool success = true;
  for (int i = 0; i < N; i++) {
    if (h_c[i] != h_a[i] + h_b[i]) {
      success = false;
      std::cout << "Error at index " << i << ": " << h_c[i]
                << " != " << (h_a[i] + h_b[i]) << std::endl;
      break;
    }
  }

  if (success) {
    std::cout << "\n✓ Vector addition test PASSED!" << std::endl;
  } else {
    std::cout << "\n✗ Vector addition test FAILED!" << std::endl;
  }

  // Cleanup
  hipFree(d_a);
  hipFree(d_b);
  hipFree(d_c);
  delete[] h_a;
  delete[] h_b;
  delete[] h_c;

  return success ? 0 : 1;
}
