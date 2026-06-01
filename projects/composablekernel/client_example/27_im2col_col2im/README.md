# Client Example: im2col and col2im Transformations

## Theory

This client example demonstrates **im2col (image-to-column) and col2im (column-to-image) transformations**. These operations are used to convert image data into a matrix form suitable for GEMM-based convolution and reconstruct images from column representations.

**Mathematical Formulation:**
- **im2col**: Rearranges image blocks into columns, mapping a 3D/4D tensor to a 2D matrix.
- **col2im**: Reverses the process, mapping a 2D matrix back to an image tensor.

**Algorithmic Background:**
- im2col is used to lower convolution to matrix multiplication (GEMM).
- col2im is used to reconstruct the original image or feature map from the column representation.
- These transformations are essential for efficient convolution implementations on GPUs.

## How to Run

### Prerequisites

Please follow the instructions in the main [Build Guide](../../README.md#building-ck) section as a prerequisite to building and running this example.

### Build and run
```bash
cd composable_kernel/client_example/27_im2col_col2im
mkdir build && cd build
cmake -DCMAKE_CXX_COMPILER=/opt/rocm/bin/hipcc ..
make -j

# Example run (image to column)
./image_to_column

# Example run (column to image)
./column_to_image
```

## Source Code Structure

### Directory Layout
```
client_example/27_im2col_col2im/
├── image_to_column.cpp         # im2col: image to column transformation
├── column_to_image.cpp         # col2im: column to image transformation
├── CMakeLists.txt              # Build configuration for the example
```

### Key Functions

- **main()** (in each `.cpp`):  
  Sets up input tensors, configures transformation parameters, launches the im2col or col2im kernel, and verifies the result.
- **im2col/col2im kernel invocation**:  
  Uses the Composable Kernel device API to launch the transformation.

---

## Additional Details

- Supports various image and patch sizes.
- Example parameters can be adjusted in the source for different workloads.

---

## Related Examples

- [52_im2col_col2im](../../example/52_im2col_col2im/README.md): im2col/col2im in the main example directory
- [09_convnd_fwd](../../example/09_convnd_fwd/README.md): N-dimensional convolution using im2col

---
[Back to Client Examples](../README.md)
