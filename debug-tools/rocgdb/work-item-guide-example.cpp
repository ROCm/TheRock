/* Example application for learning work-item debugging commands.
 *
 * This simple matrix addition kernel demonstrates how to use the
 * work-item debugging commands in ROCgdb.
 *
 * Compile with:
 *   hipcc -g work-item-guide-example.cpp -o work-item-guide-example
 *
 * Debug with:
 *   rocgdb ./work-item-guide-example
 */

#include <hip/hip_runtime.h>
#include <stdio.h>
#include <stdlib.h>

#define N 128  // Matrix dimension (128x128)
#define BLOCK_SIZE 16  // 16x16 threads per block

/* Simple matrix addition kernel */
__global__ void
matrix_add (float *C, const float *A, const float *B, int n)
{
  // Calculate global row and column from block and thread indices
  int row = hipBlockIdx_y * hipBlockDim_y + hipThreadIdx_y;
  int col = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;

  // Check bounds
  if (row < n && col < n)
    {
      int idx = row * n + col;

      // Store coordinates for debugging
      int my_block_x = hipBlockIdx_x;
      int my_block_y = hipBlockIdx_y;
      int my_thread_x = hipThreadIdx_x;
      int my_thread_y = hipThreadIdx_y;

      // Perform addition
      C[idx] = A[idx] + B[idx];

      // Special marker for debugging specific work-item
      // Set breakpoint here and use: work-item (2,3,0)[8,8,0]
      if (my_block_x == 2 && my_block_y == 3 &&
          my_thread_x == 8 && my_thread_y == 8)
        {
          printf ("Debug marker: work-item (2,3,0)[8,8,0] processed element [%d][%d] = %f\n",
                  row, col, C[idx]);
        }
    }
}

int
main (int argc, char **argv)
{
  float *h_A, *h_B, *h_C;  // Host matrices
  float *d_A, *d_B, *d_C;  // Device matrices
  size_t bytes = N * N * sizeof (float);

  // Allocate host memory
  h_A = (float *) malloc (bytes);
  h_B = (float *) malloc (bytes);
  h_C = (float *) malloc (bytes);

  // Initialize matrices
  for (int i = 0; i < N * N; i++)
    {
      h_A[i] = (float) i;
      h_B[i] = (float) (i * 2);
    }

  // Allocate device memory
  hipMalloc (&d_A, bytes);
  hipMalloc (&d_B, bytes);
  hipMalloc (&d_C, bytes);

  // Copy to device
  hipMemcpy (d_A, h_A, bytes, hipMemcpyHostToDevice);
  hipMemcpy (d_B, h_B, bytes, hipMemcpyHostToDevice);

  // Launch configuration
  dim3 threads_per_block (BLOCK_SIZE, BLOCK_SIZE);
  dim3 blocks_per_grid ((N + BLOCK_SIZE - 1) / BLOCK_SIZE,
                        (N + BLOCK_SIZE - 1) / BLOCK_SIZE);

  printf ("Launching kernel with grid(%d,%d) block(%d,%d)\n",
          blocks_per_grid.x, blocks_per_grid.y,
          threads_per_block.x, threads_per_block.y);

  printf ("\nTry these debugging commands in rocgdb:\n");
  printf ("  1. Set breakpoint: break matrix_add\n");
  printf ("  2. Run program: run\n");
  printf ("  3. Select work-item: work-item (2,3,0)[8,8,0]\n");
  printf ("  4. Print coordinates: print my_block_x, my_thread_y\n");
  printf ("  5. List all work-items: info work-items\n");
  printf ("  6. Check convenience vars: print $_work_item_block_x\n\n");

  // Launch kernel
  hipLaunchKernelGGL (matrix_add,
                      blocks_per_grid,
                      threads_per_block,
                      0, 0,  // shared mem, stream
                      d_C, d_A, d_B, N);

  // Wait for completion
  hipDeviceSynchronize ();

  // Copy result back
  hipMemcpy (h_C, d_C, bytes, hipMemcpyDeviceToHost);

  // Verify a few elements
  printf ("Verification (first few elements):\n");
  for (int i = 0; i < 5; i++)
    {
      float expected = h_A[i] + h_B[i];
      printf ("  C[%d] = %f (expected %f) %s\n",
              i, h_C[i], expected,
              (h_C[i] == expected) ? "✓" : "✗");
    }

  // Verify the special element computed by work-item (2,3,0)[8,8,0]
  // block(2,3) thread[8,8] -> row = 3*16+8 = 56, col = 2*16+8 = 40
  int special_row = 3 * BLOCK_SIZE + 8;
  int special_col = 2 * BLOCK_SIZE + 8;
  int special_idx = special_row * N + special_col;
  float special_expected = h_A[special_idx] + h_B[special_idx];

  printf ("\nSpecial work-item (2,3,0)[8,8,0] computed:\n");
  printf ("  Element [%d][%d] = %f (expected %f) %s\n",
          special_row, special_col,
          h_C[special_idx], special_expected,
          (h_C[special_idx] == special_expected) ? "✓" : "✗");

  // Cleanup
  free (h_A);
  free (h_B);
  free (h_C);
  hipFree (d_A);
  hipFree (d_B);
  hipFree (d_C);

  printf ("\nMatrix addition completed successfully!\n");

  return 0;
}

/*
 * Example ROCgdb Session:
 *
 * $ rocgdb ./work-item-guide-example
 * (gdb) break matrix_add
 * Breakpoint 1 at 0x...: file work-item-guide-example.cpp, line 23.
 *
 * (gdb) run
 * ...
 * Breakpoint 1, matrix_add(...) at work-item-guide-example.cpp:23
 *
 * (gdb) work-item (2,3,0)[8,8,0]
 * [Switching to work-item (2,3,0)[8,8,0], wave 42, lane 25]
 *
 * (gdb) print my_block_x, my_block_y
 * $1 = 2
 * $2 = 3
 *
 * (gdb) print my_thread_x, my_thread_y
 * $3 = 8
 * $4 = 8
 *
 * (gdb) print row, col
 * $5 = 56
 * $6 = 40
 *
 * (gdb) print $_work_item_block_x
 * $7 = 2
 *
 * (gdb) print $_work_item_global_id
 * $8 = 2248
 *
 * (gdb) info work-items -block 2,3,0
 * Wave  Lane  State  Block      Thread     Global-ID
 * ...
 * *42    25    A      (2,3,0)    [8,8,0]    2248
 * ...
 *
 * (gdb) continue
 * ...
 */
