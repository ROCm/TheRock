// Host-only dispatch smoke: does not validate HIP code generation or correctness.
#include <algorithm>
#include <initializer_list>
#include <cassert>
#include <cstdlib>
#include <cstddef>
using rocblas_int=int; using rocblas_stride=long long;
using rocblas_handle=void*;using rocblas_status=int;using hipStream_t=void*;
struct rocblas_float_complex {float r,i;};struct rocblas_double_complex {double r,i;};
struct dim3{dim3(int,int,int){}};
constexpr int rocblas_status_success=0;
int observed_dimension=0;bool observed_pivot=false;
inline void rocblas_get_stream(rocblas_handle,hipStream_t*){}
#define GETF2_OPTIM_NGRP 1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1
#define ROCSOLVER_UNREACHABLE() std::abort()
template<int DIM,typename T,typename...Args> void getf2_small_kernel(Args...){observed_dimension=DIM;observed_pivot=true;}
template<int DIM,typename T,typename...Args> void getf2_npvt_small_kernel(Args...){observed_dimension=DIM;observed_pivot=false;}
#define ROCSOLVER_LAUNCH_KERNEL(kernel, grid, block, lmem, stream, ...) kernel(__VA_ARGS__)
#define ROCSOLVER_SHARD_GETF2_SMALL 1
/** launcher of getf2_small_kernel **/
#ifdef ROCSOLVER_SHARD_GETF2_SMALL
template <int SHARD, typename T, typename I, typename INFO, typename U>
rocblas_status getf2_run_small_shard(
#else
template <typename T, typename I, typename INFO, typename U>
rocblas_status getf2_run_small(
#endif
rocblas_handle handle,
                               const I m,
                               const I n,
                               U A,
                               const rocblas_stride shiftA,
                               const I lda,
                               const rocblas_stride strideA,
                               I* ipiv,
                               const rocblas_stride shiftP,
                               const rocblas_stride strideP,
                               INFO* info,
                               const I batch_count,
                               const bool pivot,
                               const I offset,
                               I* permut_idx,
                               const rocblas_stride stride)
{
#ifdef ROCSOLVER_SHARD_GETF2_SMALL
#define RUN_LUFACT_SMALL(DIM) \
    if constexpr(((DIM) - 1) / 16 == SHARD) { \
        if(pivot) \
            ROCSOLVER_LAUNCH_KERNEL((getf2_small_kernel<DIM, T>), grid, block, lmemsize, stream, m, A, \
                                    shiftA, lda, strideA, ipiv, shiftP, strideP, info, batch_count, \
                                    offset, permut_idx, stride); \
        else \
            ROCSOLVER_LAUNCH_KERNEL((getf2_npvt_small_kernel<DIM, T>), grid, block, lmemsize, stream, \
                                    m, A, shiftA, lda, strideA, info, batch_count, offset); \
    } else { ROCSOLVER_UNREACHABLE(); }
#else
#define RUN_LUFACT_SMALL(DIM)                                                                      \
    if(pivot)                                                                                      \
        ROCSOLVER_LAUNCH_KERNEL((getf2_small_kernel<DIM, T>), grid, block, lmemsize, stream, m, A, \
                                shiftA, lda, strideA, ipiv, shiftP, strideP, info, batch_count,    \
                                offset, permut_idx, stride);                                       \
    else                                                                                           \
        ROCSOLVER_LAUNCH_KERNEL((getf2_npvt_small_kernel<DIM, T>), grid, block, lmemsize, stream,  \
                                m, A, shiftA, lda, strideA, info, batch_count, offset)
#endif

    // determine sizes
    I opval[] = {GETF2_OPTIM_NGRP};
    I ngrp = (batch_count < 2 || m > 32) ? 1 : opval[m - 1];
    I blocks = (batch_count - 1) / ngrp + 1;
    I nthds = m;
    I msize;
    if(pivot)
        msize = std::max(m, n);
    else
        msize = n + 1;

    // prepare kernel launch
    // batch blocks go on grid.x (limit 2^31), not grid.y: grid.y is capped at
    // 65536 on some archs (e.g. gfx1201), which overflows for large batches.
    dim3 grid(blocks, 1, 1);
    dim3 block(nthds, ngrp, 1);
    size_t lmemsize = msize * ngrp * sizeof(T);
    hipStream_t stream;
    rocblas_get_stream(handle, &stream);

    // instantiate cases to make number of columns n known at compile time
    // this should allow loop unrolling.
    // kernel launch
    switch(n)
    {
    case 1: RUN_LUFACT_SMALL(1); break;
    case 2: RUN_LUFACT_SMALL(2); break;
    case 3: RUN_LUFACT_SMALL(3); break;
    case 4: RUN_LUFACT_SMALL(4); break;
    case 5: RUN_LUFACT_SMALL(5); break;
    case 6: RUN_LUFACT_SMALL(6); break;
    case 7: RUN_LUFACT_SMALL(7); break;
    case 8: RUN_LUFACT_SMALL(8); break;
    case 9: RUN_LUFACT_SMALL(9); break;
    case 10: RUN_LUFACT_SMALL(10); break;
    case 11: RUN_LUFACT_SMALL(11); break;
    case 12: RUN_LUFACT_SMALL(12); break;
    case 13: RUN_LUFACT_SMALL(13); break;
    case 14: RUN_LUFACT_SMALL(14); break;
    case 15: RUN_LUFACT_SMALL(15); break;
    case 16: RUN_LUFACT_SMALL(16); break;
    case 17: RUN_LUFACT_SMALL(17); break;
    case 18: RUN_LUFACT_SMALL(18); break;
    case 19: RUN_LUFACT_SMALL(19); break;
    case 20: RUN_LUFACT_SMALL(20); break;
    case 21: RUN_LUFACT_SMALL(21); break;
    case 22: RUN_LUFACT_SMALL(22); break;
    case 23: RUN_LUFACT_SMALL(23); break;
    case 24: RUN_LUFACT_SMALL(24); break;
    case 25: RUN_LUFACT_SMALL(25); break;
    case 26: RUN_LUFACT_SMALL(26); break;
    case 27: RUN_LUFACT_SMALL(27); break;
    case 28: RUN_LUFACT_SMALL(28); break;
    case 29: RUN_LUFACT_SMALL(29); break;
    case 30: RUN_LUFACT_SMALL(30); break;
    case 31: RUN_LUFACT_SMALL(31); break;
    case 32: RUN_LUFACT_SMALL(32); break;
    case 33: RUN_LUFACT_SMALL(33); break;
    case 34: RUN_LUFACT_SMALL(34); break;
    case 35: RUN_LUFACT_SMALL(35); break;
    case 36: RUN_LUFACT_SMALL(36); break;
    case 37: RUN_LUFACT_SMALL(37); break;
    case 38: RUN_LUFACT_SMALL(38); break;
    case 39: RUN_LUFACT_SMALL(39); break;
    case 40: RUN_LUFACT_SMALL(40); break;
    case 41: RUN_LUFACT_SMALL(41); break;
    case 42: RUN_LUFACT_SMALL(42); break;
    case 43: RUN_LUFACT_SMALL(43); break;
    case 44: RUN_LUFACT_SMALL(44); break;
    case 45: RUN_LUFACT_SMALL(45); break;
    case 46: RUN_LUFACT_SMALL(46); break;
    case 47: RUN_LUFACT_SMALL(47); break;
    case 48: RUN_LUFACT_SMALL(48); break;
    case 49: RUN_LUFACT_SMALL(49); break;
    case 50: RUN_LUFACT_SMALL(50); break;
    case 51: RUN_LUFACT_SMALL(51); break;
    case 52: RUN_LUFACT_SMALL(52); break;
    case 53: RUN_LUFACT_SMALL(53); break;
    case 54: RUN_LUFACT_SMALL(54); break;
    case 55: RUN_LUFACT_SMALL(55); break;
    case 56: RUN_LUFACT_SMALL(56); break;
    case 57: RUN_LUFACT_SMALL(57); break;
    case 58: RUN_LUFACT_SMALL(58); break;
    case 59: RUN_LUFACT_SMALL(59); break;
    case 60: RUN_LUFACT_SMALL(60); break;
    case 61: RUN_LUFACT_SMALL(61); break;
    case 62: RUN_LUFACT_SMALL(62); break;
    case 63: RUN_LUFACT_SMALL(63); break;
    case 64: RUN_LUFACT_SMALL(64); break;
    default: ROCSOLVER_UNREACHABLE();
    }

    return rocblas_status_success;
}

#undef RUN_LUFACT_SMALL

#ifdef ROCSOLVER_SHARD_GETF2_SMALL
#define ROCSOLVER_INSTANTIATE_GETF2_SMALL_SHARD(SHARD, T, I, INFO, U) \
    template rocblas_status getf2_run_small_shard<SHARD, T, I, INFO, U>( \
        rocblas_handle handle, const I m, const I n, U A, const rocblas_stride shiftA, \
        const I lda, const rocblas_stride strideA, I* ipiv, const rocblas_stride shiftP, \
        const rocblas_stride strideP, INFO* info, const I batch_count, const bool pivot, \
        const I offset, I* permut_idx, const rocblas_stride stride)

// Keep every original scalar/batch specialization while distributing the
// 64 dimension-specialized device kernels across four independent commands.
extern ROCSOLVER_INSTANTIATE_GETF2_SMALL_SHARD(0, rocblas_float_complex, rocblas_int, rocblas_int, rocblas_float_complex*);
extern ROCSOLVER_INSTANTIATE_GETF2_SMALL_SHARD(1, rocblas_float_complex, rocblas_int, rocblas_int, rocblas_float_complex*);
extern ROCSOLVER_INSTANTIATE_GETF2_SMALL_SHARD(2, rocblas_float_complex, rocblas_int, rocblas_int, rocblas_float_complex*);
extern ROCSOLVER_INSTANTIATE_GETF2_SMALL_SHARD(3, rocblas_float_complex, rocblas_int, rocblas_int, rocblas_float_complex*);
extern ROCSOLVER_INSTANTIATE_GETF2_SMALL_SHARD(0, rocblas_float_complex, rocblas_int, rocblas_int, rocblas_float_complex* const*);
extern ROCSOLVER_INSTANTIATE_GETF2_SMALL_SHARD(1, rocblas_float_complex, rocblas_int, rocblas_int, rocblas_float_complex* const*);
extern ROCSOLVER_INSTANTIATE_GETF2_SMALL_SHARD(2, rocblas_float_complex, rocblas_int, rocblas_int, rocblas_float_complex* const*);
extern ROCSOLVER_INSTANTIATE_GETF2_SMALL_SHARD(3, rocblas_float_complex, rocblas_int, rocblas_int, rocblas_float_complex* const*);
extern ROCSOLVER_INSTANTIATE_GETF2_SMALL_SHARD(0, double, rocblas_int, rocblas_int, double*);
extern ROCSOLVER_INSTANTIATE_GETF2_SMALL_SHARD(1, double, rocblas_int, rocblas_int, double*);
extern ROCSOLVER_INSTANTIATE_GETF2_SMALL_SHARD(2, double, rocblas_int, rocblas_int, double*);
extern ROCSOLVER_INSTANTIATE_GETF2_SMALL_SHARD(3, double, rocblas_int, rocblas_int, double*);
extern ROCSOLVER_INSTANTIATE_GETF2_SMALL_SHARD(0, double, rocblas_int, rocblas_int, double* const*);
extern ROCSOLVER_INSTANTIATE_GETF2_SMALL_SHARD(1, double, rocblas_int, rocblas_int, double* const*);
extern ROCSOLVER_INSTANTIATE_GETF2_SMALL_SHARD(2, double, rocblas_int, rocblas_int, double* const*);
extern ROCSOLVER_INSTANTIATE_GETF2_SMALL_SHARD(3, double, rocblas_int, rocblas_int, double* const*);
extern ROCSOLVER_INSTANTIATE_GETF2_SMALL_SHARD(0, float, rocblas_int, rocblas_int, float*);
extern ROCSOLVER_INSTANTIATE_GETF2_SMALL_SHARD(1, float, rocblas_int, rocblas_int, float*);
extern ROCSOLVER_INSTANTIATE_GETF2_SMALL_SHARD(2, float, rocblas_int, rocblas_int, float*);
extern ROCSOLVER_INSTANTIATE_GETF2_SMALL_SHARD(3, float, rocblas_int, rocblas_int, float*);
extern ROCSOLVER_INSTANTIATE_GETF2_SMALL_SHARD(0, float, rocblas_int, rocblas_int, float* const*);
extern ROCSOLVER_INSTANTIATE_GETF2_SMALL_SHARD(1, float, rocblas_int, rocblas_int, float* const*);
extern ROCSOLVER_INSTANTIATE_GETF2_SMALL_SHARD(2, float, rocblas_int, rocblas_int, float* const*);
extern ROCSOLVER_INSTANTIATE_GETF2_SMALL_SHARD(3, float, rocblas_int, rocblas_int, float* const*);
extern ROCSOLVER_INSTANTIATE_GETF2_SMALL_SHARD(0, rocblas_double_complex, rocblas_int, rocblas_int, rocblas_double_complex*);
extern ROCSOLVER_INSTANTIATE_GETF2_SMALL_SHARD(1, rocblas_double_complex, rocblas_int, rocblas_int, rocblas_double_complex*);
extern ROCSOLVER_INSTANTIATE_GETF2_SMALL_SHARD(2, rocblas_double_complex, rocblas_int, rocblas_int, rocblas_double_complex*);
extern ROCSOLVER_INSTANTIATE_GETF2_SMALL_SHARD(3, rocblas_double_complex, rocblas_int, rocblas_int, rocblas_double_complex*);
extern ROCSOLVER_INSTANTIATE_GETF2_SMALL_SHARD(0, rocblas_double_complex, rocblas_int, rocblas_int, rocblas_double_complex* const*);
extern ROCSOLVER_INSTANTIATE_GETF2_SMALL_SHARD(1, rocblas_double_complex, rocblas_int, rocblas_int, rocblas_double_complex* const*);
extern ROCSOLVER_INSTANTIATE_GETF2_SMALL_SHARD(2, rocblas_double_complex, rocblas_int, rocblas_int, rocblas_double_complex* const*);
extern ROCSOLVER_INSTANTIATE_GETF2_SMALL_SHARD(3, rocblas_double_complex, rocblas_int, rocblas_int, rocblas_double_complex* const*);

template <typename T, typename I, typename INFO, typename U>
rocblas_status getf2_run_small(rocblas_handle handle,
                               const I m,
                               const I n,
                               U A,
                               const rocblas_stride shiftA,
                               const I lda,
                               const rocblas_stride strideA,
                               I* ipiv,
                               const rocblas_stride shiftP,
                               const rocblas_stride strideP,
                               INFO* info,
                               const I batch_count,
                               const bool pivot,
                               const I offset,
                               I* permut_idx,
                               const rocblas_stride stride)
{
    if(n <= 16)
        return getf2_run_small_shard<0, T, I, INFO, U>(handle, m, n, A, shiftA, lda, strideA, ipiv, shiftP, strideP, info, batch_count, pivot, offset, permut_idx, stride);
    if(n <= 32)
        return getf2_run_small_shard<1, T, I, INFO, U>(handle, m, n, A, shiftA, lda, strideA, ipiv, shiftP, strideP, info, batch_count, pivot, offset, permut_idx, stride);
    if(n <= 48)
        return getf2_run_small_shard<2, T, I, INFO, U>(handle, m, n, A, shiftA, lda, strideA, ipiv, shiftP, strideP, info, batch_count, pivot, offset, permut_idx, stride);
        return getf2_run_small_shard<3, T, I, INFO, U>(handle, m, n, A, shiftA, lda, strideA, ipiv, shiftP, strideP, info, batch_count, pivot, offset, permut_idx, stride);
}
#endif


ROCSOLVER_INSTANTIATE_GETF2_SMALL_SHARD(0, float, int, int, float*);
ROCSOLVER_INSTANTIATE_GETF2_SMALL_SHARD(1, float, int, int, float*);
ROCSOLVER_INSTANTIATE_GETF2_SMALL_SHARD(2, float, int, int, float*);
ROCSOLVER_INSTANTIATE_GETF2_SMALL_SHARD(3, float, int, int, float*);
ROCSOLVER_INSTANTIATE_GETF2_SMALL_SHARD(0, float, int, int, float* const*);
ROCSOLVER_INSTANTIATE_GETF2_SMALL_SHARD(1, float, int, int, float* const*);
ROCSOLVER_INSTANTIATE_GETF2_SMALL_SHARD(2, float, int, int, float* const*);
ROCSOLVER_INSTANTIATE_GETF2_SMALL_SHARD(3, float, int, int, float* const*);
ROCSOLVER_INSTANTIATE_GETF2_SMALL_SHARD(0, double, int, int, double*);
ROCSOLVER_INSTANTIATE_GETF2_SMALL_SHARD(1, double, int, int, double*);
ROCSOLVER_INSTANTIATE_GETF2_SMALL_SHARD(2, double, int, int, double*);
ROCSOLVER_INSTANTIATE_GETF2_SMALL_SHARD(3, double, int, int, double*);
ROCSOLVER_INSTANTIATE_GETF2_SMALL_SHARD(0, double, int, int, double* const*);
ROCSOLVER_INSTANTIATE_GETF2_SMALL_SHARD(1, double, int, int, double* const*);
ROCSOLVER_INSTANTIATE_GETF2_SMALL_SHARD(2, double, int, int, double* const*);
ROCSOLVER_INSTANTIATE_GETF2_SMALL_SHARD(3, double, int, int, double* const*);
ROCSOLVER_INSTANTIATE_GETF2_SMALL_SHARD(0, rocblas_float_complex, int, int, rocblas_float_complex*);
ROCSOLVER_INSTANTIATE_GETF2_SMALL_SHARD(1, rocblas_float_complex, int, int, rocblas_float_complex*);
ROCSOLVER_INSTANTIATE_GETF2_SMALL_SHARD(2, rocblas_float_complex, int, int, rocblas_float_complex*);
ROCSOLVER_INSTANTIATE_GETF2_SMALL_SHARD(3, rocblas_float_complex, int, int, rocblas_float_complex*);
ROCSOLVER_INSTANTIATE_GETF2_SMALL_SHARD(0, rocblas_float_complex, int, int, rocblas_float_complex* const*);
ROCSOLVER_INSTANTIATE_GETF2_SMALL_SHARD(1, rocblas_float_complex, int, int, rocblas_float_complex* const*);
ROCSOLVER_INSTANTIATE_GETF2_SMALL_SHARD(2, rocblas_float_complex, int, int, rocblas_float_complex* const*);
ROCSOLVER_INSTANTIATE_GETF2_SMALL_SHARD(3, rocblas_float_complex, int, int, rocblas_float_complex* const*);
ROCSOLVER_INSTANTIATE_GETF2_SMALL_SHARD(0, rocblas_double_complex, int, int, rocblas_double_complex*);
ROCSOLVER_INSTANTIATE_GETF2_SMALL_SHARD(1, rocblas_double_complex, int, int, rocblas_double_complex*);
ROCSOLVER_INSTANTIATE_GETF2_SMALL_SHARD(2, rocblas_double_complex, int, int, rocblas_double_complex*);
ROCSOLVER_INSTANTIATE_GETF2_SMALL_SHARD(3, rocblas_double_complex, int, int, rocblas_double_complex*);
ROCSOLVER_INSTANTIATE_GETF2_SMALL_SHARD(0, rocblas_double_complex, int, int, rocblas_double_complex* const*);
ROCSOLVER_INSTANTIATE_GETF2_SMALL_SHARD(1, rocblas_double_complex, int, int, rocblas_double_complex* const*);
ROCSOLVER_INSTANTIATE_GETF2_SMALL_SHARD(2, rocblas_double_complex, int, int, rocblas_double_complex* const*);
ROCSOLVER_INSTANTIATE_GETF2_SMALL_SHARD(3, rocblas_double_complex, int, int, rocblas_double_complex* const*);
int main(){
for(int n=1;n<=64;++n)for(bool p:{false,true}){
    getf2_run_small<float,int,int,float*>(nullptr,32,n,nullptr,0,32,0,nullptr,0,0,nullptr,1,p,0,nullptr,0);
    assert(observed_dimension==n && observed_pivot==p);
  }
for(int n=1;n<=64;++n)for(bool p:{false,true}){
    getf2_run_small<float,int,int,float* const*>(nullptr,32,n,nullptr,0,32,0,nullptr,0,0,nullptr,1,p,0,nullptr,0);
    assert(observed_dimension==n && observed_pivot==p);
  }
for(int n=1;n<=64;++n)for(bool p:{false,true}){
    getf2_run_small<double,int,int,double*>(nullptr,32,n,nullptr,0,32,0,nullptr,0,0,nullptr,1,p,0,nullptr,0);
    assert(observed_dimension==n && observed_pivot==p);
  }
for(int n=1;n<=64;++n)for(bool p:{false,true}){
    getf2_run_small<double,int,int,double* const*>(nullptr,32,n,nullptr,0,32,0,nullptr,0,0,nullptr,1,p,0,nullptr,0);
    assert(observed_dimension==n && observed_pivot==p);
  }
for(int n=1;n<=64;++n)for(bool p:{false,true}){
    getf2_run_small<rocblas_float_complex,int,int,rocblas_float_complex*>(nullptr,32,n,nullptr,0,32,0,nullptr,0,0,nullptr,1,p,0,nullptr,0);
    assert(observed_dimension==n && observed_pivot==p);
  }
for(int n=1;n<=64;++n)for(bool p:{false,true}){
    getf2_run_small<rocblas_float_complex,int,int,rocblas_float_complex* const*>(nullptr,32,n,nullptr,0,32,0,nullptr,0,0,nullptr,1,p,0,nullptr,0);
    assert(observed_dimension==n && observed_pivot==p);
  }
for(int n=1;n<=64;++n)for(bool p:{false,true}){
    getf2_run_small<rocblas_double_complex,int,int,rocblas_double_complex*>(nullptr,32,n,nullptr,0,32,0,nullptr,0,0,nullptr,1,p,0,nullptr,0);
    assert(observed_dimension==n && observed_pivot==p);
  }
for(int n=1;n<=64;++n)for(bool p:{false,true}){
    getf2_run_small<rocblas_double_complex,int,int,rocblas_double_complex* const*>(nullptr,32,n,nullptr,0,32,0,nullptr,0,0,nullptr,1,p,0,nullptr,0);
    assert(observed_dimension==n && observed_pivot==p);
  }
}
