!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
! Copyright (c) 2020-2022 Advanced Micro Devices, Inc. All rights reserved.
!
! Permission is hereby granted, free of charge, to any person obtaining a copy
! of this software and associated documentation files (the "Software"), to deal
! in the Software without restriction, including without limitation the rights
! to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
! copies of the Software, and to permit persons to whom the Software is
! furnished to do so, subject to the following conditions:
!
! The above copyright notice and this permission notice shall be included in
! all copies or substantial portions of the Software.
!
! THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
! IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
! FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
! AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
! LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
! OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
! THE SOFTWARE.
!
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

!--------!
!  Aux   !
!--------!
function hipblasSetVectorFortran(n, elemSize, x, incx, y, incy) &
    bind(c, name='hipblasSetVectorFortran')
    use iso_c_binding
    use hipblas_enums
    implicit none
    integer(kind(HIPBLAS_STATUS_SUCCESS)) :: hipblasSetVectorFortran
    integer(c_int), value :: n
    integer(c_int), value :: elemSize
    type(c_ptr), value :: x
    integer(c_int), value :: incx
    type(c_ptr), value :: y
    integer(c_int), value :: incy
            hipblasSetVectorFortran = &
        hipblasSetVector(n, elemSize, x, incx, y, incy)
end function hipblasSetVectorFortran

function hipblasGetVectorFortran(n, elemSize, x, incx, y, incy) &
    bind(c, name='hipblasGetVectorFortran')
    use iso_c_binding
    use hipblas_enums
    implicit none
    integer(kind(HIPBLAS_STATUS_SUCCESS)) :: hipblasGetVectorFortran
    integer(c_int), value :: n
    integer(c_int), value :: elemSize
    type(c_ptr), value :: x
    integer(c_int), value :: incx
    type(c_ptr), value :: y
    integer(c_int), value :: incy
            hipblasGetVectorFortran = &
        hipblasGetVector(n, elemSize, x, incx, y, incy)
end function hipblasGetVectorFortran

function hipblasSetMatrixFortran(rows, cols, elemSize, A, lda, B, ldb) &
    bind(c, name='hipblasSetMatrixFortran')
    use iso_c_binding
    use hipblas_enums
    implicit none
    integer(kind(HIPBLAS_STATUS_SUCCESS)) :: hipblasSetMatrixFortran
    integer(c_int), value :: rows
    integer(c_int), value :: cols
    integer(c_int), value :: elemSize
    type(c_ptr), value :: A
    integer(c_int), value :: lda
    type(c_ptr), value :: B
    integer(c_int), value :: ldb
            hipblasSetMatrixFortran = &
        hipblasSetMatrix(rows, cols, elemSize, A, lda, B, ldb)
end function hipblasSetMatrixFortran

function hipblasGetMatrixFortran(rows, cols, elemSize, A, lda, B, ldb) &
    bind(c, name='hipblasGetMatrixFortran')
    use iso_c_binding
    use hipblas_enums
    implicit none
    integer(kind(HIPBLAS_STATUS_SUCCESS)) :: hipblasGetMatrixFortran
    integer(c_int), value :: rows
    integer(c_int), value :: cols
    integer(c_int), value :: elemSize
    type(c_ptr), value :: A
    integer(c_int), value :: lda
    type(c_ptr), value :: B
    integer(c_int), value :: ldb
            hipblasGetMatrixFortran = &
        hipblasGetMatrix(rows, cols, elemSize, A, lda, B, ldb)
end function hipblasGetMatrixFortran

function hipblasSetVectorAsyncFortran(n, elemSize, x, incx, y, incy, stream) &
    bind(c, name='hipblasSetVectorAsyncFortran')
    use iso_c_binding
    use hipblas_enums
    implicit none
    integer(kind(HIPBLAS_STATUS_SUCCESS)) :: hipblasSetVectorAsyncFortran
    integer(c_int), value :: n
    integer(c_int), value :: elemSize
    type(c_ptr), value :: x
    integer(c_int), value :: incx
    type(c_ptr), value :: y
    integer(c_int), value :: incy
    type(c_ptr), value :: stream
            hipblasSetVectorAsyncFortran = &
        hipblasSetVectorAsync(n, elemSize, x, incx, y, incy, stream)
end function hipblasSetVectorAsyncFortran

function hipblasGetVectorAsyncFortran(n, elemSize, x, incx, y, incy, stream) &
    bind(c, name='hipblasGetVectorAsyncFortran')
    use iso_c_binding
    use hipblas_enums
    implicit none
    integer(kind(HIPBLAS_STATUS_SUCCESS)) :: hipblasGetVectorAsyncFortran
    integer(c_int), value :: n
    integer(c_int), value :: elemSize
    type(c_ptr), value :: x
    integer(c_int), value :: incx
    type(c_ptr), value :: y
    integer(c_int), value :: incy
    type(c_ptr), value :: stream
            hipblasGetVectorAsyncFortran = &
        hipblasGetVectorAsync(n, elemSize, x, incx, y, incy, stream)
end function hipblasGetVectorAsyncFortran

function hipblasSetMatrixAsyncFortran(rows, cols, elemSize, A, lda, B, ldb, stream) &
    bind(c, name='hipblasSetMatrixAsyncFortran')
    use iso_c_binding
    use hipblas_enums
    implicit none
    integer(kind(HIPBLAS_STATUS_SUCCESS)) :: hipblasSetMatrixAsyncFortran
    integer(c_int), value :: rows
    integer(c_int), value :: cols
    integer(c_int), value :: elemSize
    type(c_ptr), value :: A
    integer(c_int), value :: lda
    type(c_ptr), value :: B
    integer(c_int), value :: ldb
    type(c_ptr), value :: stream
            hipblasSetMatrixAsyncFortran = &
        hipblasSetMatrixAsync(rows, cols, elemSize, A, lda, B, ldb, stream)
end function hipblasSetMatrixAsyncFortran

function hipblasGetMatrixAsyncFortran(rows, cols, elemSize, A, lda, B, ldb, stream) &
    bind(c, name='hipblasGetMatrixAsyncFortran')
    use iso_c_binding
    use hipblas_enums
    implicit none
    integer(kind(HIPBLAS_STATUS_SUCCESS)) :: hipblasGetMatrixAsyncFortran
    integer(c_int), value :: rows
    integer(c_int), value :: cols
    integer(c_int), value :: elemSize
    type(c_ptr), value :: A
    integer(c_int), value :: lda
    type(c_ptr), value :: B
    integer(c_int), value :: ldb
    type(c_ptr), value :: stream
            hipblasGetMatrixAsyncFortran = &
        hipblasGetMatrixAsync(rows, cols, elemSize, A, lda, B, ldb, stream)
end function hipblasGetMatrixAsyncFortran

function hipblasSetAtomicsModeFortran(handle, atomics_mode) &
    bind(c, name='hipblasSetAtomicsModeFortran')
    use iso_c_binding
    use hipblas_enums
    implicit none
    integer(kind(HIPBLAS_STATUS_SUCCESS)) :: hipblasSetAtomicsModeFortran
    type(c_ptr), value :: handle
    integer(kind(HIPBLAS_ATOMICS_ALLOWED)), value :: atomics_mode
            hipblasSetAtomicsModeFortran = &
        hipblasSetAtomicsMode(handle, atomics_mode)
end function hipblasSetAtomicsModeFortran

function hipblasGetAtomicsModeFortran(handle, atomics_mode) &
    bind(c, name='hipblasGetAtomicsModeFortran')
    use iso_c_binding
    use hipblas_enums
    implicit none
    integer(kind(HIPBLAS_STATUS_SUCCESS)) :: hipblasGetAtomicsModeFortran
    type(c_ptr), value :: handle
    type(c_ptr), value :: atomics_mode
            hipblasGetAtomicsModeFortran = &
        hipblasGetAtomicsMode(handle, atomics_mode)
end function hipblasGetAtomicsModeFortran

!--------!
! blas 1 !
!--------!

! scal
function hipblasSscalFortran(handle, n, alpha, x, incx) &
    bind(c, name='hipblasSscalFortran')
    use iso_c_binding
    use hipblas_enums
    implicit none
    integer(kind(HIPBLAS_STATUS_SUCCESS)) :: hipblasSscalFortran
    type(c_ptr), value :: handle
    integer(c_int), value :: n
    type(c_ptr), value :: alpha
    type(c_ptr), value :: x
    integer(c_int), value :: incx
            hipblasSscalFortran = &
        hipblasSscal(handle, n, alpha, x, incx)
    return
end function hipblasSscalFortran

function hipblasDscalFortran(handle, n, alpha, x, incx) &
    bind(c, name='hipblasDscalFortran')
    use iso_c_binding
    use hipblas_enums
    implicit none
    integer(kind(HIPBLAS_STATUS_SUCCESS)) :: hipblasDscalFortran
    type(c_ptr), value :: handle
    integer(c_int), value :: n
    type(c_ptr), value :: alpha
    type(c_ptr), value :: x
    integer(c_int), value :: incx
            hipblasDscalFortran = &
        hipblasDscal(handle, n, alpha, x, incx)
    return
end function hipblasDscalFortran

function hipblasCscalFortran(handle, n, alpha, x, incx) &
    bind(c, name='hipblasCscalFortran')
    use iso_c_binding
    use hipblas_enums
    implicit none
    integer(kind(HIPBLAS_STATUS_SUCCESS)) :: hipblasCscalFortran
    type(c_ptr), value :: handle
    integer(c_int), value :: n
    type(c_ptr), value :: alpha
    type(c_ptr), value :: x
    integer(c_int), value :: incx
            hipblasCscalFortran = &
        hipblasCscal(handle, n, alpha, x, incx)
    return
end function hipblasCscalFortran

function hipblasZscalFortran(handle, n, alpha, x, incx) &
    bind(c, name='hipblasZscalFortran')
    use iso_c_binding
    use hipblas_enums
    implicit none
    integer(kind(HIPBLAS_STATUS_SUCCESS)) :: hipblasZscalFortran
    type(c_ptr), value :: handle
    integer(c_int), value :: n
    type(c_ptr), value :: alpha
    type(c_ptr), value :: x
    integer(c_int), value :: incx
            hipblasZscalFortran = &
        hipblasZscal(handle, n, alpha, x, incx)
    return
end function hipblasZscalFortran

function hipblasCsscalFortran(handle, n, alpha, x, incx) &
    bind(c, name='hipblasCsscalFortran')
    use iso_c_binding
    use hipblas_enums
    implicit none
    integer(kind(HIPBLAS_STATUS_SUCCESS)) :: hipblasCsscalFortran
    type(c_ptr), value :: handle
    integer(c_int), value :: n
    type(c_ptr), value :: alpha
    type(c_ptr), value :: x
    integer(c_int), value :: incx
            hipblasCsscalFortran = &
        hipblasCsscal(handle, n, alpha, x, incx)
    return
end function hipblasCsscalFortran

function hipblasZdscalFortran(handle, n, alpha, x, incx) &
    bind(c, name='hipblasZdscalFortran')
    use iso_c_binding
    use hipblas_enums
    implicit none
    integer(kind(HIPBLAS_STATUS_SUCCESS)) :: hipblasZdscalFortran
    type(c_ptr), value :: handle
    integer(c_int), value :: n
    type(c_ptr), value :: alpha
    type(c_ptr), value :: x
    integer(c_int), value :: incx
            hipblasZdscalFortran = &
        hipblasZdscal(handle, n, alpha, x, incx)
    return
end function hipblasZdscalFortran

! scalBatched
function hipblasSscalBatchedFortran(handle, n, alpha, x, incx, batch_count) &
    bind(c, name='hipblasSscalBatchedFortran')
    use iso_c_binding
    use hipblas_enums
    implicit none
    integer(kind(HIPBLAS_STATUS_SUCCESS)) :: hipblasSscalBatchedFortran
    type(c_ptr), value :: handle
    integer(c_int), value :: n
    type(c_ptr), value :: alpha
    type(c_ptr), value :: x
    integer(c_int), value :: incx
    integer(c_int), value :: batch_count
            hipblasSscalBatchedFortran = &
        hipblasSscalBatched(handle, n, alpha, x, incx, batch_count)
    return
end function hipblasSscalBatchedFortran

function hipblasDscalBatchedFortran(handle, n, alpha, x, incx, batch_count) &
    bind(c, name='hipblasDscalBatchedFortran')
    use iso_c_binding
    use hipblas_enums
    implicit none
    integer(kind(HIPBLAS_STATUS_SUCCESS)) :: hipblasDscalBatchedFortran
    type(c_ptr), value :: handle
    integer(c_int), value :: n
    type(c_ptr), value :: alpha
    type(c_ptr), value :: x
    integer(c_int), value :: incx
    integer(c_int), value :: batch_count
            hipblasDscalBatchedFortran = &
        hipblasDscalBatched(handle, n, alpha, x, incx, batch_count)
    return
end function hipblasDscalBatchedFortran

function hipblasCscalBatchedFortran(handle, n, alpha, x, incx, batch_count) &
    bind(c, name='hipblasCscalBatchedFortran')
    use iso_c_binding
    use hipblas_enums
    implicit none
    integer(kind(HIPBLAS_STATUS_SUCCESS)) :: hipblasCscalBatchedFortran
    type(c_ptr), value :: handle
    integer(c_int), value :: n
    type(c_ptr), value :: alpha
    type(c_ptr), value :: x
    integer(c_int), value :: incx
    integer(c_int), value :: batch_count
            hipblasCscalBatchedFortran = &
        hipblasCscalBatched(handle, n, alpha, x, incx, batch_count)
    return
end function hipblasCscalBatchedFortran

function hipblasZscalBatchedFortran(handle, n, alpha, x, incx, batch_count) &
    bind(c, name='hipblasZscalBatchedFortran')
    use iso_c_binding
    use hipblas_enums
    implicit none
    integer(kind(HIPBLAS_STATUS_SUCCESS)) :: hipblasZscalBatchedFortran
    type(c_ptr), value :: handle
    integer(c_int), value :: n
    type(c_ptr), value :: alpha
    type(c_ptr), value :: x
    integer(c_int), value :: incx
    integer(c_int), value :: batch_count
            hipblasZscalBatchedFortran = &
        hipblasZscalBatched(handle, n, alpha, x, incx, batch_count)
    return
end function hipblasZscalBatchedFortran

function hipblasCsscalBatchedFortran(handle, n, alpha, x, incx, batch_count) &
    bind(c, name='hipblasCsscalBatchedFortran')
    use iso_c_binding
    use hipblas_enums
    implicit none
    integer(kind(HIPBLAS_STATUS_SUCCESS)) :: hipblasCsscalBatchedFortran
    type(c_ptr), value :: handle
    integer(c_int), value :: n
    type(c_ptr), value :: alpha
    type(c_ptr), value :: x
    integer(c_int), value :: incx
    integer(c_int), value :: batch_count
            hipblasCsscalBatchedFortran = &
        hipblasCsscalBatched(handle, n, alpha, x, incx, batch_count)
    return
end function hipblasCsscalBatchedFortran

function hipblasZdscalBatchedFortran(handle, n, alpha, x, incx, batch_count) &
    bind(c, name='hipblasZdscalBatchedFortran')
    use iso_c_binding
    use hipblas_enums
    implicit none
    integer(kind(HIPBLAS_STATUS_SUCCESS)) :: hipblasZdscalBatchedFortran
    type(c_ptr), value :: handle
    integer(c_int), value :: n
    type(c_ptr), value :: alpha
    type(c_ptr), value :: x
    integer(c_int), value :: incx
    integer(c_int), value :: batch_count
            hipblasZdscalBatchedFortran = &
        hipblasZdscalBatched(handle, n, alpha, x, incx, batch_count)
    return
end function hipblasZdscalBatchedFortran

! scalStridedBatched
function hipblasSscalStridedBatchedFortran(handle, n, alpha, x, incx, stride_x, batch_count) &
    bind(c, name='hipblasSscalStridedBatchedFortran')
    use iso_c_binding
    use hipblas_enums
    implicit none
    integer(kind(HIPBLAS_STATUS_SUCCESS)) :: hipblasSscalStridedBatchedFortran
    type(c_ptr), value :: handle
    integer(c_int), value :: n
    type(c_ptr), value :: alpha
    type(c_ptr), value :: x
    integer(c_int), value :: incx
    integer(c_int64_t), value :: stride_x
    integer(c_int), value :: batch_count
            hipblasSscalStridedBatchedFortran = &
        hipblasSscalStridedBatched(handle, n, alpha, x, incx, stride_x, batch_count)
    return
end function hipblasSscalStridedBatchedFortran

function hipblasDscalStridedBatchedFortran(handle, n, alpha, x, incx, stride_x, batch_count) &
    bind(c, name='hipblasDscalStridedBatchedFortran')
    use iso_c_binding
    use hipblas_enums
    implicit none
    integer(kind(HIPBLAS_STATUS_SUCCESS)) :: hipblasDscalStridedBatchedFortran
    type(c_ptr), value :: handle
    integer(c_int), value :: n
    type(c_ptr), value :: alpha
    type(c_ptr), value :: x
    integer(c_int), value :: incx
    integer(c_int64_t), value :: stride_x
    integer(c_int), value :: batch_count
            hipblasDscalStridedBatchedFortran = &
        hipblasDscalStridedBatched(handle, n, alpha, x, incx, stride_x, batch_count)
    return
end function hipblasDscalStridedBatchedFortran

function hipblasCscalStridedBatchedFortran(handle, n, alpha, x, incx, stride_x, batch_count) &
    bind(c, name='hipblasCscalStridedBatchedFortran')
    use iso_c_binding
    use hipblas_enums
    implicit none
    integer(kind(HIPBLAS_STATUS_SUCCESS)) :: hipblasCscalStridedBatchedFortran
    type(c_ptr), value :: handle
    integer(c_int), value :: n
    type(c_ptr), value :: alpha
    type(c_ptr), value :: x
    integer(c_int), value :: incx
    integer(c_int64_t), value :: stride_x
    integer(c_int), value :: batch_count
            hipblasCscalStridedBatchedFortran = &
        hipblasCscalStridedBatched(handle, n, alpha, x, incx, stride_x, batch_count)
    return
end function hipblasCscalStridedBatchedFortran

function hipblasZscalStridedBatchedFortran(handle, n, alpha, x, incx, stride_x, batch_count) &
    bind(c, name='hipblasZscalStridedBatchedFortran')
    use iso_c_binding
    use hipblas_enums
    implicit none
    integer(kind(HIPBLAS_STATUS_SUCCESS)) :: hipblasZscalStridedBatchedFortran
    type(c_ptr), value :: handle
    integer(c_int), value :: n
    type(c_ptr), value :: alpha
    type(c_ptr), value :: x
    integer(c_int), value :: incx
    integer(c_int64_t), value :: stride_x
    integer(c_int), value :: batch_count
            hipblasZscalStridedBatchedFortran = &
        hipblasZscalStridedBatched(handle, n, alpha, x, incx, stride_x, batch_count)
    return
end function hipblasZscalStridedBatchedFortran

function hipblasCsscalStridedBatchedFortran(handle, n, alpha, x, incx, stride_x, batch_count) &
    bind(c, name='hipblasCsscalStridedBatchedFortran')
    use iso_c_binding
    use hipblas_enums
    implicit none
    integer(kind(HIPBLAS_STATUS_SUCCESS)) :: hipblasCsscalStridedBatchedFortran
    type(c_ptr), value :: handle
    integer(c_int), value :: n
    type(c_ptr), value :: alpha
    type(c_ptr), value :: x
    integer(c_int), value :: incx
    integer(c_int64_t), value :: stride_x
    integer(c_int), value :: batch_count
            hipblasCsscalStridedBatchedFortran = &
        hipblasCsscalStridedBatched(handle, n, alpha, x, incx, stride_x, batch_count)
    return
end function hipblasCsscalStridedBatchedFortran

function hipblasZdscalStridedBatchedFortran(handle, n, alpha, x, incx, stride_x, batch_count) &
    bind(c, name='hipblasZdscalStridedBatchedFortran')
    use iso_c_binding
    use hipblas_enums
    implicit none
    integer(kind(HIPBLAS_STATUS_SUCCESS)) :: hipblasZdscalStridedBatchedFortran
    type(c_ptr), value :: handle
    integer(c_int), value :: n
    type(c_ptr), value :: alpha
    type(c_ptr), value :: x
    integer(c_int), value :: incx
    integer(c_int64_t), value :: stride_x
    integer(c_int), value :: batch_count
            hipblasZdscalStridedBatchedFortran = &
        hipblasZdscalStridedBatched(handle, n, alpha, x, incx, stride_x, batch_count)
    return
end function hipblasZdscalStridedBatchedFortran

! copy
function hipblasScopyFortran(handle, n, x, incx, y, incy) &
    bind(c, name='hipblasScopyFortran')
    use iso_c_binding
    use hipblas_enums
    implicit none
    integer(kind(HIPBLAS_STATUS_SUCCESS)) :: hipblasScopyFortran
    type(c_ptr), value :: handle
    integer(c_int), value :: n
    type(c_ptr), value :: x
    integer(c_int), value :: incx
    type(c_ptr), value :: y
    integer(c_int), value :: incy
            hipblasScopyFortran = &
        hipblasScopy(handle, n, x, incx, y, incy)
    return
end function hipblasScopyFortran

function hipblasDcopyFortran(handle, n, x, incx, y, incy) &
    bind(c, name='hipblasDcopyFortran')
    use iso_c_binding
    use hipblas_enums
    implicit none
    integer(kind(HIPBLAS_STATUS_SUCCESS)) :: hipblasDcopyFortran
    type(c_ptr), value :: handle
    integer(c_int), value :: n
    type(c_ptr), value :: x
    integer(c_int), value :: incx
    type(c_ptr), value :: y
    integer(c_int), value :: incy
            hipblasDcopyFortran = &
        hipblasDcopy(handle, n, x, incx, y, incy)
    return
end function hipblasDcopyFortran

function hipblasCcopyFortran(handle, n, x, incx, y, incy) &
    bind(c, name='hipblasCcopyFortran')
    use iso_c_binding
    use hipblas_enums
    implicit none
    integer(kind(HIPBLAS_STATUS_SUCCESS)) :: hipblasCcopyFortran
    type(c_ptr), value :: handle
    integer(c_int), value :: n
    type(c_ptr), value :: x
    integer(c_int), value :: incx
    type(c_ptr), value :: y
    integer(c_int), value :: incy
            hipblasCcopyFortran = &
        hipblasCcopy(handle, n, x, incx, y, incy)
    return
end function hipblasCcopyFortran

function hipblasZcopyFortran(handle, n, x, incx, y, incy) &
    bind(c, name='hipblasZcopyFortran')
    use iso_c_binding
    use hipblas_enums
    implicit none
    integer(kind(HIPBLAS_STATUS_SUCCESS)) :: hipblasZcopyFortran
    type(c_ptr), value :: handle
    integer(c_int), value :: n
    type(c_ptr), value :: x
    integer(c_int), value :: incx
    type(c_ptr), value :: y
    integer(c_int), value :: incy
            hipblasZcopyFortran = &
        hipblasZcopy(handle, n, x, incx, y, incy)
    return
end function hipblasZcopyFortran

! copyBatched
function hipblasScopyBatchedFortran(handle, n, x, incx, y, incy, batch_count) &
    bind(c, name='hipblasScopyBatchedFortran')
    use iso_c_binding
    use hipblas_enums
    implicit none
    integer(kind(HIPBLAS_STATUS_SUCCESS)) :: hipblasScopyBatchedFortran
    type(c_ptr), value :: handle
    integer(c_int), value :: n
    type(c_ptr), value :: x
    integer(c_int), value :: incx
    type(c_ptr), value :: y
    integer(c_int), value :: incy
    integer(c_int), value :: batch_count
            hipblasScopyBatchedFortran = &
        hipblasScopyBatched(handle, n, x, incx, y, incy, batch_count)
    return
end function hipblasScopyBatchedFortran

function hipblasDcopyBatchedFortran(handle, n, x, incx, y, incy, batch_count) &
    bind(c, name='hipblasDcopyBatchedFortran')
    use iso_c_binding
    use hipblas_enums
    implicit none
    integer(kind(HIPBLAS_STATUS_SUCCESS)) :: hipblasDcopyBatchedFortran
    type(c_ptr), value :: handle
    integer(c_int), value :: n
    type(c_ptr), value :: x
    integer(c_int), value :: incx
    type(c_ptr), value :: y
    integer(c_int), value :: incy
    integer(c_int), value :: batch_count
            hipblasDcopyBatchedFortran = &
        hipblasDcopyBatched(handle, n, x, incx, y, incy, batch_count)
    return
end function hipblasDcopyBatchedFortran

function hipblasCcopyBatchedFortran(handle, n, x, incx, y, incy, batch_count) &
    bind(c, name='hipblasCcopyBatchedFortran')
    use iso_c_binding
    use hipblas_enums
    implicit none
    integer(kind(HIPBLAS_STATUS_SUCCESS)) :: hipblasCcopyBatchedFortran
    type(c_ptr), value :: handle
    integer(c_int), value :: n
    type(c_ptr), value :: x
    integer(c_int), value :: incx
    type(c_ptr), value :: y
    integer(c_int), value :: incy
    integer(c_int), value :: batch_count
            hipblasCcopyBatchedFortran = &
        hipblasCcopyBatched(handle, n, x, incx, y, incy, batch_count)
    return
end function hipblasCcopyBatchedFortran

function hipblasZcopyBatchedFortran(handle, n, x, incx, y, incy, batch_count) &
    bind(c, name='hipblasZcopyBatchedFortran')
    use iso_c_binding
    use hipblas_enums
    implicit none
    integer(kind(HIPBLAS_STATUS_SUCCESS)) :: hipblasZcopyBatchedFortran
    type(c_ptr), value :: handle
    integer(c_int), value :: n
    type(c_ptr), value :: x
    integer(c_int), value :: incx
    type(c_ptr), value :: y
    integer(c_int), value :: incy
    integer(c_int), value :: batch_count
            hipblasZcopyBatchedFortran = &
        hipblasZcopyBatched(handle, n, x, incx, y, incy, batch_count)
    return
end function hipblasZcopyBatchedFortran

! copyStridedBatched
function hipblasScopyStridedBatchedFortran(handle, n, x, incx, stride_x, y, incy, stride_y, batch_count) &
    bind(c, name='hipblasScopyStridedBatchedFortran')
    use iso_c_binding
    use hipblas_enums
    implicit none
    integer(kind(HIPBLAS_STATUS_SUCCESS)) :: hipblasScopyStridedBatchedFortran
    type(c_ptr), value :: handle
    integer(c_int), value :: n
    type(c_ptr), value :: x
    integer(c_int), value :: incx
    integer(c_int64_t), value :: stride_x
    type(c_ptr), value :: y
    integer(c_int), value :: incy
    integer(c_int64_t), value :: stride_y
    integer(c_int), value :: batch_count
            hipblasScopyStridedBatchedFortran = &
        hipblasScopyStridedBatched(handle, n, x, incx, stride_x, y, incy, stride_y, batch_count)
    return
end function hipblasScopyStridedBatchedFortran

function hipblasDcopyStridedBatchedFortran(handle, n, x, incx, stride_x, y, incy, stride_y, batch_count) &
    bind(c, name='hipblasDcopyStridedBatchedFortran')
    use iso_c_binding
    use hipblas_enums
    implicit none
    integer(kind(HIPBLAS_STATUS_SUCCESS)) :: hipblasDcopyStridedBatchedFortran
    type(c_ptr), value :: handle
    integer(c_int), value :: n
    type(c_ptr), value :: x
    integer(c_int), value :: incx
    integer(c_int64_t), value :: stride_x
    type(c_ptr), value :: y
    integer(c_int), value :: incy
    integer(c_int64_t), value :: stride_y
    integer(c_int), value :: batch_count
            hipblasDcopyStridedBatchedFortran = &
        hipblasDcopyStridedBatched(handle, n, x, incx, stride_x, y, incy, stride_y, batch_count)
    return
end function hipblasDcopyStridedBatchedFortran

function hipblasCcopyStridedBatchedFortran(handle, n, x, incx, stride_x, y, incy, stride_y, batch_count) &
    bind(c, name='hipblasCcopyStridedBatchedFortran')
    use iso_c_binding
    use hipblas_enums
    implicit none
    integer(kind(HIPBLAS_STATUS_SUCCESS)) :: hipblasCcopyStridedBatchedFortran
    type(c_ptr), value :: handle
    integer(c_int), value :: n
    type(c_ptr), value :: x
    integer(c_int), value :: incx
    integer(c_int64_t), value :: stride_x
    type(c_ptr), value :: y
    integer(c_int), value :: incy
    integer(c_int64_t), value :: stride_y
    integer(c_int), value :: batch_count
            hipblasCcopyStridedBatchedFortran = &
        hipblasCcopyStridedBatched(handle, n, x, incx, stride_x, y, incy, stride_y, batch_count)
    return
end function hipblasCcopyStridedBatchedFortran

function hipblasZcopyStridedBatchedFortran(handle, n, x, incx, stride_x, y, incy, stride_y, batch_count) &
    bind(c, name='hipblasZcopyStridedBatchedFortran')
    use iso_c_binding
    use hipblas_enums
    implicit none
    integer(kind(HIPBLAS_STATUS_SUCCESS)) :: hipblasZcopyStridedBatchedFortran
    type(c_ptr), value :: handle
    integer(c_int), value :: n
    type(c_ptr), value :: x
    integer(c_int), value :: incx
    integer(c_int64_t), value :: stride_x
    type(c_ptr), value :: y
    integer(c_int), value :: incy
    integer(c_int64_t), value :: stride_y
    integer(c_int), value :: batch_count
            hipblasZcopyStridedBatchedFortran = &
        hipblasZcopyStridedBatched(handle, n, x, incx, stride_x, y, incy, stride_y, batch_count)
    return
end function hipblasZcopyStridedBatchedFortran

! dot
function hipblasSdotFortran(handle, n, x, incx, y, incy, result) &
    bind(c, name='hipblasSdotFortran')
    use iso_c_binding
    use hipblas_enums
    implicit none
    integer(kind(HIPBLAS_STATUS_SUCCESS)) :: hipblasSdotFortran
    type(c_ptr), value :: handle
    integer(c_int), value :: n
    type(c_ptr), value :: x
    integer(c_int), value :: incx
    type(c_ptr), value :: y
    integer(c_int), value :: incy
    type(c_ptr), value :: result
            hipblasSdotFortran = &
        hipblasSdot(handle, n, x, incx, y, incy, result)
    return
end function hipblasSdotFortran

function hipblasDdotFortran(handle, n, x, incx, y, incy, result) &
    bind(c, name='hipblasDdotFortran')
    use iso_c_binding
    use hipblas_enums
    implicit none
    integer(kind(HIPBLAS_STATUS_SUCCESS)) :: hipblasDdotFortran
    type(c_ptr), value :: handle
    integer(c_int), value :: n
    type(c_ptr), value :: x
    integer(c_int), value :: incx
    type(c_ptr), value :: y
    integer(c_int), value :: incy
    type(c_ptr), value :: result
            hipblasDdotFortran = &
        hipblasDdot(handle, n, x, incx, y, incy, result)
    return
end function hipblasDdotFortran

function hipblasHdotFortran(handle, n, x, incx, y, incy, result) &
    bind(c, name='hipblasHdotFortran')
    use iso_c_binding
    use hipblas_enums
    implicit none
    integer(kind(HIPBLAS_STATUS_SUCCESS)) :: hipblasHdotFortran
    type(c_ptr), value :: handle
    integer(c_int), value :: n
    type(c_ptr), value :: x
    integer(c_int), value :: incx
    type(c_ptr), value :: y
    integer(c_int), value :: incy
    type(c_ptr), value :: result
            hipblasHdotFortran = &
        hipblasHdot(handle, n, x, incx, y, incy, result)
    return
end function hipblasHdotFortran

function hipblasBfdotFortran(handle, n, x, incx, y, incy, result) &
    bind(c, name='hipblasBfdotFortran')
    use iso_c_binding
    use hipblas_enums
    implicit none
    integer(kind(HIPBLAS_STATUS_SUCCESS)) :: hipblasBfdotFortran
    type(c_ptr), value :: handle
    integer(c_int), value :: n
    type(c_ptr), value :: x
    integer(c_int), value :: incx
    type(c_ptr), value :: y
    integer(c_int), value :: incy
    type(c_ptr), value :: result
            hipblasBfdotFortran = &
        hipblasBfdot(handle, n, x, incx, y, incy, result)
    return
end function hipblasBfdotFortran

function hipblasCdotuFortran(handle, n, x, incx, y, incy, result) &
    bind(c, name='hipblasCdotuFortran')
    use iso_c_binding
    use hipblas_enums
    implicit none
    integer(kind(HIPBLAS_STATUS_SUCCESS)) :: hipblasCdotuFortran
    type(c_ptr), value :: handle
    integer(c_int), value :: n
    type(c_ptr), value :: x
    integer(c_int), value :: incx
    type(c_ptr), value :: y
    integer(c_int), value :: incy
    type(c_ptr), value :: result
            hipblasCdotuFortran = &
        hipblasCdotu(handle, n, x, incx, y, incy, result)
    return
end function hipblasCdotuFortran

function hipblasCdotcFortran(handle, n, x, incx, y, incy, result) &
    bind(c, name='hipblasCdotcFortran')
    use iso_c_binding
    use hipblas_enums
    implicit none
    integer(kind(HIPBLAS_STATUS_SUCCESS)) :: hipblasCdotcFortran
    type(c_ptr), value :: handle
    integer(c_int), value :: n
    type(c_ptr), value :: x
    integer(c_int), value :: incx
    type(c_ptr), value :: y
    integer(c_int), value :: incy
    type(c_ptr), value :: result
            hipblasCdotcFortran = &
        hipblasCdotc(handle, n, x, incx, y, incy, result)
    return
end function hipblasCdotcFortran

function hipblasZdotuFortran(handle, n, x, incx, y, incy, result) &
    bind(c, name='hipblasZdotuFortran')
    use iso_c_binding
    use hipblas_enums
    implicit none
    integer(kind(HIPBLAS_STATUS_SUCCESS)) :: hipblasZdotuFortran
    type(c_ptr), value :: handle
    integer(c_int), value :: n
    type(c_ptr), value :: x
    integer(c_int), value :: incx
    type(c_ptr), value :: y
    integer(c_int), value :: incy
    type(c_ptr), value :: result
            hipblasZdotuFortran = &
        hipblasZdotu(handle, n, x, incx, y, incy, result)
    return
end function hipblasZdotuFortran

function hipblasZdotcFortran(handle, n, x, incx, y, incy, result) &
    bind(c, name='hipblasZdotcFortran')
    use iso_c_binding
    use hipblas_enums
    implicit none
    integer(kind(HIPBLAS_STATUS_SUCCESS)) :: hipblasZdotcFortran
    type(c_ptr), value :: handle
    integer(c_int), value :: n
    type(c_ptr), value :: x
    integer(c_int), value :: incx
    type(c_ptr), value :: y
    integer(c_int), value :: incy
    type(c_ptr), value :: result
            hipblasZdotcFortran = &
        hipblasZdotc(handle, n, x, incx, y, incy, result)
    return
end function hipblasZdotcFortran

! dotBatched
function hipblasSdotBatchedFortran(handle, n, x, incx, y, incy, batch_count, result) &
    bind(c, name='hipblasSdotBatchedFortran')
    use iso_c_binding
    use hipblas_enums
    implicit none
    integer(kind(HIPBLAS_STATUS_SUCCESS)) :: hipblasSdotBatchedFortran
    type(c_ptr), value :: handle
    integer(c_int), value :: n
    type(c_ptr), value :: x
    integer(c_int), value :: incx
    type(c_ptr), value :: y
    integer(c_int), value :: incy
    integer(c_int), value :: batch_count
    type(c_ptr), value :: result
            hipblasSdotBatchedFortran = &
        hipblasSdotBatched(handle, n, x, incx, y, incy, batch_count, result)
    return
end function hipblasSdotBatchedFortran

function hipblasDdotBatchedFortran(handle, n, x, incx, y, incy, batch_count, result) &
    bind(c, name='hipblasDdotBatchedFortran')
    use iso_c_binding
    use hipblas_enums
    implicit none
    integer(kind(HIPBLAS_STATUS_SUCCESS)) :: hipblasDdotBatchedFortran
    type(c_ptr), value :: handle
    integer(c_int), value :: n
    type(c_ptr), value :: x
    integer(c_int), value :: incx
    type(c_ptr), value :: y
    integer(c_int), value :: incy
    integer(c_int), value :: batch_count
    type(c_ptr), value :: result
            hipblasDdotBatchedFortran = &
        hipblasDdotBatched(handle, n, x, incx, y, incy, batch_count, result)
    return
end function hipblasDdotBatchedFortran

function hipblasHdotBatchedFortran(handle, n, x, incx, y, incy, batch_count, result) &
    bind(c, name='hipblasHdotBatchedFortran')
    use iso_c_binding
    use hipblas_enums
    implicit none
    integer(kind(HIPBLAS_STATUS_SUCCESS)) :: hipblasHdotBatchedFortran
    type(c_ptr), value :: handle
    integer(c_int), value :: n
    type(c_ptr), value :: x
    integer(c_int), value :: incx
    type(c_ptr), value :: y
    integer(c_int), value :: incy
    integer(c_int), value :: batch_count
    type(c_ptr), value :: result
            hipblasHdotBatchedFortran = &
        hipblasHdotBatched(handle, n, x, incx, y, incy, batch_count, result)
    return
end function hipblasHdotBatchedFortran

function hipblasBfdotBatchedFortran(handle, n, x, incx, y, incy, batch_count, result) &
    bind(c, name='hipblasBfdotBatchedFortran')
    use iso_c_binding
    use hipblas_enums
    implicit none
    integer(kind(HIPBLAS_STATUS_SUCCESS)) :: hipblasBfdotBatchedFortran
    type(c_ptr), value :: handle
    integer(c_int), value :: n
    type(c_ptr), value :: x
    integer(c_int), value :: incx
    type(c_ptr), value :: y
    integer(c_int), value :: incy
    integer(c_int), value :: batch_count
    type(c_ptr), value :: result
            hipblasBfdotBatchedFortran = &
        hipblasBfdotBatched(handle, n, x, incx, y, incy, batch_count, result)
    return
end function hipblasBfdotBatchedFortran

function hipblasCdotuBatchedFortran(handle, n, x, incx, y, incy, batch_count, result) &
    bind(c, name='hipblasCdotuBatchedFortran')
    use iso_c_binding
    use hipblas_enums
    implicit none
    integer(kind(HIPBLAS_STATUS_SUCCESS)) :: hipblasCdotuBatchedFortran
    type(c_ptr), value :: handle
    integer(c_int), value :: n
    type(c_ptr), value :: x
    integer(c_int), value :: incx
    type(c_ptr), value :: y
    integer(c_int), value :: incy
    integer(c_int), value :: batch_count
    type(c_ptr), value :: result
            hipblasCdotuBatchedFortran = &
        hipblasCdotuBatched(handle, n, x, incx, y, incy, batch_count, result)
    return
end function hipblasCdotuBatchedFortran

function hipblasCdotcBatchedFortran(handle, n, x, incx, y, incy, batch_count, result) &
    bind(c, name='hipblasCdotcBatchedFortran')
    use iso_c_binding
    use hipblas_enums
    implicit none
    integer(kind(HIPBLAS_STATUS_SUCCESS)) :: hipblasCdotcBatchedFortran
    type(c_ptr), value :: handle
    integer(c_int), value :: n
    type(c_ptr), value :: x
    integer(c_int), value :: incx
    type(c_ptr), value :: y
    integer(c_int), value :: incy
    integer(c_int), value :: batch_count
    type(c_ptr), value :: result
            hipblasCdotcBatchedFortran = &
        hipblasCdotcBatched(handle, n, x, incx, y, incy, batch_count, result)
    return
end function hipblasCdotcBatchedFortran

function hipblasZdotuBatchedFortran(handle, n, x, incx, y, incy, batch_count, result) &
    bind(c, name='hipblasZdotuBatchedFortran')
    use iso_c_binding
    use hipblas_enums
    implicit none
    integer(kind(HIPBLAS_STATUS_SUCCESS)) :: hipblasZdotuBatchedFortran
    type(c_ptr), value :: handle
    integer(c_int), value :: n
    type(c_ptr), value :: x
    integer(c_int), value :: incx
    type(c_ptr), value :: y
    integer(c_int), value :: incy
    integer(c_int), value :: batch_count
    type(c_ptr), value :: result
            hipblasZdotuBatchedFortran = &
        hipblasZdotuBatched(handle, n, x, incx, y, incy, batch_count, result)
    return
end function hipblasZdotuBatchedFortran

function hipblasZdotcBatchedFortran(handle, n, x, incx, y, incy, batch_count, result) &
    bind(c, name='hipblasZdotcBatchedFortran')
    use iso_c_binding
    use hipblas_enums
    implicit none
    integer(kind(HIPBLAS_STATUS_SUCCESS)) :: hipblasZdotcBatchedFortran
    type(c_ptr), value :: handle
    integer(c_int), value :: n
    type(c_ptr), value :: x
    integer(c_int), value :: incx
    type(c_ptr), value :: y
    integer(c_int), value :: incy
    integer(c_int), value :: batch_count
    type(c_ptr), value :: result
            hipblasZdotcBatchedFortran = &
        hipblasZdotcBatched(handle, n, x, incx, y, incy, batch_count, result)
    return
end function hipblasZdotcBatchedFortran

! dotStridedBatched
function hipblasSdotStridedBatchedFortran(handle, n, x, incx, stride_x, y, incy, stride_y, batch_count, result) &
    bind(c, name='hipblasSdotStridedBatchedFortran')
    use iso_c_binding
    use hipblas_enums
    implicit none
    integer(kind(HIPBLAS_STATUS_SUCCESS)) :: hipblasSdotStridedBatchedFortran
    type(c_ptr), value :: handle
    integer(c_int), value :: n
    type(c_ptr), value :: x
    integer(c_int), value :: incx
    integer(c_int64_t), value :: stride_x
    type(c_ptr), value :: y
    integer(c_int), value :: incy
    integer(c_int64_t), value :: stride_y
    integer(c_int), value :: batch_count
    type(c_ptr), value :: result
            hipblasSdotStridedBatchedFortran = &
        hipblasSdotStridedBatched(handle, n, x, incx, stride_x, y, incy, stride_y, batch_count, result)
    return
end function hipblasSdotStridedBatchedFortran

function hipblasDdotStridedBatchedFortran(handle, n, x, incx, stride_x, y, incy, stride_y, batch_count, result) &
    bind(c, name='hipblasDdotStridedBatchedFortran')
    use iso_c_binding
    use hipblas_enums
    implicit none
    integer(kind(HIPBLAS_STATUS_SUCCESS)) :: hipblasDdotStridedBatchedFortran
    type(c_ptr), value :: handle
    integer(c_int), value :: n
    type(c_ptr), value :: x
    integer(c_int), value :: incx
    integer(c_int64_t), value :: stride_x
    type(c_ptr), value :: y
    integer(c_int), value :: incy
    integer(c_int64_t), value :: stride_y
    integer(c_int), value :: batch_count
    type(c_ptr), value :: result
            hipblasDdotStridedBatchedFortran = &
        hipblasDdotStridedBatched(handle, n, x, incx, stride_x, y, incy, stride_y, batch_count, result)
    return
end function hipblasDdotStridedBatchedFortran

function hipblasHdotStridedBatchedFortran(handle, n, x, incx, stride_x, y, incy, stride_y, batch_count, result) &
    bind(c, name='hipblasHdotStridedBatchedFortran')
    use iso_c_binding
    use hipblas_enums
    implicit none
    integer(kind(HIPBLAS_STATUS_SUCCESS)) :: hipblasHdotStridedBatchedFortran
    type(c_ptr), value :: handle
    integer(c_int), value :: n
    type(c_ptr), value :: x
    integer(c_int), value :: incx
    integer(c_int64_t), value :: stride_x
    type(c_ptr), value :: y
    integer(c_int), value :: incy
    integer(c_int64_t), value :: stride_y
    integer(c_int), value :: batch_count
    type(c_ptr), value :: result
            hipblasHdotStridedBatchedFortran = &
        hipblasHdotStridedBatched(handle, n, x, incx, stride_x, y, incy, stride_y, batch_count, result)
    return
end function hipblasHdotStridedBatchedFortran

function hipblasBfdotStridedBatchedFortran(handle, n, x, incx, stride_x, y, incy, stride_y, batch_count, result) &
    bind(c, name='hipblasBfdotStridedBatchedFortran')
    use iso_c_binding
    use hipblas_enums
    implicit none
    integer(kind(HIPBLAS_STATUS_SUCCESS)) :: hipblasBfdotStridedBatchedFortran
    type(c_ptr), value :: handle
    integer(c_int), value :: n
    type(c_ptr), value :: x
    integer(c_int), value :: incx
    integer(c_int64_t), value :: stride_x
    type(c_ptr), value :: y
    integer(c_int), value :: incy
    integer(c_int64_t), value :: stride_y
    integer(c_int), value :: batch_count
    type(c_ptr), value :: result
            hipblasBfdotStridedBatchedFortran = &
        hipblasBfdotStridedBatched(handle, n, x, incx, stride_x, y, incy, stride_y, batch_count, result)
    return
end function hipblasBfdotStridedBatchedFortran

function hipblasCdotuStridedBatchedFortran(handle, n, x, incx, stride_x, y, incy, stride_y, batch_count, result) &
    bind(c, name='hipblasCdotuStridedBatchedFortran')
    use iso_c_binding
    use hipblas_enums
    implicit none
    integer(kind(HIPBLAS_STATUS_SUCCESS)) :: hipblasCdotuStridedBatchedFortran
    type(c_ptr), value :: handle
    integer(c_int), value :: n
    type(c_ptr), value :: x
    integer(c_int), value :: incx
    integer(c_int64_t), value :: stride_x
    type(c_ptr), value :: y
    integer(c_int), value :: incy
    integer(c_int64_t), value :: stride_y
    integer(c_int), value :: batch_count
    type(c_ptr), value :: result
            hipblasCdotuStridedBatchedFortran = &
        hipblasCdotuStridedBatched(handle, n, x, incx, stride_x, y, incy, stride_y, batch_count, result)
    return
end function hipblasCdotuStridedBatchedFortran

function hipblasCdotcStridedBatchedFortran(handle, n, x, incx, stride_x, y, incy, stride_y, batch_count, result) &
    bind(c, name='hipblasCdotcStridedBatchedFortran')
    use iso_c_binding
    use hipblas_enums
    implicit none
    integer(kind(HIPBLAS_STATUS_SUCCESS)) :: hipblasCdotcStridedBatchedFortran
    type(c_ptr), value :: handle
    integer(c_int), value :: n
    type(c_ptr), value :: x
    integer(c_int), value :: incx
    integer(c_int64_t), value :: stride_x
    type(c_ptr), value :: y
    integer(c_int), value :: incy
    integer(c_int64_t), value :: stride_y
    integer(c_int), value :: batch_count
    type(c_ptr), value :: result
            hipblasCdotcStridedBatchedFortran = &
        hipblasCdotcStridedBatched(handle, n, x, incx, stride_x, y, incy, stride_y, batch_count, result)
    return
end function hipblasCdotcStridedBatchedFortran

function hipblasZdotuStridedBatchedFortran(handle, n, x, incx, stride_x, y, incy, stride_y, batch_count, result) &
    bind(c, name='hipblasZdotuStridedBatchedFortran')
    use iso_c_binding
    use hipblas_enums
    implicit none
    integer(kind(HIPBLAS_STATUS_SUCCESS)) :: hipblasZdotuStridedBatchedFortran
    type(c_ptr), value :: handle
    integer(c_int), value :: n
    type(c_ptr), value :: x
    integer(c_int), value :: incx
    integer(c_int64_t), value :: stride_x
    type(c_ptr), value :: y
    integer(c_int), value :: incy
    integer(c_int64_t), value :: stride_y
    integer(c_int), value :: batch_count
    type(c_ptr), value :: result
            hipblasZdotuStridedBatchedFortran = &
        hipblasZdotuStridedBatched(handle, n, x, incx, stride_x, y, incy, stride_y, batch_count, result)
    return
end function hipblasZdotuStridedBatchedFortran

function hipblasZdotcStridedBatchedFortran(handle, n, x, incx, stride_x, y, incy, stride_y, batch_count, result) &
    bind(c, name='hipblasZdotcStridedBatchedFortran')
    use iso_c_binding
    use hipblas_enums
    implicit none
    integer(kind(HIPBLAS_STATUS_SUCCESS)) :: hipblasZdotcStridedBatchedFortran
    type(c_ptr), value :: handle
    integer(c_int), value :: n
    type(c_ptr), value :: x
    integer(c_int), value :: incx
    integer(c_int64_t), value :: stride_x
    type(c_ptr), value :: y
    integer(c_int), value :: incy
    integer(c_int64_t), value :: stride_y
    integer(c_int), value :: batch_count
    type(c_ptr), value :: result
            hipblasZdotcStridedBatchedFortran = &
        hipblasZdotcStridedBatched(handle, n, x, incx, stride_x, y, incy, stride_y, batch_count, result)
    return
end function hipblasZdotcStridedBatchedFortran

! swap
function hipblasSswapFortran(handle, n, x, incx, y, incy) &
    bind(c, name='hipblasSswapFortran')
    use iso_c_binding
    use hipblas_enums
    implicit none
    integer(kind(HIPBLAS_STATUS_SUCCESS)) :: hipblasSswapFortran
    type(c_ptr), value :: handle
    integer(c_int), value :: n
    type(c_ptr), value :: x
    integer(c_int), value :: incx
    type(c_ptr), value :: y
    integer(c_int), value :: incy
            hipblasSswapFortran = &
        hipblasSswap(handle, n, x, incx, y, incy)
    return
end function hipblasSswapFortran

function hipblasDswapFortran(handle, n, x, incx, y, incy) &
    bind(c, name='hipblasDswapFortran')
    use iso_c_binding
    use hipblas_enums
    implicit none
    integer(kind(HIPBLAS_STATUS_SUCCESS)) :: hipblasDswapFortran
    type(c_ptr), value :: handle
    integer(c_int), value :: n
    type(c_ptr), value :: x
    integer(c_int), value :: incx
    type(c_ptr), value :: y
    integer(c_int), value :: incy
            hipblasDswapFortran = &
        hipblasDswap(handle, n, x, incx, y, incy)
    return
end function hipblasDswapFortran

function hipblasCswapFortran(handle, n, x, incx, y, incy) &
    bind(c, name='hipblasCswapFortran')
    use iso_c_binding
    use hipblas_enums
    implicit none
    integer(kind(HIPBLAS_STATUS_SUCCESS)) :: hipblasCswapFortran
    type(c_ptr), value :: handle
    integer(c_int), value :: n
    type(c_ptr), value :: x
    integer(c_int), value :: incx
    type(c_ptr), value :: y
    integer(c_int), value :: incy
            hipblasCswapFortran = &
        hipblasCswap(handle, n, x, incx, y, incy)
    return
end function hipblasCswapFortran

function hipblasZswapFortran(handle, n, x, incx, y, incy) &
    bind(c, name='hipblasZswapFortran')
    use iso_c_binding
    use hipblas_enums
    implicit none
    integer(kind(HIPBLAS_STATUS_SUCCESS)) :: hipblasZswapFortran
    type(c_ptr), value :: handle
    integer(c_int), value :: n
    type(c_ptr), value :: x
    integer(c_int), value :: incx
    type(c_ptr), value :: y
    integer(c_int), value :: incy
            hipblasZswapFortran = &
        hipblasZswap(handle, n, x, incx, y, incy)
    return
end function hipblasZswapFortran

! swapBatched
function hipblasSswapBatchedFortran(handle, n, x, incx, y, incy, batch_count) &
    bind(c, name='hipblasSswapBatchedFortran')
    use iso_c_binding
    use hipblas_enums
    implicit none
    integer(kind(HIPBLAS_STATUS_SUCCESS)) :: hipblasSswapBatchedFortran
    type(c_ptr), value :: handle
    integer(c_int), value :: n
    type(c_ptr), value :: x
    integer(c_int), value :: incx
    type(c_ptr), value :: y
    integer(c_int), value :: incy
    integer(c_int), value :: batch_count
            hipblasSswapBatchedFortran = &
        hipblasSswapBatched(handle, n, x, incx, y, incy, batch_count)
    return
end function hipblasSswapBatchedFortran

function hipblasDswapBatchedFortran(handle, n, x, incx, y, incy, batch_count) &
    bind(c, name='hipblasDswapBatchedFortran')
    use iso_c_binding
    use hipblas_enums
    implicit none
    integer(kind(HIPBLAS_STATUS_SUCCESS)) :: hipblasDswapBatchedFortran
    type(c_ptr), value :: handle
    integer(c_int), value :: n
    type(c_ptr), value :: x
    integer(c_int), value :: incx
    type(c_ptr), value :: y
    integer(c_int), value :: incy
    integer(c_int), value :: batch_count
            hipblasDswapBatchedFortran = &
        hipblasDswapBatched(handle, n, x, incx, y, incy, batch_count)
    return
end function hipblasDswapBatchedFortran

function hipblasCswapBatchedFortran(handle, n, x, incx, y, incy, batch_count) &
    bind(c, name='hipblasCswapBatchedFortran')
    use iso_c_binding
    use hipblas_enums
    implicit none
    integer(kind(HIPBLAS_STATUS_SUCCESS)) :: hipblasCswapBatchedFortran
    type(c_ptr), value :: handle
    integer(c_int), value :: n
    type(c_ptr), value :: x
    integer(c_int), value :: incx
    type(c_ptr), value :: y
    integer(c_int), value :: incy
    integer(c_int), value :: batch_count
            hipblasCswapBatchedFortran = &
        hipblasCswapBatched(handle, n, x, incx, y, incy, batch_count)
    return
end function hipblasCswapBatchedFortran

function hipblasZswapBatchedFortran(handle, n, x, incx, y, incy, batch_count) &
    bind(c, name='hipblasZswapBatchedFortran')
    use iso_c_binding
    use hipblas_enums
    implicit none
    integer(kind(HIPBLAS_STATUS_SUCCESS)) :: hipblasZswapBatchedFortran
    type(c_ptr), value :: handle
    integer(c_int), value :: n
    type(c_ptr), value :: x
    integer(c_int), value :: incx
    type(c_ptr), value :: y
    integer(c_int), value :: incy
    integer(c_int), value :: batch_count
            hipblasZswapBatchedFortran = &
        hipblasZswapBatched(handle, n, x, incx, y, incy, batch_count)
    return
end function hipblasZswapBatchedFortran

! swapStridedBatched
function hipblasSswapStridedBatchedFortran(handle, n, x, incx, stride_x, y, incy, stride_y, batch_count) &
    bind(c, name='hipblasSswapStridedBatchedFortran')
    use iso_c_binding
    use hipblas_enums
    implicit none
    integer(kind(HIPBLAS_STATUS_SUCCESS)) :: hipblasSswapStridedBatchedFortran
    type(c_ptr), value :: handle
    integer(c_int), value :: n
    type(c_ptr), value :: x
    integer(c_int), value :: incx
    integer(c_int64_t), value :: stride_x
    type(c_ptr), value :: y
    integer(c_int), value :: incy
    integer(c_int64_t), value :: stride_y
    integer(c_int), value :: batch_count
            hipblasSswapStridedBatchedFortran = &
        hipblasSswapStridedBatched(handle, n, x, incx, stride_x, y, incy, stride_y, batch_count)
    return
end function hipblasSswapStridedBatchedFortran

function hipblasDswapStridedBatchedFortran(handle, n, x, incx, stride_x, y, incy, stride_y, batch_count) &
    bind(c, name='hipblasDswapStridedBatchedFortran')
    use iso_c_binding
    use hipblas_enums
    implicit none
    integer(kind(HIPBLAS_STATUS_SUCCESS)) :: hipblasDswapStridedBatchedFortran
    type(c_ptr), value :: handle
    integer(c_int), value :: n
    type(c_ptr), value :: x
    integer(c_int), value :: incx
    integer(c_int64_t), value :: stride_x
    type(c_ptr), value :: y
    integer(c_int), value :: incy
    integer(c_int64_t), value :: stride_y
    integer(c_int), value :: batch_count
            hipblasDswapStridedBatchedFortran = &
        hipblasDswapStridedBatched(handle, n, x, incx, stride_x, y, incy, stride_y, batch_count)
    return
end function hipblasDswapStridedBatchedFortran

function hipblasCswapStridedBatchedFortran(handle, n, x, incx, stride_x, y, incy, stride_y, batch_count) &
    bind(c, name='hipblasCswapStridedBatchedFortran')
    use iso_c_binding
    use hipblas_enums
    implicit none
    integer(kind(HIPBLAS_STATUS_SUCCESS)) :: hipblasCswapStridedBatchedFortran
    type(c_ptr), value :: handle
    integer(c_int), value :: n
    type(c_ptr), value :: x
    integer(c_int), value :: incx
    integer(c_int64_t), value :: stride_x
    type(c_ptr), value :: y
    integer(c_int), value :: incy
    integer(c_int64_t), value :: stride_y
    integer(c_int), value :: batch_count
            hipblasCswapStridedBatchedFortran = &
        hipblasCswapStridedBatched(handle, n, x, incx, stride_x, y, incy, stride_y, batch_count)
    return
end function hipblasCswapStridedBatchedFortran

function hipblasZswapStridedBatchedFortran(handle, n, x, incx, stride_x, y, incy, stride_y, batch_count) &
    bind(c, name='hipblasZswapStridedBatchedFortran')
    use iso_c_binding
    use hipblas_enums
    implicit none
    integer(kind(HIPBLAS_STATUS_SUCCESS)) :: hipblasZswapStridedBatchedFortran
    type(c_ptr), value :: handle
    integer(c_int), value :: n
    type(c_ptr), value :: x
    integer(c_int), value :: incx
    integer(c_int64_t), value :: stride_x
    type(c_ptr), value :: y
    integer(c_int), value :: incy
    integer(c_int64_t), value :: stride_y
    integer(c_int), value :: batch_count
            hipblasZswapStridedBatchedFortran = &
        hipblasZswapStridedBatched(handle, n, x, incx, stride_x, y, incy, stride_y, batch_count)
    return
end function hipblasZswapStridedBatchedFortran

! axpy
function hipblasHaxpyFortran(handle, n, alpha, x, incx, y, incy) &
    bind(c, name='hipblasHaxpyFortran')
    use iso_c_binding
    use hipblas_enums
    implicit none
    integer(kind(HIPBLAS_STATUS_SUCCESS)) :: hipblasHaxpyFortran
    type(c_ptr), value :: handle
    integer(c_int), value :: n
    type(c_ptr), value :: alpha
    type(c_ptr), value :: x
    integer(c_int), value :: incx
    type(c_ptr), value :: y
    integer(c_int), value :: incy
            hipblasHaxpyFortran = &
        hipblasHaxpy(handle, n, alpha, x, incx, y, incy)
    return
end function hipblasHaxpyFortran

function hipblasSaxpyFortran(handle, n, alpha, x, incx, y, incy) &
    bind(c, name='hipblasSaxpyFortran')
    use iso_c_binding
    use hipblas_enums
    implicit none
    integer(kind(HIPBLAS_STATUS_SUCCESS)) :: hipblasSaxpyFortran
    type(c_ptr), value :: handle
    integer(c_int), value :: n
    type(c_ptr), value :: alpha
    type(c_ptr), value :: x
    integer(c_int), value :: incx
    type(c_ptr), value :: y
    integer(c_int), value :: incy
            hipblasSaxpyFortran = &
        hipblasSaxpy(handle, n, alpha, x, incx, y, incy)
    return
end function hipblasSaxpyFortran

function hipblasDaxpyFortran(handle, n, alpha, x, incx, y, incy) &
    bind(c, name='hipblasDaxpyFortran')
    use iso_c_binding
    use hipblas_enums
    implicit none
    integer(kind(HIPBLAS_STATUS_SUCCESS)) :: hipblasDaxpyFortran
    type(c_ptr), value :: handle
    integer(c_int), value :: n
    type(c_ptr), value :: alpha
    type(c_ptr), value :: x
    integer(c_int), value :: incx
    type(c_ptr), value :: y
    integer(c_int), value :: incy
            hipblasDaxpyFortran = &
        hipblasDaxpy(handle, n, alpha, x, incx, y, incy)
    return
end function hipblasDaxpyFortran

function hipblasCaxpyFortran(handle, n, alpha, x, incx, y, incy) &
    bind(c, name='hipblasCaxpyFortran')
    use iso_c_binding
    use hipblas_enums
    implicit none
    integer(kind(HIPBLAS_STATUS_SUCCESS)) :: hipblasCaxpyFortran
    type(c_ptr), value :: handle
    integer(c_int), value :: n
    type(c_ptr), value :: alpha
    type(c_ptr), value :: x
    integer(c_int), value :: incx
    type(c_ptr), value :: y
    integer(c_int), value :: incy
            hipblasCaxpyFortran = &
        hipblasCaxpy(handle, n, alpha, x, incx, y, incy)
    return
end function hipblasCaxpyFortran

function hipblasZaxpyFortran(handle, n, alpha, x, incx, y, incy) &
    bind(c, name='hipblasZaxpyFortran')
    use iso_c_binding
    use hipblas_enums
    implicit none
    integer(kind(HIPBLAS_STATUS_SUCCESS)) :: hipblasZaxpyFortran
    type(c_ptr), value :: handle
    integer(c_int), value :: n
    type(c_ptr), value :: alpha
    type(c_ptr), value :: x
    integer(c_int), value :: incx
    type(c_ptr), value :: y
    integer(c_int), value :: incy
            hipblasZaxpyFortran = &
        hipblasZaxpy(handle, n, alpha, x, incx, y, incy)
    return
end function hipblasZaxpyFortran

! axpyBatched
function hipblasHaxpyBatchedFortran(handle, n, alpha, x, incx, y, incy, batch_count) &
    bind(c, name='hipblasHaxpyBatchedFortran')
    use iso_c_binding
    use hipblas_enums
    implicit none
    integer(kind(HIPBLAS_STATUS_SUCCESS)) :: hipblasHaxpyBatchedFortran
    type(c_ptr), value :: handle
    integer(c_int), value :: n
    type(c_ptr), value :: alpha
    type(c_ptr), value :: x
    integer(c_int), value :: incx
    type(c_ptr), value :: y
    integer(c_int), value :: incy
    integer(c_int), value :: batch_count
            hipblasHaxpyBatchedFortran = &
        hipblasHaxpyBatched(handle, n, alpha, x, incx, y, incy, batch_count)
    return
end function hipblasHaxpyBatchedFortran

function hipblasSaxpyBatchedFortran(handle, n, alpha, x, incx, y, incy, batch_count) &
    bind(c, name='hipblasSaxpyBatchedFortran')
    use iso_c_binding
    use hipblas_enums
    implicit none
    integer(kind(HIPBLAS_STATUS_SUCCESS)) :: hipblasSaxpyBatchedFortran
    type(c_ptr), value :: handle
    integer(c_int), value :: n
    type(c_ptr), value :: alpha
    type(c_ptr), value :: x
    integer(c_int), value :: incx
    type(c_ptr), value :: y
    integer(c_int), value :: incy
    integer(c_int), value :: batch_count
            hipblasSaxpyBatchedFortran = &
        hipblasSaxpyBatched(handle, n, alpha, x, incx, y, incy, batch_count)
    return
end function hipblasSaxpyBatchedFortran

function hipblasDaxpyBatchedFortran(handle, n, alpha, x, incx, y, incy, batch_count) &
    bind(c, name='hipblasDaxpyBatchedFortran')
    use iso_c_binding
    use hipblas_enums
    implicit none
    integer(kind(HIPBLAS_STATUS_SUCCESS)) :: hipblasDaxpyBatchedFortran
    type(c_ptr), value :: handle
    integer(c_int), value :: n
    type(c_ptr), value :: alpha
    type(c_ptr), value :: x
    integer(c_int), value :: incx
    type(c_ptr), value :: y
    integer(c_int), value :: incy
    integer(c_int), value :: batch_count
            hipblasDaxpyBatchedFortran = &
        hipblasDaxpyBatched(handle, n, alpha, x, incx, y, incy, batch_count)
    return
end function hipblasDaxpyBatchedFortran

function hipblasCaxpyBatchedFortran(handle, n, alpha, x, incx, y, incy, batch_count) &
    bind(c, name='hipblasCaxpyBatchedFortran')
    use iso_c_binding
    use hipblas_enums
    implicit none
    integer(kind(HIPBLAS_STATUS_SUCCESS)) :: hipblasCaxpyBatchedFortran
    type(c_ptr), value :: handle
    integer(c_int), value :: n
    type(c_ptr), value :: alpha
    type(c_ptr), value :: x
    integer(c_int), value :: incx
    type(c_ptr), value :: y
    integer(c_int), value :: incy
    integer(c_int), value :: batch_count
            hipblasCaxpyBatchedFortran = &
        hipblasCaxpyBatched(handle, n, alpha, x, incx, y, incy, batch_count)
    return
end function hipblasCaxpyBatchedFortran

function hipblasZaxpyBatchedFortran(handle, n, alpha, x, incx, y, incy, batch_count) &
    bind(c, name='hipblasZaxpyBatchedFortran')
    use iso_c_binding
    use hipblas_enums
    implicit none
    integer(kind(HIPBLAS_STATUS_SUCCESS)) :: hipblasZaxpyBatchedFortran
    type(c_ptr), value :: handle
    integer(c_int), value :: n
    type(c_ptr), value :: alpha
    type(c_ptr), value :: x
    integer(c_int), value :: incx
    type(c_ptr), value :: y
    integer(c_int), value :: incy
    integer(c_int), value :: batch_count
            hipblasZaxpyBatchedFortran = &
        hipblasZaxpyBatched(handle, n, alpha, x, incx, y, incy, batch_count)
    return
end function hipblasZaxpyBatchedFortran

! axpyStridedBatched
function hipblasHaxpyStridedBatchedFortran(handle, n, alpha, x, incx, stride_x, y, incy, stride_y, batch_count) &
    bind(c, name='hipblasHaxpyStridedBatchedFortran')
    use iso_c_binding
    use hipblas_enums
    implicit none
    integer(kind(HIPBLAS_STATUS_SUCCESS)) :: hipblasHaxpyStridedBatchedFortran
    type(c_ptr), value :: handle
    integer(c_int), value :: n
    type(c_ptr), value :: alpha
    type(c_ptr), value :: x
    integer(c_int), value :: incx
    integer(c_int64_t), value :: stride_x
    type(c_ptr), value :: y
    integer(c_int), value :: incy
    integer(c_int64_t), value :: stride_y
    integer(c_int), value :: batch_count
            hipblasHaxpyStridedBatchedFortran = &
        hipblasHaxpyStridedBatched(handle, n, alpha, x, incx, stride_x, y, incy, stride_y, batch_count)
    return
end function hipblasHaxpyStridedBatchedFortran

function hipblasSaxpyStridedBatchedFortran(handle, n, alpha, x, incx, stride_x, y, incy, stride_y, batch_count) &
    bind(c, name='hipblasSaxpyStridedBatchedFortran')
    use iso_c_binding
    use hipblas_enums
    implicit none
    integer(kind(HIPBLAS_STATUS_SUCCESS)) :: hipblasSaxpyStridedBatchedFortran
    type(c_ptr), value :: handle
    integer(c_int), value :: n
    type(c_ptr), value :: alpha
    type(c_ptr), value :: x
    integer(c_int), value :: incx
    integer(c_int64_t), value :: stride_x
    type(c_ptr), value :: y
    integer(c_int), value :: incy
    integer(c_int64_t), value :: stride_y
    integer(c_int), value :: batch_count
            hipblasSaxpyStridedBatchedFortran = &
        hipblasSaxpyStridedBatched(handle, n, alpha, x, incx, stride_x, y, incy, stride_y, batch_count)
    return
end function hipblasSaxpyStridedBatchedFortran

function hipblasDaxpyStridedBatchedFortran(handle, n, alpha, x, incx, stride_x, y, incy, stride_y, batch_count) &
    bind(c, name='hipblasDaxpyStridedBatchedFortran')
    use iso_c_binding
    use hipblas_enums
    implicit none
    integer(kind(HIPBLAS_STATUS_SUCCESS)) :: hipblasDaxpyStridedBatchedFortran
    type(c_ptr), value :: handle
    integer(c_int), value :: n
    type(c_ptr), value :: alpha
    type(c_ptr), value :: x
    integer(c_int), value :: incx
    integer(c_int64_t), value :: stride_x
    type(c_ptr), value :: y
    integer(c_int), value :: incy
    integer(c_int64_t), value :: stride_y
    integer(c_int), value :: batch_count
            hipblasDaxpyStridedBatchedFortran = &
        hipblasDaxpyStridedBatched(handle, n, alpha, x, incx, stride_x, y, incy, stride_y, batch_count)
    return
end function hipblasDaxpyStridedBatchedFortran

function hipblasCaxpyStridedBatchedFortran(handle, n, alpha, x, incx, stride_x, y, incy, stride_y, batch_count) &
    bind(c, name='hipblasCaxpyStridedBatchedFortran')
    use iso_c_binding
    use hipblas_enums
    implicit none
    integer(kind(HIPBLAS_STATUS_SUCCESS)) :: hipblasCaxpyStridedBatchedFortran
    type(c_ptr), value :: handle
    integer(c_int), value :: n
    type(c_ptr), value :: alpha
    type(c_ptr), value :: x
    integer(c_int), value :: incx
    integer(c_int64_t), value :: stride_x
    type(c_ptr), value :: y
    integer(c_int), value :: incy
    integer(c_int64_t), value :: stride_y
    integer(c_int), value :: batch_count
            hipblasCaxpyStridedBatchedFortran = &
        hipblasCaxpyStridedBatched(handle, n, alpha, x, incx, stride_x, y, incy, stride_y, batch_count)
    return
end function hipblasCaxpyStridedBatchedFortran

function hipblasZaxpyStridedBatchedFortran(handle, n, alpha, x, incx, stride_x, y, incy, stride_y, batch_count) &
    bind(c, name='hipblasZaxpyStridedBatchedFortran')
    use iso_c_binding
    use hipblas_enums
    implicit none
    integer(kind(HIPBLAS_STATUS_SUCCESS)) :: hipblasZaxpyStridedBatchedFortran
    type(c_ptr), value :: handle
    integer(c_int), value :: n
    type(c_ptr), value :: alpha
    type(c_ptr), value :: x
    integer(c_int), value :: incx
    integer(c_int64_t), value :: stride_x
    type(c_ptr), value :: y
    integer(c_int), value :: incy
    integer(c_int64_t), value :: stride_y
    integer(c_int), value :: batch_count
            hipblasZaxpyStridedBatchedFortran = &
        hipblasZaxpyStridedBatched(handle, n, alpha, x, incx, stride_x, y, incy, stride_y, batch_count)
    return
end function hipblasZaxpyStridedBatchedFortran

! asum
function hipblasSasumFortran(handle, n, x, incx, result) &
    bind(c, name='hipblasSasumFortran')
    use iso_c_binding
    use hipblas_enums
    implicit none
    integer(kind(HIPBLAS_STATUS_SUCCESS)) :: hipblasSasumFortran
    type(c_ptr), value :: handle
    integer(c_int), value :: n
    type(c_ptr), value :: x
    integer(c_int), value :: incx
    type(c_ptr), value :: result
            hipblasSasumFortran = &
        hipblasSasum(handle, n, x, incx, result)
    return
end function hipblasSasumFortran

function hipblasDasumFortran(handle, n, x, incx, result) &
    bind(c, name='hipblasDasumFortran')
    use iso_c_binding
    use hipblas_enums
    implicit none
    integer(kind(HIPBLAS_STATUS_SUCCESS)) :: hipblasDasumFortran
    type(c_ptr), value :: handle
    integer(c_int), value :: n
    type(c_ptr), value :: x
    integer(c_int), value :: incx
    type(c_ptr), value :: result
            hipblasDasumFortran = &
        hipblasDasum(handle, n, x, incx, result)
    return
end function hipblasDasumFortran

function hipblasScasumFortran(handle, n, x, incx, result) &
    bind(c, name='hipblasScasumFortran')
    use iso_c_binding
    use hipblas_enums
    implicit none
    integer(kind(HIPBLAS_STATUS_SUCCESS)) :: hipblasScasumFortran
    type(c_ptr), value :: handle
    integer(c_int), value :: n
    type(c_ptr), value :: x
    integer(c_int), value :: incx
    type(c_ptr), value :: result
            hipblasScasumFortran = &
        hipblasScasum(handle, n, x, incx, result)
    return
end function hipblasScasumFortran

function hipblasDzasumFortran(handle, n, x, incx, result) &
    bind(c, name='hipblasDzasumFortran')
    use iso_c_binding
    use hipblas_enums
    implicit none
    integer(kind(HIPBLAS_STATUS_SUCCESS)) :: hipblasDzasumFortran
    type(c_ptr), value :: handle
    integer(c_int), value :: n
    type(c_ptr), value :: x
    integer(c_int), value :: incx
    type(c_ptr), value :: result
            hipblasDzasumFortran = &
        hipblasDzasum(handle, n, x, incx, result)
    return
end function hipblasDzasumFortran

! asumBatched
function hipblasSasumBatchedFortran(handle, n, x, incx, batch_count, result) &
    bind(c, name='hipblasSasumBatchedFortran')
    use iso_c_binding
    use hipblas_enums
    implicit none
    integer(kind(HIPBLAS_STATUS_SUCCESS)) :: hipblasSasumBatchedFortran
    type(c_ptr), value :: handle
    integer(c_int), value :: n
    type(c_ptr), value :: x
    integer(c_int), value :: incx
    integer(c_int), value :: batch_count
    type(c_ptr), value :: result
            hipblasSasumBatchedFortran = &
        hipblasSasumBatched(handle, n, x, incx, batch_count, result)
    return
end function hipblasSasumBatchedFortran

function hipblasDasumBatchedFortran(handle, n, x, incx, batch_count, result) &
    bind(c, name='hipblasDasumBatchedFortran')
    use iso_c_binding
    use hipblas_enums
    implicit none
    integer(kind(HIPBLAS_STATUS_SUCCESS)) :: hipblasDasumBatchedFortran
    type(c_ptr), value :: handle
    integer(c_int), value :: n
    type(c_ptr), value :: x
    integer(c_int), value :: incx
    integer(c_int), value :: batch_count
    type(c_ptr), value :: result
            hipblasDasumBatchedFortran = &
        hipblasDasumBatched(handle, n, x, incx, batch_count, result)
    return
end function hipblasDasumBatchedFortran

function hipblasScasumBatchedFortran(handle, n, x, incx, batch_count, result) &
    bind(c, name='hipblasScasumBatchedFortran')
    use iso_c_binding
    use hipblas_enums
    implicit none
    integer(kind(HIPBLAS_STATUS_SUCCESS)) :: hipblasScasumBatchedFortran
    type(c_ptr), value :: handle
    integer(c_int), value :: n
    type(c_ptr), value :: x
    integer(c_int), value :: incx
    integer(c_int), value :: batch_count
    type(c_ptr), value :: result
            hipblasScasumBatchedFortran = &
        hipblasScasumBatched(handle, n, x, incx, batch_count, result)
    return
end function hipblasScasumBatchedFortran

function hipblasDzasumBatchedFortran(handle, n, x, incx, batch_count, result) &
    bind(c, name='hipblasDzasumBatchedFortran')
    use iso_c_binding
    use hipblas_enums
    implicit none
    integer(kind(HIPBLAS_STATUS_SUCCESS)) :: hipblasDzasumBatchedFortran
    type(c_ptr), value :: handle
    integer(c_int), value :: n
    type(c_ptr), value :: x
    integer(c_int), value :: incx
    integer(c_int), value :: batch_count
    type(c_ptr), value :: result
            hipblasDzasumBatchedFortran = &
        hipblasDzasumBatched(handle, n, x, incx, batch_count, result)
    return
end function hipblasDzasumBatchedFortran

! asumStridedBatched
function hipblasSasumStridedBatchedFortran(handle, n, x, incx, stride_x, batch_count, result) &
    bind(c, name='hipblasSasumStridedBatchedFortran')
    use iso_c_binding
    use hipblas_enums
    implicit none
    integer(kind(HIPBLAS_STATUS_SUCCESS)) :: hipblasSasumStridedBatchedFortran
    type(c_ptr), value :: handle
    integer(c_int), value :: n
    type(c_ptr), value :: x
    integer(c_int), value :: incx
    integer(c_int64_t), value :: stride_x
    integer(c_int), value :: batch_count
    type(c_ptr), value :: result
            hipblasSasumStridedBatchedFortran = &
        hipblasSasumStridedBatched(handle, n, x, incx, stride_x, batch_count, result)
    return
end function hipblasSasumStridedBatchedFortran

function hipblasDasumStridedBatchedFortran(handle, n, x, incx, stride_x, batch_count, result) &
    bind(c, name='hipblasDasumStridedBatchedFortran')
    use iso_c_binding
    use hipblas_enums
    implicit none
    integer(kind(HIPBLAS_STATUS_SUCCESS)) :: hipblasDasumStridedBatchedFortran
    type(c_ptr), value :: handle
    integer(c_int), value :: n
    type(c_ptr), value :: x
    integer(c_int), value :: incx
    integer(c_int64_t), value :: stride_x
    integer(c_int), value :: batch_count
    type(c_ptr), value :: result
            hipblasDasumStridedBatchedFortran = &
        hipblasDasumStridedBatched(handle, n, x, incx, stride_x, batch_count, result)
    return
end function hipblasDasumStridedBatchedFortran

function hipblasScasumStridedBatchedFortran(handle, n, x, incx, stride_x, batch_count, result) &
    bind(c, name='hipblasScasumStridedBatchedFortran')
    use iso_c_binding
    use hipblas_enums
    implicit none
    integer(kind(HIPBLAS_STATUS_SUCCESS)) :: hipblasScasumStridedBatchedFortran
    type(c_ptr), value :: handle
    integer(c_int), value :: n
    type(c_ptr), value :: x
    integer(c_int), value :: incx
    integer(c_int64_t), value :: stride_x
    integer(c_int), value :: batch_count
    type(c_ptr), value :: result
            hipblasScasumStridedBatchedFortran = &
        hipblasScasumStridedBatched(handle, n, x, incx, stride_x, batch_count, result)
    return
end function hipblasScasumStridedBatchedFortran

function hipblasDzasumStridedBatchedFortran(handle, n, x, incx, stride_x, batch_count, result) &
    bind(c, name='hipblasDzasumStridedBatchedFortran')
    use iso_c_binding
    use hipblas_enums
    implicit none
    integer(kind(HIPBLAS_STATUS_SUCCESS)) :: hipblasDzasumStridedBatchedFortran
    type(c_ptr), value :: handle
    integer(c_int), value :: n
    type(c_ptr), value :: x
    integer(c_int), value :: incx
    integer(c_int64_t), value :: stride_x
    integer(c_int), value :: batch_count
    type(c_ptr), value :: result
            hipblasDzasumStridedBatchedFortran = &
        hipblasDzasumStridedBatched(handle, n, x, incx, stride_x, batch_count, result)
    return
end function hipblasDzasumStridedBatchedFortran

! nrm2
function hipblasSnrm2Fortran(handle, n, x, incx, result) &
    bind(c, name='hipblasSnrm2Fortran')
    use iso_c_binding
    use hipblas_enums
    implicit none
    integer(kind(HIPBLAS_STATUS_SUCCESS)) :: hipblasSnrm2Fortran
    type(c_ptr), value :: handle
    integer(c_int), value :: n
    type(c_ptr), value :: x
    integer(c_int), value :: incx
    type(c_ptr), value :: result
            hipblasSnrm2Fortran = &
        hipblasSnrm2(handle, n, x, incx, result)
    return
end function hipblasSnrm2Fortran

function hipblasDnrm2Fortran(handle, n, x, incx, result) &
    bind(c, name='hipblasDnrm2Fortran')
    use iso_c_binding
    use hipblas_enums
    implicit none
    integer(kind(HIPBLAS_STATUS_SUCCESS)) :: hipblasDnrm2Fortran
    type(c_ptr), value :: handle
    integer(c_int), value :: n
    type(c_ptr), value :: x
    integer(c_int), value :: incx
    type(c_ptr), value :: result
            hipblasDnrm2Fortran = &
        hipblasDnrm2(handle, n, x, incx, result)
    return
end function hipblasDnrm2Fortran

function hipblasScnrm2Fortran(handle, n, x, incx, result) &
    bind(c, name='hipblasScnrm2Fortran')
    use iso_c_binding
    use hipblas_enums
    implicit none
    integer(kind(HIPBLAS_STATUS_SUCCESS)) :: hipblasScnrm2Fortran
    type(c_ptr), value :: handle
    integer(c_int), value :: n
    type(c_ptr), value :: x
    integer(c_int), value :: incx
    type(c_ptr), value :: result
            hipblasScnrm2Fortran = &
        hipblasScnrm2(handle, n, x, incx, result)
    return
end function hipblasScnrm2Fortran

function hipblasDznrm2Fortran(handle, n, x, incx, result) &
    bind(c, name='hipblasDznrm2Fortran')
    use iso_c_binding
    use hipblas_enums
    implicit none
    integer(kind(HIPBLAS_STATUS_SUCCESS)) :: hipblasDznrm2Fortran
    type(c_ptr), value :: handle
    integer(c_int), value :: n
    type(c_ptr), value :: x
    integer(c_int), value :: incx
    type(c_ptr), value :: result
            hipblasDznrm2Fortran = &
        hipblasDznrm2(handle, n, x, incx, result)
    return
end function hipblasDznrm2Fortran

! nrm2Batched
function hipblasSnrm2BatchedFortran(handle, n, x, incx, batch_count, result) &
    bind(c, name='hipblasSnrm2BatchedFortran')
    use iso_c_binding
    use hipblas_enums
    implicit none
    integer(kind(HIPBLAS_STATUS_SUCCESS)) :: hipblasSnrm2BatchedFortran
    type(c_ptr), value :: handle
    integer(c_int), value :: n
    type(c_ptr), value :: x
    integer(c_int), value :: incx
    integer(c_int), value :: batch_count
    type(c_ptr), value :: result
            hipblasSnrm2BatchedFortran = &
        hipblasSnrm2Batched(handle, n, x, incx, batch_count, result)
    return
end function hipblasSnrm2BatchedFortran

function hipblasDnrm2BatchedFortran(handle, n, x, incx, batch_count, result) &
    bind(c, name='hipblasDnrm2BatchedFortran')
    use iso_c_binding
    use hipblas_enums
    implicit none
    integer(kind(HIPBLAS_STATUS_SUCCESS)) :: hipblasDnrm2BatchedFortran
    type(c_ptr), value :: handle
    integer(c_int), value :: n
    type(c_ptr), value :: x
    integer(c_int), value :: incx
    integer(c_int), value :: batch_count
    type(c_ptr), value :: result
            hipblasDnrm2BatchedFortran = &
        hipblasDnrm2Batched(handle, n, x, incx, batch_count, result)
    return
end function hipblasDnrm2BatchedFortran

function hipblasScnrm2BatchedFortran(handle, n, x, incx, batch_count, result) &
    bind(c, name='hipblasScnrm2BatchedFortran')
    use iso_c_binding
    use hipblas_enums
    implicit none
    integer(kind(HIPBLAS_STATUS_SUCCESS)) :: hipblasScnrm2BatchedFortran
    type(c_ptr), value :: handle
    integer(c_int), value :: n
    type(c_ptr), value :: x
    integer(c_int), value :: incx
    integer(c_int), value :: batch_count
    type(c_ptr), value :: result
            hipblasScnrm2BatchedFortran = &
        hipblasScnrm2Batched(handle, n, x, incx, batch_count, result)
    return
end function hipblasScnrm2BatchedFortran

function hipblasDznrm2BatchedFortran(handle, n, x, incx, batch_count, result) &
    bind(c, name='hipblasDznrm2BatchedFortran')
    use iso_c_binding
    use hipblas_enums
    implicit none
    integer(kind(HIPBLAS_STATUS_SUCCESS)) :: hipblasDznrm2BatchedFortran
    type(c_ptr), value :: handle
    integer(c_int), value :: n
    type(c_ptr), value :: x
    integer(c_int), value :: incx
    integer(c_int), value :: batch_count
    type(c_ptr), value :: result
            hipblasDznrm2BatchedFortran = &
        hipblasDznrm2Batched(handle, n, x, incx, batch_count, result)
    return
end function hipblasDznrm2BatchedFortran

! nrm2StridedBatched
function hipblasSnrm2StridedBatchedFortran(handle, n, x, incx, stride_x, batch_count, result) &
    bind(c, name='hipblasSnrm2StridedBatchedFortran')
    use iso_c_binding
    use hipblas_enums
    implicit none
    integer(kind(HIPBLAS_STATUS_SUCCESS)) :: hipblasSnrm2StridedBatchedFortran
    type(c_ptr), value :: handle
    integer(c_int), value :: n
    type(c_ptr), value :: x
    integer(c_int), value :: incx
    integer(c_int64_t), value :: stride_x
    integer(c_int), value :: batch_count
    type(c_ptr), value :: result
            hipblasSnrm2StridedBatchedFortran = &
        hipblasSnrm2StridedBatched(handle, n, x, incx, stride_x, batch_count, result)
    return
end function hipblasSnrm2StridedBatchedFortran

function hipblasDnrm2StridedBatchedFortran(handle, n, x, incx, stride_x, batch_count, result) &
    bind(c, name='hipblasDnrm2StridedBatchedFortran')
    use iso_c_binding
    use hipblas_enums
    implicit none
    integer(kind(HIPBLAS_STATUS_SUCCESS)) :: hipblasDnrm2StridedBatchedFortran
    type(c_ptr), value :: handle
    integer(c_int), value :: n
    type(c_ptr), value :: x
    integer(c_int), value :: incx
    integer(c_int64_t), value :: stride_x
    integer(c_int), value :: batch_count
    type(c_ptr), value :: result
            hipblasDnrm2StridedBatchedFortran = &
        hipblasDnrm2StridedBatched(handle, n, x, incx, stride_x, batch_count, result)
    return
end function hipblasDnrm2StridedBatchedFortran

function hipblasScnrm2StridedBatchedFortran(handle, n, x, incx, stride_x, batch_count, result) &
    bind(c, name='hipblasScnrm2StridedBatchedFortran')
    use iso_c_binding
    use hipblas_enums
    implicit none
    integer(kind(HIPBLAS_STATUS_SUCCESS)) :: hipblasScnrm2StridedBatchedFortran
    type(c_ptr), value :: handle
    integer(c_int), value :: n
    type(c_ptr), value :: x
    integer(c_int), value :: incx
    integer(c_int64_t), value :: stride_x
    integer(c_int), value :: batch_count
    type(c_ptr), value :: result
            hipblasScnrm2StridedBatchedFortran = &
        hipblasScnrm2StridedBatched(handle, n, x, incx, stride_x, batch_count, result)
    return
end function hipblasScnrm2StridedBatchedFortran

function hipblasDznrm2StridedBatchedFortran(handle, n, x, incx, stride_x, batch_count, result) &
    bind(c, name='hipblasDznrm2StridedBatchedFortran')
    use iso_c_binding
    use hipblas_enums
    implicit none
    integer(kind(HIPBLAS_STATUS_SUCCESS)) :: hipblasDznrm2StridedBatchedFortran
    type(c_ptr), value :: handle
    integer(c_int), value :: n
    type(c_ptr), value :: x
    integer(c_int), value :: incx
    integer(c_int64_t), value :: stride_x
    integer(c_int), value :: batch_count
    type(c_ptr), value :: result
            hipblasDznrm2StridedBatchedFortran = &
        hipblasDznrm2StridedBatched(handle, n, x, incx, stride_x, batch_count, result)
    return
end function hipblasDznrm2StridedBatchedFortran

! amax
function hipblasIsamaxFortran(handle, n, x, incx, result) &
    bind(c, name='hipblasIsamaxFortran')
    use iso_c_binding
    use hipblas_enums
    implicit none
    integer(kind(HIPBLAS_STATUS_SUCCESS)) :: hipblasIsamaxFortran
    type(c_ptr), value :: handle
    integer(c_int), value :: n
    type(c_ptr), value :: x
    integer(c_int), value :: incx
    type(c_ptr), value :: result
            hipblasIsamaxFortran = &
        hipblasIsamax(handle, n, x, incx, result)
    return
end function hipblasIsamaxFortran

function hipblasIdamaxFortran(handle, n, x, incx, result) &
    bind(c, name='hipblasIdamaxFortran')
    use iso_c_binding
    use hipblas_enums
    implicit none
    integer(kind(HIPBLAS_STATUS_SUCCESS)) :: hipblasIdamaxFortran
    type(c_ptr), value :: handle
    integer(c_int), value :: n
    type(c_ptr), value :: x
    integer(c_int), value :: incx
    type(c_ptr), value :: result
            hipblasIdamaxFortran = &
        hipblasIdamax(handle, n, x, incx, result)
    return
end function hipblasIdamaxFortran

function hipblasIcamaxFortran(handle, n, x, incx, result) &
    bind(c, name='hipblasIcamaxFortran')
    use iso_c_binding
    use hipblas_enums
    implicit none
    integer(kind(HIPBLAS_STATUS_SUCCESS)) :: hipblasIcamaxFortran
    type(c_ptr), value :: handle
    integer(c_int), value :: n
    type(c_ptr), value :: x
    integer(c_int), value :: incx
    type(c_ptr), value :: result
            hipblasIcamaxFortran = &
        hipblasIcamax(handle, n, x, incx, result)
    return
end function hipblasIcamaxFortran

function hipblasIzamaxFortran(handle, n, x, incx, result) &
    bind(c, name='hipblasIzamaxFortran')
    use iso_c_binding
    use hipblas_enums
    implicit none
    integer(kind(HIPBLAS_STATUS_SUCCESS)) :: hipblasIzamaxFortran
    type(c_ptr), value :: handle
    integer(c_int), value :: n
    type(c_ptr), value :: x
    integer(c_int), value :: incx
    type(c_ptr), value :: result
            hipblasIzamaxFortran = &
        hipblasIzamax(handle, n, x, incx, result)
    return
end function hipblasIzamaxFortran

! amaxBatched
function hipblasIsamaxBatchedFortran(handle, n, x, incx, batch_count, result) &
    bind(c, name='hipblasIsamaxBatchedFortran')
    use iso_c_binding
    use hipblas_enums
    implicit none
    integer(kind(HIPBLAS_STATUS_SUCCESS)) :: hipblasIsamaxBatchedFortran
    type(c_ptr), value :: handle
    integer(c_int), value :: n
    type(c_ptr), value :: x
    integer(c_int), value :: incx
    integer(c_int), value :: batch_count
    type(c_ptr), value :: result
            hipblasIsamaxBatchedFortran = &
        hipblasIsamaxBatched(handle, n, x, incx, batch_count, result)
    return
end function hipblasIsamaxBatchedFortran

function hipblasIdamaxBatchedFortran(handle, n, x, incx, batch_count, result) &
    bind(c, name='hipblasIdamaxBatchedFortran')
    use iso_c_binding
    use hipblas_enums
    implicit none
    integer(kind(HIPBLAS_STATUS_SUCCESS)) :: hipblasIdamaxBatchedFortran
    type(c_ptr), value :: handle
    integer(c_int), value :: n
    type(c_ptr), value :: x
    integer(c_int), value :: incx
    integer(c_int), value :: batch_count
    type(c_ptr), value :: result
            hipblasIdamaxBatchedFortran = &
        hipblasIdamaxBatched(handle, n, x, incx, batch_count, result)
    return
end function hipblasIdamaxBatchedFortran

function hipblasIcamaxBatchedFortran(handle, n, x, incx, batch_count, result) &
    bind(c, name='hipblasIcamaxBatchedFortran')
    use iso_c_binding
    use hipblas_enums
    implicit none
    integer(kind(HIPBLAS_STATUS_SUCCESS)) :: hipblasIcamaxBatchedFortran
    type(c_ptr), value :: handle
    integer(c_int), value :: n
    type(c_ptr), value :: x
    integer(c_int), value :: incx
    integer(c_int), value :: batch_count
    type(c_ptr), value :: result
            hipblasIcamaxBatchedFortran = &
        hipblasIcamaxBatched(handle, n, x, incx, batch_count, result)
    return
end function hipblasIcamaxBatchedFortran

function hipblasIzamaxBatchedFortran(handle, n, x, incx, batch_count, result) &
    bind(c, name='hipblasIzamaxBatchedFortran')
    use iso_c_binding
    use hipblas_enums
    implicit none
    integer(kind(HIPBLAS_STATUS_SUCCESS)) :: hipblasIzamaxBatchedFortran
    type(c_ptr), value :: handle
    integer(c_int), value :: n
    type(c_ptr), value :: x
    integer(c_int), value :: incx
    integer(c_int), value :: batch_count
    type(c_ptr), value :: result
            hipblasIzamaxBatchedFortran = &
        hipblasIzamaxBatched(handle, n, x, incx, batch_count, result)
    return
end function hipblasIzamaxBatchedFortran

! amaxStridedBatched
function hipblasIsamaxStridedBatchedFortran(handle, n, x, incx, stride_x, batch_count, result) &
    bind(c, name='hipblasIsamaxStridedBatchedFortran')
    use iso_c_binding
    use hipblas_enums
    implicit none
    integer(kind(HIPBLAS_STATUS_SUCCESS)) :: hipblasIsamaxStridedBatchedFortran
    type(c_ptr), value :: handle
    integer(c_int), value :: n
    type(c_ptr), value :: x
    integer(c_int), value :: incx
    integer(c_int64_t), value :: stride_x
    integer(c_int), value :: batch_count
    type(c_ptr), value :: result
            hipblasIsamaxStridedBatchedFortran = &
        hipblasIsamaxStridedBatched(handle, n, x, incx, stride_x, batch_count, result)
    return
end function hipblasIsamaxStridedBatchedFortran

function hipblasIdamaxStridedBatchedFortran(handle, n, x, incx, stride_x, batch_count, result) &
    bind(c, name='hipblasIdamaxStridedBatchedFortran')
    use iso_c_binding
    use hipblas_enums
    implicit none
    integer(kind(HIPBLAS_STATUS_SUCCESS)) :: hipblasIdamaxStridedBatchedFortran
    type(c_ptr), value :: handle
    integer(c_int), value :: n
    type(c_ptr), value :: x
    integer(c_int), value :: incx
    integer(c_int64_t), value :: stride_x
    integer(c_int), value :: batch_count
    type(c_ptr), value :: result
            hipblasIdamaxStridedBatchedFortran = &
        hipblasIdamaxStridedBatched(handle, n, x, incx, stride_x, batch_count, result)
    return
end function hipblasIdamaxStridedBatchedFortran

function hipblasIcamaxStridedBatchedFortran(handle, n, x, incx, stride_x, batch_count, result) &
    bind(c, name='hipblasIcamaxStridedBatchedFortran')
    use iso_c_binding
    use hipblas_enums
    implicit none
    integer(kind(HIPBLAS_STATUS_SUCCESS)) :: hipblasIcamaxStridedBatchedFortran
    type(c_ptr), value :: handle
    integer(c_int), value :: n
    type(c_ptr), value :: x
    integer(c_int), value :: incx
    integer(c_int64_t), value :: stride_x
    integer(c_int), value :: batch_count
    type(c_ptr), value :: result
            hipblasIcamaxStridedBatchedFortran = &
        hipblasIcamaxStridedBatched(handle, n, x, incx, stride_x, batch_count, result)
    return
end function hipblasIcamaxStridedBatchedFortran

function hipblasIzamaxStridedBatchedFortran(handle, n, x, incx, stride_x, batch_count, result) &
    bind(c, name='hipblasIzamaxStridedBatchedFortran')
    use iso_c_binding
    use hipblas_enums
    implicit none
    integer(kind(HIPBLAS_STATUS_SUCCESS)) :: hipblasIzamaxStridedBatchedFortran
    type(c_ptr), value :: handle
    integer(c_int), value :: n
    type(c_ptr), value :: x
    integer(c_int), value :: incx
    integer(c_int64_t), value :: stride_x
    integer(c_int), value :: batch_count
    type(c_ptr), value :: result
            hipblasIzamaxStridedBatchedFortran = &
        hipblasIzamaxStridedBatched(handle, n, x, incx, stride_x, batch_count, result)
    return
end function hipblasIzamaxStridedBatchedFortran

! amin
function hipblasIsaminFortran(handle, n, x, incx, result) &
    bind(c, name='hipblasIsaminFortran')
    use iso_c_binding
    use hipblas_enums
    implicit none
    integer(kind(HIPBLAS_STATUS_SUCCESS)) :: hipblasIsaminFortran
    type(c_ptr), value :: handle
    integer(c_int), value :: n
    type(c_ptr), value :: x
    integer(c_int), value :: incx
    type(c_ptr), value :: result
            hipblasIsaminFortran = &
        hipblasIsamin(handle, n, x, incx, result)
    return
end function hipblasIsaminFortran

function hipblasIdaminFortran(handle, n, x, incx, result) &
    bind(c, name='hipblasIdaminFortran')
    use iso_c_binding
    use hipblas_enums
    implicit none
    integer(kind(HIPBLAS_STATUS_SUCCESS)) :: hipblasIdaminFortran
    type(c_ptr), value :: handle
    integer(c_int), value :: n
    type(c_ptr), value :: x
    integer(c_int), value :: incx
    type(c_ptr), value :: result
            hipblasIdaminFortran = &
        hipblasIdamin(handle, n, x, incx, result)
    return
end function hipblasIdaminFortran

function hipblasIcaminFortran(handle, n, x, incx, result) &
    bind(c, name='hipblasIcaminFortran')
    use iso_c_binding
    use hipblas_enums
    implicit none
    integer(kind(HIPBLAS_STATUS_SUCCESS)) :: hipblasIcaminFortran
    type(c_ptr), value :: handle
    integer(c_int), value :: n
    type(c_ptr), value :: x
    integer(c_int), value :: incx
    type(c_ptr), value :: result
            hipblasIcaminFortran = &
        hipblasIcamin(handle, n, x, incx, result)
    return
end function hipblasIcaminFortran

function hipblasIzaminFortran(handle, n, x, incx, result) &
    bind(c, name='hipblasIzaminFortran')
    use iso_c_binding
    use hipblas_enums
    implicit none
    integer(kind(HIPBLAS_STATUS_SUCCESS)) :: hipblasIzaminFortran
    type(c_ptr), value :: handle
    integer(c_int), value :: n
    type(c_ptr), value :: x
    integer(c_int), value :: incx
    type(c_ptr), value :: result
            hipblasIzaminFortran = &
        hipblasIzamin(handle, n, x, incx, result)
    return
end function hipblasIzaminFortran

! aminBatched
function hipblasIsaminBatchedFortran(handle, n, x, incx, batch_count, result) &
    bind(c, name='hipblasIsaminBatchedFortran')
    use iso_c_binding
    use hipblas_enums
    implicit none
    integer(kind(HIPBLAS_STATUS_SUCCESS)) :: hipblasIsaminBatchedFortran
    type(c_ptr), value :: handle
    integer(c_int), value :: n
    type(c_ptr), value :: x
    integer(c_int), value :: incx
    integer(c_int), value :: batch_count
    type(c_ptr), value :: result
            hipblasIsaminBatchedFortran = &
        hipblasIsaminBatched(handle, n, x, incx, batch_count, result)
    return
end function hipblasIsaminBatchedFortran

function hipblasIdaminBatchedFortran(handle, n, x, incx, batch_count, result) &
    bind(c, name='hipblasIdaminBatchedFortran')
    use iso_c_binding
    use hipblas_enums
    implicit none
    integer(kind(HIPBLAS_STATUS_SUCCESS)) :: hipblasIdaminBatchedFortran
    type(c_ptr), value :: handle
    integer(c_int), value :: n
    type(c_ptr), value :: x
    integer(c_int), value :: incx
    integer(c_int), value :: batch_count
    type(c_ptr), value :: result
            hipblasIdaminBatchedFortran = &
        hipblasIdaminBatched(handle, n, x, incx, batch_count, result)
    return
end function hipblasIdaminBatchedFortran

function hipblasIcaminBatchedFortran(handle, n, x, incx, batch_count, result) &
    bind(c, name='hipblasIcaminBatchedFortran')
    use iso_c_binding
    use hipblas_enums
    implicit none
    integer(kind(HIPBLAS_STATUS_SUCCESS)) :: hipblasIcaminBatchedFortran
    type(c_ptr), value :: handle
    integer(c_int), value :: n
    type(c_ptr), value :: x
    integer(c_int), value :: incx
    integer(c_int), value :: batch_count
    type(c_ptr), value :: result
            hipblasIcaminBatchedFortran = &
        hipblasIcaminBatched(handle, n, x, incx, batch_count, result)
    return
end function hipblasIcaminBatchedFortran

function hipblasIzaminBatchedFortran(handle, n, x, incx, batch_count, result) &
    bind(c, name='hipblasIzaminBatchedFortran')
    use iso_c_binding
    use hipblas_enums
    implicit none
    integer(kind(HIPBLAS_STATUS_SUCCESS)) :: hipblasIzaminBatchedFortran
    type(c_ptr), value :: handle
    integer(c_int), value :: n
    type(c_ptr), value :: x
    integer(c_int), value :: incx
    integer(c_int), value :: batch_count
    type(c_ptr), value :: result
            hipblasIzaminBatchedFortran = &
        hipblasIzaminBatched(handle, n, x, incx, batch_count, result)
    return
end function hipblasIzaminBatchedFortran

! aminStridedBatched
function hipblasIsaminStridedBatchedFortran(handle, n, x, incx, stride_x, batch_count, result) &
    bind(c, name='hipblasIsaminStridedBatchedFortran')
    use iso_c_binding
    use hipblas_enums
    implicit none
    integer(kind(HIPBLAS_STATUS_SUCCESS)) :: hipblasIsaminStridedBatchedFortran
    type(c_ptr), value :: handle
    integer(c_int), value :: n
    type(c_ptr), value :: x
    integer(c_int), value :: incx
    integer(c_int64_t), value :: stride_x
    integer(c_int), value :: batch_count
    type(c_ptr), value :: result
            hipblasIsaminStridedBatchedFortran = &
        hipblasIsaminStridedBatched(handle, n, x, incx, stride_x, batch_count, result)
    return
end function hipblasIsaminStridedBatchedFortran

function hipblasIdaminStridedBatchedFortran(handle, n, x, incx, stride_x, batch_count, result) &
    bind(c, name='hipblasIdaminStridedBatchedFortran')
    use iso_c_binding
    use hipblas_enums
    implicit none
    integer(kind(HIPBLAS_STATUS_SUCCESS)) :: hipblasIdaminStridedBatchedFortran
    type(c_ptr), value :: handle
    integer(c_int), value :: n
    type(c_ptr), value :: x
    integer(c_int), value :: incx
    integer(c_int64_t), value :: stride_x
    integer(c_int), value :: batch_count
    type(c_ptr), value :: result
            hipblasIdaminStridedBatchedFortran = &
        hipblasIdaminStridedBatched(handle, n, x, incx, stride_x, batch_count, result)
    return
end function hipblasIdaminStridedBatchedFortran

function hipblasIcaminStridedBatchedFortran(handle, n, x, incx, stride_x, batch_count, result) &
    bind(c, name='hipblasIcaminStridedBatchedFortran')
    use iso_c_binding
    use hipblas_enums
    implicit none
    integer(kind(HIPBLAS_STATUS_SUCCESS)) :: hipblasIcaminStridedBatchedFortran
    type(c_ptr), value :: handle
    integer(c_int), value :: n
    type(c_ptr), value :: x
    integer(c_int), value :: incx
    integer(c_int64_t), value :: stride_x
    integer(c_int), value :: batch_count
    type(c_ptr), value :: result
            hipblasIcaminStridedBatchedFortran = &
        hipblasIcaminStridedBatched(handle, n, x, incx, stride_x, batch_count, result)
    return
end function hipblasIcaminStridedBatchedFortran

function hipblasIzaminStridedBatchedFortran(handle, n, x, incx, stride_x, batch_count, result) &
    bind(c, name='hipblasIzaminStridedBatchedFortran')
    use iso_c_binding
    use hipblas_enums
    implicit none
    integer(kind(HIPBLAS_STATUS_SUCCESS)) :: hipblasIzaminStridedBatchedFortran
    type(c_ptr), value :: handle
    integer(c_int), value :: n
    type(c_ptr), value :: x
    integer(c_int), value :: incx
    integer(c_int64_t), value :: stride_x
    integer(c_int), value :: batch_count
    type(c_ptr), value :: result
            hipblasIzaminStridedBatchedFortran = &
        hipblasIzaminStridedBatched(handle, n, x, incx, stride_x, batch_count, result)
    return
end function hipblasIzaminStridedBatchedFortran

! rot
function hipblasSrotFortran(handle, n, x, incx, y, incy, c, s) &
    bind(c, name='hipblasSrotFortran')
    use iso_c_binding
    use hipblas_enums
    implicit none
    integer(kind(HIPBLAS_STATUS_SUCCESS)) :: hipblasSrotFortran
    type(c_ptr), value :: handle
    integer(c_int), value :: n
    type(c_ptr), value :: x
    integer(c_int), value :: incx
    type(c_ptr), value :: y
    integer(c_int), value :: incy
    type(c_ptr), value :: c
    type(c_ptr), value :: s
            hipblasSrotFortran = &
        hipblasSrot(handle, n, x, incx, y, incy, c, s)
    return
end function hipblasSrotFortran

function hipblasDrotFortran(handle, n, x, incx, y, incy, c, s) &
    bind(c, name='hipblasDrotFortran')
    use iso_c_binding
    use hipblas_enums
    implicit none
    integer(kind(HIPBLAS_STATUS_SUCCESS)) :: hipblasDrotFortran
    type(c_ptr), value :: handle
    integer(c_int), value :: n
    type(c_ptr), value :: x
    integer(c_int), value :: incx
    type(c_ptr), value :: y
    integer(c_int), value :: incy
    type(c_ptr), value :: c
    type(c_ptr), value :: s
            hipblasDrotFortran = &
        hipblasDrot(handle, n, x, incx, y, incy, c, s)
    return
end function hipblasDrotFortran

function hipblasCrotFortran(handle, n, x, incx, y, incy, c, s) &
    bind(c, name='hipblasCrotFortran')
    use iso_c_binding
    use hipblas_enums
    implicit none
    integer(kind(HIPBLAS_STATUS_SUCCESS)) :: hipblasCrotFortran
    type(c_ptr), value :: handle
    integer(c_int), value :: n
    type(c_ptr), value :: x
    integer(c_int), value :: incx
    type(c_ptr), value :: y
    integer(c_int), value :: incy
    type(c_ptr), value :: c
    type(c_ptr), value :: s
            hipblasCrotFortran = &
        hipblasCrot(handle, n, x, incx, y, incy, c, s)
    return
end function hipblasCrotFortran

function hipblasCsrotFortran(handle, n, x, incx, y, incy, c, s) &
    bind(c, name='hipblasCsrotFortran')
    use iso_c_binding
    use hipblas_enums
    implicit none
    integer(kind(HIPBLAS_STATUS_SUCCESS)) :: hipblasCsrotFortran
    type(c_ptr), value :: handle
    integer(c_int), value :: n
    type(c_ptr), value :: x
    integer(c_int), value :: incx
    type(c_ptr), value :: y
    integer(c_int), value :: incy
    type(c_ptr), value :: c
    type(c_ptr), value :: s
            hipblasCsrotFortran = &
        hipblasCsrot(handle, n, x, incx, y, incy, c, s)
    return
end function hipblasCsrotFortran

function hipblasZrotFortran(handle, n, x, incx, y, incy, c, s) &
    bind(c, name='hipblasZrotFortran')
    use iso_c_binding
    use hipblas_enums
    implicit none
    integer(kind(HIPBLAS_STATUS_SUCCESS)) :: hipblasZrotFortran
    type(c_ptr), value :: handle
    integer(c_int), value :: n
    type(c_ptr), value :: x
    integer(c_int), value :: incx
    type(c_ptr), value :: y
    integer(c_int), value :: incy
    type(c_ptr), value :: c
    type(c_ptr), value :: s
            hipblasZrotFortran = &
        hipblasZrot(handle, n, x, incx, y, incy, c, s)
    return
end function hipblasZrotFortran

function hipblasZdrotFortran(handle, n, x, incx, y, incy, c, s) &
    bind(c, name='hipblasZdrotFortran')
    use iso_c_binding
    use hipblas_enums
    implicit none
    integer(kind(HIPBLAS_STATUS_SUCCESS)) :: hipblasZdrotFortran
    type(c_ptr), value :: handle
    integer(c_int), value :: n
    type(c_ptr), value :: x
    integer(c_int), value :: incx
    type(c_ptr), value :: y
    integer(c_int), value :: incy
    type(c_ptr), value :: c
    type(c_ptr), value :: s
            hipblasZdrotFortran = &
        hipblasZdrot(handle, n, x, incx, y, incy, c, s)
    return
end function hipblasZdrotFortran

! rotBatched
function hipblasSrotBatchedFortran(handle, n, x, incx, y, incy, c, s, batch_count) &
    bind(c, name='hipblasSrotBatchedFortran')
    use iso_c_binding
    use hipblas_enums
    implicit none
    integer(kind(HIPBLAS_STATUS_SUCCESS)) :: hipblasSrotBatchedFortran
    type(c_ptr), value :: handle
    integer(c_int), value :: n
    type(c_ptr), value :: x
    integer(c_int), value :: incx
    type(c_ptr), value :: y
    integer(c_int), value :: incy
    type(c_ptr), value :: c
    type(c_ptr), value :: s
    integer(c_int), value :: batch_count
            hipblasSrotBatchedFortran = &
        hipblasSrotBatched(handle, n, x, incx, y, incy, c, s, batch_count)
    return
end function hipblasSrotBatchedFortran

function hipblasDrotBatchedFortran(handle, n, x, incx, y, incy, c, s, batch_count) &
    bind(c, name='hipblasDrotBatchedFortran')
    use iso_c_binding
    use hipblas_enums
    implicit none
    integer(kind(HIPBLAS_STATUS_SUCCESS)) :: hipblasDrotBatchedFortran
    type(c_ptr), value :: handle
    integer(c_int), value :: n
    type(c_ptr), value :: x
    integer(c_int), value :: incx
    type(c_ptr), value :: y
    integer(c_int), value :: incy
    type(c_ptr), value :: c
    type(c_ptr), value :: s
    integer(c_int), value :: batch_count
            hipblasDrotBatchedFortran = &
        hipblasDrotBatched(handle, n, x, incx, y, incy, c, s, batch_count)
    return
end function hipblasDrotBatchedFortran

function hipblasCrotBatchedFortran(handle, n, x, incx, y, incy, c, s, batch_count) &
    bind(c, name='hipblasCrotBatchedFortran')
    use iso_c_binding
    use hipblas_enums
    implicit none
    integer(kind(HIPBLAS_STATUS_SUCCESS)) :: hipblasCrotBatchedFortran
    type(c_ptr), value :: handle
    integer(c_int), value :: n
    type(c_ptr), value :: x
    integer(c_int), value :: incx
    type(c_ptr), value :: y
    integer(c_int), value :: incy
    type(c_ptr), value :: c
    type(c_ptr), value :: s
    integer(c_int), value :: batch_count
            hipblasCrotBatchedFortran = &
        hipblasCrotBatched(handle, n, x, incx, y, incy, c, s, batch_count)
    return
end function hipblasCrotBatchedFortran

function hipblasCsrotBatchedFortran(handle, n, x, incx, y, incy, c, s, batch_count) &
    bind(c, name='hipblasCsrotBatchedFortran')
    use iso_c_binding
    use hipblas_enums
    implicit none
    integer(kind(HIPBLAS_STATUS_SUCCESS)) :: hipblasCsrotBatchedFortran
    type(c_ptr), value :: handle
    integer(c_int), value :: n
    type(c_ptr), value :: x
    integer(c_int), value :: incx
    type(c_ptr), value :: y
    integer(c_int), value :: incy
    type(c_ptr), value :: c
    type(c_ptr), value :: s
    integer(c_int), value :: batch_count
            hipblasCsrotBatchedFortran = &
        hipblasCsrotBatched(handle, n, x, incx, y, incy, c, s, batch_count)
    return
end function hipblasCsrotBatchedFortran

function hipblasZrotBatchedFortran(handle, n, x, incx, y, incy, c, s, batch_count) &
    bind(c, name='hipblasZrotBatchedFortran')
    use iso_c_binding
    use hipblas_enums
    implicit none
    integer(kind(HIPBLAS_STATUS_SUCCESS)) :: hipblasZrotBatchedFortran
    type(c_ptr), value :: handle
    integer(c_int), value :: n
    type(c_ptr), value :: x
    integer(c_int), value :: incx
    type(c_ptr), value :: y
    integer(c_int), value :: incy
    type(c_ptr), value :: c
    type(c_ptr), value :: s
    integer(c_int), value :: batch_count
            hipblasZrotBatchedFortran = &
        hipblasZrotBatched(handle, n, x, incx, y, incy, c, s, batch_count)
    return
end function hipblasZrotBatchedFortran

function hipblasZdrotBatchedFortran(handle, n, x, incx, y, incy, c, s, batch_count) &
    bind(c, name='hipblasZdrotBatchedFortran')
    use iso_c_binding
    use hipblas_enums
    implicit none
    integer(kind(HIPBLAS_STATUS_SUCCESS)) :: hipblasZdrotBatchedFortran
    type(c_ptr), value :: handle
    integer(c_int), value :: n
    type(c_ptr), value :: x
    integer(c_int), value :: incx
    type(c_ptr), value :: y
    integer(c_int), value :: incy
    type(c_ptr), value :: c
    type(c_ptr), value :: s
    integer(c_int), value :: batch_count
            hipblasZdrotBatchedFortran = &
        hipblasZdrotBatched(handle, n, x, incx, y, incy, c, s, batch_count)
    return
end function hipblasZdrotBatchedFortran

! rotStridedBatched
function hipblasSrotStridedBatchedFortran(handle, n, x, incx, stride_x, y, incy, stride_y, c, s, batch_count) &
    bind(c, name='hipblasSrotStridedBatchedFortran')
    use iso_c_binding
    use hipblas_enums
    implicit none
    integer(kind(HIPBLAS_STATUS_SUCCESS)) :: hipblasSrotStridedBatchedFortran
    type(c_ptr), value :: handle
    integer(c_int), value :: n
    type(c_ptr), value :: x
    integer(c_int), value :: incx
    integer(c_int64_t), value :: stride_x
    type(c_ptr), value :: y
    integer(c_int), value :: incy
    integer(c_int64_t), value :: stride_y
    type(c_ptr), value :: c
    type(c_ptr), value :: s
    integer(c_int), value :: batch_count
            hipblasSrotStridedBatchedFortran = &
        hipblasSrotStridedBatched(handle, n, x, incx, stride_x, y, incy, stride_y, c, s, batch_count)
    return
end function hipblasSrotStridedBatchedFortran

function hipblasDrotStridedBatchedFortran(handle, n, x, incx, stride_x, y, incy, stride_y, c, s, batch_count) &
    bind(c, name='hipblasDrotStridedBatchedFortran')
    use iso_c_binding
    use hipblas_enums
    implicit none
    integer(kind(HIPBLAS_STATUS_SUCCESS)) :: hipblasDrotStridedBatchedFortran
    type(c_ptr), value :: handle
    integer(c_int), value :: n
    type(c_ptr), value :: x
    integer(c_int), value :: incx
    integer(c_int64_t), value :: stride_x
    type(c_ptr), value :: y
    integer(c_int), value :: incy
    integer(c_int64_t), value :: stride_y
    type(c_ptr), value :: c
    type(c_ptr), value :: s
    integer(c_int), value :: batch_count
            hipblasDrotStridedBatchedFortran = &
        hipblasDrotStridedBatched(handle, n, x, incx, stride_x, y, incy, stride_y, c, s, batch_count)
    return
end function hipblasDrotStridedBatchedFortran

function hipblasCrotStridedBatchedFortran(handle, n, x, incx, stride_x, y, incy, stride_y, c, s, batch_count) &
    bind(c, name='hipblasCrotStridedBatchedFortran')
    use iso_c_binding
    use hipblas_enums
    implicit none
    integer(kind(HIPBLAS_STATUS_SUCCESS)) :: hipblasCrotStridedBatchedFortran
    type(c_ptr), value :: handle
    integer(c_int), value :: n
    type(c_ptr), value :: x
    integer(c_int), value :: incx
    integer(c_int64_t), value :: stride_x
    type(c_ptr), value :: y
    integer(c_int), value :: incy
    integer(c_int64_t), value :: stride_y
    type(c_ptr), value :: c
    type(c_ptr), value :: s
    integer(c_int), value :: batch_count
            hipblasCrotStridedBatchedFortran = &
        hipblasCrotStridedBatched(handle, n, x, incx, stride_x, y, incy, stride_y, c, s, batch_count)
    return
end function hipblasCrotStridedBatchedFortran

function hipblasCsrotStridedBatchedFortran(handle, n, x, incx, stride_x, y, incy, stride_y, c, s, batch_count) &
    bind(c, name='hipblasCsrotStridedBatchedFortran')
    use iso_c_binding
    use hipblas_enums
    implicit none
    integer(kind(HIPBLAS_STATUS_SUCCESS)) :: hipblasCsrotStridedBatchedFortran
    type(c_ptr), value :: handle
    integer(c_int), value :: n
    type(c_ptr), value :: x
    integer(c_int), value :: incx
    integer(c_int64_t), value :: stride_x
    type(c_ptr), value :: y
    integer(c_int), value :: incy
    integer(c_int64_t), value :: stride_y
    type(c_ptr), value :: c
    type(c_ptr), value :: s
    integer(c_int), value :: batch_count
            hipblasCsrotStridedBatchedFortran = &
        hipblasCsrotStridedBatched(handle, n, x, incx, stride_x, y, incy, stride_y, c, s, batch_count)
    return
end function hipblasCsrotStridedBatchedFortran

function hipblasZrotStridedBatchedFortran(handle, n, x, incx, stride_x, y, incy, stride_y, c, s, batch_count) &
    bind(c, name='hipblasZrotStridedBatchedFortran')
    use iso_c_binding
    use hipblas_enums
    implicit none
    integer(kind(HIPBLAS_STATUS_SUCCESS)) :: hipblasZrotStridedBatchedFortran
    type(c_ptr), value :: handle
    integer(c_int), value :: n
    type(c_ptr), value :: x
    integer(c_int), value :: incx
    integer(c_int64_t), value :: stride_x
    type(c_ptr), value :: y
    integer(c_int), value :: incy
    integer(c_int64_t), value :: stride_y
    type(c_ptr), value :: c
    type(c_ptr), value :: s
    integer(c_int), value :: batch_count
            hipblasZrotStridedBatchedFortran = &
        hipblasZrotStridedBatched(handle, n, x, incx, stride_x, y, incy, stride_y, c, s, batch_count)
    return
end function hipblasZrotStridedBatchedFortran

function hipblasZdrotStridedBatchedFortran(handle, n, x, incx, stride_x, y, incy, stride_y, c, s, batch_count) &
    bind(c, name='hipblasZdrotStridedBatchedFortran')
    use iso_c_binding
    use hipblas_enums
    implicit none
    integer(kind(HIPBLAS_STATUS_SUCCESS)) :: hipblasZdrotStridedBatchedFortran
    type(c_ptr), value :: handle
    integer(c_int), value :: n
    type(c_ptr), value :: x
    integer(c_int), value :: incx
    integer(c_int64_t), value :: stride_x
    type(c_ptr), value :: y
    integer(c_int), value :: incy
    integer(c_int64_t), value :: stride_y
    type(c_ptr), value :: c
    type(c_ptr), value :: s
    integer(c_int), value :: batch_count
            hipblasZdrotStridedBatchedFortran = &
        hipblasZdrotStridedBatched(handle, n, x, incx, stride_x, y, incy, stride_y, c, s, batch_count)
    return
end function hipblasZdrotStridedBatchedFortran

! rotg
function hipblasSrotgFortran(handle, a, b, c, s) &
    bind(c, name='hipblasSrotgFortran')
    use iso_c_binding
    use hipblas_enums
    implicit none
    integer(kind(HIPBLAS_STATUS_SUCCESS)) :: hipblasSrotgFortran
    type(c_ptr), value :: handle
    type(c_ptr), value :: a
    type(c_ptr), value :: b
    type(c_ptr), value :: c
    type(c_ptr), value :: s
            hipblasSrotgFortran = &
        hipblasSrotg(handle, a, b, c, s)
    return
end function hipblasSrotgFortran

function hipblasDrotgFortran(handle, a, b, c, s) &
    bind(c, name='hipblasDrotgFortran')
    use iso_c_binding
    use hipblas_enums
    implicit none
    integer(kind(HIPBLAS_STATUS_SUCCESS)) :: hipblasDrotgFortran
    type(c_ptr), value :: handle
    type(c_ptr), value :: a
    type(c_ptr), value :: b
    type(c_ptr), value :: c
    type(c_ptr), value :: s
            hipblasDrotgFortran = &
        hipblasDrotg(handle, a, b, c, s)
    return
end function hipblasDrotgFortran

function hipblasCrotgFortran(handle, a, b, c, s) &
    bind(c, name='hipblasCrotgFortran')
    use iso_c_binding
    use hipblas_enums
    implicit none
    integer(kind(HIPBLAS_STATUS_SUCCESS)) :: hipblasCrotgFortran
    type(c_ptr), value :: handle
    type(c_ptr), value :: a
    type(c_ptr), value :: b
    type(c_ptr), value :: c
    type(c_ptr), value :: s
            hipblasCrotgFortran = &
        hipblasCrotg(handle, a, b, c, s)
    return
end function hipblasCrotgFortran

function hipblasZrotgFortran(handle, a, b, c, s) &
    bind(c, name='hipblasZrotgFortran')
    use iso_c_binding
    use hipblas_enums
    implicit none
    integer(kind(HIPBLAS_STATUS_SUCCESS)) :: hipblasZrotgFortran
    type(c_ptr), value :: handle
    type(c_ptr), value :: a
    type(c_ptr), value :: b
    type(c_ptr), value :: c
    type(c_ptr), value :: s
            hipblasZrotgFortran = &
        hipblasZrotg(handle, a, b, c, s)
    return
end function hipblasZrotgFortran

! rotgBatched
function hipblasSrotgBatchedFortran(handle, a, b, c, s, batch_count) &
    bind(c, name='hipblasSrotgBatchedFortran')
    use iso_c_binding
    use hipblas_enums
    implicit none
    integer(kind(HIPBLAS_STATUS_SUCCESS)) :: hipblasSrotgBatchedFortran
    type(c_ptr), value :: handle
    type(c_ptr), value :: a
    type(c_ptr), value :: b
    type(c_ptr), value :: c
    type(c_ptr), value :: s
    integer(c_int), value :: batch_count
            hipblasSrotgBatchedFortran = &
        hipblasSrotgBatched(handle, a, b, c, s, batch_count)
    return
end function hipblasSrotgBatchedFortran

function hipblasDrotgBatchedFortran(handle, a, b, c, s, batch_count) &
    bind(c, name='hipblasDrotgBatchedFortran')
    use iso_c_binding
    use hipblas_enums
    implicit none
    integer(kind(HIPBLAS_STATUS_SUCCESS)) :: hipblasDrotgBatchedFortran
    type(c_ptr), value :: handle
    type(c_ptr), value :: a
    type(c_ptr), value :: b
    type(c_ptr), value :: c
    type(c_ptr), value :: s
    integer(c_int), value :: batch_count
            hipblasDrotgBatchedFortran = &
        hipblasDrotgBatched(handle, a, b, c, s, batch_count)
    return
end function hipblasDrotgBatchedFortran

function hipblasCrotgBatchedFortran(handle, a, b, c, s, batch_count) &
    bind(c, name='hipblasCrotgBatchedFortran')
    use iso_c_binding
    use hipblas_enums
    implicit none
    integer(kind(HIPBLAS_STATUS_SUCCESS)) :: hipblasCrotgBatchedFortran
    type(c_ptr), value :: handle
    type(c_ptr), value :: a
    type(c_ptr), value :: b
    type(c_ptr), value :: c
    type(c_ptr), value :: s
    integer(c_int), value :: batch_count
            hipblasCrotgBatchedFortran = &
        hipblasCrotgBatched(handle, a, b, c, s, batch_count)
    return
end function hipblasCrotgBatchedFortran

function hipblasZrotgBatchedFortran(handle, a, b, c, s, batch_count) &
    bind(c, name='hipblasZrotgBatchedFortran')
    use iso_c_binding
    use hipblas_enums
    implicit none
    integer(kind(HIPBLAS_STATUS_SUCCESS)) :: hipblasZrotgBatchedFortran
    type(c_ptr), value :: handle
    type(c_ptr), value :: a
    type(c_ptr), value :: b
    type(c_ptr), value :: c
    type(c_ptr), value :: s
    integer(c_int), value :: batch_count
            hipblasZrotgBatchedFortran = &
        hipblasZrotgBatched(handle, a, b, c, s, batch_count)
    return
end function hipblasZrotgBatchedFortran

! rotgStridedBatched
function hipblasSrotgStridedBatchedFortran(handle, a, stride_a, b, stride_b, c, stride_c, s, stride_s, batch_count) &
    bind(c, name='hipblasSrotgStridedBatchedFortran')
    use iso_c_binding
    use hipblas_enums
    implicit none
    integer(kind(HIPBLAS_STATUS_SUCCESS)) :: hipblasSrotgStridedBatchedFortran
    type(c_ptr), value :: handle
    type(c_ptr), value :: a
    integer(c_int64_t), value :: stride_a
    type(c_ptr), value :: b
    integer(c_int64_t), value :: stride_b
    type(c_ptr), value :: c
    integer(c_int64_t), value :: stride_c
    type(c_ptr), value :: s
    integer(c_int64_t), value :: stride_s
    integer(c_int), value :: batch_count
            hipblasSrotgStridedBatchedFortran = &
        hipblasSrotgStridedBatched(handle, a, stride_a, b, stride_b, c, stride_c, s, stride_s, batch_count)
    return
end function hipblasSrotgStridedBatchedFortran

function hipblasDrotgStridedBatchedFortran(handle, a, stride_a, b, stride_b, c, stride_c, s, stride_s, batch_count) &
    bind(c, name='hipblasDrotgStridedBatchedFortran')
    use iso_c_binding
    use hipblas_enums
    implicit none
    integer(kind(HIPBLAS_STATUS_SUCCESS)) :: hipblasDrotgStridedBatchedFortran
    type(c_ptr), value :: handle
    type(c_ptr), value :: a
    integer(c_int64_t), value :: stride_a
    type(c_ptr), value :: b
    integer(c_int64_t), value :: stride_b
    type(c_ptr), value :: c
    integer(c_int64_t), value :: stride_c
    type(c_ptr), value :: s
    integer(c_int64_t), value :: stride_s
    integer(c_int), value :: batch_count
            hipblasDrotgStridedBatchedFortran = &
        hipblasDrotgStridedBatched(handle, a, stride_a, b, stride_b, c, stride_c, s, stride_s, batch_count)
    return
end function hipblasDrotgStridedBatchedFortran

function hipblasCrotgStridedBatchedFortran(handle, a, stride_a, b, stride_b, c, stride_c, s, stride_s, batch_count) &
    bind(c, name='hipblasCrotgStridedBatchedFortran')
    use iso_c_binding
    use hipblas_enums
    implicit none
    integer(kind(HIPBLAS_STATUS_SUCCESS)) :: hipblasCrotgStridedBatchedFortran
    type(c_ptr), value :: handle
    type(c_ptr), value :: a
    integer(c_int64_t), value :: stride_a
    type(c_ptr), value :: b
    integer(c_int64_t), value :: stride_b
    type(c_ptr), value :: c
    integer(c_int64_t), value :: stride_c
    type(c_ptr), value :: s
    integer(c_int64_t), value :: stride_s
    integer(c_int), value :: batch_count
            hipblasCrotgStridedBatchedFortran = &
        hipblasCrotgStridedBatched(handle, a, stride_a, b, stride_b, c, stride_c, s, stride_s, batch_count)
    return
end function hipblasCrotgStridedBatchedFortran

function hipblasZrotgStridedBatchedFortran(handle, a, stride_a, b, stride_b, c, stride_c, s, stride_s, batch_count) &
    bind(c, name='hipblasZrotgStridedBatchedFortran')
    use iso_c_binding
    use hipblas_enums
    implicit none
    integer(kind(HIPBLAS_STATUS_SUCCESS)) :: hipblasZrotgStridedBatchedFortran
    type(c_ptr), value :: handle
    type(c_ptr), value :: a
    integer(c_int64_t), value :: stride_a
    type(c_ptr), value :: b
    integer(c_int64_t), value :: stride_b
    type(c_ptr), value :: c
    integer(c_int64_t), value :: stride_c
    type(c_ptr), value :: s
    integer(c_int64_t), value :: stride_s
    integer(c_int), value :: batch_count
            hipblasZrotgStridedBatchedFortran = &
        hipblasZrotgStridedBatched(handle, a, stride_a, b, stride_b, c, stride_c, s, stride_s, batch_count)
    return
end function hipblasZrotgStridedBatchedFortran

! rotm
function hipblasSrotmFortran(handle, n, x, incx, y, incy, param) &
    bind(c, name='hipblasSrotmFortran')
    use iso_c_binding
    use hipblas_enums
    implicit none
    integer(kind(HIPBLAS_STATUS_SUCCESS)) :: hipblasSrotmFortran
    type(c_ptr), value :: handle
    integer(c_int), value :: n
    type(c_ptr), value :: x
    integer(c_int), value :: incx
    type(c_ptr), value :: y
    integer(c_int), value :: incy
    type(c_ptr), value :: param
            hipblasSrotmFortran = &
        hipblasSrotm(handle, n, x, incx, y, incy, param)
    return
end function hipblasSrotmFortran

function hipblasDrotmFortran(handle, n, x, incx, y, incy, param) &
    bind(c, name='hipblasDrotmFortran')
    use iso_c_binding
    use hipblas_enums
    implicit none
    integer(kind(HIPBLAS_STATUS_SUCCESS)) :: hipblasDrotmFortran
    type(c_ptr), value :: handle
    integer(c_int), value :: n
    type(c_ptr), value :: x
    integer(c_int), value :: incx
    type(c_ptr), value :: y
    integer(c_int), value :: incy
    type(c_ptr), value :: param
            hipblasDrotmFortran = &
        hipblasDrotm(handle, n, x, incx, y, incy, param)
    return
end function hipblasDrotmFortran

! rotmBatched
function hipblasSrotmBatchedFortran(handle, n, x, incx, y, incy, param, batch_count) &
    bind(c, name='hipblasSrotmBatchedFortran')
    use iso_c_binding
    use hipblas_enums
    implicit none
    integer(kind(HIPBLAS_STATUS_SUCCESS)) :: hipblasSrotmBatchedFortran
    type(c_ptr), value :: handle
    integer(c_int), value :: n
    type(c_ptr), value :: x
    integer(c_int), value :: incx
    type(c_ptr), value :: y
    integer(c_int), value :: incy
    type(c_ptr), value :: param
    integer(c_int), value :: batch_count
            hipblasSrotmBatchedFortran = &
        hipblasSrotmBatched(handle, n, x, incx, y, incy, param, batch_count)
    return
end function hipblasSrotmBatchedFortran

function hipblasDrotmBatchedFortran(handle, n, x, incx, y, incy, param, batch_count) &
    bind(c, name='hipblasDrotmBatchedFortran')
    use iso_c_binding
    use hipblas_enums
    implicit none
    integer(kind(HIPBLAS_STATUS_SUCCESS)) :: hipblasDrotmBatchedFortran
    type(c_ptr), value :: handle
    integer(c_int), value :: n
    type(c_ptr), value :: x
    integer(c_int), value :: incx
    type(c_ptr), value :: y
    integer(c_int), value :: incy
    type(c_ptr), value :: param
    integer(c_int), value :: batch_count
            hipblasDrotmBatchedFortran = &
        hipblasDrotmBatched(handle, n, x, incx, y, incy, param, batch_count)
    return
end function hipblasDrotmBatchedFortran

! rotmStridedBatched
function hipblasSrotmStridedBatchedFortran(handle, n, x, incx, stride_x, y, incy, stride_y, param, &
                                            stride_param, batch_count) &
    bind(c, name='hipblasSrotmStridedBatchedFortran')
    use iso_c_binding
    use hipblas_enums
    implicit none
    integer(kind(HIPBLAS_STATUS_SUCCESS)) :: hipblasSrotmStridedBatchedFortran
    type(c_ptr), value :: handle
    integer(c_int), value :: n
    type(c_ptr), value :: x
    integer(c_int), value :: incx
    integer(c_int64_t), value :: stride_x
    type(c_ptr), value :: y
    integer(c_int), value :: incy
    integer(c_int64_t), value :: stride_y
    type(c_ptr), value :: param
    integer(c_int64_t), value :: stride_param
    integer(c_int), value :: batch_count
            hipblasSrotmStridedBatchedFortran = &
        hipblasSrotmStridedBatched(handle, n, x, incx, stride_x, y, incy, stride_y, param, &
                                    stride_param, batch_count)
    return
end function hipblasSrotmStridedBatchedFortran

function hipblasDrotmStridedBatchedFortran(handle, n, x, incx, stride_x, y, incy, stride_y, param, &
                                            stride_param, batch_count) &
    bind(c, name='hipblasDrotmStridedBatchedFortran')
    use iso_c_binding
    use hipblas_enums
    implicit none
    integer(kind(HIPBLAS_STATUS_SUCCESS)) :: hipblasDrotmStridedBatchedFortran
    type(c_ptr), value :: handle
    integer(c_int), value :: n
    type(c_ptr), value :: x
    integer(c_int), value :: incx
    integer(c_int64_t), value :: stride_x
    type(c_ptr), value :: y
    integer(c_int), value :: incy
    integer(c_int64_t), value :: stride_y
    type(c_ptr), value :: param
    integer(c_int64_t), value :: stride_param
    integer(c_int), value :: batch_count
            hipblasDrotmStridedBatchedFortran = &
        hipblasDrotmStridedBatched(handle, n, x, incx, stride_x, y, incy, stride_y, param, &
                                    stride_param, batch_count)
    return
end function hipblasDrotmStridedBatchedFortran

! rotmg
function hipblasSrotmgFortran(handle, d1, d2, x1, y1, param) &
    bind(c, name='hipblasSrotmgFortran')
    use iso_c_binding
    use hipblas_enums
    implicit none
    integer(kind(HIPBLAS_STATUS_SUCCESS)) :: hipblasSrotmgFortran
    type(c_ptr), value :: handle
    type(c_ptr), value :: d1
    type(c_ptr), value :: d2
    type(c_ptr), value :: x1
    type(c_ptr), value :: y1
    type(c_ptr), value :: param
            hipblasSrotmgFortran = &
        hipblasSrotmg(handle, d1, d2, x1, y1, param)
    return
end function hipblasSrotmgFortran

function hipblasDrotmgFortran(handle, d1, d2, x1, y1, param) &
    bind(c, name='hipblasDrotmgFortran')
    use iso_c_binding
    use hipblas_enums
    implicit none
    integer(kind(HIPBLAS_STATUS_SUCCESS)) :: hipblasDrotmgFortran
    type(c_ptr), value :: handle
    type(c_ptr), value :: d1
    type(c_ptr), value :: d2
    type(c_ptr), value :: x1
    type(c_ptr), value :: y1
    type(c_ptr), value :: param
            hipblasDrotmgFortran = &
        hipblasDrotmg(handle, d1, d2, x1, y1, param)
    return
end function hipblasDrotmgFortran

! rotmgBatched
function hipblasSrotmgBatchedFortran(handle, d1, d2, x1, y1, param, batch_count) &
    bind(c, name='hipblasSrotmgBatchedFortran')
    use iso_c_binding
    use hipblas_enums
    implicit none
    integer(kind(HIPBLAS_STATUS_SUCCESS)) :: hipblasSrotmgBatchedFortran
    type(c_ptr), value :: handle
    type(c_ptr), value :: d1
    type(c_ptr), value :: d2
    type(c_ptr), value :: x1
    type(c_ptr), value :: y1
    type(c_ptr), value :: param
    integer(c_int), value :: batch_count
            hipblasSrotmgBatchedFortran = &
        hipblasSrotmgBatched(handle, d1, d2, x1, y1, param, batch_count)
    return
end function hipblasSrotmgBatchedFortran

function hipblasDrotmgBatchedFortran(handle, d1, d2, x1, y1, param, batch_count) &
    bind(c, name='hipblasDrotmgBatchedFortran')
    use iso_c_binding
    use hipblas_enums
    implicit none
    integer(kind(HIPBLAS_STATUS_SUCCESS)) :: hipblasDrotmgBatchedFortran
    type(c_ptr), value :: handle
    type(c_ptr), value :: d1
    type(c_ptr), value :: d2
    type(c_ptr), value :: x1
    type(c_ptr), value :: y1
    type(c_ptr), value :: param
    integer(c_int), value :: batch_count
            hipblasDrotmgBatchedFortran = &
        hipblasDrotmgBatched(handle, d1, d2, x1, y1, param, batch_count)
    return
end function hipblasDrotmgBatchedFortran

! rotmgStridedBatched
function hipblasSrotmgStridedBatchedFortran(handle, d1, stride_d1, d2, stride_d2, x1, stride_x1, &
                                            y1, stride_y1, param, stride_param, batch_count) &
    bind(c, name='hipblasSrotmgStridedBatchedFortran')
    use iso_c_binding
    use hipblas_enums
    implicit none
    integer(kind(HIPBLAS_STATUS_SUCCESS)) :: hipblasSrotmgStridedBatchedFortran
    type(c_ptr), value :: handle
    type(c_ptr), value :: d1
    integer(c_int64_t), value :: stride_d1
    type(c_ptr), value :: d2
    integer(c_int64_t), value :: stride_d2
    type(c_ptr), value :: x1
    integer(c_int64_t), value :: stride_x1
    type(c_ptr), value :: y1
    integer(c_int64_t), value :: stride_y1
    type(c_ptr), value :: param
    integer(c_int64_t), value :: stride_param
    integer(c_int), value :: batch_count
            hipblasSrotmgStridedBatchedFortran = &
        hipblasSrotmgStridedBatched(handle, d1, stride_d1, d2, stride_d2, x1, stride_x1, y1, stride_y1, &
                                    param, stride_param, batch_count)
    return
end function hipblasSrotmgStridedBatchedFortran

function hipblasDrotmgStridedBatchedFortran(handle, d1, stride_d1, d2, stride_d2, x1, stride_x1, &
                                            y1, stride_y1, param, stride_param, batch_count) &
    bind(c, name='hipblasDrotmgStridedBatchedFortran')
    use iso_c_binding
    use hipblas_enums
    implicit none
    integer(kind(HIPBLAS_STATUS_SUCCESS)) :: hipblasDrotmgStridedBatchedFortran
    type(c_ptr), value :: handle
    type(c_ptr), value :: d1
    integer(c_int64_t), value :: stride_d1
    type(c_ptr), value :: d2
    integer(c_int64_t), value :: stride_d2
    type(c_ptr), value :: x1
    integer(c_int64_t), value :: stride_x1
    type(c_ptr), value :: y1
    integer(c_int64_t), value :: stride_y1
    type(c_ptr), value :: param
    integer(c_int64_t), value :: stride_param
    integer(c_int), value :: batch_count
            hipblasDrotmgStridedBatchedFortran = &
        hipblasDrotmgStridedBatched(handle, d1, stride_d1, d2, stride_d2, x1, stride_x1, y1, stride_y1, &
                                    param, stride_param, batch_count)
    return
end function hipblasDrotmgStridedBatchedFortran

!--------!
! blas 2 !
!--------!

! gbmv
function hipblasSgbmvFortran(handle, trans, m, n, kl, ku, alpha, A, lda, x, incx, beta, y, incy) &
    bind(c, name='hipblasSgbmvFortran')
    use iso_c_binding
    use hipblas_enums
    implicit none
    integer(kind(HIPBLAS_STATUS_SUCCESS)) :: hipblasSgbmvFortran
    type(c_ptr), value :: handle
    integer(kind(HIPBLAS_OP_N)), value :: trans
    integer(c_int), value :: m
    integer(c_int), value :: n
    integer(c_int), value :: kl
    integer(c_int), value :: ku
    type(c_ptr), value :: alpha
    type(c_ptr), value :: A
    integer(c_int), value :: lda
    type(c_ptr), value :: x
    integer(c_int), value :: incx
    type(c_ptr), value :: beta
    type(c_ptr), value :: y
    integer(c_int), value :: incy
            hipblasSgbmvFortran = &
        hipblasSgbmv(handle, trans, m, n, kl, ku, alpha, A, lda, x, incx, beta, y, incy)
end function hipblasSgbmvFortran

function hipblasDgbmvFortran(handle, trans, m, n, kl, ku, alpha, A, lda, x, incx, beta, y, incy) &
    bind(c, name='hipblasDgbmvFortran')
    use iso_c_binding
    use hipblas_enums
    implicit none
    integer(kind(HIPBLAS_STATUS_SUCCESS)) :: hipblasDgbmvFortran
    type(c_ptr), value :: handle
    integer(kind(HIPBLAS_OP_N)), value :: trans
    integer(c_int), value :: m
    integer(c_int), value :: n
    integer(c_int), value :: kl
    integer(c_int), value :: ku
    type(c_ptr), value :: alpha
    type(c_ptr), value :: A
    integer(c_int), value :: lda
    type(c_ptr), value :: x
    integer(c_int), value :: incx
    type(c_ptr), value :: beta
    type(c_ptr), value :: y
    integer(c_int), value :: incy
            hipblasDgbmvFortran = &
        hipblasDgbmv(handle, trans, m, n, kl, ku, alpha, A, lda, x, incx, beta, y, incy)
end function hipblasDgbmvFortran

function hipblasCgbmvFortran(handle, trans, m, n, kl, ku, alpha, A, lda, x, incx, beta, y, incy) &
    bind(c, name='hipblasCgbmvFortran')
    use iso_c_binding
    use hipblas_enums
    implicit none
    integer(kind(HIPBLAS_STATUS_SUCCESS)) :: hipblasCgbmvFortran
    type(c_ptr), value :: handle
    integer(kind(HIPBLAS_OP_N)), value :: trans
    integer(c_int), value :: m
    integer(c_int), value :: n
    integer(c_int), value :: kl
    integer(c_int), value :: ku
    type(c_ptr), value :: alpha
    type(c_ptr), value :: A
    integer(c_int), value :: lda
    type(c_ptr), value :: x
    integer(c_int), value :: incx
    type(c_ptr), value :: beta
    type(c_ptr), value :: y
    integer(c_int), value :: incy
            hipblasCgbmvFortran = &
        hipblasCgbmv(handle, trans, m, n, kl, ku, alpha, A, lda, x, incx, beta, y, incy)
end function hipblasCgbmvFortran

function hipblasZgbmvFortran(handle, trans, m, n, kl, ku, alpha, A, lda, x, incx, beta, y, incy) &
    bind(c, name='hipblasZgbmvFortran')
    use iso_c_binding
    use hipblas_enums
    implicit none
    integer(kind(HIPBLAS_STATUS_SUCCESS)) :: hipblasZgbmvFortran
    type(c_ptr), value :: handle
    integer(kind(HIPBLAS_OP_N)), value :: trans
    integer(c_int), value :: m
    integer(c_int), value :: n
    integer(c_int), value :: kl
    integer(c_int), value :: ku
    type(c_ptr), value :: alpha
    type(c_ptr), value :: A
    integer(c_int), value :: lda
    type(c_ptr), value :: x
    integer(c_int), value :: incx
    type(c_ptr), value :: beta
    type(c_ptr), value :: y
    integer(c_int), value :: incy
            hipblasZgbmvFortran = &
        hipblasZgbmv(handle, trans, m, n, kl, ku, alpha, A, lda, x, incx, beta, y, incy)
end function hipblasZgbmvFortran

! gbmvBatched
function hipblasSgbmvBatchedFortran(handle, trans, m, n, kl, ku, alpha, A, lda, &
                                    x, incx, beta, y, incy, batch_count) &
    bind(c, name='hipblasSgbmvBatchedFortran')
    use iso_c_binding
    use hipblas_enums
    implicit none
    integer(kind(HIPBLAS_STATUS_SUCCESS)) :: hipblasSgbmvBatchedFortran
    type(c_ptr), value :: handle
    integer(kind(HIPBLAS_OP_N)), value :: trans
    integer(c_int), value :: m
    integer(c_int), value :: n
    integer(c_int), value :: kl
    integer(c_int), value :: ku
    type(c_ptr), value :: alpha
    type(c_ptr), value :: A
    integer(c_int), value :: lda
    type(c_ptr), value :: x
    integer(c_int), value :: incx
    type(c_ptr), value :: beta
    type(c_ptr), value :: y
    integer(c_int), value :: incy
    integer(c_int), value :: batch_count
            hipblasSgbmvBatchedFortran = &
        hipblasSgbmvBatched(handle, trans, m, n, kl, ku, alpha, A, lda, x, incx, &
                            beta, y, incy, batch_count)
end function hipblasSgbmvBatchedFortran

function hipblasDgbmvBatchedFortran(handle, trans, m, n, kl, ku, alpha, A, lda, &
                                    x, incx, beta, y, incy, batch_count) &
    bind(c, name='hipblasDgbmvBatchedFortran')
    use iso_c_binding
    use hipblas_enums
    implicit none
    integer(kind(HIPBLAS_STATUS_SUCCESS)) :: hipblasDgbmvBatchedFortran
    type(c_ptr), value :: handle
    integer(kind(HIPBLAS_OP_N)), value :: trans
    integer(c_int), value :: m
    integer(c_int), value :: n
    integer(c_int), value :: kl
    integer(c_int), value :: ku
    type(c_ptr), value :: alpha
    type(c_ptr), value :: A
    integer(c_int), value :: lda
    type(c_ptr), value :: x
    integer(c_int), value :: incx
    type(c_ptr), value :: beta
    type(c_ptr), value :: y
    integer(c_int), value :: incy
    integer(c_int), value :: batch_count
            hipblasDgbmvBatchedFortran = &
        hipblasDgbmvBatched(handle, trans, m, n, kl, ku, alpha, A, lda, x, incx, &
                            beta, y, incy, batch_count)
end function hipblasDgbmvBatchedFortran

function hipblasCgbmvBatchedFortran(handle, trans, m, n, kl, ku, alpha, A, lda, &
                                    x, incx, beta, y, incy, batch_count) &
    bind(c, name='hipblasCgbmvBatchedFortran')
    use iso_c_binding
    use hipblas_enums
    implicit none
    integer(kind(HIPBLAS_STATUS_SUCCESS)) :: hipblasCgbmvBatchedFortran
    type(c_ptr), value :: handle
    integer(kind(HIPBLAS_OP_N)), value :: trans
    integer(c_int), value :: m
    integer(c_int), value :: n
    integer(c_int), value :: kl
    integer(c_int), value :: ku
    type(c_ptr), value :: alpha
    type(c_ptr), value :: A
    integer(c_int), value :: lda
    type(c_ptr), value :: x
    integer(c_int), value :: incx
    type(c_ptr), value :: beta
    type(c_ptr), value :: y
    integer(c_int), value :: incy
    integer(c_int), value :: batch_count
            hipblasCgbmvBatchedFortran = &
        hipblasCgbmvBatched(handle, trans, m, n, kl, ku, alpha, A, lda, x, incx, &
                            beta, y, incy, batch_count)
end function hipblasCgbmvBatchedFortran

function hipblasZgbmvBatchedFortran(handle, trans, m, n, kl, ku, alpha, A, lda, &
                                    x, incx, beta, y, incy, batch_count) &
    bind(c, name='hipblasZgbmvBatchedFortran')
    use iso_c_binding
    use hipblas_enums
    implicit none
    integer(kind(HIPBLAS_STATUS_SUCCESS)) :: hipblasZgbmvBatchedFortran
    type(c_ptr), value :: handle
    integer(kind(HIPBLAS_OP_N)), value :: trans
    integer(c_int), value :: m
    integer(c_int), value :: n
    integer(c_int), value :: kl
    integer(c_int), value :: ku
    type(c_ptr), value :: alpha
    type(c_ptr), value :: A
    integer(c_int), value :: lda
    type(c_ptr), value :: x
    integer(c_int), value :: incx
    type(c_ptr), value :: beta
    type(c_ptr), value :: y
    integer(c_int), value :: incy
    integer(c_int), value :: batch_count
            hipblasZgbmvBatchedFortran = &
        hipblasZgbmvBatched(handle, trans, m, n, kl, ku, alpha, A, lda, x, incx, &
                            beta, y, incy, batch_count)
end function hipblasZgbmvBatchedFortran

! gbmvStridedBatched
function hipblasSgbmvStridedBatchedFortran(handle, trans, m, n, kl, ku, alpha, A, lda, stride_A, &
                                            x, incx, stride_x, beta, y, incy, stride_y, batch_count) &
    bind(c, name='hipblasSgbmvStridedBatchedFortran')
    use iso_c_binding
    use hipblas_enums
    implicit none
    integer(kind(HIPBLAS_STATUS_SUCCESS)) :: hipblasSgbmvStridedBatchedFortran
    type(c_ptr), value :: handle
    integer(kind(HIPBLAS_OP_N)), value :: trans
    integer(c_int), value :: m
    integer(c_int), value :: n
    integer(c_int), value :: kl
    integer(c_int), value :: ku
    type(c_ptr), value :: alpha
    type(c_ptr), value :: A
    integer(c_int), value :: lda
    integer(c_int64_t), value :: stride_A
    type(c_ptr), value :: x
    integer(c_int), value :: incx
    integer(c_int64_t), value :: stride_x
    type(c_ptr), value :: beta
    type(c_ptr), value :: y
    integer(c_int), value :: incy
    integer(c_int64_t), value :: stride_y
    integer(c_int), value :: batch_count
            hipblasSgbmvStridedBatchedFortran = &
        hipblasSgbmvStridedBatched(handle, trans, m, n, kl, ku, alpha, A, lda, stride_A, x, incx, stride_x, &
                                    beta, y, incy, stride_y, batch_count)
end function hipblasSgbmvStridedBatchedFortran

function hipblasDgbmvStridedBatchedFortran(handle, trans, m, n, kl, ku, alpha, A, lda, stride_A, &
                                            x, incx, stride_x, beta, y, incy, stride_y, batch_count) &
    bind(c, name='hipblasDgbmvStridedBatchedFortran')
    use iso_c_binding
    use hipblas_enums
    implicit none
    integer(kind(HIPBLAS_STATUS_SUCCESS)) :: hipblasDgbmvStridedBatchedFortran
    type(c_ptr), value :: handle
    integer(kind(HIPBLAS_OP_N)), value :: trans
    integer(c_int), value :: m
    integer(c_int), value :: n
    integer(c_int), value :: kl
    integer(c_int), value :: ku
    type(c_ptr), value :: alpha
    type(c_ptr), value :: A
    integer(c_int), value :: lda
    integer(c_int64_t), value :: stride_A
    type(c_ptr), value :: x
    integer(c_int), value :: incx
    integer(c_int64_t), value :: stride_x
    type(c_ptr), value :: beta
    type(c_ptr), value :: y
    integer(c_int), value :: incy
    integer(c_int64_t), value :: stride_y
    integer(c_int), value :: batch_count
            hipblasDgbmvStridedBatchedFortran = &
        hipblasDgbmvStridedBatched(handle, trans, m, n, kl, ku, alpha, A, lda, stride_A, x, incx, stride_x, &
                                    beta, y, incy, stride_y, batch_count)
end function hipblasDgbmvStridedBatchedFortran

function hipblasCgbmvStridedBatchedFortran(handle, trans, m, n, kl, ku, alpha, A, lda, stride_A, &
                                            x, incx, stride_x, beta, y, incy, stride_y, batch_count) &
    bind(c, name='hipblasCgbmvStridedBatchedFortran')
    use iso_c_binding
    use hipblas_enums
    implicit none
    integer(kind(HIPBLAS_STATUS_SUCCESS)) :: hipblasCgbmvStridedBatchedFortran
    type(c_ptr), value :: handle
    integer(kind(HIPBLAS_OP_N)), value :: trans
    integer(c_int), value :: m
    integer(c_int), value :: n
    integer(c_int), value :: kl
    integer(c_int), value :: ku
    type(c_ptr), value :: alpha
    type(c_ptr), value :: A
    integer(c_int), value :: lda
    integer(c_int64_t), value :: stride_A
    type(c_ptr), value :: x
    integer(c_int), value :: incx
    integer(c_int64_t), value :: stride_x
    type(c_ptr), value :: beta
    type(c_ptr), value :: y
    integer(c_int), value :: incy
    integer(c_int64_t), value :: stride_y
    integer(c_int), value :: batch_count
            hipblasCgbmvStridedBatchedFortran = &
        hipblasCgbmvStridedBatched(handle, trans, m, n, kl, ku, alpha, A, lda, stride_A, x, incx, stride_x, &
                                    beta, y, incy, stride_y, batch_count)
end function hipblasCgbmvStridedBatchedFortran

function hipblasZgbmvStridedBatchedFortran(handle, trans, m, n, kl, ku, alpha, A, lda, stride_A, &
                                            x, incx, stride_x, beta, y, incy, stride_y, batch_count) &
    bind(c, name='hipblasZgbmvStridedBatchedFortran')
    use iso_c_binding
    use hipblas_enums
    implicit none
    integer(kind(HIPBLAS_STATUS_SUCCESS)) :: hipblasZgbmvStridedBatchedFortran
    type(c_ptr), value :: handle
    integer(kind(HIPBLAS_OP_N)), value :: trans
    integer(c_int), value :: m
    integer(c_int), value :: n
    integer(c_int), value :: kl
    integer(c_int), value :: ku
    type(c_ptr), value :: alpha
    type(c_ptr), value :: A
    integer(c_int), value :: lda
    integer(c_int64_t), value :: stride_A
    type(c_ptr), value :: x
    integer(c_int), value :: incx
    integer(c_int64_t), value :: stride_x
    type(c_ptr), value :: beta
    type(c_ptr), value :: y
    integer(c_int), value :: incy
    integer(c_int64_t), value :: stride_y
    integer(c_int), value :: batch_count
            hipblasZgbmvStridedBatchedFortran = &
        hipblasZgbmvStridedBatched(handle, trans, m, n, kl, ku, alpha, A, lda, stride_A, x, incx, stride_x, &
                                    beta, y, incy, stride_y, batch_count)
end function hipblasZgbmvStridedBatchedFortran

! gemv
function hipblasSgemvFortran(handle, trans, m, n, alpha, A, lda, &
                                x, incx, beta, y, incy) &
    bind(c, name='hipblasSgemvFortran')
    use iso_c_binding
    use hipblas_enums
    implicit none
    integer(kind(HIPBLAS_STATUS_SUCCESS)) :: hipblasSgemvFortran
    type(c_ptr), value :: handle
    integer(kind(HIPBLAS_OP_N)), value :: trans
    integer(c_int), value :: m
    integer(c_int), value :: n
    type(c_ptr), value :: alpha
    type(c_ptr), value :: A
    integer(c_int), value :: lda
    type(c_ptr), value :: x
    integer(c_int), value :: incx
    type(c_ptr), value :: beta
    type(c_ptr), value :: y
    integer(c_int), value :: incy
            hipblasSgemvFortran = &
        hipblasSgemv(handle, trans, m, n, alpha, A, lda, x, incx, beta, y, incy)
end function hipblasSgemvFortran

function hipblasDgemvFortran(handle, trans, m, n, alpha, A, lda, &
                                x, incx, beta, y, incy) &
    bind(c, name='hipblasDgemvFortran')
    use iso_c_binding
    use hipblas_enums
    implicit none
    integer(kind(HIPBLAS_STATUS_SUCCESS)) :: hipblasDgemvFortran
    type(c_ptr), value :: handle
    integer(kind(HIPBLAS_OP_N)), value :: trans
    integer(c_int), value :: m
    integer(c_int), value :: n
    type(c_ptr), value :: alpha
    type(c_ptr), value :: A
    integer(c_int), value :: lda
    type(c_ptr), value :: x
    integer(c_int), value :: incx
    type(c_ptr), value :: beta
    type(c_ptr), value :: y
    integer(c_int), value :: incy
            hipblasDgemvFortran = &
        hipblasDgemv(handle, trans, m, n, alpha, A, lda, x, incx, beta, y, incy)
end function hipblasDgemvFortran

function hipblasCgemvFortran(handle, trans, m, n, alpha, A, lda, &
                                x, incx, beta, y, incy) &
    bind(c, name='hipblasCgemvFortran')
    use iso_c_binding
    use hipblas_enums
    implicit none
    integer(kind(HIPBLAS_STATUS_SUCCESS)) :: hipblasCgemvFortran
    type(c_ptr), value :: handle
    integer(kind(HIPBLAS_OP_N)), value :: trans
    integer(c_int), value :: m
    integer(c_int), value :: n
    type(c_ptr), value :: alpha
    type(c_ptr), value :: A
    integer(c_int), value :: lda
    type(c_ptr), value :: x
    integer(c_int), value :: incx
    type(c_ptr), value :: beta
    type(c_ptr), value :: y
    integer(c_int), value :: incy
            hipblasCgemvFortran = &
        hipblasCgemv(handle, trans, m, n, alpha, A, lda, x, incx, beta, y, incy)
end function hipblasCgemvFortran

function hipblasZgemvFortran(handle, trans, m, n, alpha, A, lda, &
                                x, incx, beta, y, incy) &
    bind(c, name='hipblasZgemvFortran')
    use iso_c_binding
    use hipblas_enums
    implicit none
    integer(kind(HIPBLAS_STATUS_SUCCESS)) :: hipblasZgemvFortran
    type(c_ptr), value :: handle
    integer(kind(HIPBLAS_OP_N)), value :: trans
    integer(c_int), value :: m
    integer(c_int), value :: n
    type(c_ptr), value :: alpha
    type(c_ptr), value :: A
    integer(c_int), value :: lda
    type(c_ptr), value :: x
    integer(c_int), value :: incx
    type(c_ptr), value :: beta
    type(c_ptr), value :: y
    integer(c_int), value :: incy
            hipblasZgemvFortran = &
        hipblasZgemv(handle, trans, m, n, alpha, A, lda, x, incx, beta, y, incy)
end function hipblasZgemvFortran

! gemvBatched
function hipblasSgemvBatchedFortran(handle, trans, m, n, alpha, A, lda, &
                                    x, incx, beta, y, incy, batch_count) &
    bind(c, name='hipblasSgemvBatchedFortran')
    use iso_c_binding
    use hipblas_enums
    implicit none
    integer(kind(HIPBLAS_STATUS_SUCCESS)) :: hipblasSgemvBatchedFortran
    type(c_ptr), value :: handle
    integer(kind(HIPBLAS_OP_N)), value :: trans
    integer(c_int), value :: m
    integer(c_int), value :: n
    type(c_ptr), value :: alpha
    type(c_ptr), value :: A
    integer(c_int), value :: lda
    type(c_ptr), value :: x
    integer(c_int), value :: incx
    type(c_ptr), value :: beta
    type(c_ptr), value :: y
    integer(c_int), value :: incy
    integer(c_int), value :: batch_count
            hipblasSgemvBatchedFortran = &
        hipblasSgemvBatched(handle, trans, m, n, alpha, A, lda, x, incx, beta, y, incy, batch_count)
end function hipblasSgemvBatchedFortran

function hipblasDgemvBatchedFortran(handle, trans, m, n, alpha, A, lda, &
                                    x, incx, beta, y, incy, batch_count) &
    bind(c, name='hipblasDgemvBatchedFortran')
    use iso_c_binding
    use hipblas_enums
    implicit none
    integer(kind(HIPBLAS_STATUS_SUCCESS)) :: hipblasDgemvBatchedFortran
    type(c_ptr), value :: handle
    integer(kind(HIPBLAS_OP_N)), value :: trans
    integer(c_int), value :: m
    integer(c_int), value :: n
    type(c_ptr), value :: alpha
    type(c_ptr), value :: A
    integer(c_int), value :: lda
    type(c_ptr), value :: x
    integer(c_int), value :: incx
    type(c_ptr), value :: beta
    type(c_ptr), value :: y
    integer(c_int), value :: incy
    integer(c_int), value :: batch_count
            hipblasDgemvBatchedFortran = &
        hipblasDgemvBatched(handle, trans, m, n, alpha, A, lda, x, incx, beta, y, incy, batch_count)
end function hipblasDgemvBatchedFortran

function hipblasCgemvBatchedFortran(handle, trans, m, n, alpha, A, lda, &
                                    x, incx, beta, y, incy, batch_count) &
    bind(c, name='hipblasCgemvBatchedFortran')
    use iso_c_binding
    use hipblas_enums
    implicit none
    integer(kind(HIPBLAS_STATUS_SUCCESS)) :: hipblasCgemvBatchedFortran
    type(c_ptr), value :: handle
    integer(kind(HIPBLAS_OP_N)), value :: trans
    integer(c_int), value :: m
    integer(c_int), value :: n
    type(c_ptr), value :: alpha
    type(c_ptr), value :: A
    integer(c_int), value :: lda
    type(c_ptr), value :: x
    integer(c_int), value :: incx
    type(c_ptr), value :: beta
    type(c_ptr), value :: y
    integer(c_int), value :: incy
    integer(c_int), value :: batch_count
            hipblasCgemvBatchedFortran = &
        hipblasCgemvBatched(handle, trans, m, n, alpha, A, lda, x, incx, beta, y, incy, batch_count)
end function hipblasCgemvBatchedFortran

function hipblasZgemvBatchedFortran(handle, trans, m, n, alpha, A, lda, &
                                    x, incx, beta, y, incy, batch_count) &
    bind(c, name='hipblasZgemvBatchedFortran')
    use iso_c_binding
    use hipblas_enums
    implicit none
    integer(kind(HIPBLAS_STATUS_SUCCESS)) :: hipblasZgemvBatchedFortran
    type(c_ptr), value :: handle
    integer(kind(HIPBLAS_OP_N)), value :: trans
    integer(c_int), value :: m
    integer(c_int), value :: n
    type(c_ptr), value :: alpha
    type(c_ptr), value :: A
    integer(c_int), value :: lda
    type(c_ptr), value :: x
    integer(c_int), value :: incx
    type(c_ptr), value :: beta
    type(c_ptr), value :: y
    integer(c_int), value :: incy
    integer(c_int), value :: batch_count
            hipblasZgemvBatchedFortran = &
        hipblasZgemvBatched(handle, trans, m, n, alpha, A, lda, x, incx, beta, y, incy, batch_count)
end function hipblasZgemvBatchedFortran

! gemvStridedBatched
function hipblasSgemvStridedBatchedFortran(handle, trans, m, n, alpha, A, lda, stride_A, &
                                            x, incx, stride_x, beta, y, incy, stride_y, batch_count) &
    bind(c, name='hipblasSgemvStridedBatchedFortran')
    use iso_c_binding
    use hipblas_enums
    implicit none
    integer(kind(HIPBLAS_STATUS_SUCCESS)) :: hipblasSgemvStridedBatchedFortran
    type(c_ptr), value :: handle
    integer(kind(HIPBLAS_OP_N)), value :: trans
    integer(c_int), value :: m
    integer(c_int), value :: n
    type(c_ptr), value :: alpha
    type(c_ptr), value :: A
    integer(c_int), value :: lda
    integer(c_int64_t), value :: stride_A
    type(c_ptr), value :: x
    integer(c_int), value :: incx
    integer(c_int64_t), value :: stride_x
    type(c_ptr), value :: beta
    type(c_ptr), value :: y
    integer(c_int), value :: incy
    integer(c_int64_t), value :: stride_y
    integer(c_int), value :: batch_count
            hipblasSgemvStridedBatchedFortran = &
        hipblasSgemvStridedBatched(handle, trans, m, n, alpha, A, lda, stride_A, &
                                    x, incx, stride_x, beta, y, incy, stride_y, batch_count)
end function hipblasSgemvStridedBatchedFortran

function hipblasDgemvStridedBatchedFortran(handle, trans, m, n, alpha, A, lda, stride_A, &
                                            x, incx, stride_x, beta, y, incy, stride_y, batch_count) &
    bind(c, name='hipblasDgemvStridedBatchedFortran')
    use iso_c_binding
    use hipblas_enums
    implicit none
    integer(kind(HIPBLAS_STATUS_SUCCESS)) :: hipblasDgemvStridedBatchedFortran
    type(c_ptr), value :: handle
    integer(kind(HIPBLAS_OP_N)), value :: trans
    integer(c_int), value :: m
    integer(c_int), value :: n
    type(c_ptr), value :: alpha
    type(c_ptr), value :: A
    integer(c_int), value :: lda
    integer(c_int64_t), value :: stride_A
    type(c_ptr), value :: x
    integer(c_int), value :: incx
    integer(c_int64_t), value :: stride_x
    type(c_ptr), value :: beta
    type(c_ptr), value :: y
    integer(c_int), value :: incy
    integer(c_int64_t), value :: stride_y
    integer(c_int), value :: batch_count
            hipblasDgemvStridedBatchedFortran = &
        hipblasDgemvStridedBatched(handle, trans, m, n, alpha, A, lda, stride_A, &
                                    x, incx, stride_x, beta, y, incy, stride_y, batch_count)
end function hipblasDgemvStridedBatchedFortran

function hipblasCgemvStridedBatchedFortran(handle, trans, m, n, alpha, A, lda, stride_A, &
                                            x, incx, stride_x, beta, y, incy, stride_y, batch_count) &
    bind(c, name='hipblasCgemvStridedBatchedFortran')
    use iso_c_binding
    use hipblas_enums
    implicit none
    integer(kind(HIPBLAS_STATUS_SUCCESS)) :: hipblasCgemvStridedBatchedFortran
    type(c_ptr), value :: handle
    integer(kind(HIPBLAS_OP_N)), value :: trans
    integer(c_int), value :: m
    integer(c_int), value :: n
    type(c_ptr), value :: alpha
    type(c_ptr), value :: A
    integer(c_int), value :: lda
    integer(c_int64_t), value :: stride_A
    type(c_ptr), value :: x
    integer(c_int), value :: incx
    integer(c_int64_t), value :: stride_x
    type(c_ptr), value :: beta
    type(c_ptr), value :: y
    integer(c_int), value :: incy
    integer(c_int64_t), value :: stride_y
    integer(c_int), value :: batch_count
            hipblasCgemvStridedBatchedFortran = &
        hipblasCgemvStridedBatched(handle, trans, m, n, alpha, A, lda, stride_A, &
                                    x, incx, stride_x, beta, y, incy, stride_y, batch_count)
end function hipblasCgemvStridedBatchedFortran

function hipblasZgemvStridedBatchedFortran(handle, trans, m, n, alpha, A, lda, stride_A, &
                                            x, incx, stride_x, beta, y, incy, stride_y, batch_count) &
    bind(c, name='hipblasZgemvStridedBatchedFortran')
    use iso_c_binding
    use hipblas_enums
    implicit none
    integer(kind(HIPBLAS_STATUS_SUCCESS)) :: hipblasZgemvStridedBatchedFortran
    type(c_ptr), value :: handle
    integer(kind(HIPBLAS_OP_N)), value :: trans
    integer(c_int), value :: m
    integer(c_int), value :: n
    type(c_ptr), value :: alpha
    type(c_ptr), value :: A
    integer(c_int), value :: lda
    integer(c_int64_t), value :: stride_A
    type(c_ptr), value :: x
    integer(c_int), value :: incx
    integer(c_int64_t), value :: stride_x
    type(c_ptr), value :: beta
    type(c_ptr), value :: y
    integer(c_int), value :: incy
    integer(c_int64_t), value :: stride_y
    integer(c_int), value :: batch_count
            hipblasZgemvStridedBatchedFortran = &
        hipblasZgemvStridedBatched(handle, trans, m, n, alpha, A, lda, stride_A, &
                                    x, incx, stride_x, beta, y, incy, stride_y, batch_count)
end function hipblasZgemvStridedBatchedFortran

! hbmv
function hipblasChbmvFortran(handle, uplo, n, k, alpha, A, lda, &
                                x, incx, beta, y, incy) &
    bind(c, name='hipblasChbmvFortran')
    use iso_c_binding
    use hipblas_enums
    implicit none
    integer(kind(HIPBLAS_STATUS_SUCCESS)) :: hipblasChbmvFortran
    type(c_ptr), value :: handle
    integer(kind(HIPBLAS_FILL_MODE_FULL)), value :: uplo
    integer(c_int), value :: n
    integer(c_int), value :: k
    type(c_ptr), value :: alpha
    type(c_ptr), value :: A
    integer(c_int), value :: lda
    type(c_ptr), value :: x
    integer(c_int), value :: incx
    type(c_ptr), value :: beta
    type(c_ptr), value :: y
    integer(c_int), value :: incy
            hipblasChbmvFortran = &
        hipblasChbmv(handle, uplo, n, k, alpha, A, lda, x, incx, beta, y, incy)
end function hipblasChbmvFortran

function hipblasZhbmvFortran(handle, uplo, n, k, alpha, A, lda, &
                                x, incx, beta, y, incy) &
    bind(c, name='hipblasZhbmvFortran')
    use iso_c_binding
    use hipblas_enums
    implicit none
    integer(kind(HIPBLAS_STATUS_SUCCESS)) :: hipblasZhbmvFortran
    type(c_ptr), value :: handle
    integer(kind(HIPBLAS_FILL_MODE_FULL)), value :: uplo
    integer(c_int), value :: n
    integer(c_int), value :: k
    type(c_ptr), value :: alpha
    type(c_ptr), value :: A
    integer(c_int), value :: lda
    type(c_ptr), value :: x
    integer(c_int), value :: incx
    type(c_ptr), value :: beta
    type(c_ptr), value :: y
    integer(c_int), value :: incy
            hipblasZhbmvFortran = &
        hipblasZhbmv(handle, uplo, n, k, alpha, A, lda, x, incx, beta, y, incy)
end function hipblasZhbmvFortran

! hbmvBatched
function hipblasChbmvBatchedFortran(handle, uplo, n, k, alpha, A, lda, &
                                    x, incx, beta, y, incy, batch_count) &
    bind(c, name='hipblasChbmvBatchedFortran')
    use iso_c_binding
    use hipblas_enums
    implicit none
    integer(kind(HIPBLAS_STATUS_SUCCESS)) :: hipblasChbmvBatchedFortran
    type(c_ptr), value :: handle
    integer(kind(HIPBLAS_FILL_MODE_FULL)), value :: uplo
    integer(c_int), value :: n
    integer(c_int), value :: k
    type(c_ptr), value :: alpha
    type(c_ptr), value :: A
    integer(c_int), value :: lda
    type(c_ptr), value :: x
    integer(c_int), value :: incx
    type(c_ptr), value :: beta
    type(c_ptr), value :: y
    integer(c_int), value :: incy
    integer(c_int), value :: batch_count
            hipblasChbmvBatchedFortran = &
        hipblasChbmvBatched(handle, uplo, n, k, alpha, A, lda, &
                            x, incx, beta, y, incy, batch_count)
end function hipblasChbmvBatchedFortran

function hipblasZhbmvBatchedFortran(handle, uplo, n, k, alpha, A, lda, &
                                    x, incx, beta, y, incy, batch_count) &
    bind(c, name='hipblasZhbmvBatchedFortran')
    use iso_c_binding
    use hipblas_enums
    implicit none
    integer(kind(HIPBLAS_STATUS_SUCCESS)) :: hipblasZhbmvBatchedFortran
    type(c_ptr), value :: handle
    integer(kind(HIPBLAS_FILL_MODE_FULL)), value :: uplo
    integer(c_int), value :: n
    integer(c_int), value :: k
    type(c_ptr), value :: alpha
    type(c_ptr), value :: A
    integer(c_int), value :: lda
    type(c_ptr), value :: x
    integer(c_int), value :: incx
    type(c_ptr), value :: beta
    type(c_ptr), value :: y
    integer(c_int), value :: incy
    integer(c_int), value :: batch_count
            hipblasZhbmvBatchedFortran = &
        hipblasZhbmvBatched(handle, uplo, n, k, alpha, A, lda, &
                            x, incx, beta, y, incy, batch_count)
end function hipblasZhbmvBatchedFortran

! hbmvStridedBatched
function hipblasChbmvStridedBatchedFortran(handle, uplo, n, k, alpha, A, lda, stride_A, &
                                            x, incx, stride_x, beta, y, incy, stride_y, batch_count) &
    bind(c, name='hipblasChbmvStridedBatchedFortran')
    use iso_c_binding
    use hipblas_enums
    implicit none
    integer(kind(HIPBLAS_STATUS_SUCCESS)) :: hipblasChbmvStridedBatchedFortran
    type(c_ptr), value :: handle
    integer(kind(HIPBLAS_FILL_MODE_FULL)), value :: uplo
    integer(c_int), value :: n
    integer(c_int), value :: k
    type(c_ptr), value :: alpha
    type(c_ptr), value :: A
    integer(c_int), value :: lda
    integer(c_int64_t), value :: stride_A
    type(c_ptr), value :: x
    integer(c_int), value :: incx
    integer(c_int64_t), value :: stride_x
    type(c_ptr), value :: beta
    type(c_ptr), value :: y
    integer(c_int), value :: incy
    integer(c_int64_t), value :: stride_y
    integer(c_int), value :: batch_count
            hipblasChbmvStridedBatchedFortran = &
        hipblasChbmvStridedBatched(handle, uplo, n, k, alpha, A, lda, stride_A, &
                                    x, incx, stride_x, beta, y, incy, stride_y, batch_count)
end function hipblasChbmvStridedBatchedFortran

function hipblasZhbmvStridedBatchedFortran(handle, uplo, n, k, alpha, A, lda, stride_A, &
                                            x, incx, stride_x, beta, y, incy, stride_y, batch_count) &
    bind(c, name='hipblasZhbmvStridedBatchedFortran')
    use iso_c_binding
    use hipblas_enums
    implicit none
    integer(kind(HIPBLAS_STATUS_SUCCESS)) :: hipblasZhbmvStridedBatchedFortran
    type(c_ptr), value :: handle
    integer(kind(HIPBLAS_FILL_MODE_FULL)), value :: uplo
    integer(c_int), value :: n
    integer(c_int), value :: k
    type(c_ptr), value :: alpha
    type(c_ptr), value :: A
    integer(c_int), value :: lda
    integer(c_int64_t), value :: stride_A
    type(c_ptr), value :: x
    integer(c_int), value :: incx
    integer(c_int64_t), value :: stride_x
    type(c_ptr), value :: beta
    type(c_ptr), value :: y
    integer(c_int), value :: incy
    integer(c_int64_t), value :: stride_y
    integer(c_int), value :: batch_count
            hipblasZhbmvStridedBatchedFortran = &
        hipblasZhbmvStridedBatched(handle, uplo, n, k, alpha, A, lda, stride_A, &
                                    x, incx, stride_x, beta, y, incy, stride_y, batch_count)
end function hipblasZhbmvStridedBatchedFortran

! hemv
function hipblasChemvFortran(handle, uplo, n, alpha, A, lda, &
                                x, incx, beta, y, incy) &
    bind(c, name='hipblasChemvFortran')
    use iso_c_binding
    use hipblas_enums
    implicit none
    integer(kind(HIPBLAS_STATUS_SUCCESS)) :: hipblasChemvFortran
    type(c_ptr), value :: handle
    integer(kind(HIPBLAS_FILL_MODE_FULL)), value :: uplo
    integer(c_int), value :: n
    type(c_ptr), value :: alpha
    type(c_ptr), value :: A
    integer(c_int), value :: lda
    type(c_ptr), value :: x
    integer(c_int), value :: incx
    type(c_ptr), value :: beta
    type(c_ptr), value :: y
    integer(c_int), value :: incy
            hipblasChemvFortran = &
        hipblasChemv(handle, uplo, n, alpha, A, lda, x, incx, beta, y, incy)
end function hipblasChemvFortran

function hipblasZhemvFortran(handle, uplo, n, alpha, A, lda, &
                                x, incx, beta, y, incy) &
    bind(c, name='hipblasZhemvFortran')
    use iso_c_binding
    use hipblas_enums
    implicit none
    integer(kind(HIPBLAS_STATUS_SUCCESS)) :: hipblasZhemvFortran
    type(c_ptr), value :: handle
    integer(kind(HIPBLAS_FILL_MODE_FULL)), value :: uplo
    integer(c_int), value :: n
    type(c_ptr), value :: alpha
    type(c_ptr), value :: A
    integer(c_int), value :: lda
    type(c_ptr), value :: x
    integer(c_int), value :: incx
    type(c_ptr), value :: beta
    type(c_ptr), value :: y
    integer(c_int), value :: incy
            hipblasZhemvFortran = &
        hipblasZhemv(handle, uplo, n, alpha, A, lda, x, incx, beta, y, incy)
end function hipblasZhemvFortran

! hemvBatched
function hipblasChemvBatchedFortran(handle, uplo, n, alpha, A, lda, &
                                    x, incx, beta, y, incy, batch_count) &
    bind(c, name='hipblasChemvBatchedFortran')
    use iso_c_binding
    use hipblas_enums
    implicit none
    integer(kind(HIPBLAS_STATUS_SUCCESS)) :: hipblasChemvBatchedFortran
    type(c_ptr), value :: handle
    integer(kind(HIPBLAS_FILL_MODE_FULL)), value :: uplo
    integer(c_int), value :: n
    type(c_ptr), value :: alpha
    type(c_ptr), value :: A
    integer(c_int), value :: lda
    type(c_ptr), value :: x
    integer(c_int), value :: incx
    type(c_ptr), value :: beta
    type(c_ptr), value :: y
    integer(c_int), value :: incy
    integer(c_int), value :: batch_count
            hipblasChemvBatchedFortran = &
        hipblasChemvBatched(handle, uplo, n, alpha, A, lda, &
                            x, incx, beta, y, incy, batch_count)
end function hipblasChemvBatchedFortran

function hipblasZhemvBatchedFortran(handle, uplo, n, alpha, A, lda, &
                                    x, incx, beta, y, incy, batch_count) &
    bind(c, name='hipblasZhemvBatchedFortran')
    use iso_c_binding
    use hipblas_enums
    implicit none
    integer(kind(HIPBLAS_STATUS_SUCCESS)) :: hipblasZhemvBatchedFortran
    type(c_ptr), value :: handle
    integer(kind(HIPBLAS_FILL_MODE_FULL)), value :: uplo
    integer(c_int), value :: n
    type(c_ptr), value :: alpha
    type(c_ptr), value :: A
    integer(c_int), value :: lda
    type(c_ptr), value :: x
    integer(c_int), value :: incx
    type(c_ptr), value :: beta
    type(c_ptr), value :: y
    integer(c_int), value :: incy
    integer(c_int), value :: batch_count
            hipblasZhemvBatchedFortran = &
        hipblasZhemvBatched(handle, uplo, n, alpha, A, lda, &
                            x, incx, beta, y, incy, batch_count)
end function hipblasZhemvBatchedFortran

! hemvStridedBatched
function hipblasChemvStridedBatchedFortran(handle, uplo, n, alpha, A, lda, stride_A, &
                                            x, incx, stride_x, beta, y, incy, stride_y, batch_count) &
    bind(c, name='hipblasChemvStridedBatchedFortran')
    use iso_c_binding
    use hipblas_enums
    implicit none
    integer(kind(HIPBLAS_STATUS_SUCCESS)) :: hipblasChemvStridedBatchedFortran
    type(c_ptr), value :: handle
    integer(kind(HIPBLAS_FILL_MODE_FULL)), value :: uplo
    integer(c_int), value :: n
    type(c_ptr), value :: alpha
    type(c_ptr), value :: A
    integer(c_int), value :: lda
    integer(c_int64_t), value :: stride_A
    type(c_ptr), value :: x
    integer(c_int), value :: incx
    integer(c_int64_t), value :: stride_x
    type(c_ptr), value :: beta
    type(c_ptr), value :: y
    integer(c_int), value :: incy
    integer(c_int64_t), value :: stride_y
    integer(c_int), value :: batch_count
            hipblasChemvStridedBatchedFortran = &
        hipblasChemvStridedBatched(handle, uplo, n, alpha, A, lda, stride_A, &
                                    x, incx, stride_x, beta, y, incy, stride_y, batch_count)
end function hipblasChemvStridedBatchedFortran

function hipblasZhemvStridedBatchedFortran(handle, uplo, n, alpha, A, lda, stride_A, &
                                            x, incx, stride_x, beta, y, incy, stride_y, batch_count) &
    bind(c, name='hipblasZhemvStridedBatchedFortran')
    use iso_c_binding
    use hipblas_enums
    implicit none
    integer(kind(HIPBLAS_STATUS_SUCCESS)) :: hipblasZhemvStridedBatchedFortran
    type(c_ptr), value :: handle
    integer(kind(HIPBLAS_FILL_MODE_FULL)), value :: uplo
    integer(c_int), value :: n
    type(c_ptr), value :: alpha
    type(c_ptr), value :: A
    integer(c_int), value :: lda
    integer(c_int64_t), value :: stride_A
    type(c_ptr), value :: x
    integer(c_int), value :: incx
    integer(c_int64_t), value :: stride_x
    type(c_ptr), value :: beta
    type(c_ptr), value :: y
    integer(c_int), value :: incy
    integer(c_int64_t), value :: stride_y
    integer(c_int), value :: batch_count
            hipblasZhemvStridedBatchedFortran = &
        hipblasZhemvStridedBatched(handle, uplo, n, alpha, A, lda, stride_A, &
                                    x, incx, stride_x, beta, y, incy, stride_y, batch_count)
end function hipblasZhemvStridedBatchedFortran

! her
function hipblasCherFortran(handle, uplo, n, alpha, &
                            x, incx, A, lda) &
    bind(c, name='hipblasCherFortran')
    use iso_c_binding
    use hipblas_enums
    implicit none
    integer(kind(HIPBLAS_STATUS_SUCCESS)) :: hipblasCherFortran
    type(c_ptr), value :: handle
    integer(kind(HIPBLAS_FILL_MODE_FULL)), value :: uplo
    integer(c_int), value :: n
    type(c_ptr), value :: alpha
    type(c_ptr), value :: x
    integer(c_int), value :: incx
    type(c_ptr), value :: A
    integer(c_int), value :: lda
            hipblasCherFortran = &
        hipblasCher(handle, uplo, n, alpha, x, incx, A, lda)
end function hipblasCherFortran

function hipblasZherFortran(handle, uplo, n, alpha, &
                            x, incx, A, lda) &
    bind(c, name='hipblasZherFortran')
    use iso_c_binding
    use hipblas_enums
    implicit none
    integer(kind(HIPBLAS_STATUS_SUCCESS)) :: hipblasZherFortran
    type(c_ptr), value :: handle
    integer(kind(HIPBLAS_FILL_MODE_FULL)), value :: uplo
    integer(c_int), value :: n
    type(c_ptr), value :: alpha
    type(c_ptr), value :: x
    integer(c_int), value :: incx
    type(c_ptr), value :: A
    integer(c_int), value :: lda
            hipblasZherFortran = &
        hipblasZher(handle, uplo, n, alpha, x, incx, A, lda)
end function hipblasZherFortran

! herBatched
function hipblasCherBatchedFortran(handle, uplo, n, alpha, &
                                    x, incx, A, lda, batch_count) &
    bind(c, name='hipblasCherBatchedFortran')
    use iso_c_binding
    use hipblas_enums
    implicit none
    integer(kind(HIPBLAS_STATUS_SUCCESS)) :: hipblasCherBatchedFortran
    type(c_ptr), value :: handle
    integer(kind(HIPBLAS_FILL_MODE_FULL)), value :: uplo
    integer(c_int), value :: n
    type(c_ptr), value :: alpha
    type(c_ptr), value :: x
    integer(c_int), value :: incx
    type(c_ptr), value :: A
    integer(c_int), value :: lda
    integer(c_int), value :: batch_count
            hipblasCherBatchedFortran = &
        hipblasCherBatched(handle, uplo, n, alpha, x, incx, A, lda, batch_count)
end function hipblasCherBatchedFortran

function hipblasZherBatchedFortran(handle, uplo, n, alpha, &
                                    x, incx, A, lda, batch_count) &
    bind(c, name='hipblasZherBatchedFortran')
    use iso_c_binding
    use hipblas_enums
    implicit none
    integer(kind(HIPBLAS_STATUS_SUCCESS)) :: hipblasZherBatchedFortran
    type(c_ptr), value :: handle
    integer(kind(HIPBLAS_FILL_MODE_FULL)), value :: uplo
    integer(c_int), value :: n
    type(c_ptr), value :: alpha
    type(c_ptr), value :: x
    integer(c_int), value :: incx
    type(c_ptr), value :: A
    integer(c_int), value :: lda
    integer(c_int), value :: batch_count
            hipblasZherBatchedFortran = &
        hipblasZherBatched(handle, uplo, n, alpha, x, incx, A, lda, batch_count)
end function hipblasZherBatchedFortran

! herStridedBatched
function hipblasCherStridedBatchedFortran(handle, uplo, n, alpha, &
                                            x, incx, stride_x, A, lda, stride_A, batch_count) &
    bind(c, name='hipblasCherStridedBatchedFortran')
    use iso_c_binding
    use hipblas_enums
    implicit none
    integer(kind(HIPBLAS_STATUS_SUCCESS)) :: hipblasCherStridedBatchedFortran
    type(c_ptr), value :: handle
    integer(kind(HIPBLAS_FILL_MODE_FULL)), value :: uplo
    integer(c_int), value :: n
    type(c_ptr), value :: alpha
    type(c_ptr), value :: x
    integer(c_int), value :: incx
    integer(c_int64_t), value :: stride_x
    type(c_ptr), value :: A
    integer(c_int), value :: lda
    integer(c_int64_t), value :: stride_A
    integer(c_int), value :: batch_count
            hipblasCherStridedBatchedFortran = &
        hipblasCherStridedBatched(handle, uplo, n, alpha, x, incx, stride_x, &
                                    A, lda, stride_A, batch_count)
end function hipblasCherStridedBatchedFortran

function hipblasZherStridedBatchedFortran(handle, uplo, n, alpha, &
                                            x, incx, stride_x, A, lda, stride_A, batch_count) &
    bind(c, name='hipblasZherStridedBatchedFortran')
    use iso_c_binding
    use hipblas_enums
    implicit none
    integer(kind(HIPBLAS_STATUS_SUCCESS)) :: hipblasZherStridedBatchedFortran
    type(c_ptr), value :: handle
    integer(kind(HIPBLAS_FILL_MODE_FULL)), value :: uplo
    integer(c_int), value :: n
    type(c_ptr), value :: alpha
    type(c_ptr), value :: x
    integer(c_int), value :: incx
    integer(c_int64_t), value :: stride_x
    type(c_ptr), value :: A
    integer(c_int), value :: lda
    integer(c_int64_t), value :: stride_A
    integer(c_int), value :: batch_count
            hipblasZherStridedBatchedFortran = &
        hipblasZherStridedBatched(handle, uplo, n, alpha, x, incx, stride_x, &
                                    A, lda, stride_A, batch_count)
end function hipblasZherStridedBatchedFortran

! her2
function hipblasCher2Fortran(handle, uplo, n, alpha, &
                                x, incx, y, incy, A, lda) &
    bind(c, name='hipblasCher2Fortran')
    use iso_c_binding
    use hipblas_enums
    implicit none
    integer(kind(HIPBLAS_STATUS_SUCCESS)) :: hipblasCher2Fortran
    type(c_ptr), value :: handle
    integer(kind(HIPBLAS_FILL_MODE_FULL)), value :: uplo
    integer(c_int), value :: n
    type(c_ptr), value :: alpha
    type(c_ptr), value :: x
    integer(c_int), value :: incx
    type(c_ptr), value :: y
    integer(c_int), value :: incy
    type(c_ptr), value :: A
    integer(c_int), value :: lda
            hipblasCher2Fortran = &
        hipblasCher2(handle, uplo, n, alpha, x, incx, &
                        y, incy, A, lda)
end function hipblasCher2Fortran

function hipblasZher2Fortran(handle, uplo, n, alpha, &
                                x, incx, y, incy, A, lda) &
    bind(c, name='hipblasZher2Fortran')
    use iso_c_binding
    use hipblas_enums
    implicit none
    integer(kind(HIPBLAS_STATUS_SUCCESS)) :: hipblasZher2Fortran
    type(c_ptr), value :: handle
    integer(kind(HIPBLAS_FILL_MODE_FULL)), value :: uplo
    integer(c_int), value :: n
    type(c_ptr), value :: alpha
    type(c_ptr), value :: x
    integer(c_int), value :: incx
    type(c_ptr), value :: y
    integer(c_int), value :: incy
    type(c_ptr), value :: A
    integer(c_int), value :: lda
            hipblasZher2Fortran = &
        hipblasZher2(handle, uplo, n, alpha, x, incx, &
                        y, incy, A, lda)
end function hipblasZher2Fortran

! her2Batched
function hipblasCher2BatchedFortran(handle, uplo, n, alpha, &
                                    x, incx, y, incy, A, lda, batch_count) &
    bind(c, name='hipblasCher2BatchedFortran')
    use iso_c_binding
    use hipblas_enums
    implicit none
    integer(kind(HIPBLAS_STATUS_SUCCESS)) :: hipblasCher2BatchedFortran
    type(c_ptr), value :: handle
    integer(kind(HIPBLAS_FILL_MODE_FULL)), value :: uplo
    integer(c_int), value :: n
    type(c_ptr), value :: alpha
    type(c_ptr), value :: x
    integer(c_int), value :: incx
    type(c_ptr), value :: y
    integer(c_int), value :: incy
    type(c_ptr), value :: A
    integer(c_int), value :: lda
    integer(c_int), value :: batch_count
            hipblasCher2BatchedFortran = &
        hipblasCher2Batched(handle, uplo, n, alpha, x, incx, &
                            y, incy, A, lda, batch_count)
end function hipblasCher2BatchedFortran

function hipblasZher2BatchedFortran(handle, uplo, n, alpha, &
                                    x, incx, y, incy, A, lda, batch_count) &
    bind(c, name='hipblasZher2BatchedFortran')
    use iso_c_binding
    use hipblas_enums
    implicit none
    integer(kind(HIPBLAS_STATUS_SUCCESS)) :: hipblasZher2BatchedFortran
    type(c_ptr), value :: handle
    integer(kind(HIPBLAS_FILL_MODE_FULL)), value :: uplo
    integer(c_int), value :: n
    type(c_ptr), value :: alpha
    type(c_ptr), value :: x
    integer(c_int), value :: incx
    type(c_ptr), value :: y
    integer(c_int), value :: incy
    type(c_ptr), value :: A
    integer(c_int), value :: lda
    integer(c_int), value :: batch_count
            hipblasZher2BatchedFortran = &
        hipblasZher2Batched(handle, uplo, n, alpha, x, incx, &
                            y, incy, A, lda, batch_count)
end function hipblasZher2BatchedFortran

! her2StridedBatched
function hipblasCher2StridedBatchedFortran(handle, uplo, n, alpha, &
                                            x, incx, stride_x, y, incy, stride_y, A, lda, stride_A, batch_count) &
    bind(c, name='hipblasCher2StridedBatchedFortran')
    use iso_c_binding
    use hipblas_enums
    implicit none
    integer(kind(HIPBLAS_STATUS_SUCCESS)) :: hipblasCher2StridedBatchedFortran
    type(c_ptr), value :: handle
    integer(kind(HIPBLAS_FILL_MODE_FULL)), value :: uplo
    integer(c_int), value :: n
    type(c_ptr), value :: alpha
    type(c_ptr), value :: x
    integer(c_int), value :: incx
    integer(c_int64_t), value :: stride_x
    type(c_ptr), value :: y
    integer(c_int), value :: incy
    integer(c_int64_t), value :: stride_y
    type(c_ptr), value :: A
    integer(c_int), value :: lda
    integer(c_int64_t), value :: stride_A
    integer(c_int), value :: batch_count
            hipblasCher2StridedBatchedFortran = &
        hipblasCher2StridedBatched(handle, uplo, n, alpha, x, incx, stride_x, &
                                    y, incy, stride_y, A, lda, stride_A, batch_count)
end function hipblasCher2StridedBatchedFortran

function hipblasZher2StridedBatchedFortran(handle, uplo, n, alpha, &
                                            x, incx, stride_x, y, incy, stride_y, A, lda, stride_A, batch_count) &
    bind(c, name='hipblasZher2StridedBatchedFortran')
    use iso_c_binding
    use hipblas_enums
    implicit none
    integer(kind(HIPBLAS_STATUS_SUCCESS)) :: hipblasZher2StridedBatchedFortran
    type(c_ptr), value :: handle
    integer(kind(HIPBLAS_FILL_MODE_FULL)), value :: uplo
    integer(c_int), value :: n
    type(c_ptr), value :: alpha
    type(c_ptr), value :: x
    integer(c_int), value :: incx
    integer(c_int64_t), value :: stride_x
    type(c_ptr), value :: y
    integer(c_int), value :: incy
    integer(c_int64_t), value :: stride_y
    type(c_ptr), value :: A
    integer(c_int), value :: lda
    integer(c_int64_t), value :: stride_A
    integer(c_int), value :: batch_count
            hipblasZher2StridedBatchedFortran = &
        hipblasZher2StridedBatched(handle, uplo, n, alpha, x, incx, stride_x, &
                                    y, incy, stride_y, A, lda, stride_A, batch_count)
end function hipblasZher2StridedBatchedFortran

! hpmv
function hipblasChpmvFortran(handle, uplo, n, alpha, AP, &
                                x, incx, beta, y, incy) &
    bind(c, name='hipblasChpmvFortran')
    use iso_c_binding
    use hipblas_enums
    implicit none
    integer(kind(HIPBLAS_STATUS_SUCCESS)) :: hipblasChpmvFortran
    type(c_ptr), value :: handle
    integer(kind(HIPBLAS_FILL_MODE_FULL)), value :: uplo
    integer(c_int), value :: n
    type(c_ptr), value :: alpha
    type(c_ptr), value :: AP
    type(c_ptr), value :: x
    integer(c_int), value :: incx
    type(c_ptr), value :: beta
    type(c_ptr), value :: y
    integer(c_int), value :: incy
            hipblasChpmvFortran = &
        hipblasChpmv(handle, uplo, n, alpha, AP, &
                        x, incx, beta, y, incy)
end function hipblasChpmvFortran

function hipblasZhpmvFortran(handle, uplo, n, alpha, AP, &
                                x, incx, beta, y, incy) &
    bind(c, name='hipblasZhpmvFortran')
    use iso_c_binding
    use hipblas_enums
    implicit none
    integer(kind(HIPBLAS_STATUS_SUCCESS)) :: hipblasZhpmvFortran
    type(c_ptr), value :: handle
    integer(kind(HIPBLAS_FILL_MODE_FULL)), value :: uplo
    integer(c_int), value :: n
    type(c_ptr), value :: alpha
    type(c_ptr), value :: AP
    type(c_ptr), value :: x
    integer(c_int), value :: incx
    type(c_ptr), value :: beta
    type(c_ptr), value :: y
    integer(c_int), value :: incy
            hipblasZhpmvFortran = &
        hipblasZhpmv(handle, uplo, n, alpha, AP, &
                        x, incx, beta, y, incy)
end function hipblasZhpmvFortran

! hpmvBatched
function hipblasChpmvBatchedFortran(handle, uplo, n, alpha, AP, &
                                    x, incx, beta, y, incy, batch_count) &
    bind(c, name='hipblasChpmvBatchedFortran')
    use iso_c_binding
    use hipblas_enums
    implicit none
    integer(kind(HIPBLAS_STATUS_SUCCESS)) :: hipblasChpmvBatchedFortran
    type(c_ptr), value :: handle
    integer(kind(HIPBLAS_FILL_MODE_FULL)), value :: uplo
    integer(c_int), value :: n
    type(c_ptr), value :: alpha
    type(c_ptr), value :: AP
    type(c_ptr), value :: x
    integer(c_int), value :: incx
    type(c_ptr), value :: beta
    type(c_ptr), value :: y
    integer(c_int), value :: incy
    integer(c_int), value :: batch_count
            hipblasChpmvBatchedFortran = &
        hipblasChpmvBatched(handle, uplo, n, alpha, AP, &
                            x, incx, beta, y, incy, batch_count)
end function hipblasChpmvBatchedFortran

function hipblasZhpmvBatchedFortran(handle, uplo, n, alpha, AP, &
                                    x, incx, beta, y, incy, batch_count) &
    bind(c, name='hipblasZhpmvBatchedFortran')
    use iso_c_binding
    use hipblas_enums
    implicit none
    integer(kind(HIPBLAS_STATUS_SUCCESS)) :: hipblasZhpmvBatchedFortran
    type(c_ptr), value :: handle
    integer(kind(HIPBLAS_FILL_MODE_FULL)), value :: uplo
    integer(c_int), value :: n
    type(c_ptr), value :: alpha
    type(c_ptr), value :: AP
    type(c_ptr), value :: x
    integer(c_int), value :: incx
    type(c_ptr), value :: beta
    type(c_ptr), value :: y
    integer(c_int), value :: incy
    integer(c_int), value :: batch_count
            hipblasZhpmvBatchedFortran = &
        hipblasZhpmvBatched(handle, uplo, n, alpha, AP, &
                            x, incx, beta, y, incy, batch_count)
end function hipblasZhpmvBatchedFortran

! hpmvStridedBatched
function hipblasChpmvStridedBatchedFortran(handle, uplo, n, alpha, AP, stride_AP, &
                                            x, incx, stride_x, beta, y, incy, stride_y, batch_count) &
    bind(c, name='hipblasChpmvStridedBatchedFortran')
    use iso_c_binding
    use hipblas_enums
    implicit none
    integer(kind(HIPBLAS_STATUS_SUCCESS)) :: hipblasChpmvStridedBatchedFortran
    type(c_ptr), value :: handle
    integer(kind(HIPBLAS_FILL_MODE_FULL)), value :: uplo
    integer(c_int), value :: n
    type(c_ptr), value :: alpha
    type(c_ptr), value :: AP
    integer(c_int64_t), value :: stride_AP
    type(c_ptr), value :: x
    integer(c_int), value :: incx
    integer(c_int64_t), value :: stride_x
    type(c_ptr), value :: beta
    type(c_ptr), value :: y
    integer(c_int), value :: incy
    integer(c_int64_t), value :: stride_y
    integer(c_int), value :: batch_count
            hipblasChpmvStridedBatchedFortran = &
        hipblasChpmvStridedBatched(handle, uplo, n, alpha, AP, stride_AP, &
                                    x, incx, stride_x, beta, y, incy, stride_y, batch_count)
end function hipblasChpmvStridedBatchedFortran

function hipblasZhpmvStridedBatchedFortran(handle, uplo, n, alpha, AP, stride_AP, &
                                            x, incx, stride_x, beta, y, incy, stride_y, batch_count) &
    bind(c, name='hipblasZhpmvStridedBatchedFortran')
    use iso_c_binding
    use hipblas_enums
    implicit none
    integer(kind(HIPBLAS_STATUS_SUCCESS)) :: hipblasZhpmvStridedBatchedFortran
    type(c_ptr), value :: handle
    integer(kind(HIPBLAS_FILL_MODE_FULL)), value :: uplo
    integer(c_int), value :: n
    type(c_ptr), value :: alpha
    type(c_ptr), value :: AP
    integer(c_int64_t), value :: stride_AP
    type(c_ptr), value :: x
    integer(c_int), value :: incx
    integer(c_int64_t), value :: stride_x
    type(c_ptr), value :: beta
    type(c_ptr), value :: y
    integer(c_int), value :: incy
    integer(c_int64_t), value :: stride_y
    integer(c_int), value :: batch_count
            hipblasZhpmvStridedBatchedFortran = &
        hipblasZhpmvStridedBatched(handle, uplo, n, alpha, AP, stride_AP, &
                                    x, incx, stride_x, beta, y, incy, stride_y, batch_count)
end function hipblasZhpmvStridedBatchedFortran

! hpr
function hipblasChprFortran(handle, uplo, n, alpha, &
                            x, incx, AP) &
    bind(c, name='hipblasChprFortran')
    use iso_c_binding
    use hipblas_enums
    implicit none
    integer(kind(HIPBLAS_STATUS_SUCCESS)) :: hipblasChprFortran
    type(c_ptr), value :: handle
    integer(kind(HIPBLAS_FILL_MODE_FULL)), value :: uplo
    integer(c_int), value :: n
    type(c_ptr), value :: alpha
    type(c_ptr), value :: x
    integer(c_int), value :: incx
    type(c_ptr), value :: AP
            hipblasChprFortran = &
        hipblasChpr(handle, uplo, n, alpha, x, incx, AP)
end function hipblasChprFortran

function hipblasZhprFortran(handle, uplo, n, alpha, &
                            x, incx, AP) &
    bind(c, name='hipblasZhprFortran')
    use iso_c_binding
    use hipblas_enums
    implicit none
    integer(kind(HIPBLAS_STATUS_SUCCESS)) :: hipblasZhprFortran
    type(c_ptr), value :: handle
    integer(kind(HIPBLAS_FILL_MODE_FULL)), value :: uplo
    integer(c_int), value :: n
    type(c_ptr), value :: alpha
    type(c_ptr), value :: x
    integer(c_int), value :: incx
    type(c_ptr), value :: AP
            hipblasZhprFortran = &
        hipblasZhpr(handle, uplo, n, alpha, x, incx, AP)
end function hipblasZhprFortran

! hprBatched
function hipblasChprBatchedFortran(handle, uplo, n, alpha, &
                                    x, incx, AP, batch_count) &
    bind(c, name='hipblasChprBatchedFortran')
    use iso_c_binding
    use hipblas_enums
    implicit none
    integer(kind(HIPBLAS_STATUS_SUCCESS)) :: hipblasChprBatchedFortran
    type(c_ptr), value :: handle
    integer(kind(HIPBLAS_FILL_MODE_FULL)), value :: uplo
    integer(c_int), value :: n
    type(c_ptr), value :: alpha
    type(c_ptr), value :: x
    integer(c_int), value :: incx
    type(c_ptr), value :: AP
    integer(c_int), value :: batch_count
            hipblasChprBatchedFortran = &
        hipblasChprBatched(handle, uplo, n, alpha, x, incx, AP, batch_count)
end function hipblasChprBatchedFortran

function hipblasZhprBatchedFortran(handle, uplo, n, alpha, &
                                    x, incx, AP, batch_count) &
    bind(c, name='hipblasZhprBatchedFortran')
    use iso_c_binding
    use hipblas_enums
    implicit none
    integer(kind(HIPBLAS_STATUS_SUCCESS)) :: hipblasZhprBatchedFortran
    type(c_ptr), value :: handle
    integer(kind(HIPBLAS_FILL_MODE_FULL)), value :: uplo
    integer(c_int), value :: n
    type(c_ptr), value :: alpha
    type(c_ptr), value :: x
    integer(c_int), value :: incx
    type(c_ptr), value :: AP
    integer(c_int), value :: batch_count
            hipblasZhprBatchedFortran = &
        hipblasZhprBatched(handle, uplo, n, alpha, x, incx, AP, batch_count)
end function hipblasZhprBatchedFortran

! hprStridedBatched
function hipblasChprStridedBatchedFortran(handle, uplo, n, alpha, &
                                            x, incx, stride_x, AP, stride_AP, batch_count) &
    bind(c, name='hipblasChprStridedBatchedFortran')
    use iso_c_binding
    use hipblas_enums
    implicit none
    integer(kind(HIPBLAS_STATUS_SUCCESS)) :: hipblasChprStridedBatchedFortran
    type(c_ptr), value :: handle
    integer(kind(HIPBLAS_FILL_MODE_FULL)), value :: uplo
    integer(c_int), value :: n
    type(c_ptr), value :: alpha
    type(c_ptr), value :: x
    integer(c_int), value :: incx
    integer(c_int64_t), value :: stride_x
    type(c_ptr), value :: AP
    integer(c_int64_t), value :: stride_AP
    integer(c_int), value :: batch_count
            hipblasChprStridedBatchedFortran = &
        hipblasChprStridedBatched(handle, uplo, n, alpha, x, incx, stride_x, &
                                    AP, stride_AP, batch_count)
end function hipblasChprStridedBatchedFortran

function hipblasZhprStridedBatchedFortran(handle, uplo, n, alpha, &
                                            x, incx, stride_x, AP, stride_AP, batch_count) &
    bind(c, name='hipblasZhprStridedBatchedFortran')
    use iso_c_binding
    use hipblas_enums
    implicit none
    integer(kind(HIPBLAS_STATUS_SUCCESS)) :: hipblasZhprStridedBatchedFortran
    type(c_ptr), value :: handle
    integer(kind(HIPBLAS_FILL_MODE_FULL)), value :: uplo
    integer(c_int), value :: n
    type(c_ptr), value :: alpha
    type(c_ptr), value :: x
    integer(c_int), value :: incx
    integer(c_int64_t), value :: stride_x
    type(c_ptr), value :: AP
    integer(c_int64_t), value :: stride_AP
    integer(c_int), value :: batch_count
            hipblasZhprStridedBatchedFortran = &
        hipblasZhprStridedBatched(handle, uplo, n, alpha, x, incx, stride_x, &
                                    AP, stride_AP, batch_count)
end function hipblasZhprStridedBatchedFortran

! hpr2
function hipblasChpr2Fortran(handle, uplo, n, alpha, &
                                x, incx, y, incy, AP) &
    bind(c, name='hipblasChpr2Fortran')
    use iso_c_binding
    use hipblas_enums
    implicit none
    integer(kind(HIPBLAS_STATUS_SUCCESS)) :: hipblasChpr2Fortran
    type(c_ptr), value :: handle
    integer(kind(HIPBLAS_FILL_MODE_FULL)), value :: uplo
    integer(c_int), value :: n
    type(c_ptr), value :: alpha
    type(c_ptr), value :: x
    integer(c_int), value :: incx
    type(c_ptr), value :: y
    integer(c_int), value :: incy
    type(c_ptr), value :: AP
            hipblasChpr2Fortran = &
        hipblasChpr2(handle, uplo, n, alpha, x, incx, &
                        y, incy, AP)
end function hipblasChpr2Fortran

function hipblasZhpr2Fortran(handle, uplo, n, alpha, &
                                x, incx, y, incy, AP) &
    bind(c, name='hipblasZhpr2Fortran')
    use iso_c_binding
    use hipblas_enums
    implicit none
    integer(kind(HIPBLAS_STATUS_SUCCESS)) :: hipblasZhpr2Fortran
    type(c_ptr), value :: handle
    integer(kind(HIPBLAS_FILL_MODE_FULL)), value :: uplo
    integer(c_int), value :: n
    type(c_ptr), value :: alpha
    type(c_ptr), value :: x
    integer(c_int), value :: incx
    type(c_ptr), value :: y
    integer(c_int), value :: incy
    type(c_ptr), value :: AP
            hipblasZhpr2Fortran = &
        hipblasZhpr2(handle, uplo, n, alpha, x, incx, &
                        y, incy, AP)
end function hipblasZhpr2Fortran

! hpr2Batched
function hipblasChpr2BatchedFortran(handle, uplo, n, alpha, &
                                    x, incx, y, incy, AP, batch_count) &
    bind(c, name='hipblasChpr2BatchedFortran')
    use iso_c_binding
    use hipblas_enums
    implicit none
    integer(kind(HIPBLAS_STATUS_SUCCESS)) :: hipblasChpr2BatchedFortran
    type(c_ptr), value :: handle
    integer(kind(HIPBLAS_FILL_MODE_FULL)), value :: uplo
    integer(c_int), value :: n
    type(c_ptr), value :: alpha
    type(c_ptr), value :: x
    integer(c_int), value :: incx
    type(c_ptr), value :: y
    integer(c_int), value :: incy
    type(c_ptr), value :: AP
    integer(c_int), value :: batch_count
            hipblasChpr2BatchedFortran = &
        hipblasChpr2Batched(handle, uplo, n, alpha, x, incx, &
                            y, incy, AP, batch_count)
end function hipblasChpr2BatchedFortran

function hipblasZhpr2BatchedFortran(handle, uplo, n, alpha, &
                                    x, incx, y, incy, AP, batch_count) &
    bind(c, name='hipblasZhpr2BatchedFortran')
    use iso_c_binding
    use hipblas_enums
    implicit none
    integer(kind(HIPBLAS_STATUS_SUCCESS)) :: hipblasZhpr2BatchedFortran
    type(c_ptr), value :: handle
    integer(kind(HIPBLAS_FILL_MODE_FULL)), value :: uplo
    integer(c_int), value :: n
    type(c_ptr), value :: alpha
    type(c_ptr), value :: x
    integer(c_int), value :: incx
    type(c_ptr), value :: y
    integer(c_int), value :: incy
    type(c_ptr), value :: AP
    integer(c_int), value :: batch_count
            hipblasZhpr2BatchedFortran = &
        hipblasZhpr2Batched(handle, uplo, n, alpha, x, incx, &
                            y, incy, AP, batch_count)
end function hipblasZhpr2BatchedFortran

! hpr2StridedBatched
function hipblasChpr2StridedBatchedFortran(handle, uplo, n, alpha, &
                                            x, incx, stride_x, y, incy, stride_y, AP, stride_AP, batch_count) &
    bind(c, name='hipblasChpr2StridedBatchedFortran')
    use iso_c_binding
    use hipblas_enums
    implicit none
    integer(kind(HIPBLAS_STATUS_SUCCESS)) :: hipblasChpr2StridedBatchedFortran
    type(c_ptr), value :: handle
    integer(kind(HIPBLAS_FILL_MODE_FULL)), value :: uplo
    integer(c_int), value :: n
    type(c_ptr), value :: alpha
    type(c_ptr), value :: x
    integer(c_int), value :: incx
    integer(c_int64_t), value :: stride_x
    type(c_ptr), value :: y
    integer(c_int), value :: incy
    integer(c_int64_t), value :: stride_y
    type(c_ptr), value :: AP
    integer(c_int64_t), value :: stride_AP
    integer(c_int), value :: batch_count
            hipblasChpr2StridedBatchedFortran = &
        hipblasChpr2StridedBatched(handle, uplo, n, alpha, x, incx, stride_x, &
                                    y, incy, stride_y, AP, stride_AP, batch_count)
end function hipblasChpr2StridedBatchedFortran

function hipblasZhpr2StridedBatchedFortran(handle, uplo, n, alpha, &
                                            x, incx, stride_x, y, incy, stride_y, AP, stride_AP, batch_count) &
    bind(c, name='hipblasZhpr2StridedBatchedFortran')
    use iso_c_binding
    use hipblas_enums
    implicit none
    integer(kind(HIPBLAS_STATUS_SUCCESS)) :: hipblasZhpr2StridedBatchedFortran
    type(c_ptr), value :: handle
    integer(kind(HIPBLAS_FILL_MODE_FULL)), value :: uplo
    integer(c_int), value :: n
    type(c_ptr), value :: alpha
    type(c_ptr), value :: x
    integer(c_int), value :: incx
    integer(c_int64_t), value :: stride_x
    type(c_ptr), value :: y
    integer(c_int), value :: incy
    integer(c_int64_t), value :: stride_y
    type(c_ptr), value :: AP
    integer(c_int64_t), value :: stride_AP
    integer(c_int), value :: batch_count
            hipblasZhpr2StridedBatchedFortran = &
        hipblasZhpr2StridedBatched(handle, uplo, n, alpha, x, incx, stride_x, &
                                    y, incy, stride_y, AP, stride_AP, batch_count)
end function hipblasZhpr2StridedBatchedFortran

! trmv
function hipblasStrmvFortran(handle, uplo, transA, diag, m, &
                                A, lda, x, incx) &
    bind(c, name='hipblasStrmvFortran')
    use iso_c_binding
    use hipblas_enums
    implicit none
    integer(kind(HIPBLAS_STATUS_SUCCESS)) :: hipblasStrmvFortran
    type(c_ptr), value :: handle
    integer(kind(HIPBLAS_FILL_MODE_FULL)), value :: uplo
    integer(kind(HIPBLAS_OP_N)), value :: transA
    integer(kind(HIPBLAS_DIAG_NON_UNIT)), value :: diag
    integer(c_int), value :: m
    type(c_ptr), value :: A
    integer(c_int), value :: lda
    type(c_ptr), value :: x
    integer(c_int), value :: incx
            hipblasStrmvFortran = &
        hipblasStrmv(handle, uplo, transA, diag, m, &
                        A, lda, x, incx)
end function hipblasStrmvFortran

function hipblasDtrmvFortran(handle, uplo, transA, diag, m, &
                                A, lda, x, incx) &
    bind(c, name='hipblasDtrmvFortran')
    use iso_c_binding
    use hipblas_enums
    implicit none
    integer(kind(HIPBLAS_STATUS_SUCCESS)) :: hipblasDtrmvFortran
    type(c_ptr), value :: handle
    integer(kind(HIPBLAS_FILL_MODE_FULL)), value :: uplo
    integer(kind(HIPBLAS_OP_N)), value :: transA
    integer(kind(HIPBLAS_DIAG_NON_UNIT)), value :: diag
    integer(c_int), value :: m
    type(c_ptr), value :: A
    integer(c_int), value :: lda
    type(c_ptr), value :: x
    integer(c_int), value :: incx
            hipblasDtrmvFortran = &
        hipblasDtrmv(handle, uplo, transA, diag, m, &
                        A, lda, x, incx)
end function hipblasDtrmvFortran

function hipblasCtrmvFortran(handle, uplo, transA, diag, m, &
                                A, lda, x, incx) &
    bind(c, name='hipblasCtrmvFortran')
    use iso_c_binding
    use hipblas_enums
    implicit none
    integer(kind(HIPBLAS_STATUS_SUCCESS)) :: hipblasCtrmvFortran
    type(c_ptr), value :: handle
    integer(kind(HIPBLAS_FILL_MODE_FULL)), value :: uplo
    integer(kind(HIPBLAS_OP_N)), value :: transA
    integer(kind(HIPBLAS_DIAG_NON_UNIT)), value :: diag
    integer(c_int), value :: m
    type(c_ptr), value :: A
    integer(c_int), value :: lda
    type(c_ptr), value :: x
    integer(c_int), value :: incx
            hipblasCtrmvFortran = &
        hipblasCtrmv(handle, uplo, transA, diag, m, &
                        A, lda, x, incx)
end function hipblasCtrmvFortran

function hipblasZtrmvFortran(handle, uplo, transA, diag, m, &
                                A, lda, x, incx) &
    bind(c, name='hipblasZtrmvFortran')
    use iso_c_binding
    use hipblas_enums
    implicit none
    integer(kind(HIPBLAS_STATUS_SUCCESS)) :: hipblasZtrmvFortran
    type(c_ptr), value :: handle
    integer(kind(HIPBLAS_FILL_MODE_FULL)), value :: uplo
    integer(kind(HIPBLAS_OP_N)), value :: transA
    integer(kind(HIPBLAS_DIAG_NON_UNIT)), value :: diag
    integer(c_int), value :: m
    type(c_ptr), value :: A
    integer(c_int), value :: lda
    type(c_ptr), value :: x
    integer(c_int), value :: incx
            hipblasZtrmvFortran = &
        hipblasZtrmv(handle, uplo, transA, diag, m, &
                        A, lda, x, incx)
end function hipblasZtrmvFortran

! trmvBatched
function hipblasStrmvBatchedFortran(handle, uplo, transA, diag, m, &
                                    A, lda, x, incx, batch_count) &
    bind(c, name='hipblasStrmvBatchedFortran')
    use iso_c_binding
    use hipblas_enums
    implicit none
    integer(kind(HIPBLAS_STATUS_SUCCESS)) :: hipblasStrmvBatchedFortran
    type(c_ptr), value :: handle
    integer(kind(HIPBLAS_FILL_MODE_FULL)), value :: uplo
    integer(kind(HIPBLAS_OP_N)), value :: transA
    integer(kind(HIPBLAS_DIAG_NON_UNIT)), value :: diag
    integer(c_int), value :: m
    type(c_ptr), value :: A
    integer(c_int), value :: lda
    type(c_ptr), value :: x
    integer(c_int), value :: incx
    integer(c_int), value :: batch_count
            hipblasStrmvBatchedFortran = &
        hipblasStrmvBatched(handle, uplo, transA, diag, m, &
                            A, lda, x, incx, batch_count)
end function hipblasStrmvBatchedFortran

function hipblasDtrmvBatchedFortran(handle, uplo, transA, diag, m, &
                                    A, lda, x, incx, batch_count) &
    bind(c, name='hipblasDtrmvBatchedFortran')
    use iso_c_binding
    use hipblas_enums
    implicit none
    integer(kind(HIPBLAS_STATUS_SUCCESS)) :: hipblasDtrmvBatchedFortran
    type(c_ptr), value :: handle
    integer(kind(HIPBLAS_FILL_MODE_FULL)), value :: uplo
    integer(kind(HIPBLAS_OP_N)), value :: transA
    integer(kind(HIPBLAS_DIAG_NON_UNIT)), value :: diag
    integer(c_int), value :: m
    type(c_ptr), value :: A
    integer(c_int), value :: lda
    type(c_ptr), value :: x
    integer(c_int), value :: incx
    integer(c_int), value :: batch_count
            hipblasDtrmvBatchedFortran = &
        hipblasDtrmvBatched(handle, uplo, transA, diag, m, &
                            A, lda, x, incx, batch_count)
end function hipblasDtrmvBatchedFortran

function hipblasCtrmvBatchedFortran(handle, uplo, transA, diag, m, &
                                    A, lda, x, incx, batch_count) &
    bind(c, name='hipblasCtrmvBatchedFortran')
    use iso_c_binding
    use hipblas_enums
    implicit none
    integer(kind(HIPBLAS_STATUS_SUCCESS)) :: hipblasCtrmvBatchedFortran
    type(c_ptr), value :: handle
    integer(kind(HIPBLAS_FILL_MODE_FULL)), value :: uplo
    integer(kind(HIPBLAS_OP_N)), value :: transA
    integer(kind(HIPBLAS_DIAG_NON_UNIT)), value :: diag
    integer(c_int), value :: m
    type(c_ptr), value :: A
    integer(c_int), value :: lda
    type(c_ptr), value :: x
    integer(c_int), value :: incx
    integer(c_int), value :: batch_count
            hipblasCtrmvBatchedFortran = &
        hipblasCtrmvBatched(handle, uplo, transA, diag, m, &
                            A, lda, x, incx, batch_count)
end function hipblasCtrmvBatchedFortran

function hipblasZtrmvBatchedFortran(handle, uplo, transA, diag, m, &
                                    A, lda, x, incx, batch_count) &
    bind(c, name='hipblasZtrmvBatchedFortran')
    use iso_c_binding
    use hipblas_enums
    implicit none
    integer(kind(HIPBLAS_STATUS_SUCCESS)) :: hipblasZtrmvBatchedFortran
    type(c_ptr), value :: handle
    integer(kind(HIPBLAS_FILL_MODE_FULL)), value :: uplo
    integer(kind(HIPBLAS_OP_N)), value :: transA
    integer(kind(HIPBLAS_DIAG_NON_UNIT)), value :: diag
    integer(c_int), value :: m
    type(c_ptr), value :: A
    integer(c_int), value :: lda
    type(c_ptr), value :: x
    integer(c_int), value :: incx
    integer(c_int), value :: batch_count
            hipblasZtrmvBatchedFortran = &
        hipblasZtrmvBatched(handle, uplo, transA, diag, m, &
                            A, lda, x, incx, batch_count)
end function hipblasZtrmvBatchedFortran

! trmvStridedBatched
function hipblasStrmvStridedBatchedFortran(handle, uplo, transA, diag, m, &
                                            A, lda, stride_A, x, incx, stride_x, batch_count) &
    bind(c, name='hipblasStrmvStridedBatchedFortran')
    use iso_c_binding
    use hipblas_enums
    implicit none
    integer(kind(HIPBLAS_STATUS_SUCCESS)) :: hipblasStrmvStridedBatchedFortran
    type(c_ptr), value :: handle
    integer(kind(HIPBLAS_FILL_MODE_FULL)), value :: uplo
    integer(kind(HIPBLAS_OP_N)), value :: transA
    integer(kind(HIPBLAS_DIAG_NON_UNIT)), value :: diag
    integer(c_int), value :: m
    type(c_ptr), value :: A
    integer(c_int), value :: lda
    integer(c_int64_t), value :: stride_A
    type(c_ptr), value :: x
    integer(c_int), value :: incx
    integer(c_int64_t), value :: stride_x
    integer(c_int), value :: batch_count
            hipblasStrmvStridedBatchedFortran = &
        hipblasStrmvStridedBatched(handle, uplo, transA, diag, m, &
                                    A, lda, stride_A, x, incx, stride_x, batch_count)
end function hipblasStrmvStridedBatchedFortran

function hipblasDtrmvStridedBatchedFortran(handle, uplo, transA, diag, m, &
                                            A, lda, stride_A, x, incx, stride_x, batch_count) &
    bind(c, name='hipblasDtrmvStridedBatchedFortran')
    use iso_c_binding
    use hipblas_enums
    implicit none
    integer(kind(HIPBLAS_STATUS_SUCCESS)) :: hipblasDtrmvStridedBatchedFortran
    type(c_ptr), value :: handle
    integer(kind(HIPBLAS_FILL_MODE_FULL)), value :: uplo
    integer(kind(HIPBLAS_OP_N)), value :: transA
    integer(kind(HIPBLAS_DIAG_NON_UNIT)), value :: diag
    integer(c_int), value :: m
    type(c_ptr), value :: A
    integer(c_int), value :: lda
    integer(c_int64_t), value :: stride_A
    type(c_ptr), value :: x
    integer(c_int), value :: incx
    integer(c_int64_t), value :: stride_x
    integer(c_int), value :: batch_count
            hipblasDtrmvStridedBatchedFortran = &
        hipblasDtrmvStridedBatched(handle, uplo, transA, diag, m, &
                                    A, lda, stride_A, x, incx, stride_x, batch_count)
end function hipblasDtrmvStridedBatchedFortran

function hipblasCtrmvStridedBatchedFortran(handle, uplo, transA, diag, m, &
                                            A, lda, stride_A, x, incx, stride_x, batch_count) &
    bind(c, name='hipblasCtrmvStridedBatchedFortran')
    use iso_c_binding
    use hipblas_enums
    implicit none
    integer(kind(HIPBLAS_STATUS_SUCCESS)) :: hipblasCtrmvStridedBatchedFortran
    type(c_ptr), value :: handle
    integer(kind(HIPBLAS_FILL_MODE_FULL)), value :: uplo
    integer(kind(HIPBLAS_OP_N)), value :: transA
    integer(kind(HIPBLAS_DIAG_NON_UNIT)), value :: diag
    integer(c_int), value :: m
    type(c_ptr), value :: A
    integer(c_int), value :: lda
    integer(c_int64_t), value :: stride_A
    type(c_ptr), value :: x
    integer(c_int), value :: incx
    integer(c_int64_t), value :: stride_x
    integer(c_int), value :: batch_count
            hipblasCtrmvStridedBatchedFortran = &
        hipblasCtrmvStridedBatched(handle, uplo, transA, diag, m, &
                                    A, lda, stride_A, x, incx, stride_x, batch_count)
end function hipblasCtrmvStridedBatchedFortran

function hipblasZtrmvStridedBatchedFortran(handle, uplo, transA, diag, m, &
                                            A, lda, stride_A, x, incx, stride_x, batch_count) &
    bind(c, name='hipblasZtrmvStridedBatchedFortran')
    use iso_c_binding
    use hipblas_enums
    implicit none
    integer(kind(HIPBLAS_STATUS_SUCCESS)) :: hipblasZtrmvStridedBatchedFortran
    type(c_ptr), value :: handle
    integer(kind(HIPBLAS_FILL_MODE_FULL)), value :: uplo
    integer(kind(HIPBLAS_OP_N)), value :: transA
    integer(kind(HIPBLAS_DIAG_NON_UNIT)), value :: diag
    integer(c_int), value :: m
    type(c_ptr), value :: A
    integer(c_int), value :: lda
    integer(c_int64_t), value :: stride_A
    type(c_ptr), value :: x
    integer(c_int), value :: incx
    integer(c_int64_t), value :: stride_x
    integer(c_int), value :: batch_count
            hipblasZtrmvStridedBatchedFortran = &
        hipblasZtrmvStridedBatched(handle, uplo, transA, diag, m, &
                                    A, lda, stride_A, x, incx, stride_x, batch_count)
end function hipblasZtrmvStridedBatchedFortran

! tpmv
function hipblasStpmvFortran(handle, uplo, transA, diag, m, &
                                AP, x, incx) &
    bind(c, name='hipblasStpmvFortran')
    use iso_c_binding
    use hipblas_enums
    implicit none
    integer(kind(HIPBLAS_STATUS_SUCCESS)) :: hipblasStpmvFortran
    type(c_ptr), value :: handle
    integer(kind(HIPBLAS_FILL_MODE_FULL)), value :: uplo
    integer(kind(HIPBLAS_OP_N)), value :: transA
    integer(kind(HIPBLAS_DIAG_NON_UNIT)), value :: diag
    integer(c_int), value :: m
    type(c_ptr), value :: AP
    type(c_ptr), value :: x
    integer(c_int), value :: incx
            hipblasStpmvFortran = &
        hipblasStpmv(handle, uplo, transA, diag, m, &
                        AP, x, incx)
end function hipblasStpmvFortran

function hipblasDtpmvFortran(handle, uplo, transA, diag, m, &
                                AP, x, incx) &
    bind(c, name='hipblasDtpmvFortran')
    use iso_c_binding
    use hipblas_enums
    implicit none
    integer(kind(HIPBLAS_STATUS_SUCCESS)) :: hipblasDtpmvFortran
    type(c_ptr), value :: handle
    integer(kind(HIPBLAS_FILL_MODE_FULL)), value :: uplo
    integer(kind(HIPBLAS_OP_N)), value :: transA
    integer(kind(HIPBLAS_DIAG_NON_UNIT)), value :: diag
    integer(c_int), value :: m
    type(c_ptr), value :: AP
    type(c_ptr), value :: x
    integer(c_int), value :: incx
            hipblasDtpmvFortran = &
        hipblasDtpmv(handle, uplo, transA, diag, m, &
                        AP, x, incx)
end function hipblasDtpmvFortran

function hipblasCtpmvFortran(handle, uplo, transA, diag, m, &
                                AP, x, incx) &
    bind(c, name='hipblasCtpmvFortran')
    use iso_c_binding
    use hipblas_enums
    implicit none
    integer(kind(HIPBLAS_STATUS_SUCCESS)) :: hipblasCtpmvFortran
    type(c_ptr), value :: handle
    integer(kind(HIPBLAS_FILL_MODE_FULL)), value :: uplo
    integer(kind(HIPBLAS_OP_N)), value :: transA
    integer(kind(HIPBLAS_DIAG_NON_UNIT)), value :: diag
    integer(c_int), value :: m
    type(c_ptr), value :: AP
    type(c_ptr), value :: x
    integer(c_int), value :: incx
            hipblasCtpmvFortran = &
        hipblasCtpmv(handle, uplo, transA, diag, m, &
                        AP, x, incx)
end function hipblasCtpmvFortran

function hipblasZtpmvFortran(handle, uplo, transA, diag, m, &
                                AP, x, incx) &
    bind(c, name='hipblasZtpmvFortran')
    use iso_c_binding
    use hipblas_enums
    implicit none
    integer(kind(HIPBLAS_STATUS_SUCCESS)) :: hipblasZtpmvFortran
    type(c_ptr), value :: handle
    integer(kind(HIPBLAS_FILL_MODE_FULL)), value :: uplo
    integer(kind(HIPBLAS_OP_N)), value :: transA
    integer(kind(HIPBLAS_DIAG_NON_UNIT)), value :: diag
    integer(c_int), value :: m
    type(c_ptr), value :: AP
    type(c_ptr), value :: x
    integer(c_int), value :: incx
            hipblasZtpmvFortran = &
        hipblasZtpmv(handle, uplo, transA, diag, m, &
                        AP, x, incx)
end function hipblasZtpmvFortran

! tpmvBatched
function hipblasStpmvBatchedFortran(handle, uplo, transA, diag, m, &
                                    AP, x, incx, batch_count) &
    bind(c, name='hipblasStpmvBatchedFortran')
    use iso_c_binding
    use hipblas_enums
    implicit none
    integer(kind(HIPBLAS_STATUS_SUCCESS)) :: hipblasStpmvBatchedFortran
    type(c_ptr), value :: handle
    integer(kind(HIPBLAS_FILL_MODE_FULL)), value :: uplo
    integer(kind(HIPBLAS_OP_N)), value :: transA
    integer(kind(HIPBLAS_DIAG_NON_UNIT)), value :: diag
    integer(c_int), value :: m
    type(c_ptr), value :: AP
    type(c_ptr), value :: x
    integer(c_int), value :: incx
    integer(c_int), value :: batch_count
            hipblasStpmvBatchedFortran = &
        hipblasStpmvBatched(handle, uplo, transA, diag, m, &
                            AP, x, incx, batch_count)
end function hipblasStpmvBatchedFortran

function hipblasDtpmvBatchedFortran(handle, uplo, transA, diag, m, &
                                    AP, x, incx, batch_count) &
    bind(c, name='hipblasDtpmvBatchedFortran')
    use iso_c_binding
    use hipblas_enums
    implicit none
    integer(kind(HIPBLAS_STATUS_SUCCESS)) :: hipblasDtpmvBatchedFortran
    type(c_ptr), value :: handle
    integer(kind(HIPBLAS_FILL_MODE_FULL)), value :: uplo
    integer(kind(HIPBLAS_OP_N)), value :: transA
    integer(kind(HIPBLAS_DIAG_NON_UNIT)), value :: diag
    integer(c_int), value :: m
    type(c_ptr), value :: AP
    type(c_ptr), value :: x
    integer(c_int), value :: incx
    integer(c_int), value :: batch_count
            hipblasDtpmvBatchedFortran = &
        hipblasDtpmvBatched(handle, uplo, transA, diag, m, &
                            AP, x, incx, batch_count)
end function hipblasDtpmvBatchedFortran

function hipblasCtpmvBatchedFortran(handle, uplo, transA, diag, m, &
                                    AP, x, incx, batch_count) &
    bind(c, name='hipblasCtpmvBatchedFortran')
    use iso_c_binding
    use hipblas_enums
    implicit none
    integer(kind(HIPBLAS_STATUS_SUCCESS)) :: hipblasCtpmvBatchedFortran
    type(c_ptr), value :: handle
    integer(kind(HIPBLAS_FILL_MODE_FULL)), value :: uplo
    integer(kind(HIPBLAS_OP_N)), value :: transA
    integer(kind(HIPBLAS_DIAG_NON_UNIT)), value :: diag
    integer(c_int), value :: m
    type(c_ptr), value :: AP
    type(c_ptr), value :: x
    integer(c_int), value :: incx
    integer(c_int), value :: batch_count
            hipblasCtpmvBatchedFortran = &
        hipblasCtpmvBatched(handle, uplo, transA, diag, m, &
                            AP, x, incx, batch_count)
end function hipblasCtpmvBatchedFortran

function hipblasZtpmvBatchedFortran(handle, uplo, transA, diag, m, &
                                    AP, x, incx, batch_count) &
    bind(c, name='hipblasZtpmvBatchedFortran')
    use iso_c_binding
    use hipblas_enums
    implicit none
    integer(kind(HIPBLAS_STATUS_SUCCESS)) :: hipblasZtpmvBatchedFortran
    type(c_ptr), value :: handle
    integer(kind(HIPBLAS_FILL_MODE_FULL)), value :: uplo
    integer(kind(HIPBLAS_OP_N)), value :: transA
    integer(kind(HIPBLAS_DIAG_NON_UNIT)), value :: diag
    integer(c_int), value :: m
    type(c_ptr), value :: AP
    type(c_ptr), value :: x
    integer(c_int), value :: incx
    integer(c_int), value :: batch_count
            hipblasZtpmvBatchedFortran = &
        hipblasZtpmvBatched(handle, uplo, transA, diag, m, &
                            AP, x, incx, batch_count)
end function hipblasZtpmvBatchedFortran

! tpmvStridedBatched
function hipblasStpmvStridedBatchedFortran(handle, uplo, transA, diag, m, &
                                            AP, stride_AP, x, incx, stride_x, batch_count) &
    bind(c, name='hipblasStpmvStridedBatchedFortran')
    use iso_c_binding
    use hipblas_enums
    implicit none
    integer(kind(HIPBLAS_STATUS_SUCCESS)) :: hipblasStpmvStridedBatchedFortran
    type(c_ptr), value :: handle
    integer(kind(HIPBLAS_FILL_MODE_FULL)), value :: uplo
    integer(kind(HIPBLAS_OP_N)), value :: transA
    integer(kind(HIPBLAS_DIAG_NON_UNIT)), value :: diag
    integer(c_int), value :: m
    type(c_ptr), value :: AP
    integer(c_int64_t), value :: stride_AP
    type(c_ptr), value :: x
    integer(c_int), value :: incx
    integer(c_int64_t), value :: stride_x
    integer(c_int), value :: batch_count
            hipblasStpmvStridedBatchedFortran = &
        hipblasStpmvStridedBatched(handle, uplo, transA, diag, m, &
                                    AP, stride_AP, x, incx, stride_x, batch_count)
end function hipblasStpmvStridedBatchedFortran

function hipblasDtpmvStridedBatchedFortran(handle, uplo, transA, diag, m, &
                                            AP, stride_AP, x, incx, stride_x, batch_count) &
    bind(c, name='hipblasDtpmvStridedBatchedFortran')
    use iso_c_binding
    use hipblas_enums
    implicit none
    integer(kind(HIPBLAS_STATUS_SUCCESS)) :: hipblasDtpmvStridedBatchedFortran
    type(c_ptr), value :: handle
    integer(kind(HIPBLAS_FILL_MODE_FULL)), value :: uplo
    integer(kind(HIPBLAS_OP_N)), value :: transA
    integer(kind(HIPBLAS_DIAG_NON_UNIT)), value :: diag
    integer(c_int), value :: m
    type(c_ptr), value :: AP
    integer(c_int64_t), value :: stride_AP
    type(c_ptr), value :: x
    integer(c_int), value :: incx
    integer(c_int64_t), value :: stride_x
    integer(c_int), value :: batch_count
            hipblasDtpmvStridedBatchedFortran = &
        hipblasDtpmvStridedBatched(handle, uplo, transA, diag, m, &
                                    AP, stride_AP, x, incx, stride_x, batch_count)
end function hipblasDtpmvStridedBatchedFortran

function hipblasCtpmvStridedBatchedFortran(handle, uplo, transA, diag, m, &
                                            AP, stride_AP, x, incx, stride_x, batch_count) &
    bind(c, name='hipblasCtpmvStridedBatchedFortran')
    use iso_c_binding
    use hipblas_enums
    implicit none
    integer(kind(HIPBLAS_STATUS_SUCCESS)) :: hipblasCtpmvStridedBatchedFortran
    type(c_ptr), value :: handle
    integer(kind(HIPBLAS_FILL_MODE_FULL)), value :: uplo
    integer(kind(HIPBLAS_OP_N)), value :: transA
    integer(kind(HIPBLAS_DIAG_NON_UNIT)), value :: diag
    integer(c_int), value :: m
    type(c_ptr), value :: AP
    integer(c_int64_t), value :: stride_AP
    type(c_ptr), value :: x
    integer(c_int), value :: incx
    integer(c_int64_t), value :: stride_x
    integer(c_int), value :: batch_count
            hipblasCtpmvStridedBatchedFortran = &
        hipblasCtpmvStridedBatched(handle, uplo, transA, diag, m, &
                                    AP, stride_AP, x, incx, stride_x, batch_count)
end function hipblasCtpmvStridedBatchedFortran

function hipblasZtpmvStridedBatchedFortran(handle, uplo, transA, diag, m, &
                                            AP, stride_AP, x, incx, stride_x, batch_count) &
    bind(c, name='hipblasZtpmvStridedBatchedFortran')
    use iso_c_binding
    use hipblas_enums
    implicit none
    integer(kind(HIPBLAS_STATUS_SUCCESS)) :: hipblasZtpmvStridedBatchedFortran
    type(c_ptr), value :: handle
    integer(kind(HIPBLAS_FILL_MODE_FULL)), value :: uplo
    integer(kind(HIPBLAS_OP_N)), value :: transA
    integer(kind(HIPBLAS_DIAG_NON_UNIT)), value :: diag
    integer(c_int), value :: m
    type(c_ptr), value :: AP
    integer(c_int64_t), value :: stride_AP
    type(c_ptr), value :: x
    integer(c_int), value :: incx
    integer(c_int64_t), value :: stride_x
    integer(c_int), value :: batch_count
            hipblasZtpmvStridedBatchedFortran = &
        hipblasZtpmvStridedBatched(handle, uplo, transA, diag, m, &
                                    AP, stride_AP, x, incx, stride_x, batch_count)
end function hipblasZtpmvStridedBatchedFortran

! tbmv
function hipblasStbmvFortran(handle, uplo, transA, diag, m, k, &
                                A, lda, x, incx) &
    bind(c, name='hipblasStbmvFortran')
    use iso_c_binding
    use hipblas_enums
    implicit none
    integer(kind(HIPBLAS_STATUS_SUCCESS)) :: hipblasStbmvFortran
    type(c_ptr), value :: handle
    integer(kind(HIPBLAS_FILL_MODE_FULL)), value :: uplo
    integer(kind(HIPBLAS_OP_N)), value :: transA
    integer(kind(HIPBLAS_DIAG_NON_UNIT)), value :: diag
    integer(c_int), value :: m
    integer(c_int), value :: k
    type(c_ptr), value :: A
    integer(c_int), value :: lda
    type(c_ptr), value :: x
    integer(c_int), value :: incx
            hipblasStbmvFortran = &
        hipblasStbmv(handle, uplo, transA, diag, m, k, &
                        A, lda, x, incx)
end function hipblasStbmvFortran

function hipblasDtbmvFortran(handle, uplo, transA, diag, m, k, &
                                A, lda, x, incx) &
    bind(c, name='hipblasDtbmvFortran')
    use iso_c_binding
    use hipblas_enums
    implicit none
    integer(kind(HIPBLAS_STATUS_SUCCESS)) :: hipblasDtbmvFortran
    type(c_ptr), value :: handle
    integer(kind(HIPBLAS_FILL_MODE_FULL)), value :: uplo
    integer(kind(HIPBLAS_OP_N)), value :: transA
    integer(kind(HIPBLAS_DIAG_NON_UNIT)), value :: diag
    integer(c_int), value :: m
    integer(c_int), value :: k
    type(c_ptr), value :: A
    integer(c_int), value :: lda
    type(c_ptr), value :: x
    integer(c_int), value :: incx
            hipblasDtbmvFortran = &
        hipblasDtbmv(handle, uplo, transA, diag, m, k, &
                        A, lda, x, incx)
end function hipblasDtbmvFortran

function hipblasCtbmvFortran(handle, uplo, transA, diag, m, k, &
                                A, lda, x, incx) &
    bind(c, name='hipblasCtbmvFortran')
    use iso_c_binding
    use hipblas_enums
    implicit none
    integer(kind(HIPBLAS_STATUS_SUCCESS)) :: hipblasCtbmvFortran
    type(c_ptr), value :: handle
    integer(kind(HIPBLAS_FILL_MODE_FULL)), value :: uplo
    integer(kind(HIPBLAS_OP_N)), value :: transA
    integer(kind(HIPBLAS_DIAG_NON_UNIT)), value :: diag
    integer(c_int), value :: m
    integer(c_int), value :: k
    type(c_ptr), value :: A
    integer(c_int), value :: lda
    type(c_ptr), value :: x
    integer(c_int), value :: incx
            hipblasCtbmvFortran = &
        hipblasCtbmv(handle, uplo, transA, diag, m, k, &
                        A, lda, x, incx)
end function hipblasCtbmvFortran

function hipblasZtbmvFortran(handle, uplo, transA, diag, m, k, &
                                A, lda, x, incx) &
    bind(c, name='hipblasZtbmvFortran')
    use iso_c_binding
    use hipblas_enums
    implicit none
    integer(kind(HIPBLAS_STATUS_SUCCESS)) :: hipblasZtbmvFortran
    type(c_ptr), value :: handle
    integer(kind(HIPBLAS_FILL_MODE_FULL)), value :: uplo
    integer(kind(HIPBLAS_OP_N)), value :: transA
    integer(kind(HIPBLAS_DIAG_NON_UNIT)), value :: diag
    integer(c_int), value :: m
    integer(c_int), value :: k
    type(c_ptr), value :: A
    integer(c_int), value :: lda
    type(c_ptr), value :: x
    integer(c_int), value :: incx
            hipblasZtbmvFortran = &
        hipblasZtbmv(handle, uplo, transA, diag, m, k, &
                        A, lda, x, incx)
end function hipblasZtbmvFortran

! tbmvBatched
function hipblasStbmvBatchedFortran(handle, uplo, transA, diag, m, k, &
                                    A, lda, x, incx, batch_count) &
    bind(c, name='hipblasStbmvBatchedFortran')
    use iso_c_binding
    use hipblas_enums
    implicit none
    integer(kind(HIPBLAS_STATUS_SUCCESS)) :: hipblasStbmvBatchedFortran
    type(c_ptr), value :: handle
    integer(kind(HIPBLAS_FILL_MODE_FULL)), value :: uplo
    integer(kind(HIPBLAS_OP_N)), value :: transA
    integer(kind(HIPBLAS_DIAG_NON_UNIT)), value :: diag
    integer(c_int), value :: m
    integer(c_int), value :: k
    type(c_ptr), value :: A
    integer(c_int), value :: lda
    type(c_ptr), value :: x
    integer(c_int), value :: incx
    integer(c_int), value :: batch_count
            hipblasStbmvBatchedFortran = &
        hipblasStbmvBatched(handle, uplo, transA, diag, m, k, &
                            A, lda, x, incx, batch_count)
end function hipblasStbmvBatchedFortran

function hipblasDtbmvBatchedFortran(handle, uplo, transA, diag, m, k, &
                                    A, lda, x, incx, batch_count) &
    bind(c, name='hipblasDtbmvBatchedFortran')
    use iso_c_binding
    use hipblas_enums
    implicit none
    integer(kind(HIPBLAS_STATUS_SUCCESS)) :: hipblasDtbmvBatchedFortran
    type(c_ptr), value :: handle
    integer(kind(HIPBLAS_FILL_MODE_FULL)), value :: uplo
    integer(kind(HIPBLAS_OP_N)), value :: transA
    integer(kind(HIPBLAS_DIAG_NON_UNIT)), value :: diag
    integer(c_int), value :: m
    integer(c_int), value :: k
    type(c_ptr), value :: A
    integer(c_int), value :: lda
    type(c_ptr), value :: x
    integer(c_int), value :: incx
    integer(c_int), value :: batch_count
            hipblasDtbmvBatchedFortran = &
        hipblasDtbmvBatched(handle, uplo, transA, diag, m, k, &
                            A, lda, x, incx, batch_count)
end function hipblasDtbmvBatchedFortran

function hipblasCtbmvBatchedFortran(handle, uplo, transA, diag, m, k, &
                                    A, lda, x, incx, batch_count) &
    bind(c, name='hipblasCtbmvBatchedFortran')
    use iso_c_binding
    use hipblas_enums
    implicit none
    integer(kind(HIPBLAS_STATUS_SUCCESS)) :: hipblasCtbmvBatchedFortran
    type(c_ptr), value :: handle
    integer(kind(HIPBLAS_FILL_MODE_FULL)), value :: uplo
    integer(kind(HIPBLAS_OP_N)), value :: transA
    integer(kind(HIPBLAS_DIAG_NON_UNIT)), value :: diag
    integer(c_int), value :: m
    integer(c_int), value :: k
    type(c_ptr), value :: A
    integer(c_int), value :: lda
    type(c_ptr), value :: x
    integer(c_int), value :: incx
    integer(c_int), value :: batch_count
            hipblasCtbmvBatchedFortran = &
        hipblasCtbmvBatched(handle, uplo, transA, diag, m, k, &
                            A, lda, x, incx, batch_count)
end function hipblasCtbmvBatchedFortran

function hipblasZtbmvBatchedFortran(handle, uplo, transA, diag, m, k, &
                                    A, lda, x, incx, batch_count) &
    bind(c, name='hipblasZtbmvBatchedFortran')
    use iso_c_binding
    use hipblas_enums
    implicit none
    integer(kind(HIPBLAS_STATUS_SUCCESS)) :: hipblasZtbmvBatchedFortran
    type(c_ptr), value :: handle
    integer(kind(HIPBLAS_FILL_MODE_FULL)), value :: uplo
    integer(kind(HIPBLAS_OP_N)), value :: transA
    integer(kind(HIPBLAS_DIAG_NON_UNIT)), value :: diag
    integer(c_int), value :: m
    integer(c_int), value :: k
    type(c_ptr), value :: A
    integer(c_int), value :: lda
    type(c_ptr), value :: x
    integer(c_int), value :: incx
    integer(c_int), value :: batch_count
            hipblasZtbmvBatchedFortran = &
        hipblasZtbmvBatched(handle, uplo, transA, diag, m, k, &
                            A, lda, x, incx, batch_count)
end function hipblasZtbmvBatchedFortran

! tbmvStridedBatched
function hipblasStbmvStridedBatchedFortran(handle, uplo, transA, diag, m, k, &
                                            A, lda, stride_A, x, incx, stride_x, batch_count) &
    bind(c, name='hipblasStbmvStridedBatchedFortran')
    use iso_c_binding
    use hipblas_enums
    implicit none
    integer(kind(HIPBLAS_STATUS_SUCCESS)) :: hipblasStbmvStridedBatchedFortran
    type(c_ptr), value :: handle
    integer(kind(HIPBLAS_FILL_MODE_FULL)), value :: uplo
    integer(kind(HIPBLAS_OP_N)), value :: transA
    integer(kind(HIPBLAS_DIAG_NON_UNIT)), value :: diag
    integer(c_int), value :: m
    integer(c_int), value :: k
    type(c_ptr), value :: A
    integer(c_int), value :: lda
    integer(c_int64_t), value :: stride_A
    type(c_ptr), value :: x
    integer(c_int), value :: incx
    integer(c_int64_t), value :: stride_x
    integer(c_int), value :: batch_count
            hipblasStbmvStridedBatchedFortran = &
        hipblasStbmvStridedBatched(handle, uplo, transA, diag, m, k, &
                                    A, lda, stride_A, x, incx, stride_x, batch_count)
end function hipblasStbmvStridedBatchedFortran

function hipblasDtbmvStridedBatchedFortran(handle, uplo, transA, diag, m, k, &
                                            A, lda, stride_A, x, incx, stride_x, batch_count) &
    bind(c, name='hipblasDtbmvStridedBatchedFortran')
    use iso_c_binding
    use hipblas_enums
    implicit none
    integer(kind(HIPBLAS_STATUS_SUCCESS)) :: hipblasDtbmvStridedBatchedFortran
    type(c_ptr), value :: handle
    integer(kind(HIPBLAS_FILL_MODE_FULL)), value :: uplo
    integer(kind(HIPBLAS_OP_N)), value :: transA
    integer(kind(HIPBLAS_DIAG_NON_UNIT)), value :: diag
    integer(c_int), value :: m
    integer(c_int), value :: k
    type(c_ptr), value :: A
    integer(c_int), value :: lda
    integer(c_int64_t), value :: stride_A
    type(c_ptr), value :: x
    integer(c_int), value :: incx
    integer(c_int64_t), value :: stride_x
    integer(c_int), value :: batch_count
            hipblasDtbmvStridedBatchedFortran = &
        hipblasDtbmvStridedBatched(handle, uplo, transA, diag, m, k, &
                                    A, lda, stride_A, x, incx, stride_x, batch_count)
end function hipblasDtbmvStridedBatchedFortran

function hipblasCtbmvStridedBatchedFortran(handle, uplo, transA, diag, m, k, &
                                            A, lda, stride_A, x, incx, stride_x, batch_count) &
    bind(c, name='hipblasCtbmvStridedBatchedFortran')
    use iso_c_binding
    use hipblas_enums
    implicit none
    integer(kind(HIPBLAS_STATUS_SUCCESS)) :: hipblasCtbmvStridedBatchedFortran
    type(c_ptr), value :: handle
    integer(kind(HIPBLAS_FILL_MODE_FULL)), value :: uplo
    integer(kind(HIPBLAS_OP_N)), value :: transA
    integer(kind(HIPBLAS_DIAG_NON_UNIT)), value :: diag
    integer(c_int), value :: m
    integer(c_int), value :: k
    type(c_ptr), value :: A
    integer(c_int), value :: lda
    integer(c_int64_t), value :: stride_A
    type(c_ptr), value :: x
    integer(c_int), value :: incx
    integer(c_int64_t), value :: stride_x
    integer(c_int), value :: batch_count
            hipblasCtbmvStridedBatchedFortran = &
        hipblasCtbmvStridedBatched(handle, uplo, transA, diag, m, k, &
                                    A, lda, stride_A, x, incx, stride_x, batch_count)
end function hipblasCtbmvStridedBatchedFortran

function hipblasZtbmvStridedBatchedFortran(handle, uplo, transA, diag, m, k, &
                                            A, lda, stride_A, x, incx, stride_x, batch_count) &
    bind(c, name='hipblasZtbmvStridedBatchedFortran')
    use iso_c_binding
    use hipblas_enums
    implicit none
    integer(kind(HIPBLAS_STATUS_SUCCESS)) :: hipblasZtbmvStridedBatchedFortran
    type(c_ptr), value :: handle
    integer(kind(HIPBLAS_FILL_MODE_FULL)), value :: uplo
    integer(kind(HIPBLAS_OP_N)), value :: transA
    integer(kind(HIPBLAS_DIAG_NON_UNIT)), value :: diag
    integer(c_int), value :: m
    integer(c_int), value :: k
    type(c_ptr), value :: A
    integer(c_int), value :: lda
    integer(c_int64_t), value :: stride_A
    type(c_ptr), value :: x
    integer(c_int), value :: incx
    integer(c_int64_t), value :: stride_x
    integer(c_int), value :: batch_count
            hipblasZtbmvStridedBatchedFortran = &
        hipblasZtbmvStridedBatched(handle, uplo, transA, diag, m, k, &
                                    A, lda, stride_A, x, incx, stride_x, batch_count)
end function hipblasZtbmvStridedBatchedFortran

! tbsv
function hipblasStbsvFortran(handle, uplo, transA, diag, n, k, &
                                A, lda, x, incx) &
    bind(c, name='hipblasStbsvFortran')
    use iso_c_binding
    use hipblas_enums
    implicit none
    integer(kind(HIPBLAS_STATUS_SUCCESS)) :: hipblasStbsvFortran
    type(c_ptr), value :: handle
    integer(kind(HIPBLAS_FILL_MODE_FULL)), value :: uplo
    integer(kind(HIPBLAS_OP_N)), value :: transA
    integer(kind(HIPBLAS_DIAG_NON_UNIT)), value :: diag
    integer(c_int), value :: n
    integer(c_int), value :: k
    type(c_ptr), value :: A
    integer(c_int), value :: lda
    type(c_ptr), value :: x
    integer(c_int), value :: incx
            hipblasStbsvFortran = &
        hipblasStbsv(handle, uplo, transA, diag, n, k, &
                        A, lda, x, incx)
end function hipblasStbsvFortran

function hipblasDtbsvFortran(handle, uplo, transA, diag, n, k, &
                                A, lda, x, incx) &
    bind(c, name='hipblasDtbsvFortran')
    use iso_c_binding
    use hipblas_enums
    implicit none
    integer(kind(HIPBLAS_STATUS_SUCCESS)) :: hipblasDtbsvFortran
    type(c_ptr), value :: handle
    integer(kind(HIPBLAS_FILL_MODE_FULL)), value :: uplo
    integer(kind(HIPBLAS_OP_N)), value :: transA
    integer(kind(HIPBLAS_DIAG_NON_UNIT)), value :: diag
    integer(c_int), value :: n
    integer(c_int), value :: k
    type(c_ptr), value :: A
    integer(c_int), value :: lda
    type(c_ptr), value :: x
    integer(c_int), value :: incx
            hipblasDtbsvFortran = &
        hipblasDtbsv(handle, uplo, transA, diag, n, k, &
                        A, lda, x, incx)
end function hipblasDtbsvFortran

function hipblasCtbsvFortran(handle, uplo, transA, diag, n, k, &
                                A, lda, x, incx) &
    bind(c, name='hipblasCtbsvFortran')
    use iso_c_binding
    use hipblas_enums
    implicit none
    integer(kind(HIPBLAS_STATUS_SUCCESS)) :: hipblasCtbsvFortran
    type(c_ptr), value :: handle
    integer(kind(HIPBLAS_FILL_MODE_FULL)), value :: uplo
    integer(kind(HIPBLAS_OP_N)), value :: transA
    integer(kind(HIPBLAS_DIAG_NON_UNIT)), value :: diag
    integer(c_int), value :: n
    integer(c_int), value :: k
    type(c_ptr), value :: A
    integer(c_int), value :: lda
    type(c_ptr), value :: x
    integer(c_int), value :: incx
            hipblasCtbsvFortran = &
        hipblasCtbsv(handle, uplo, transA, diag, n, k, &
                        A, lda, x, incx)
end function hipblasCtbsvFortran

function hipblasZtbsvFortran(handle, uplo, transA, diag, n, k, &
                                A, lda, x, incx) &
    bind(c, name='hipblasZtbsvFortran')
    use iso_c_binding
    use hipblas_enums
    implicit none
    integer(kind(HIPBLAS_STATUS_SUCCESS)) :: hipblasZtbsvFortran
    type(c_ptr), value :: handle
    integer(kind(HIPBLAS_FILL_MODE_FULL)), value :: uplo
    integer(kind(HIPBLAS_OP_N)), value :: transA
    integer(kind(HIPBLAS_DIAG_NON_UNIT)), value :: diag
    integer(c_int), value :: n
    integer(c_int), value :: k
    type(c_ptr), value :: A
    integer(c_int), value :: lda
    type(c_ptr), value :: x
    integer(c_int), value :: incx
            hipblasZtbsvFortran = &
        hipblasZtbsv(handle, uplo, transA, diag, n, k, &
                        A, lda, x, incx)
end function hipblasZtbsvFortran

! tbsvBatched
function hipblasStbsvBatchedFortran(handle, uplo, transA, diag, n, k, &
                                    A, lda, x, incx, batch_count) &
    bind(c, name='hipblasStbsvBatchedFortran')
    use iso_c_binding
    use hipblas_enums
    implicit none
    integer(kind(HIPBLAS_STATUS_SUCCESS)) :: hipblasStbsvBatchedFortran
    type(c_ptr), value :: handle
    integer(kind(HIPBLAS_FILL_MODE_FULL)), value :: uplo
    integer(kind(HIPBLAS_OP_N)), value :: transA
    integer(kind(HIPBLAS_DIAG_NON_UNIT)), value :: diag
    integer(c_int), value :: n
    integer(c_int), value :: k
    type(c_ptr), value :: A
    integer(c_int), value :: lda
    type(c_ptr), value :: x
    integer(c_int), value :: incx
    integer(c_int), value :: batch_count
            hipblasStbsvBatchedFortran = &
        hipblasStbsvBatched(handle, uplo, transA, diag, n, k, &
                            A, lda, x, incx, batch_count)
end function hipblasStbsvBatchedFortran

function hipblasDtbsvBatchedFortran(handle, uplo, transA, diag, n, k, &
                                    A, lda, x, incx, batch_count) &
    bind(c, name='hipblasDtbsvBatchedFortran')
    use iso_c_binding
    use hipblas_enums
    implicit none
    integer(kind(HIPBLAS_STATUS_SUCCESS)) :: hipblasDtbsvBatchedFortran
    type(c_ptr), value :: handle
    integer(kind(HIPBLAS_FILL_MODE_FULL)), value :: uplo
    integer(kind(HIPBLAS_OP_N)), value :: transA
    integer(kind(HIPBLAS_DIAG_NON_UNIT)), value :: diag
    integer(c_int), value :: n
    integer(c_int), value :: k
    type(c_ptr), value :: A
    integer(c_int), value :: lda
    type(c_ptr), value :: x
    integer(c_int), value :: incx
    integer(c_int), value :: batch_count
            hipblasDtbsvBatchedFortran = &
        hipblasDtbsvBatched(handle, uplo, transA, diag, n, k, &
                            A, lda, x, incx, batch_count)
end function hipblasDtbsvBatchedFortran

function hipblasCtbsvBatchedFortran(handle, uplo, transA, diag, n, k, &
                                    A, lda, x, incx, batch_count) &
    bind(c, name='hipblasCtbsvBatchedFortran')
    use iso_c_binding
    use hipblas_enums
    implicit none
    integer(kind(HIPBLAS_STATUS_SUCCESS)) :: hipblasCtbsvBatchedFortran
    type(c_ptr), value :: handle
    integer(kind(HIPBLAS_FILL_MODE_FULL)), value :: uplo
    integer(kind(HIPBLAS_OP_N)), value :: transA
    integer(kind(HIPBLAS_DIAG_NON_UNIT)), value :: diag
    integer(c_int), value :: n
    integer(c_int), value :: k
    type(c_ptr), value :: A
    integer(c_int), value :: lda
    type(c_ptr), value :: x
    integer(c_int), value :: incx
    integer(c_int), value :: batch_count
            hipblasCtbsvBatchedFortran = &
        hipblasCtbsvBatched(handle, uplo, transA, diag, n, k, &
                            A, lda, x, incx, batch_count)
end function hipblasCtbsvBatchedFortran

function hipblasZtbsvBatchedFortran(handle, uplo, transA, diag, n, k, &
                                    A, lda, x, incx, batch_count) &
    bind(c, name='hipblasZtbsvBatchedFortran')
    use iso_c_binding
    use hipblas_enums
    implicit none
    integer(kind(HIPBLAS_STATUS_SUCCESS)) :: hipblasZtbsvBatchedFortran
    type(c_ptr), value :: handle
    integer(kind(HIPBLAS_FILL_MODE_FULL)), value :: uplo
    integer(kind(HIPBLAS_OP_N)), value :: transA
    integer(kind(HIPBLAS_DIAG_NON_UNIT)), value :: diag
    integer(c_int), value :: n
    integer(c_int), value :: k
    type(c_ptr), value :: A
    integer(c_int), value :: lda
    type(c_ptr), value :: x
    integer(c_int), value :: incx
    integer(c_int), value :: batch_count
            hipblasZtbsvBatchedFortran = &
        hipblasZtbsvBatched(handle, uplo, transA, diag, n, k, &
                            A, lda, x, incx, batch_count)
end function hipblasZtbsvBatchedFortran

! tbsvStridedBatched
function hipblasStbsvStridedBatchedFortran(handle, uplo, transA, diag, n, k, &
                                            A, lda, stride_A, x, incx, stride_x, batch_count) &
    bind(c, name='hipblasStbsvStridedBatchedFortran')
    use iso_c_binding
    use hipblas_enums
    implicit none
    integer(kind(HIPBLAS_STATUS_SUCCESS)) :: hipblasStbsvStridedBatchedFortran
    type(c_ptr), value :: handle
    integer(kind(HIPBLAS_FILL_MODE_FULL)), value :: uplo
    integer(kind(HIPBLAS_OP_N)), value :: transA
    integer(kind(HIPBLAS_DIAG_NON_UNIT)), value :: diag
    integer(c_int), value :: n
    integer(c_int), value :: k
    type(c_ptr), value :: A
    integer(c_int), value :: lda
    integer(c_int64_t), value :: stride_A
    type(c_ptr), value :: x
    integer(c_int), value :: incx
    integer(c_int64_t), value :: stride_x
    integer(c_int), value :: batch_count
            hipblasStbsvStridedBatchedFortran = &
        hipblasStbsvStridedBatched(handle, uplo, transA, diag, n, k, &
                                    A, lda, stride_A, x, incx, stride_x, batch_count)
end function hipblasStbsvStridedBatchedFortran

function hipblasDtbsvStridedBatchedFortran(handle, uplo, transA, diag, n, k, &
                                            A, lda, stride_A, x, incx, stride_x, batch_count) &
    bind(c, name='hipblasDtbsvStridedBatchedFortran')
    use iso_c_binding
    use hipblas_enums
    implicit none
    integer(kind(HIPBLAS_STATUS_SUCCESS)) :: hipblasDtbsvStridedBatchedFortran
    type(c_ptr), value :: handle
    integer(kind(HIPBLAS_FILL_MODE_FULL)), value :: uplo
    integer(kind(HIPBLAS_OP_N)), value :: transA
    integer(kind(HIPBLAS_DIAG_NON_UNIT)), value :: diag
    integer(c_int), value :: n
    integer(c_int), value :: k
    type(c_ptr), value :: A
    integer(c_int), value :: lda
    integer(c_int64_t), value :: stride_A
    type(c_ptr), value :: x
    integer(c_int), value :: incx
    integer(c_int64_t), value :: stride_x
    integer(c_int), value :: batch_count
            hipblasDtbsvStridedBatchedFortran = &
        hipblasDtbsvStridedBatched(handle, uplo, transA, diag, n, k, &
                                    A, lda, stride_A, x, incx, stride_x, batch_count)
end function hipblasDtbsvStridedBatchedFortran

function hipblasCtbsvStridedBatchedFortran(handle, uplo, transA, diag, n, k, &
                                            A, lda, stride_A, x, incx, stride_x, batch_count) &
    bind(c, name='hipblasCtbsvStridedBatchedFortran')
    use iso_c_binding
    use hipblas_enums
    implicit none
    integer(kind(HIPBLAS_STATUS_SUCCESS)) :: hipblasCtbsvStridedBatchedFortran
    type(c_ptr), value :: handle
    integer(kind(HIPBLAS_FILL_MODE_FULL)), value :: uplo
    integer(kind(HIPBLAS_OP_N)), value :: transA
    integer(kind(HIPBLAS_DIAG_NON_UNIT)), value :: diag
    integer(c_int), value :: n
    integer(c_int), value :: k
    type(c_ptr), value :: A
    integer(c_int), value :: lda
    integer(c_int64_t), value :: stride_A
    type(c_ptr), value :: x
    integer(c_int), value :: incx
    integer(c_int64_t), value :: stride_x
    integer(c_int), value :: batch_count
            hipblasCtbsvStridedBatchedFortran = &
        hipblasCtbsvStridedBatched(handle, uplo, transA, diag, n, k, &
                                    A, lda, stride_A, x, incx, stride_x, batch_count)
end function hipblasCtbsvStridedBatchedFortran

function hipblasZtbsvStridedBatchedFortran(handle, uplo, transA, diag, n, k, &
                                            A, lda, stride_A, x, incx, stride_x, batch_count) &
    bind(c, name='hipblasZtbsvStridedBatchedFortran')
    use iso_c_binding
    use hipblas_enums
    implicit none
    integer(kind(HIPBLAS_STATUS_SUCCESS)) :: hipblasZtbsvStridedBatchedFortran
    type(c_ptr), value :: handle
    integer(kind(HIPBLAS_FILL_MODE_FULL)), value :: uplo
    integer(kind(HIPBLAS_OP_N)), value :: transA
    integer(kind(HIPBLAS_DIAG_NON_UNIT)), value :: diag
    integer(c_int), value :: n
    integer(c_int), value :: k
    type(c_ptr), value :: A
    integer(c_int), value :: lda
    integer(c_int64_t), value :: stride_A
    type(c_ptr), value :: x
    integer(c_int), value :: incx
    integer(c_int64_t), value :: stride_x
    integer(c_int), value :: batch_count
            hipblasZtbsvStridedBatchedFortran = &
        hipblasZtbsvStridedBatched(handle, uplo, transA, diag, n, k, &
                                    A, lda, stride_A, x, incx, stride_x, batch_count)
end function hipblasZtbsvStridedBatchedFortran

! tpsv
function hipblasStpsvFortran(handle, uplo, transA, diag, n, &
                                AP, x, incx) &
    bind(c, name='hipblasStpsvFortran')
    use iso_c_binding
    use hipblas_enums
    implicit none
    integer(kind(HIPBLAS_STATUS_SUCCESS)) :: hipblasStpsvFortran
    type(c_ptr), value :: handle
    integer(kind(HIPBLAS_FILL_MODE_FULL)), value :: uplo
    integer(kind(HIPBLAS_OP_N)), value :: transA
    integer(kind(HIPBLAS_DIAG_NON_UNIT)), value :: diag
    integer(c_int), value :: n
    type(c_ptr), value :: AP
    type(c_ptr), value :: x
    integer(c_int), value :: incx
            hipblasStpsvFortran = &
        hipblasStpsv(handle, uplo, transA, diag, n, &
                        AP, x, incx)
end function hipblasStpsvFortran

function hipblasDtpsvFortran(handle, uplo, transA, diag, n, &
                                AP, x, incx) &
    bind(c, name='hipblasDtpsvFortran')
    use iso_c_binding
    use hipblas_enums
    implicit none
    integer(kind(HIPBLAS_STATUS_SUCCESS)) :: hipblasDtpsvFortran
    type(c_ptr), value :: handle
    integer(kind(HIPBLAS_FILL_MODE_FULL)), value :: uplo
    integer(kind(HIPBLAS_OP_N)), value :: transA
    integer(kind(HIPBLAS_DIAG_NON_UNIT)), value :: diag
    integer(c_int), value :: n
    type(c_ptr), value :: AP
    type(c_ptr), value :: x
    integer(c_int), value :: incx
            hipblasDtpsvFortran = &
        hipblasDtpsv(handle, uplo, transA, diag, n, &
                        AP, x, incx)
end function hipblasDtpsvFortran

function hipblasCtpsvFortran(handle, uplo, transA, diag, n, &
                                AP, x, incx) &
    bind(c, name='hipblasCtpsvFortran')
    use iso_c_binding
    use hipblas_enums
    implicit none
    integer(kind(HIPBLAS_STATUS_SUCCESS)) :: hipblasCtpsvFortran
    type(c_ptr), value :: handle
    integer(kind(HIPBLAS_FILL_MODE_FULL)), value :: uplo
    integer(kind(HIPBLAS_OP_N)), value :: transA
    integer(kind(HIPBLAS_DIAG_NON_UNIT)), value :: diag
    integer(c_int), value :: n
    type(c_ptr), value :: AP
    type(c_ptr), value :: x
    integer(c_int), value :: incx
            hipblasCtpsvFortran = &
        hipblasCtpsv(handle, uplo, transA, diag, n, &
                        AP, x, incx)
end function hipblasCtpsvFortran

function hipblasZtpsvFortran(handle, uplo, transA, diag, n, &
                                AP, x, incx) &
    bind(c, name='hipblasZtpsvFortran')
    use iso_c_binding
    use hipblas_enums
    implicit none
    integer(kind(HIPBLAS_STATUS_SUCCESS)) :: hipblasZtpsvFortran
    type(c_ptr), value :: handle
    integer(kind(HIPBLAS_FILL_MODE_FULL)), value :: uplo
    integer(kind(HIPBLAS_OP_N)), value :: transA
    integer(kind(HIPBLAS_DIAG_NON_UNIT)), value :: diag
    integer(c_int), value :: n
    type(c_ptr), value :: AP
    type(c_ptr), value :: x
    integer(c_int), value :: incx
            hipblasZtpsvFortran = &
        hipblasZtpsv(handle, uplo, transA, diag, n, &
                        AP, x, incx)
end function hipblasZtpsvFortran

! tpsvBatched
function hipblasStpsvBatchedFortran(handle, uplo, transA, diag, n, &
                                    AP, x, incx, batch_count) &
    bind(c, name='hipblasStpsvBatchedFortran')
    use iso_c_binding
    use hipblas_enums
    implicit none
    integer(kind(HIPBLAS_STATUS_SUCCESS)) :: hipblasStpsvBatchedFortran
    type(c_ptr), value :: handle
    integer(kind(HIPBLAS_FILL_MODE_FULL)), value :: uplo
    integer(kind(HIPBLAS_OP_N)), value :: transA
    integer(kind(HIPBLAS_DIAG_NON_UNIT)), value :: diag
    integer(c_int), value :: n
    type(c_ptr), value :: AP
    type(c_ptr), value :: x
    integer(c_int), value :: incx
    integer(c_int), value :: batch_count
            hipblasStpsvBatchedFortran = &
        hipblasStpsvBatched(handle, uplo, transA, diag, n, &
                            AP, x, incx, batch_count)
end function hipblasStpsvBatchedFortran

function hipblasDtpsvBatchedFortran(handle, uplo, transA, diag, n, &
                                    AP, x, incx, batch_count) &
    bind(c, name='hipblasDtpsvBatchedFortran')
    use iso_c_binding
    use hipblas_enums
    implicit none
    integer(kind(HIPBLAS_STATUS_SUCCESS)) :: hipblasDtpsvBatchedFortran
    type(c_ptr), value :: handle
    integer(kind(HIPBLAS_FILL_MODE_FULL)), value :: uplo
    integer(kind(HIPBLAS_OP_N)), value :: transA
    integer(kind(HIPBLAS_DIAG_NON_UNIT)), value :: diag
    integer(c_int), value :: n
    type(c_ptr), value :: AP
    type(c_ptr), value :: x
    integer(c_int), value :: incx
    integer(c_int), value :: batch_count
            hipblasDtpsvBatchedFortran = &
        hipblasDtpsvBatched(handle, uplo, transA, diag, n, &
                            AP, x, incx, batch_count)
end function hipblasDtpsvBatchedFortran

function hipblasCtpsvBatchedFortran(handle, uplo, transA, diag, n, &
                                    AP, x, incx, batch_count) &
    bind(c, name='hipblasCtpsvBatchedFortran')
    use iso_c_binding
    use hipblas_enums
    implicit none
    integer(kind(HIPBLAS_STATUS_SUCCESS)) :: hipblasCtpsvBatchedFortran
    type(c_ptr), value :: handle
    integer(kind(HIPBLAS_FILL_MODE_FULL)), value :: uplo
    integer(kind(HIPBLAS_OP_N)), value :: transA
    integer(kind(HIPBLAS_DIAG_NON_UNIT)), value :: diag
    integer(c_int), value :: n
    type(c_ptr), value :: AP
    type(c_ptr), value :: x
    integer(c_int), value :: incx
    integer(c_int), value :: batch_count
            hipblasCtpsvBatchedFortran = &
        hipblasCtpsvBatched(handle, uplo, transA, diag, n, &
                            AP, x, incx, batch_count)
end function hipblasCtpsvBatchedFortran

function hipblasZtpsvBatchedFortran(handle, uplo, transA, diag, n, &
                                    AP, x, incx, batch_count) &
    bind(c, name='hipblasZtpsvBatchedFortran')
    use iso_c_binding
    use hipblas_enums
    implicit none
    integer(kind(HIPBLAS_STATUS_SUCCESS)) :: hipblasZtpsvBatchedFortran
    type(c_ptr), value :: handle
    integer(kind(HIPBLAS_FILL_MODE_FULL)), value :: uplo
    integer(kind(HIPBLAS_OP_N)), value :: transA
    integer(kind(HIPBLAS_DIAG_NON_UNIT)), value :: diag
    integer(c_int), value :: n
    type(c_ptr), value :: AP
    type(c_ptr), value :: x
    integer(c_int), value :: incx
    integer(c_int), value :: batch_count
            hipblasZtpsvBatchedFortran = &
        hipblasZtpsvBatched(handle, uplo, transA, diag, n, &
                            AP, x, incx, batch_count)
end function hipblasZtpsvBatchedFortran

! tpsvStridedBatched
function hipblasStpsvStridedBatchedFortran(handle, uplo, transA, diag, n, &
                                            AP, stride_AP, x, incx, stride_x, batch_count) &
    bind(c, name='hipblasStpsvStridedBatchedFortran')
    use iso_c_binding
    use hipblas_enums
    implicit none
    integer(kind(HIPBLAS_STATUS_SUCCESS)) :: hipblasStpsvStridedBatchedFortran
    type(c_ptr), value :: handle
    integer(kind(HIPBLAS_FILL_MODE_FULL)), value :: uplo
    integer(kind(HIPBLAS_OP_N)), value :: transA
    integer(kind(HIPBLAS_DIAG_NON_UNIT)), value :: diag
    integer(c_int), value :: n
    type(c_ptr), value :: AP
    integer(c_int64_t), value :: stride_AP
    type(c_ptr), value :: x
    integer(c_int), value :: incx
    integer(c_int64_t), value :: stride_x
    integer(c_int), value :: batch_count
            hipblasStpsvStridedBatchedFortran = &
        hipblasStpsvStridedBatched(handle, uplo, transA, diag, n, &
                                    AP, stride_AP, x, incx, stride_x, batch_count)
end function hipblasStpsvStridedBatchedFortran

function hipblasDtpsvStridedBatchedFortran(handle, uplo, transA, diag, n, &
                                            AP, stride_AP, x, incx, stride_x, batch_count) &
    bind(c, name='hipblasDtpsvStridedBatchedFortran')
    use iso_c_binding
    use hipblas_enums
    implicit none
    integer(kind(HIPBLAS_STATUS_SUCCESS)) :: hipblasDtpsvStridedBatchedFortran
    type(c_ptr), value :: handle
    integer(kind(HIPBLAS_FILL_MODE_FULL)), value :: uplo
    integer(kind(HIPBLAS_OP_N)), value :: transA
    integer(kind(HIPBLAS_DIAG_NON_UNIT)), value :: diag
    integer(c_int), value :: n
    type(c_ptr), value :: AP
    integer(c_int64_t), value :: stride_AP
    type(c_ptr), value :: x
    integer(c_int), value :: incx
    integer(c_int64_t), value :: stride_x
    integer(c_int), value :: batch_count
            hipblasDtpsvStridedBatchedFortran = &
        hipblasDtpsvStridedBatched(handle, uplo, transA, diag, n, &
                                    AP, stride_AP, x, incx, stride_x, batch_count)
end function hipblasDtpsvStridedBatchedFortran

function hipblasCtpsvStridedBatchedFortran(handle, uplo, transA, diag, n, &
                                            AP, stride_AP, x, incx, stride_x, batch_count) &
    bind(c, name='hipblasCtpsvStridedBatchedFortran')
    use iso_c_binding
    use hipblas_enums
    implicit none
    integer(kind(HIPBLAS_STATUS_SUCCESS)) :: hipblasCtpsvStridedBatchedFortran
    type(c_ptr), value :: handle
    integer(kind(HIPBLAS_FILL_MODE_FULL)), value :: uplo
    integer(kind(HIPBLAS_OP_N)), value :: transA
    integer(kind(HIPBLAS_DIAG_NON_UNIT)), value :: diag
    integer(c_int), value :: n
    type(c_ptr), value :: AP
    integer(c_int64_t), value :: stride_AP
    type(c_ptr), value :: x
    integer(c_int), value :: incx
    integer(c_int64_t), value :: stride_x
    integer(c_int), value :: batch_count
            hipblasCtpsvStridedBatchedFortran = &
        hipblasCtpsvStridedBatched(handle, uplo, transA, diag, n, &
                                    AP, stride_AP, x, incx, stride_x, batch_count)
end function hipblasCtpsvStridedBatchedFortran

function hipblasZtpsvStridedBatchedFortran(handle, uplo, transA, diag, n, &
                                            AP, stride_AP, x, incx, stride_x, batch_count) &
    bind(c, name='hipblasZtpsvStridedBatchedFortran')
    use iso_c_binding
    use hipblas_enums
    implicit none
    integer(kind(HIPBLAS_STATUS_SUCCESS)) :: hipblasZtpsvStridedBatchedFortran
    type(c_ptr), value :: handle
    integer(kind(HIPBLAS_FILL_MODE_FULL)), value :: uplo
    integer(kind(HIPBLAS_OP_N)), value :: transA
    integer(kind(HIPBLAS_DIAG_NON_UNIT)), value :: diag
    integer(c_int), value :: n
    type(c_ptr), value :: AP
    integer(c_int64_t), value :: stride_AP
    type(c_ptr), value :: x
    integer(c_int), value :: incx
    integer(c_int64_t), value :: stride_x
    integer(c_int), value :: batch_count
            hipblasZtpsvStridedBatchedFortran = &
        hipblasZtpsvStridedBatched(handle, uplo, transA, diag, n, &
                                    AP, stride_AP, x, incx, stride_x, batch_count)
end function hipblasZtpsvStridedBatchedFortran

! symv
function hipblasSsymvFortran(handle, uplo, n, alpha, A, lda, &
                                x, incx, beta, y, incy) &
    bind(c, name='hipblasSsymvFortran')
    use iso_c_binding
    use hipblas_enums
    implicit none
    integer(kind(HIPBLAS_STATUS_SUCCESS)) :: hipblasSsymvFortran
    type(c_ptr), value :: handle
    integer(kind(HIPBLAS_FILL_MODE_FULL)), value :: uplo
    integer(c_int), value :: n
    type(c_ptr), value :: alpha
    type(c_ptr), value :: A
    integer(c_int), value :: lda
    type(c_ptr), value :: x
    integer(c_int), value :: incx
    type(c_ptr), value :: beta
    type(c_ptr), value :: y
    integer(c_int), value :: incy
            hipblasSsymvFortran = &
        hipblasSsymv(handle, uplo, n, alpha, &
                        A, lda, x, incx, beta, y, incy)
end function hipblasSsymvFortran

function hipblasDsymvFortran(handle, uplo, n, alpha, A, lda, &
                                x, incx, beta, y, incy) &
    bind(c, name='hipblasDsymvFortran')
    use iso_c_binding
    use hipblas_enums
    implicit none
    integer(kind(HIPBLAS_STATUS_SUCCESS)) :: hipblasDsymvFortran
    type(c_ptr), value :: handle
    integer(kind(HIPBLAS_FILL_MODE_FULL)), value :: uplo
    integer(c_int), value :: n
    type(c_ptr), value :: alpha
    type(c_ptr), value :: A
    integer(c_int), value :: lda
    type(c_ptr), value :: x
    integer(c_int), value :: incx
    type(c_ptr), value :: beta
    type(c_ptr), value :: y
    integer(c_int), value :: incy
            hipblasDsymvFortran = &
        hipblasDsymv(handle, uplo, n, alpha, &
                        A, lda, x, incx, beta, y, incy)
end function hipblasDsymvFortran

function hipblasCsymvFortran(handle, uplo, n, alpha, A, lda, &
                                x, incx, beta, y, incy) &
    bind(c, name='hipblasCsymvFortran')
    use iso_c_binding
    use hipblas_enums
    implicit none
    integer(kind(HIPBLAS_STATUS_SUCCESS)) :: hipblasCsymvFortran
    type(c_ptr), value :: handle
    integer(kind(HIPBLAS_FILL_MODE_FULL)), value :: uplo
    integer(c_int), value :: n
    type(c_ptr), value :: alpha
    type(c_ptr), value :: A
    integer(c_int), value :: lda
    type(c_ptr), value :: x
    integer(c_int), value :: incx
    type(c_ptr), value :: beta
    type(c_ptr), value :: y
    integer(c_int), value :: incy
            hipblasCsymvFortran = &
        hipblasCsymv(handle, uplo, n, alpha, &
                        A, lda, x, incx, beta, y, incy)
end function hipblasCsymvFortran

function hipblasZsymvFortran(handle, uplo, n, alpha, A, lda, &
                                x, incx, beta, y, incy) &
    bind(c, name='hipblasZsymvFortran')
    use iso_c_binding
    use hipblas_enums
    implicit none
    integer(kind(HIPBLAS_STATUS_SUCCESS)) :: hipblasZsymvFortran
    type(c_ptr), value :: handle
    integer(kind(HIPBLAS_FILL_MODE_FULL)), value :: uplo
    integer(c_int), value :: n
    type(c_ptr), value :: alpha
    type(c_ptr), value :: A
    integer(c_int), value :: lda
    type(c_ptr), value :: x
    integer(c_int), value :: incx
    type(c_ptr), value :: beta
    type(c_ptr), value :: y
    integer(c_int), value :: incy
            hipblasZsymvFortran = &
        hipblasZsymv(handle, uplo, n, alpha, &
                        A, lda, x, incx, beta, y, incy)
end function hipblasZsymvFortran

! symvBatched
function hipblasSsymvBatchedFortran(handle, uplo, n, alpha, A, lda, &
                                    x, incx, beta, y, incy, batch_count) &
    bind(c, name='hipblasSsymvBatchedFortran')
    use iso_c_binding
    use hipblas_enums
    implicit none
    integer(kind(HIPBLAS_STATUS_SUCCESS)) :: hipblasSsymvBatchedFortran
    type(c_ptr), value :: handle
    integer(kind(HIPBLAS_FILL_MODE_FULL)), value :: uplo
    integer(c_int), value :: n
    type(c_ptr), value :: alpha
    type(c_ptr), value :: A
    integer(c_int), value :: lda
    type(c_ptr), value :: x
    integer(c_int), value :: incx
    type(c_ptr), value :: beta
    type(c_ptr), value :: y
    integer(c_int), value :: incy
    integer(c_int), value :: batch_count
            hipblasSsymvBatchedFortran = &
        hipblasSsymvBatched(handle, uplo, n, alpha, &
                            A, lda, x, incx, beta, y, incy, batch_count)
end function hipblasSsymvBatchedFortran

function hipblasDsymvBatchedFortran(handle, uplo, n, alpha, A, lda, &
                                    x, incx, beta, y, incy, batch_count) &
    bind(c, name='hipblasDsymvBatchedFortran')
    use iso_c_binding
    use hipblas_enums
    implicit none
    integer(kind(HIPBLAS_STATUS_SUCCESS)) :: hipblasDsymvBatchedFortran
    type(c_ptr), value :: handle
    integer(kind(HIPBLAS_FILL_MODE_FULL)), value :: uplo
    integer(c_int), value :: n
    type(c_ptr), value :: alpha
    type(c_ptr), value :: A
    integer(c_int), value :: lda
    type(c_ptr), value :: x
    integer(c_int), value :: incx
    type(c_ptr), value :: beta
    type(c_ptr), value :: y
    integer(c_int), value :: incy
    integer(c_int), value :: batch_count
            hipblasDsymvBatchedFortran = &
        hipblasDsymvBatched(handle, uplo, n, alpha, &
                            A, lda, x, incx, beta, y, incy, batch_count)
end function hipblasDsymvBatchedFortran

function hipblasCsymvBatchedFortran(handle, uplo, n, alpha, A, lda, &
                                    x, incx, beta, y, incy, batch_count) &
    bind(c, name='hipblasCsymvBatchedFortran')
    use iso_c_binding
    use hipblas_enums
    implicit none
    integer(kind(HIPBLAS_STATUS_SUCCESS)) :: hipblasCsymvBatchedFortran
    type(c_ptr), value :: handle
    integer(kind(HIPBLAS_FILL_MODE_FULL)), value :: uplo
    integer(c_int), value :: n
    type(c_ptr), value :: alpha
    type(c_ptr), value :: A
    integer(c_int), value :: lda
    type(c_ptr), value :: x
    integer(c_int), value :: incx
    type(c_ptr), value :: beta
    type(c_ptr), value :: y
    integer(c_int), value :: incy
    integer(c_int), value :: batch_count
            hipblasCsymvBatchedFortran = &
        hipblasCsymvBatched(handle, uplo, n, alpha, &
                            A, lda, x, incx, beta, y, incy, batch_count)
end function hipblasCsymvBatchedFortran

function hipblasZsymvBatchedFortran(handle, uplo, n, alpha, A, lda, &
                                    x, incx, beta, y, incy, batch_count) &
    bind(c, name='hipblasZsymvBatchedFortran')
    use iso_c_binding
    use hipblas_enums
    implicit none
    integer(kind(HIPBLAS_STATUS_SUCCESS)) :: hipblasZsymvBatchedFortran
    type(c_ptr), value :: handle
    integer(kind(HIPBLAS_FILL_MODE_FULL)), value :: uplo
    integer(c_int), value :: n
    type(c_ptr), value :: alpha
    type(c_ptr), value :: A
    integer(c_int), value :: lda
    type(c_ptr), value :: x
    integer(c_int), value :: incx
    type(c_ptr), value :: beta
    type(c_ptr), value :: y
    integer(c_int), value :: incy
    integer(c_int), value :: batch_count
            hipblasZsymvBatchedFortran = &
        hipblasZsymvBatched(handle, uplo, n, alpha, &
                            A, lda, x, incx, beta, y, incy, batch_count)
end function hipblasZsymvBatchedFortran

! symvStridedBatched
function hipblasSsymvStridedBatchedFortran(handle, uplo, n, alpha, A, lda, stride_A, &
                                            x, incx, stride_x, beta, y, incy, stride_y, batch_count) &
    bind(c, name='hipblasSsymvStridedBatchedFortran')
    use iso_c_binding
    use hipblas_enums
    implicit none
    integer(kind(HIPBLAS_STATUS_SUCCESS)) :: hipblasSsymvStridedBatchedFortran
    type(c_ptr), value :: handle
    integer(kind(HIPBLAS_FILL_MODE_FULL)), value :: uplo
    integer(c_int), value :: n
    type(c_ptr), value :: alpha
    type(c_ptr), value :: A
    integer(c_int), value :: lda
    integer(c_int64_t), value :: stride_A
    type(c_ptr), value :: x
    integer(c_int), value :: incx
    integer(c_int64_t), value :: stride_x
    type(c_ptr), value :: beta
    type(c_ptr), value :: y
    integer(c_int), value :: incy
    integer(c_int64_t), value :: stride_y
    integer(c_int), value :: batch_count
            hipblasSsymvStridedBatchedFortran = &
        hipblasSsymvStridedBatched(handle, uplo, n, alpha, &
                                    A, lda, stride_A, x, incx, stride_x, beta, y, incy, stride_y, batch_count)
end function hipblasSsymvStridedBatchedFortran

function hipblasDsymvStridedBatchedFortran(handle, uplo, n, alpha, A, lda, stride_A, &
                                            x, incx, stride_x, beta, y, incy, stride_y, batch_count) &
    bind(c, name='hipblasDsymvStridedBatchedFortran')
    use iso_c_binding
    use hipblas_enums
    implicit none
    integer(kind(HIPBLAS_STATUS_SUCCESS)) :: hipblasDsymvStridedBatchedFortran
    type(c_ptr), value :: handle
    integer(kind(HIPBLAS_FILL_MODE_FULL)), value :: uplo
    integer(c_int), value :: n
    type(c_ptr), value :: alpha
    type(c_ptr), value :: A
    integer(c_int), value :: lda
    integer(c_int64_t), value :: stride_A
    type(c_ptr), value :: x
    integer(c_int), value :: incx
    integer(c_int64_t), value :: stride_x
    type(c_ptr), value :: beta
    type(c_ptr), value :: y
    integer(c_int), value :: incy
    integer(c_int64_t), value :: stride_y
    integer(c_int), value :: batch_count
            hipblasDsymvStridedBatchedFortran = &
        hipblasDsymvStridedBatched(handle, uplo, n, alpha, &
                                    A, lda, stride_A, x, incx, stride_x, beta, y, incy, stride_y, batch_count)
end function hipblasDsymvStridedBatchedFortran

function hipblasCsymvStridedBatchedFortran(handle, uplo, n, alpha, A, lda, stride_A, &
                                            x, incx, stride_x, beta, y, incy, stride_y, batch_count) &
    bind(c, name='hipblasCsymvStridedBatchedFortran')
    use iso_c_binding
    use hipblas_enums
    implicit none
    integer(kind(HIPBLAS_STATUS_SUCCESS)) :: hipblasCsymvStridedBatchedFortran
    type(c_ptr), value :: handle
    integer(kind(HIPBLAS_FILL_MODE_FULL)), value :: uplo
    integer(c_int), value :: n
    type(c_ptr), value :: alpha
    type(c_ptr), value :: A
    integer(c_int), value :: lda
    integer(c_int64_t), value :: stride_A
    type(c_ptr), value :: x
    integer(c_int), value :: incx
    integer(c_int64_t), value :: stride_x
    type(c_ptr), value :: beta
    type(c_ptr), value :: y
    integer(c_int), value :: incy
    integer(c_int64_t), value :: stride_y
    integer(c_int), value :: batch_count
            hipblasCsymvStridedBatchedFortran = &
        hipblasCsymvStridedBatched(handle, uplo, n, alpha, &
                                    A, lda, stride_A, x, incx, stride_x, beta, y, incy, stride_y, batch_count)
end function hipblasCsymvStridedBatchedFortran

function hipblasZsymvStridedBatchedFortran(handle, uplo, n, alpha, A, lda, stride_A, &
                                            x, incx, stride_x, beta, y, incy, stride_y, batch_count) &
    bind(c, name='hipblasZsymvStridedBatchedFortran')
    use iso_c_binding
    use hipblas_enums
    implicit none
    integer(kind(HIPBLAS_STATUS_SUCCESS)) :: hipblasZsymvStridedBatchedFortran
    type(c_ptr), value :: handle
    integer(kind(HIPBLAS_FILL_MODE_FULL)), value :: uplo
    integer(c_int), value :: n
    type(c_ptr), value :: alpha
    type(c_ptr), value :: A
    integer(c_int), value :: lda
    integer(c_int64_t), value :: stride_A
    type(c_ptr), value :: x
    integer(c_int), value :: incx
    integer(c_int64_t), value :: stride_x
    type(c_ptr), value :: beta
    type(c_ptr), value :: y
    integer(c_int), value :: incy
    integer(c_int64_t), value :: stride_y
    integer(c_int), value :: batch_count
            hipblasZsymvStridedBatchedFortran = &
        hipblasZsymvStridedBatched(handle, uplo, n, alpha, &
                                    A, lda, stride_A, x, incx, stride_x, beta, y, incy, stride_y, batch_count)
end function hipblasZsymvStridedBatchedFortran

! spmv
function hipblasSspmvFortran(handle, uplo, n, alpha, AP, &
                                x, incx, beta, y, incy) &
    bind(c, name='hipblasSspmvFortran')
    use iso_c_binding
    use hipblas_enums
    implicit none
    integer(kind(HIPBLAS_STATUS_SUCCESS)) :: hipblasSspmvFortran
    type(c_ptr), value :: handle
    integer(kind(HIPBLAS_FILL_MODE_FULL)), value :: uplo
    integer(c_int), value :: n
    type(c_ptr), value :: alpha
    type(c_ptr), value :: AP
    type(c_ptr), value :: x
    integer(c_int), value :: incx
    type(c_ptr), value :: beta
    type(c_ptr), value :: y
    integer(c_int), value :: incy
            hipblasSspmvFortran = &
        hipblasSspmv(handle, uplo, n, alpha, &
                        AP, x, incx, beta, y, incy)
end function hipblasSspmvFortran

function hipblasDspmvFortran(handle, uplo, n, alpha, AP, &
                                x, incx, beta, y, incy) &
    bind(c, name='hipblasDspmvFortran')
    use iso_c_binding
    use hipblas_enums
    implicit none
    integer(kind(HIPBLAS_STATUS_SUCCESS)) :: hipblasDspmvFortran
    type(c_ptr), value :: handle
    integer(kind(HIPBLAS_FILL_MODE_FULL)), value :: uplo
    integer(c_int), value :: n
    type(c_ptr), value :: alpha
    type(c_ptr), value :: AP
    type(c_ptr), value :: x
    integer(c_int), value :: incx
    type(c_ptr), value :: beta
    type(c_ptr), value :: y
    integer(c_int), value :: incy
            hipblasDspmvFortran = &
        hipblasDspmv(handle, uplo, n, alpha, &
                        AP, x, incx, beta, y, incy)
end function hipblasDspmvFortran

! spmvBatched
function hipblasSspmvBatchedFortran(handle, uplo, n, alpha, AP, &
                                    x, incx, beta, y, incy, batch_count) &
    bind(c, name='hipblasSspmvBatchedFortran')
    use iso_c_binding
    use hipblas_enums
    implicit none
    integer(kind(HIPBLAS_STATUS_SUCCESS)) :: hipblasSspmvBatchedFortran
    type(c_ptr), value :: handle
    integer(kind(HIPBLAS_FILL_MODE_FULL)), value :: uplo
    integer(c_int), value :: n
    type(c_ptr), value :: alpha
    type(c_ptr), value :: AP
    type(c_ptr), value :: x
    integer(c_int), value :: incx
    type(c_ptr), value :: beta
    type(c_ptr), value :: y
    integer(c_int), value :: incy
    integer(c_int), value :: batch_count
            hipblasSspmvBatchedFortran = &
        hipblasSspmvBatched(handle, uplo, n, alpha, &
                            AP, x, incx, beta, y, incy, batch_count)
end function hipblasSspmvBatchedFortran

function hipblasDspmvBatchedFortran(handle, uplo, n, alpha, AP, &
                                    x, incx, beta, y, incy, batch_count) &
    bind(c, name='hipblasDspmvBatchedFortran')
    use iso_c_binding
    use hipblas_enums
    implicit none
    integer(kind(HIPBLAS_STATUS_SUCCESS)) :: hipblasDspmvBatchedFortran
    type(c_ptr), value :: handle
    integer(kind(HIPBLAS_FILL_MODE_FULL)), value :: uplo
    integer(c_int), value :: n
    type(c_ptr), value :: alpha
    type(c_ptr), value :: AP
    type(c_ptr), value :: x
    integer(c_int), value :: incx
    type(c_ptr), value :: beta
    type(c_ptr), value :: y
    integer(c_int), value :: incy
    integer(c_int), value :: batch_count
            hipblasDspmvBatchedFortran = &
        hipblasDspmvBatched(handle, uplo, n, alpha, &
                            AP, x, incx, beta, y, incy, batch_count)
end function hipblasDspmvBatchedFortran

! spmvStridedBatched
function hipblasSspmvStridedBatchedFortran(handle, uplo, n, alpha, AP, stride_AP, &
                                            x, incx, stride_x, beta, y, incy, stride_y, batch_count) &
    bind(c, name='hipblasSspmvStridedBatchedFortran')
    use iso_c_binding
    use hipblas_enums
    implicit none
    integer(kind(HIPBLAS_STATUS_SUCCESS)) :: hipblasSspmvStridedBatchedFortran
    type(c_ptr), value :: handle
    integer(kind(HIPBLAS_FILL_MODE_FULL)), value :: uplo
    integer(c_int), value :: n
    type(c_ptr), value :: alpha
    type(c_ptr), value :: AP
    integer(c_int64_t), value :: stride_AP
    type(c_ptr), value :: x
    integer(c_int), value :: incx
    integer(c_int64_t), value :: stride_x
    type(c_ptr), value :: beta
    type(c_ptr), value :: y
    integer(c_int), value :: incy
    integer(c_int64_t), value :: stride_y
    integer(c_int), value :: batch_count
            hipblasSspmvStridedBatchedFortran = &
        hipblasSspmvStridedBatched(handle, uplo, n, alpha, &
                                    AP, stride_AP, x, incx, stride_x, beta, y, incy, stride_y, batch_count)
end function hipblasSspmvStridedBatchedFortran

function hipblasDspmvStridedBatchedFortran(handle, uplo, n, alpha, AP, stride_AP, &
                                            x, incx, stride_x, beta, y, incy, stride_y, batch_count) &
    bind(c, name='hipblasDspmvStridedBatchedFortran')
    use iso_c_binding
    use hipblas_enums
    implicit none
    integer(kind(HIPBLAS_STATUS_SUCCESS)) :: hipblasDspmvStridedBatchedFortran
    type(c_ptr), value :: handle
    integer(kind(HIPBLAS_FILL_MODE_FULL)), value :: uplo
    integer(c_int), value :: n
    type(c_ptr), value :: alpha
    type(c_ptr), value :: AP
    integer(c_int64_t), value :: stride_AP
    type(c_ptr), value :: x
    integer(c_int), value :: incx
    integer(c_int64_t), value :: stride_x
    type(c_ptr), value :: beta
    type(c_ptr), value :: y
    integer(c_int), value :: incy
    integer(c_int64_t), value :: stride_y
    integer(c_int), value :: batch_count
            hipblasDspmvStridedBatchedFortran = &
        hipblasDspmvStridedBatched(handle, uplo, n, alpha, &
                                    AP, stride_AP, x, incx, stride_x, beta, y, incy, stride_y, batch_count)
end function hipblasDspmvStridedBatchedFortran

! sbmv
function hipblasSsbmvFortran(handle, uplo, n, k, alpha, A, lda, &
                                x, incx, beta, y, incy) &
    bind(c, name='hipblasSsbmvFortran')
    use iso_c_binding
    use hipblas_enums
    implicit none
    integer(kind(HIPBLAS_STATUS_SUCCESS)) :: hipblasSsbmvFortran
    type(c_ptr), value :: handle
    integer(kind(HIPBLAS_FILL_MODE_FULL)), value :: uplo
    integer(c_int), value :: n
    integer(c_int), value :: k
    type(c_ptr), value :: alpha
    type(c_ptr), value :: A
    integer(c_int), value :: lda
    type(c_ptr), value :: x
    integer(c_int), value :: incx
    type(c_ptr), value :: beta
    type(c_ptr), value :: y
    integer(c_int), value :: incy
            hipblasSsbmvFortran = &
        hipblasSsbmv(handle, uplo, n, k, alpha, &
                        A, lda, x, incx, beta, y, incy)
end function hipblasSsbmvFortran

function hipblasDsbmvFortran(handle, uplo, n, k, alpha, A, lda, &
                                x, incx, beta, y, incy) &
    bind(c, name='hipblasDsbmvFortran')
    use iso_c_binding
    use hipblas_enums
    implicit none
    integer(kind(HIPBLAS_STATUS_SUCCESS)) :: hipblasDsbmvFortran
    type(c_ptr), value :: handle
    integer(kind(HIPBLAS_FILL_MODE_FULL)), value :: uplo
    integer(c_int), value :: n
    integer(c_int), value :: k
    type(c_ptr), value :: alpha
    type(c_ptr), value :: A
    integer(c_int), value :: lda
    type(c_ptr), value :: x
    integer(c_int), value :: incx
    type(c_ptr), value :: beta
    type(c_ptr), value :: y
    integer(c_int), value :: incy
            hipblasDsbmvFortran = &
        hipblasDsbmv(handle, uplo, n, k, alpha, &
                        A, lda, x, incx, beta, y, incy)
end function hipblasDsbmvFortran

! sbmvBatched
function hipblasSsbmvBatchedFortran(handle, uplo, n, k, alpha, A, lda, &
                                    x, incx, beta, y, incy, batch_count) &
    bind(c, name='hipblasSsbmvBatchedFortran')
    use iso_c_binding
    use hipblas_enums
    implicit none
    integer(kind(HIPBLAS_STATUS_SUCCESS)) :: hipblasSsbmvBatchedFortran
    type(c_ptr), value :: handle
    integer(kind(HIPBLAS_FILL_MODE_FULL)), value :: uplo
    integer(c_int), value :: n
    integer(c_int), value :: k
    type(c_ptr), value :: alpha
    type(c_ptr), value :: A
    integer(c_int), value :: lda
    type(c_ptr), value :: x
    integer(c_int), value :: incx
    type(c_ptr), value :: beta
    type(c_ptr), value :: y
    integer(c_int), value :: incy
    integer(c_int), value :: batch_count
            hipblasSsbmvBatchedFortran = &
        hipblasSsbmvBatched(handle, uplo, n, k, alpha, &
                            A, lda, x, incx, beta, y, incy, batch_count)
end function hipblasSsbmvBatchedFortran

function hipblasDsbmvBatchedFortran(handle, uplo, n, k, alpha, A, lda, &
                                    x, incx, beta, y, incy, batch_count) &
    bind(c, name='hipblasDsbmvBatchedFortran')
    use iso_c_binding
    use hipblas_enums
    implicit none
    integer(kind(HIPBLAS_STATUS_SUCCESS)) :: hipblasDsbmvBatchedFortran
    type(c_ptr), value :: handle
    integer(kind(HIPBLAS_FILL_MODE_FULL)), value :: uplo
    integer(c_int), value :: n
    integer(c_int), value :: k
    type(c_ptr), value :: alpha
    type(c_ptr), value :: A
    integer(c_int), value :: lda
    type(c_ptr), value :: x
    integer(c_int), value :: incx
    type(c_ptr), value :: beta
    type(c_ptr), value :: y
    integer(c_int), value :: incy
    integer(c_int), value :: batch_count
            hipblasDsbmvBatchedFortran = &
        hipblasDsbmvBatched(handle, uplo, n, k, alpha, &
                            A, lda, x, incx, beta, y, incy, batch_count)
end function hipblasDsbmvBatchedFortran

! sbmvStridedBatched
function hipblasSsbmvStridedBatchedFortran(handle, uplo, n, k, alpha, A, lda, stride_A, &
                                            x, incx, stride_x, beta, y, incy, stride_y, batch_count) &
    bind(c, name='hipblasSsbmvStridedBatchedFortran')
    use iso_c_binding
    use hipblas_enums
    implicit none
    integer(kind(HIPBLAS_STATUS_SUCCESS)) :: hipblasSsbmvStridedBatchedFortran
    type(c_ptr), value :: handle
    integer(kind(HIPBLAS_FILL_MODE_FULL)), value :: uplo
    integer(c_int), value :: n
    integer(c_int), value :: k
    type(c_ptr), value :: alpha
    type(c_ptr), value :: A
    integer(c_int), value :: lda
    integer(c_int64_t), value :: stride_A
    type(c_ptr), value :: x
    integer(c_int), value :: incx
    integer(c_int64_t), value :: stride_x
    type(c_ptr), value :: beta
    type(c_ptr), value :: y
    integer(c_int), value :: incy
    integer(c_int64_t), value :: stride_y
    integer(c_int), value :: batch_count
            hipblasSsbmvStridedBatchedFortran = &
        hipblasSsbmvStridedBatched(handle, uplo, n, k, alpha, &
                                    A, lda, stride_A, x, incx, stride_x, beta, y, incy, stride_y, batch_count)
end function hipblasSsbmvStridedBatchedFortran

function hipblasDsbmvStridedBatchedFortran(handle, uplo, n, k, alpha, A, lda, stride_A, &
                                            x, incx, stride_x, beta, y, incy, stride_y, batch_count) &
    bind(c, name='hipblasDsbmvStridedBatchedFortran')
    use iso_c_binding
    use hipblas_enums
    implicit none
    integer(kind(HIPBLAS_STATUS_SUCCESS)) :: hipblasDsbmvStridedBatchedFortran
    type(c_ptr), value :: handle
    integer(kind(HIPBLAS_FILL_MODE_FULL)), value :: uplo
    integer(c_int), value :: n
    integer(c_int), value :: k
    type(c_ptr), value :: alpha
    type(c_ptr), value :: A
    integer(c_int), value :: lda
    integer(c_int64_t), value :: stride_A
    type(c_ptr), value :: x
    integer(c_int), value :: incx
    integer(c_int64_t), value :: stride_x
    type(c_ptr), value :: beta
    type(c_ptr), value :: y
    integer(c_int), value :: incy
    integer(c_int64_t), value :: stride_y
    integer(c_int), value :: batch_count
            hipblasDsbmvStridedBatchedFortran = &
        hipblasDsbmvStridedBatched(handle, uplo, n, k, alpha, &
                                    A, lda, stride_A, x, incx, stride_x, beta, y, incy, stride_y, batch_count)
end function hipblasDsbmvStridedBatchedFortran

! ger
function hipblasSgerFortran(handle, m, n, alpha, x, incx, &
                            y, incy, A, lda) &
    bind(c, name='hipblasSgerFortran')
    use iso_c_binding
    use hipblas_enums
    implicit none
    integer(kind(HIPBLAS_STATUS_SUCCESS)) :: hipblasSgerFortran
    type(c_ptr), value :: handle
    integer(c_int), value :: m
    integer(c_int), value :: n
    type(c_ptr), value :: alpha
    type(c_ptr), value :: x
    integer(c_int), value :: incx
    type(c_ptr), value :: y
    integer(c_int), value :: incy
    type(c_ptr), value :: A
    integer(c_int), value :: lda
            hipblasSgerFortran = &
        hipblasSger(handle, m, n, alpha, &
                    x, incx, y, incy, A, lda)
end function hipblasSgerFortran

function hipblasDgerFortran(handle, m, n, alpha, x, incx, &
                            y, incy, A, lda) &
    bind(c, name='hipblasDgerFortran')
    use iso_c_binding
    use hipblas_enums
    implicit none
    integer(kind(HIPBLAS_STATUS_SUCCESS)) :: hipblasDgerFortran
    type(c_ptr), value :: handle
    integer(c_int), value :: m
    integer(c_int), value :: n
    type(c_ptr), value :: alpha
    type(c_ptr), value :: x
    integer(c_int), value :: incx
    type(c_ptr), value :: y
    integer(c_int), value :: incy
    type(c_ptr), value :: A
    integer(c_int), value :: lda
            hipblasDgerFortran = &
        hipblasDger(handle, m, n, alpha, &
                    x, incx, y, incy, A, lda)
end function hipblasDgerFortran

function hipblasCgeruFortran(handle, m, n, alpha, x, incx, &
                                y, incy, A, lda) &
    bind(c, name='hipblasCgeruFortran')
    use iso_c_binding
    use hipblas_enums
    implicit none
    integer(kind(HIPBLAS_STATUS_SUCCESS)) :: hipblasCgeruFortran
    type(c_ptr), value :: handle
    integer(c_int), value :: m
    integer(c_int), value :: n
    type(c_ptr), value :: alpha
    type(c_ptr), value :: x
    integer(c_int), value :: incx
    type(c_ptr), value :: y
    integer(c_int), value :: incy
    type(c_ptr), value :: A
    integer(c_int), value :: lda
            hipblasCgeruFortran = &
        hipblasCgeru(handle, m, n, alpha, &
                        x, incx, y, incy, A, lda)
end function hipblasCgeruFortran

function hipblasCgercFortran(handle, m, n, alpha, x, incx, &
                                y, incy, A, lda) &
    bind(c, name='hipblasCgercFortran')
    use iso_c_binding
    use hipblas_enums
    implicit none
    integer(kind(HIPBLAS_STATUS_SUCCESS)) :: hipblasCgercFortran
    type(c_ptr), value :: handle
    integer(c_int), value :: m
    integer(c_int), value :: n
    type(c_ptr), value :: alpha
    type(c_ptr), value :: x
    integer(c_int), value :: incx
    type(c_ptr), value :: y
    integer(c_int), value :: incy
    type(c_ptr), value :: A
    integer(c_int), value :: lda
            hipblasCgercFortran = &
        hipblasCgerc(handle, m, n, alpha, &
                        x, incx, y, incy, A, lda)
end function hipblasCgercFortran

function hipblasZgeruFortran(handle, m, n, alpha, x, incx, &
                                y, incy, A, lda) &
    bind(c, name='hipblasZgeruFortran')
    use iso_c_binding
    use hipblas_enums
    implicit none
    integer(kind(HIPBLAS_STATUS_SUCCESS)) :: hipblasZgeruFortran
    type(c_ptr), value :: handle
    integer(c_int), value :: m
    integer(c_int), value :: n
    type(c_ptr), value :: alpha
    type(c_ptr), value :: x
    integer(c_int), value :: incx
    type(c_ptr), value :: y
    integer(c_int), value :: incy
    type(c_ptr), value :: A
    integer(c_int), value :: lda
            hipblasZgeruFortran = &
        hipblasZgeru(handle, m, n, alpha, &
                        x, incx, y, incy, A, lda)
end function hipblasZgeruFortran

function hipblasZgercFortran(handle, m, n, alpha, x, incx, &
                                y, incy, A, lda) &
    bind(c, name='hipblasZgercFortran')
    use iso_c_binding
    use hipblas_enums
    implicit none
    integer(kind(HIPBLAS_STATUS_SUCCESS)) :: hipblasZgercFortran
    type(c_ptr), value :: handle
    integer(c_int), value :: m
    integer(c_int), value :: n
    type(c_ptr), value :: alpha
    type(c_ptr), value :: x
    integer(c_int), value :: incx
    type(c_ptr), value :: y
    integer(c_int), value :: incy
    type(c_ptr), value :: A
    integer(c_int), value :: lda
            hipblasZgercFortran = &
        hipblasZgerc(handle, m, n, alpha, &
                        x, incx, y, incy, A, lda)
end function hipblasZgercFortran

! gerBatched
function hipblasSgerBatchedFortran(handle, m, n, alpha, x, incx, &
                                    y, incy, A, lda, batch_count) &
    bind(c, name='hipblasSgerBatchedFortran')
    use iso_c_binding
    use hipblas_enums
    implicit none
    integer(kind(HIPBLAS_STATUS_SUCCESS)) :: hipblasSgerBatchedFortran
    type(c_ptr), value :: handle
    integer(c_int), value :: m
    integer(c_int), value :: n
    type(c_ptr), value :: alpha
    type(c_ptr), value :: x
    integer(c_int), value :: incx
    type(c_ptr), value :: y
    integer(c_int), value :: incy
    type(c_ptr), value :: A
    integer(c_int), value :: lda
    integer(c_int), value :: batch_count
            hipblasSgerBatchedFortran = &
        hipblasSgerBatched(handle, m, n, alpha, &
                            x, incx, y, incy, A, lda, batch_count)
end function hipblasSgerBatchedFortran

function hipblasDgerBatchedFortran(handle, m, n, alpha, x, incx, &
                                    y, incy, A, lda, batch_count) &
    bind(c, name='hipblasDgerBatchedFortran')
    use iso_c_binding
    use hipblas_enums
    implicit none
    integer(kind(HIPBLAS_STATUS_SUCCESS)) :: hipblasDgerBatchedFortran
    type(c_ptr), value :: handle
    integer(c_int), value :: m
    integer(c_int), value :: n
    type(c_ptr), value :: alpha
    type(c_ptr), value :: x
    integer(c_int), value :: incx
    type(c_ptr), value :: y
    integer(c_int), value :: incy
    type(c_ptr), value :: A
    integer(c_int), value :: lda
    integer(c_int), value :: batch_count
            hipblasDgerBatchedFortran = &
        hipblasDgerBatched(handle, m, n, alpha, &
                            x, incx, y, incy, A, lda, batch_count)
end function hipblasDgerBatchedFortran

function hipblasCgeruBatchedFortran(handle, m, n, alpha, x, incx, &
                                    y, incy, A, lda, batch_count) &
    bind(c, name='hipblasCgeruBatchedFortran')
    use iso_c_binding
    use hipblas_enums
    implicit none
    integer(kind(HIPBLAS_STATUS_SUCCESS)) :: hipblasCgeruBatchedFortran
    type(c_ptr), value :: handle
    integer(c_int), value :: m
    integer(c_int), value :: n
    type(c_ptr), value :: alpha
    type(c_ptr), value :: x
    integer(c_int), value :: incx
    type(c_ptr), value :: y
    integer(c_int), value :: incy
    type(c_ptr), value :: A
    integer(c_int), value :: lda
    integer(c_int), value :: batch_count
            hipblasCgeruBatchedFortran = &
        hipblasCgeruBatched(handle, m, n, alpha, &
                            x, incx, y, incy, A, lda, batch_count)
end function hipblasCgeruBatchedFortran

function hipblasCgercBatchedFortran(handle, m, n, alpha, x, incx, &
                                    y, incy, A, lda, batch_count) &
    bind(c, name='hipblasCgercBatchedFortran')
    use iso_c_binding
    use hipblas_enums
    implicit none
    integer(kind(HIPBLAS_STATUS_SUCCESS)) :: hipblasCgercBatchedFortran
    type(c_ptr), value :: handle
    integer(c_int), value :: m
    integer(c_int), value :: n
    type(c_ptr), value :: alpha
    type(c_ptr), value :: x
    integer(c_int), value :: incx
    type(c_ptr), value :: y
    integer(c_int), value :: incy
    type(c_ptr), value :: A
    integer(c_int), value :: lda
    integer(c_int), value :: batch_count
            hipblasCgercBatchedFortran = &
        hipblasCgercBatched(handle, m, n, alpha, &
                            x, incx, y, incy, A, lda, batch_count)
end function hipblasCgercBatchedFortran

function hipblasZgeruBatchedFortran(handle, m, n, alpha, x, incx, &
                                    y, incy, A, lda, batch_count) &
    bind(c, name='hipblasZgeruBatchedFortran')
    use iso_c_binding
    use hipblas_enums
    implicit none
    integer(kind(HIPBLAS_STATUS_SUCCESS)) :: hipblasZgeruBatchedFortran
    type(c_ptr), value :: handle
    integer(c_int), value :: m
    integer(c_int), value :: n
    type(c_ptr), value :: alpha
    type(c_ptr), value :: x
    integer(c_int), value :: incx
    type(c_ptr), value :: y
    integer(c_int), value :: incy
    type(c_ptr), value :: A
    integer(c_int), value :: lda
    integer(c_int), value :: batch_count
            hipblasZgeruBatchedFortran = &
        hipblasZgeruBatched(handle, m, n, alpha, &
                            x, incx, y, incy, A, lda, batch_count)
end function hipblasZgeruBatchedFortran

function hipblasZgercBatchedFortran(handle, m, n, alpha, x, incx, &
                                    y, incy, A, lda, batch_count) &
    bind(c, name='hipblasZgercBatchedFortran')
    use iso_c_binding
    use hipblas_enums
    implicit none
    integer(kind(HIPBLAS_STATUS_SUCCESS)) :: hipblasZgercBatchedFortran
    type(c_ptr), value :: handle
    integer(c_int), value :: m
    integer(c_int), value :: n
    type(c_ptr), value :: alpha
    type(c_ptr), value :: x
    integer(c_int), value :: incx
    type(c_ptr), value :: y
    integer(c_int), value :: incy
    type(c_ptr), value :: A
    integer(c_int), value :: lda
    integer(c_int), value :: batch_count
            hipblasZgercBatchedFortran = &
        hipblasZgercBatched(handle, m, n, alpha, &
                            x, incx, y, incy, A, lda, batch_count)
end function hipblasZgercBatchedFortran

! gerStridedBatched
function hipblasSgerStridedBatchedFortran(handle, m, n, alpha, x, incx, stride_x, &
                                            y, incy, stride_y, A, lda, stride_A, batch_count) &
    bind(c, name='hipblasSgerStridedBatchedFortran')
    use iso_c_binding
    use hipblas_enums
    implicit none
    integer(kind(HIPBLAS_STATUS_SUCCESS)) :: hipblasSgerStridedBatchedFortran
    type(c_ptr), value :: handle
    integer(c_int), value :: m
    integer(c_int), value :: n
    type(c_ptr), value :: alpha
    type(c_ptr), value :: x
    integer(c_int), value :: incx
    integer(c_int64_t), value :: stride_x
    type(c_ptr), value :: y
    integer(c_int), value :: incy
    integer(c_int64_t), value :: stride_y
    type(c_ptr), value :: A
    integer(c_int), value :: lda
    integer(c_int64_t), value :: stride_A
    integer(c_int), value :: batch_count
            hipblasSgerStridedBatchedFortran = &
        hipblasSgerStridedBatched(handle, m, n, alpha, &
                                    x, incx, stride_x, y, incy, stride_y, A, lda, stride_A, batch_count)
end function hipblasSgerStridedBatchedFortran

function hipblasDgerStridedBatchedFortran(handle, m, n, alpha, x, incx, stride_x, &
                                            y, incy, stride_y, A, lda, stride_A, batch_count) &
    bind(c, name='hipblasDgerStridedBatchedFortran')
    use iso_c_binding
    use hipblas_enums
    implicit none
    integer(kind(HIPBLAS_STATUS_SUCCESS)) :: hipblasDgerStridedBatchedFortran
    type(c_ptr), value :: handle
    integer(c_int), value :: m
    integer(c_int), value :: n
    type(c_ptr), value :: alpha
    type(c_ptr), value :: x
    integer(c_int), value :: incx
    integer(c_int64_t), value :: stride_x
    type(c_ptr), value :: y
    integer(c_int), value :: incy
    integer(c_int64_t), value :: stride_y
    type(c_ptr), value :: A
    integer(c_int), value :: lda
    integer(c_int64_t), value :: stride_A
    integer(c_int), value :: batch_count
            hipblasDgerStridedBatchedFortran = &
        hipblasDgerStridedBatched(handle, m, n, alpha, &
                                    x, incx, stride_x, y, incy, stride_y, A, lda, stride_A, batch_count)
end function hipblasDgerStridedBatchedFortran

function hipblasCgeruStridedBatchedFortran(handle, m, n, alpha, x, incx, stride_x, &
                                            y, incy, stride_y, A, lda, stride_A, batch_count) &
    bind(c, name='hipblasCgeruStridedBatchedFortran')
    use iso_c_binding
    use hipblas_enums
    implicit none
    integer(kind(HIPBLAS_STATUS_SUCCESS)) :: hipblasCgeruStridedBatchedFortran
    type(c_ptr), value :: handle
    integer(c_int), value :: m
    integer(c_int), value :: n
    type(c_ptr), value :: alpha
    type(c_ptr), value :: x
    integer(c_int), value :: incx
    integer(c_int64_t), value :: stride_x
    type(c_ptr), value :: y
    integer(c_int), value :: incy
    integer(c_int64_t), value :: stride_y
    type(c_ptr), value :: A
    integer(c_int), value :: lda
    integer(c_int64_t), value :: stride_A
    integer(c_int), value :: batch_count
            hipblasCgeruStridedBatchedFortran = &
        hipblasCgeruStridedBatched(handle, m, n, alpha, &
                                    x, incx, stride_x, y, incy, stride_y, A, lda, stride_A, batch_count)
end function hipblasCgeruStridedBatchedFortran

function hipblasCgercStridedBatchedFortran(handle, m, n, alpha, x, incx, stride_x, &
                                            y, incy, stride_y, A, lda, stride_A, batch_count) &
    bind(c, name='hipblasCgercStridedBatchedFortran')
    use iso_c_binding
    use hipblas_enums
    implicit none
    integer(kind(HIPBLAS_STATUS_SUCCESS)) :: hipblasCgercStridedBatchedFortran
    type(c_ptr), value :: handle
    integer(c_int), value :: m
    integer(c_int), value :: n
    type(c_ptr), value :: alpha
    type(c_ptr), value :: x
    integer(c_int), value :: incx
    integer(c_int64_t), value :: stride_x
    type(c_ptr), value :: y
    integer(c_int), value :: incy
    integer(c_int64_t), value :: stride_y
    type(c_ptr), value :: A
    integer(c_int), value :: lda
    integer(c_int64_t), value :: stride_A
    integer(c_int), value :: batch_count
            hipblasCgercStridedBatchedFortran = &
        hipblasCgercStridedBatched(handle, m, n, alpha, &
                                    x, incx, stride_x, y, incy, stride_y, A, lda, stride_A, batch_count)
end function hipblasCgercStridedBatchedFortran

function hipblasZgeruStridedBatchedFortran(handle, m, n, alpha, x, incx, stride_x, &
                                            y, incy, stride_y, A, lda, stride_A, batch_count) &
    bind(c, name='hipblasZgeruStridedBatchedFortran')
    use iso_c_binding
    use hipblas_enums
    implicit none
    integer(kind(HIPBLAS_STATUS_SUCCESS)) :: hipblasZgeruStridedBatchedFortran
    type(c_ptr), value :: handle
    integer(c_int), value :: m
    integer(c_int), value :: n
    type(c_ptr), value :: alpha
    type(c_ptr), value :: x
    integer(c_int), value :: incx
    integer(c_int64_t), value :: stride_x
    type(c_ptr), value :: y
    integer(c_int), value :: incy
    integer(c_int64_t), value :: stride_y
    type(c_ptr), value :: A
    integer(c_int), value :: lda
    integer(c_int64_t), value :: stride_A
    integer(c_int), value :: batch_count
            hipblasZgeruStridedBatchedFortran = &
        hipblasZgeruStridedBatched(handle, m, n, alpha, &
                                    x, incx, stride_x, y, incy, stride_y, A, lda, stride_A, batch_count)
end function hipblasZgeruStridedBatchedFortran

function hipblasZgercStridedBatchedFortran(handle, m, n, alpha, x, incx, stride_x, &
                                            y, incy, stride_y, A, lda, stride_A, batch_count) &
    bind(c, name='hipblasZgercStridedBatchedFortran')
    use iso_c_binding
    use hipblas_enums
    implicit none
    integer(kind(HIPBLAS_STATUS_SUCCESS)) :: hipblasZgercStridedBatchedFortran
    type(c_ptr), value :: handle
    integer(c_int), value :: m
    integer(c_int), value :: n
    type(c_ptr), value :: alpha
    type(c_ptr), value :: x
    integer(c_int), value :: incx
    integer(c_int64_t), value :: stride_x
    type(c_ptr), value :: y
    integer(c_int), value :: incy
    integer(c_int64_t), value :: stride_y
    type(c_ptr), value :: A
    integer(c_int), value :: lda
    integer(c_int64_t), value :: stride_A
    integer(c_int), value :: batch_count
            hipblasZgercStridedBatchedFortran = &
        hipblasZgercStridedBatched(handle, m, n, alpha, &
                                    x, incx, stride_x, y, incy, stride_y, A, lda, stride_A, batch_count)
end function hipblasZgercStridedBatchedFortran

! spr
function hipblasSsprFortran(handle, uplo, n, alpha, x, incx, AP) &
    bind(c, name='hipblasSsprFortran')
    use iso_c_binding
    use hipblas_enums
    implicit none
    integer(kind(HIPBLAS_STATUS_SUCCESS)) :: hipblasSsprFortran
    type(c_ptr), value :: handle
    integer(kind(HIPBLAS_FILL_MODE_FULL)), value :: uplo
    integer(c_int), value :: n
    type(c_ptr), value :: alpha
    type(c_ptr), value :: x
    integer(c_int), value :: incx
    type(c_ptr), value :: AP
            hipblasSsprFortran = &
        hipblasSspr(handle, uplo, n, alpha, &
                    x, incx, AP)
end function hipblasSsprFortran

function hipblasDsprFortran(handle, uplo, n, alpha, x, incx, AP) &
    bind(c, name='hipblasDsprFortran')
    use iso_c_binding
    use hipblas_enums
    implicit none
    integer(kind(HIPBLAS_STATUS_SUCCESS)) :: hipblasDsprFortran
    type(c_ptr), value :: handle
    integer(kind(HIPBLAS_FILL_MODE_FULL)), value :: uplo
    integer(c_int), value :: n
    type(c_ptr), value :: alpha
    type(c_ptr), value :: x
    integer(c_int), value :: incx
    type(c_ptr), value :: AP
            hipblasDsprFortran = &
        hipblasDspr(handle, uplo, n, alpha, &
                    x, incx, AP)
end function hipblasDsprFortran

function hipblasCsprFortran(handle, uplo, n, alpha, x, incx, AP) &
    bind(c, name='hipblasCsprFortran')
    use iso_c_binding
    use hipblas_enums
    implicit none
    integer(kind(HIPBLAS_STATUS_SUCCESS)) :: hipblasCsprFortran
    type(c_ptr), value :: handle
    integer(kind(HIPBLAS_FILL_MODE_FULL)), value :: uplo
    integer(c_int), value :: n
    type(c_ptr), value :: alpha
    type(c_ptr), value :: x
    integer(c_int), value :: incx
    type(c_ptr), value :: AP
            hipblasCsprFortran = &
        hipblasCspr(handle, uplo, n, alpha, &
                    x, incx, AP)
end function hipblasCsprFortran

function hipblasZsprFortran(handle, uplo, n, alpha, x, incx, AP) &
    bind(c, name='hipblasZsprFortran')
    use iso_c_binding
    use hipblas_enums
    implicit none
    integer(kind(HIPBLAS_STATUS_SUCCESS)) :: hipblasZsprFortran
    type(c_ptr), value :: handle
    integer(kind(HIPBLAS_FILL_MODE_FULL)), value :: uplo
    integer(c_int), value :: n
    type(c_ptr), value :: alpha
    type(c_ptr), value :: x
    integer(c_int), value :: incx
    type(c_ptr), value :: AP
            hipblasZsprFortran = &
        hipblasZspr(handle, uplo, n, alpha, &
                    x, incx, AP)
end function hipblasZsprFortran

! sprBatched
function hipblasSsprBatchedFortran(handle, uplo, n, alpha, x, incx, AP, batch_count) &
    bind(c, name='hipblasSsprBatchedFortran')
    use iso_c_binding
    use hipblas_enums
    implicit none
    integer(kind(HIPBLAS_STATUS_SUCCESS)) :: hipblasSsprBatchedFortran
    type(c_ptr), value :: handle
    integer(kind(HIPBLAS_FILL_MODE_FULL)), value :: uplo
    integer(c_int), value :: n
    type(c_ptr), value :: alpha
    type(c_ptr), value :: x
    integer(c_int), value :: incx
    type(c_ptr), value :: AP
    integer(c_int), value :: batch_count
            hipblasSsprBatchedFortran = &
        hipblasSsprBatched(handle, uplo, n, alpha, &
                            x, incx, AP, batch_count)
end function hipblasSsprBatchedFortran

function hipblasDsprBatchedFortran(handle, uplo, n, alpha, x, incx, AP, batch_count) &
    bind(c, name='hipblasDsprBatchedFortran')
    use iso_c_binding
    use hipblas_enums
    implicit none
    integer(kind(HIPBLAS_STATUS_SUCCESS)) :: hipblasDsprBatchedFortran
    type(c_ptr), value :: handle
    integer(kind(HIPBLAS_FILL_MODE_FULL)), value :: uplo
    integer(c_int), value :: n
    type(c_ptr), value :: alpha
    type(c_ptr), value :: x
    integer(c_int), value :: incx
    type(c_ptr), value :: AP
    integer(c_int), value :: batch_count
            hipblasDsprBatchedFortran = &
        hipblasDsprBatched(handle, uplo, n, alpha, &
                            x, incx, AP, batch_count)
end function hipblasDsprBatchedFortran

function hipblasCsprBatchedFortran(handle, uplo, n, alpha, x, incx, AP, batch_count) &
    bind(c, name='hipblasCsprBatchedFortran')
    use iso_c_binding
    use hipblas_enums
    implicit none
    integer(kind(HIPBLAS_STATUS_SUCCESS)) :: hipblasCsprBatchedFortran
    type(c_ptr), value :: handle
    integer(kind(HIPBLAS_FILL_MODE_FULL)), value :: uplo
    integer(c_int), value :: n
    type(c_ptr), value :: alpha
    type(c_ptr), value :: x
    integer(c_int), value :: incx
    type(c_ptr), value :: AP
    integer(c_int), value :: batch_count
            hipblasCsprBatchedFortran = &
        hipblasCsprBatched(handle, uplo, n, alpha, &
                            x, incx, AP, batch_count)
end function hipblasCsprBatchedFortran

function hipblasZsprBatchedFortran(handle, uplo, n, alpha, x, incx, AP, batch_count) &
    bind(c, name='hipblasZsprBatchedFortran')
    use iso_c_binding
    use hipblas_enums
    implicit none
    integer(kind(HIPBLAS_STATUS_SUCCESS)) :: hipblasZsprBatchedFortran
    type(c_ptr), value :: handle
    integer(kind(HIPBLAS_FILL_MODE_FULL)), value :: uplo
    integer(c_int), value :: n
    type(c_ptr), value :: alpha
    type(c_ptr), value :: x
    integer(c_int), value :: incx
    type(c_ptr), value :: AP
    integer(c_int), value :: batch_count
            hipblasZsprBatchedFortran = &
        hipblasZsprBatched(handle, uplo, n, alpha, &
                            x, incx, AP, batch_count)
end function hipblasZsprBatchedFortran

! sprStridedBatched
function hipblasSsprStridedBatchedFortran(handle, uplo, n, alpha, x, incx, stride_x, &
                                            AP, stride_AP, batch_count) &
    bind(c, name='hipblasSsprStridedBatchedFortran')
    use iso_c_binding
    use hipblas_enums
    implicit none
    integer(kind(HIPBLAS_STATUS_SUCCESS)) :: hipblasSsprStridedBatchedFortran
    type(c_ptr), value :: handle
    integer(kind(HIPBLAS_FILL_MODE_FULL)), value :: uplo
    integer(c_int), value :: n
    type(c_ptr), value :: alpha
    type(c_ptr), value :: x
    integer(c_int), value :: incx
    integer(c_int64_t), value :: stride_x
    type(c_ptr), value :: AP
    integer(c_int64_t), value :: stride_AP
    integer(c_int), value :: batch_count
            hipblasSsprStridedBatchedFortran = &
        hipblasSsprStridedBatched(handle, uplo, n, alpha, &
                                    x, incx, stride_x, AP, stride_AP, batch_count)
end function hipblasSsprStridedBatchedFortran

function hipblasDsprStridedBatchedFortran(handle, uplo, n, alpha, x, incx, stride_x, &
                                            AP, stride_AP, batch_count) &
    bind(c, name='hipblasDsprStridedBatchedFortran')
    use iso_c_binding
    use hipblas_enums
    implicit none
    integer(kind(HIPBLAS_STATUS_SUCCESS)) :: hipblasDsprStridedBatchedFortran
    type(c_ptr), value :: handle
    integer(kind(HIPBLAS_FILL_MODE_FULL)), value :: uplo
    integer(c_int), value :: n
    type(c_ptr), value :: alpha
    type(c_ptr), value :: x
    integer(c_int), value :: incx
    integer(c_int64_t), value :: stride_x
    type(c_ptr), value :: AP
    integer(c_int64_t), value :: stride_AP
    integer(c_int), value :: batch_count
            hipblasDsprStridedBatchedFortran = &
        hipblasDsprStridedBatched(handle, uplo, n, alpha, &
                                    x, incx, stride_x, AP, stride_AP, batch_count)
end function hipblasDsprStridedBatchedFortran

function hipblasCsprStridedBatchedFortran(handle, uplo, n, alpha, x, incx, stride_x, &
                                            AP, stride_AP, batch_count) &
    bind(c, name='hipblasCsprStridedBatchedFortran')
    use iso_c_binding
    use hipblas_enums
    implicit none
    integer(kind(HIPBLAS_STATUS_SUCCESS)) :: hipblasCsprStridedBatchedFortran
    type(c_ptr), value :: handle
    integer(kind(HIPBLAS_FILL_MODE_FULL)), value :: uplo
    integer(c_int), value :: n
    type(c_ptr), value :: alpha
    type(c_ptr), value :: x
    integer(c_int), value :: incx
    integer(c_int64_t), value :: stride_x
    type(c_ptr), value :: AP
    integer(c_int64_t), value :: stride_AP
    integer(c_int), value :: batch_count
            hipblasCsprStridedBatchedFortran = &
        hipblasCsprStridedBatched(handle, uplo, n, alpha, &
                                    x, incx, stride_x, AP, stride_AP, batch_count)
end function hipblasCsprStridedBatchedFortran

function hipblasZsprStridedBatchedFortran(handle, uplo, n, alpha, x, incx, stride_x, &
                                            AP, stride_AP, batch_count) &
    bind(c, name='hipblasZsprStridedBatchedFortran')
    use iso_c_binding
    use hipblas_enums
    implicit none
    integer(kind(HIPBLAS_STATUS_SUCCESS)) :: hipblasZsprStridedBatchedFortran
    type(c_ptr), value :: handle
    integer(kind(HIPBLAS_FILL_MODE_FULL)), value :: uplo
    integer(c_int), value :: n
    type(c_ptr), value :: alpha
    type(c_ptr), value :: x
    integer(c_int), value :: incx
    integer(c_int64_t), value :: stride_x
    type(c_ptr), value :: AP
    integer(c_int64_t), value :: stride_AP
    integer(c_int), value :: batch_count
            hipblasZsprStridedBatchedFortran = &
        hipblasZsprStridedBatched(handle, uplo, n, alpha, &
                                    x, incx, stride_x, AP, stride_AP, batch_count)
end function hipblasZsprStridedBatchedFortran

! spr2
function hipblasSspr2Fortran(handle, uplo, n, alpha, x, incx, &
                                y, incy, AP) &
    bind(c, name='hipblasSspr2Fortran')
    use iso_c_binding
    use hipblas_enums
    implicit none
    integer(kind(HIPBLAS_STATUS_SUCCESS)) :: hipblasSspr2Fortran
    type(c_ptr), value :: handle
    integer(kind(HIPBLAS_FILL_MODE_FULL)), value :: uplo
    integer(c_int), value :: n
    type(c_ptr), value :: alpha
    type(c_ptr), value :: x
    integer(c_int), value :: incx
    type(c_ptr), value :: y
    integer(c_int), value :: incy
    type(c_ptr), value :: AP
            hipblasSspr2Fortran = &
        hipblasSspr2(handle, uplo, n, alpha, &
                        x, incx, y, incy, AP)
end function hipblasSspr2Fortran

function hipblasDspr2Fortran(handle, uplo, n, alpha, x, incx, &
                                y, incy, AP) &
    bind(c, name='hipblasDspr2Fortran')
    use iso_c_binding
    use hipblas_enums
    implicit none
    integer(kind(HIPBLAS_STATUS_SUCCESS)) :: hipblasDspr2Fortran
    type(c_ptr), value :: handle
    integer(kind(HIPBLAS_FILL_MODE_FULL)), value :: uplo
    integer(c_int), value :: n
    type(c_ptr), value :: alpha
    type(c_ptr), value :: x
    integer(c_int), value :: incx
    type(c_ptr), value :: y
    integer(c_int), value :: incy
    type(c_ptr), value :: AP
            hipblasDspr2Fortran = &
        hipblasDspr2(handle, uplo, n, alpha, &
                        x, incx, y, incy, AP)
end function hipblasDspr2Fortran

! spr2Batched
function hipblasSspr2BatchedFortran(handle, uplo, n, alpha, x, incx, &
                                    y, incy, AP, batch_count) &
    bind(c, name='hipblasSspr2BatchedFortran')
    use iso_c_binding
    use hipblas_enums
    implicit none
    integer(kind(HIPBLAS_STATUS_SUCCESS)) :: hipblasSspr2BatchedFortran
    type(c_ptr), value :: handle
    integer(kind(HIPBLAS_FILL_MODE_FULL)), value :: uplo
    integer(c_int), value :: n
    type(c_ptr), value :: alpha
    type(c_ptr), value :: x
    integer(c_int), value :: incx
    type(c_ptr), value :: y
    integer(c_int), value :: incy
    type(c_ptr), value :: AP
    integer(c_int), value :: batch_count
            hipblasSspr2BatchedFortran = &
        hipblasSspr2Batched(handle, uplo, n, alpha, &
                            x, incx, y, incy, AP, batch_count)
end function hipblasSspr2BatchedFortran

function hipblasDspr2BatchedFortran(handle, uplo, n, alpha, x, incx, &
                                    y, incy, AP, batch_count) &
    bind(c, name='hipblasDspr2BatchedFortran')
    use iso_c_binding
    use hipblas_enums
    implicit none
    integer(kind(HIPBLAS_STATUS_SUCCESS)) :: hipblasDspr2BatchedFortran
    type(c_ptr), value :: handle
    integer(kind(HIPBLAS_FILL_MODE_FULL)), value :: uplo
    integer(c_int), value :: n
    type(c_ptr), value :: alpha
    type(c_ptr), value :: x
    integer(c_int), value :: incx
    type(c_ptr), value :: y
    integer(c_int), value :: incy
    type(c_ptr), value :: AP
    integer(c_int), value :: batch_count
            hipblasDspr2BatchedFortran = &
        hipblasDspr2Batched(handle, uplo, n, alpha, &
                            x, incx, y, incy, AP, batch_count)
end function hipblasDspr2BatchedFortran

! spr2StridedBatched
function hipblasSspr2StridedBatchedFortran(handle, uplo, n, alpha, x, incx, stride_x, &
                                            y, incy, stride_y, AP, stride_AP, batch_count) &
    bind(c, name='hipblasSspr2StridedBatchedFortran')
    use iso_c_binding
    use hipblas_enums
    implicit none
    integer(kind(HIPBLAS_STATUS_SUCCESS)) :: hipblasSspr2StridedBatchedFortran
    type(c_ptr), value :: handle
    integer(kind(HIPBLAS_FILL_MODE_FULL)), value :: uplo
    integer(c_int), value :: n
    type(c_ptr), value :: alpha
    type(c_ptr), value :: x
    integer(c_int), value :: incx
    integer(c_int64_t), value :: stride_x
    type(c_ptr), value :: y
    integer(c_int), value :: incy
    integer(c_int64_t), value :: stride_y
    type(c_ptr), value :: AP
    integer(c_int64_t), value :: stride_AP
    integer(c_int), value :: batch_count
            hipblasSspr2StridedBatchedFortran = &
        hipblasSspr2StridedBatched(handle, uplo, n, alpha, &
                                    x, incx, stride_x, y, incy, stride_y, AP, stride_AP, batch_count)
end function hipblasSspr2StridedBatchedFortran

function hipblasDspr2StridedBatchedFortran(handle, uplo, n, alpha, x, incx, stride_x, &
                                            y, incy, stride_y, AP, stride_AP, batch_count) &
    bind(c, name='hipblasDspr2StridedBatchedFortran')
    use iso_c_binding
    use hipblas_enums
    implicit none
    integer(kind(HIPBLAS_STATUS_SUCCESS)) :: hipblasDspr2StridedBatchedFortran
    type(c_ptr), value :: handle
    integer(kind(HIPBLAS_FILL_MODE_FULL)), value :: uplo
    integer(c_int), value :: n
    type(c_ptr), value :: alpha
    type(c_ptr), value :: x
    integer(c_int), value :: incx
    integer(c_int64_t), value :: stride_x
    type(c_ptr), value :: y
    integer(c_int), value :: incy
    integer(c_int64_t), value :: stride_y
    type(c_ptr), value :: AP
    integer(c_int64_t), value :: stride_AP
    integer(c_int), value :: batch_count
            hipblasDspr2StridedBatchedFortran = &
        hipblasDspr2StridedBatched(handle, uplo, n, alpha, &
                                    x, incx, stride_x, y, incy, stride_y, AP, stride_AP, batch_count)
end function hipblasDspr2StridedBatchedFortran

! syr
function hipblasSsyrFortran(handle, uplo, n, alpha, x, incx, A, lda) &
    bind(c, name='hipblasSsyrFortran')
    use iso_c_binding
    use hipblas_enums
    implicit none
    integer(kind(HIPBLAS_STATUS_SUCCESS)) :: hipblasSsyrFortran
    type(c_ptr), value :: handle
    integer(kind(HIPBLAS_FILL_MODE_FULL)), value :: uplo
    integer(c_int), value :: n
    type(c_ptr), value :: alpha
    type(c_ptr), value :: x
    integer(c_int), value :: incx
    type(c_ptr), value :: A
    integer(c_int), value :: lda
            hipblasSsyrFortran = &
        hipblasSsyr(handle, uplo, n, alpha, &
                    x, incx, A, lda)
end function hipblasSsyrFortran

function hipblasDsyrFortran(handle, uplo, n, alpha, x, incx, A, lda) &
    bind(c, name='hipblasDsyrFortran')
    use iso_c_binding
    use hipblas_enums
    implicit none
    integer(kind(HIPBLAS_STATUS_SUCCESS)) :: hipblasDsyrFortran
    type(c_ptr), value :: handle
    integer(kind(HIPBLAS_FILL_MODE_FULL)), value :: uplo
    integer(c_int), value :: n
    type(c_ptr), value :: alpha
    type(c_ptr), value :: x
    integer(c_int), value :: incx
    type(c_ptr), value :: A
    integer(c_int), value :: lda
            hipblasDsyrFortran = &
        hipblasDsyr(handle, uplo, n, alpha, &
                    x, incx, A, lda)
end function hipblasDsyrFortran

function hipblasCsyrFortran(handle, uplo, n, alpha, x, incx, A, lda) &
    bind(c, name='hipblasCsyrFortran')
    use iso_c_binding
    use hipblas_enums
    implicit none
    integer(kind(HIPBLAS_STATUS_SUCCESS)) :: hipblasCsyrFortran
    type(c_ptr), value :: handle
    integer(kind(HIPBLAS_FILL_MODE_FULL)), value :: uplo
    integer(c_int), value :: n
    type(c_ptr), value :: alpha
    type(c_ptr), value :: x
    integer(c_int), value :: incx
    type(c_ptr), value :: A
    integer(c_int), value :: lda
            hipblasCsyrFortran = &
        hipblasCsyr(handle, uplo, n, alpha, &
                    x, incx, A, lda)
end function hipblasCsyrFortran

function hipblasZsyrFortran(handle, uplo, n, alpha, x, incx, A, lda) &
    bind(c, name='hipblasZsyrFortran')
    use iso_c_binding
    use hipblas_enums
    implicit none
    integer(kind(HIPBLAS_STATUS_SUCCESS)) :: hipblasZsyrFortran
    type(c_ptr), value :: handle
    integer(kind(HIPBLAS_FILL_MODE_FULL)), value :: uplo
    integer(c_int), value :: n
    type(c_ptr), value :: alpha
    type(c_ptr), value :: x
    integer(c_int), value :: incx
    type(c_ptr), value :: A
    integer(c_int), value :: lda
            hipblasZsyrFortran = &
        hipblasZsyr(handle, uplo, n, alpha, &
                    x, incx, A, lda)
end function hipblasZsyrFortran

! syrBatched
function hipblasSsyrBatchedFortran(handle, uplo, n, alpha, x, incx, A, lda, batch_count) &
    bind(c, name='hipblasSsyrBatchedFortran')
    use iso_c_binding
    use hipblas_enums
    implicit none
    integer(kind(HIPBLAS_STATUS_SUCCESS)) :: hipblasSsyrBatchedFortran
    type(c_ptr), value :: handle
    integer(kind(HIPBLAS_FILL_MODE_FULL)), value :: uplo
    integer(c_int), value :: n
    type(c_ptr), value :: alpha
    type(c_ptr), value :: x
    integer(c_int), value :: incx
    type(c_ptr), value :: A
    integer(c_int), value :: lda
    integer(c_int), value :: batch_count
            hipblasSsyrBatchedFortran = &
        hipblasSsyrBatched(handle, uplo, n, alpha, &
                            x, incx, A, lda, batch_count)
end function hipblasSsyrBatchedFortran

function hipblasDsyrBatchedFortran(handle, uplo, n, alpha, x, incx, A, lda, batch_count) &
    bind(c, name='hipblasDsyrBatchedFortran')
    use iso_c_binding
    use hipblas_enums
    implicit none
    integer(kind(HIPBLAS_STATUS_SUCCESS)) :: hipblasDsyrBatchedFortran
    type(c_ptr), value :: handle
    integer(kind(HIPBLAS_FILL_MODE_FULL)), value :: uplo
    integer(c_int), value :: n
    type(c_ptr), value :: alpha
    type(c_ptr), value :: x
    integer(c_int), value :: incx
    type(c_ptr), value :: A
    integer(c_int), value :: lda
    integer(c_int), value :: batch_count
            hipblasDsyrBatchedFortran = &
        hipblasDsyrBatched(handle, uplo, n, alpha, &
                            x, incx, A, lda, batch_count)
end function hipblasDsyrBatchedFortran

function hipblasCsyrBatchedFortran(handle, uplo, n, alpha, x, incx, A, lda, batch_count) &
    bind(c, name='hipblasCsyrBatchedFortran')
    use iso_c_binding
    use hipblas_enums
    implicit none
    integer(kind(HIPBLAS_STATUS_SUCCESS)) :: hipblasCsyrBatchedFortran
    type(c_ptr), value :: handle
    integer(kind(HIPBLAS_FILL_MODE_FULL)), value :: uplo
    integer(c_int), value :: n
    type(c_ptr), value :: alpha
    type(c_ptr), value :: x
    integer(c_int), value :: incx
    type(c_ptr), value :: A
    integer(c_int), value :: lda
    integer(c_int), value :: batch_count
            hipblasCsyrBatchedFortran = &
        hipblasCsyrBatched(handle, uplo, n, alpha, &
                            x, incx, A, lda, batch_count)
end function hipblasCsyrBatchedFortran

function hipblasZsyrBatchedFortran(handle, uplo, n, alpha, x, incx, A, lda, batch_count) &
    bind(c, name='hipblasZsyrBatchedFortran')
    use iso_c_binding
    use hipblas_enums
    implicit none
    integer(kind(HIPBLAS_STATUS_SUCCESS)) :: hipblasZsyrBatchedFortran
    type(c_ptr), value :: handle
    integer(kind(HIPBLAS_FILL_MODE_FULL)), value :: uplo
    integer(c_int), value :: n
    type(c_ptr), value :: alpha
    type(c_ptr), value :: x
    integer(c_int), value :: incx
    type(c_ptr), value :: A
    integer(c_int), value :: lda
    integer(c_int), value :: batch_count
            hipblasZsyrBatchedFortran = &
        hipblasZsyrBatched(handle, uplo, n, alpha, &
                            x, incx, A, lda, batch_count)
end function hipblasZsyrBatchedFortran

! syrStridedBatched
function hipblasSsyrStridedBatchedFortran(handle, uplo, n, alpha, x, incx, stride_x, &
                                            A, lda, stride_A, batch_count) &
    bind(c, name='hipblasSsyrStridedBatchedFortran')
    use iso_c_binding
    use hipblas_enums
    implicit none
    integer(kind(HIPBLAS_STATUS_SUCCESS)) :: hipblasSsyrStridedBatchedFortran
    type(c_ptr), value :: handle
    integer(kind(HIPBLAS_FILL_MODE_FULL)), value :: uplo
    integer(c_int), value :: n
    type(c_ptr), value :: alpha
    type(c_ptr), value :: x
    integer(c_int), value :: incx
    integer(c_int64_t), value :: stride_x
    type(c_ptr), value :: A
    integer(c_int), value :: lda
    integer(c_int64_t), value :: stride_A
    integer(c_int), value :: batch_count
            hipblasSsyrStridedBatchedFortran = &
        hipblasSsyrStridedBatched(handle, uplo, n, alpha, &
                                    x, incx, stride_x, A, lda, stride_A, batch_count)
end function hipblasSsyrStridedBatchedFortran

function hipblasDsyrStridedBatchedFortran(handle, uplo, n, alpha, x, incx, stride_x, &
                                            A, lda, stride_A, batch_count) &
    bind(c, name='hipblasDsyrStridedBatchedFortran')
    use iso_c_binding
    use hipblas_enums
    implicit none
    integer(kind(HIPBLAS_STATUS_SUCCESS)) :: hipblasDsyrStridedBatchedFortran
    type(c_ptr), value :: handle
    integer(kind(HIPBLAS_FILL_MODE_FULL)), value :: uplo
    integer(c_int), value :: n
    type(c_ptr), value :: alpha
    type(c_ptr), value :: x
    integer(c_int), value :: incx
    integer(c_int64_t), value :: stride_x
    type(c_ptr), value :: A
    integer(c_int), value :: lda
    integer(c_int64_t), value :: stride_A
    integer(c_int), value :: batch_count
            hipblasDsyrStridedBatchedFortran = &
        hipblasDsyrStridedBatched(handle, uplo, n, alpha, &
                                    x, incx, stride_x, A, lda, stride_A, batch_count)
end function hipblasDsyrStridedBatchedFortran

function hipblasCsyrStridedBatchedFortran(handle, uplo, n, alpha, x, incx, stride_x, &
                                            A, lda, stride_A, batch_count) &
    bind(c, name='hipblasCsyrStridedBatchedFortran')
    use iso_c_binding
    use hipblas_enums
    implicit none
    integer(kind(HIPBLAS_STATUS_SUCCESS)) :: hipblasCsyrStridedBatchedFortran
    type(c_ptr), value :: handle
    integer(kind(HIPBLAS_FILL_MODE_FULL)), value :: uplo
    integer(c_int), value :: n
    type(c_ptr), value :: alpha
    type(c_ptr), value :: x
    integer(c_int), value :: incx
    integer(c_int64_t), value :: stride_x
    type(c_ptr), value :: A
    integer(c_int), value :: lda
    integer(c_int64_t), value :: stride_A
    integer(c_int), value :: batch_count
            hipblasCsyrStridedBatchedFortran = &
        hipblasCsyrStridedBatched(handle, uplo, n, alpha, &
                                    x, incx, stride_x, A, lda, stride_A, batch_count)
end function hipblasCsyrStridedBatchedFortran

function hipblasZsyrStridedBatchedFortran(handle, uplo, n, alpha, x, incx, stride_x, &
                                            A, lda, stride_A, batch_count) &
    bind(c, name='hipblasZsyrStridedBatchedFortran')
    use iso_c_binding
    use hipblas_enums
    implicit none
    integer(kind(HIPBLAS_STATUS_SUCCESS)) :: hipblasZsyrStridedBatchedFortran
    type(c_ptr), value :: handle
    integer(kind(HIPBLAS_FILL_MODE_FULL)), value :: uplo
    integer(c_int), value :: n
    type(c_ptr), value :: alpha
    type(c_ptr), value :: x
    integer(c_int), value :: incx
    integer(c_int64_t), value :: stride_x
    type(c_ptr), value :: A
    integer(c_int), value :: lda
    integer(c_int64_t), value :: stride_A
    integer(c_int), value :: batch_count
            hipblasZsyrStridedBatchedFortran = &
        hipblasZsyrStridedBatched(handle, uplo, n, alpha, &
                                    x, incx, stride_x, A, lda, stride_A, batch_count)
end function hipblasZsyrStridedBatchedFortran

! syr2
function hipblasSsyr2Fortran(handle, uplo, n, alpha, x, incx, &
                                y, incy, A, lda) &
    bind(c, name='hipblasSsyr2Fortran')
    use iso_c_binding
    use hipblas_enums
    implicit none
    integer(kind(HIPBLAS_STATUS_SUCCESS)) :: hipblasSsyr2Fortran
    type(c_ptr), value :: handle
    integer(kind(HIPBLAS_FILL_MODE_FULL)), value :: uplo
    integer(c_int), value :: n
    type(c_ptr), value :: alpha
    type(c_ptr), value :: x
    integer(c_int), value :: incx
    type(c_ptr), value :: y
    integer(c_int), value :: incy
    type(c_ptr), value :: A
    integer(c_int), value :: lda
            hipblasSsyr2Fortran = &
        hipblasSsyr2(handle, uplo, n, alpha, &
                        x, incx, y, incy, A, lda)
end function hipblasSsyr2Fortran

function hipblasDsyr2Fortran(handle, uplo, n, alpha, x, incx, &
                                y, incy, A, lda) &
    bind(c, name='hipblasDsyr2Fortran')
    use iso_c_binding
    use hipblas_enums
    implicit none
    integer(kind(HIPBLAS_STATUS_SUCCESS)) :: hipblasDsyr2Fortran
    type(c_ptr), value :: handle
    integer(kind(HIPBLAS_FILL_MODE_FULL)), value :: uplo
    integer(c_int), value :: n
    type(c_ptr), value :: alpha
    type(c_ptr), value :: x
    integer(c_int), value :: incx
    type(c_ptr), value :: y
    integer(c_int), value :: incy
    type(c_ptr), value :: A
    integer(c_int), value :: lda
            hipblasDsyr2Fortran = &
        hipblasDsyr2(handle, uplo, n, alpha, &
                        x, incx, y, incy, A, lda)
end function hipblasDsyr2Fortran

function hipblasCsyr2Fortran(handle, uplo, n, alpha, x, incx, &
                                y, incy, A, lda) &
    bind(c, name='hipblasCsyr2Fortran')
    use iso_c_binding
    use hipblas_enums
    implicit none
    integer(kind(HIPBLAS_STATUS_SUCCESS)) :: hipblasCsyr2Fortran
    type(c_ptr), value :: handle
    integer(kind(HIPBLAS_FILL_MODE_FULL)), value :: uplo
    integer(c_int), value :: n
    type(c_ptr), value :: alpha
    type(c_ptr), value :: x
    integer(c_int), value :: incx
    type(c_ptr), value :: y
    integer(c_int), value :: incy
    type(c_ptr), value :: A
    integer(c_int), value :: lda
            hipblasCsyr2Fortran = &
        hipblasCsyr2(handle, uplo, n, alpha, &
                        x, incx, y, incy, A, lda)
end function hipblasCsyr2Fortran

function hipblasZsyr2Fortran(handle, uplo, n, alpha, x, incx, &
                                y, incy, A, lda) &
    bind(c, name='hipblasZsyr2Fortran')
    use iso_c_binding
    use hipblas_enums
    implicit none
    integer(kind(HIPBLAS_STATUS_SUCCESS)) :: hipblasZsyr2Fortran
    type(c_ptr), value :: handle
    integer(kind(HIPBLAS_FILL_MODE_FULL)), value :: uplo
    integer(c_int), value :: n
    type(c_ptr), value :: alpha
    type(c_ptr), value :: x
    integer(c_int), value :: incx
    type(c_ptr), value :: y
    integer(c_int), value :: incy
    type(c_ptr), value :: A
    integer(c_int), value :: lda
            hipblasZsyr2Fortran = &
        hipblasZsyr2(handle, uplo, n, alpha, &
                        x, incx, y, incy, A, lda)
end function hipblasZsyr2Fortran

! syr2Batched
function hipblasSsyr2BatchedFortran(handle, uplo, n, alpha, x, incx, &
                                    y, incy, A, lda, batch_count) &
    bind(c, name='hipblasSsyr2BatchedFortran')
    use iso_c_binding
    use hipblas_enums
    implicit none
    integer(kind(HIPBLAS_STATUS_SUCCESS)) :: hipblasSsyr2BatchedFortran
    type(c_ptr), value :: handle
    integer(kind(HIPBLAS_FILL_MODE_FULL)), value :: uplo
    integer(c_int), value :: n
    type(c_ptr), value :: alpha
    type(c_ptr), value :: x
    integer(c_int), value :: incx
    type(c_ptr), value :: y
    integer(c_int), value :: incy
    type(c_ptr), value :: A
    integer(c_int), value :: lda
    integer(c_int), value :: batch_count
            hipblasSsyr2BatchedFortran = &
        hipblasSsyr2Batched(handle, uplo, n, alpha, &
                            x, incx, y, incy, A, lda, batch_count)
end function hipblasSsyr2BatchedFortran

function hipblasDsyr2BatchedFortran(handle, uplo, n, alpha, x, incx, &
                                    y, incy, A, lda, batch_count) &
    bind(c, name='hipblasDsyr2BatchedFortran')
    use iso_c_binding
    use hipblas_enums
    implicit none
    integer(kind(HIPBLAS_STATUS_SUCCESS)) :: hipblasDsyr2BatchedFortran
    type(c_ptr), value :: handle
    integer(kind(HIPBLAS_FILL_MODE_FULL)), value :: uplo
    integer(c_int), value :: n
    type(c_ptr), value :: alpha
    type(c_ptr), value :: x
    integer(c_int), value :: incx
    type(c_ptr), value :: y
    integer(c_int), value :: incy
    type(c_ptr), value :: A
    integer(c_int), value :: lda
    integer(c_int), value :: batch_count
            hipblasDsyr2BatchedFortran = &
        hipblasDsyr2Batched(handle, uplo, n, alpha, &
                            x, incx, y, incy, A, lda, batch_count)
end function hipblasDsyr2BatchedFortran

function hipblasCsyr2BatchedFortran(handle, uplo, n, alpha, x, incx, &
                                    y, incy, A, lda, batch_count) &
    bind(c, name='hipblasCsyr2BatchedFortran')
    use iso_c_binding
    use hipblas_enums
    implicit none
    integer(kind(HIPBLAS_STATUS_SUCCESS)) :: hipblasCsyr2BatchedFortran
    type(c_ptr), value :: handle
    integer(kind(HIPBLAS_FILL_MODE_FULL)), value :: uplo
    integer(c_int), value :: n
    type(c_ptr), value :: alpha
    type(c_ptr), value :: x
    integer(c_int), value :: incx
    type(c_ptr), value :: y
    integer(c_int), value :: incy
    type(c_ptr), value :: A
    integer(c_int), value :: lda
    integer(c_int), value :: batch_count
            hipblasCsyr2BatchedFortran = &
        hipblasCsyr2Batched(handle, uplo, n, alpha, &
                            x, incx, y, incy, A, lda, batch_count)
end function hipblasCsyr2BatchedFortran

function hipblasZsyr2BatchedFortran(handle, uplo, n, alpha, x, incx, &
                                    y, incy, A, lda, batch_count) &
    bind(c, name='hipblasZsyr2BatchedFortran')
    use iso_c_binding
    use hipblas_enums
    implicit none
    integer(kind(HIPBLAS_STATUS_SUCCESS)) :: hipblasZsyr2BatchedFortran
    type(c_ptr), value :: handle
    integer(kind(HIPBLAS_FILL_MODE_FULL)), value :: uplo
    integer(c_int), value :: n
    type(c_ptr), value :: alpha
    type(c_ptr), value :: x
    integer(c_int), value :: incx
    type(c_ptr), value :: y
    integer(c_int), value :: incy
    type(c_ptr), value :: A
    integer(c_int), value :: lda
    integer(c_int), value :: batch_count
            hipblasZsyr2BatchedFortran = &
        hipblasZsyr2Batched(handle, uplo, n, alpha, &
                            x, incx, y, incy, A, lda, batch_count)
end function hipblasZsyr2BatchedFortran

! syr2StridedBatched
function hipblasSsyr2StridedBatchedFortran(handle, uplo, n, alpha, x, incx, stride_x, &
                                            y, incy, stride_y, A, lda, stride_A, batch_count) &
    bind(c, name='hipblasSsyr2StridedBatchedFortran')
    use iso_c_binding
    use hipblas_enums
    implicit none
    integer(kind(HIPBLAS_STATUS_SUCCESS)) :: hipblasSsyr2StridedBatchedFortran
    type(c_ptr), value :: handle
    integer(kind(HIPBLAS_FILL_MODE_FULL)), value :: uplo
    integer(c_int), value :: n
    type(c_ptr), value :: alpha
    type(c_ptr), value :: x
    integer(c_int), value :: incx
    integer(c_int64_t), value :: stride_x
    type(c_ptr), value :: y
    integer(c_int), value :: incy
    integer(c_int64_t), value :: stride_y
    type(c_ptr), value :: A
    integer(c_int), value :: lda
    integer(c_int64_t), value :: stride_A
    integer(c_int), value :: batch_count
            hipblasSsyr2StridedBatchedFortran = &
        hipblasSsyr2StridedBatched(handle, uplo, n, alpha, &
                                    x, incx, stride_x, y, incy, stride_y, A, lda, stride_A, batch_count)
end function hipblasSsyr2StridedBatchedFortran

function hipblasDsyr2StridedBatchedFortran(handle, uplo, n, alpha, x, incx, stride_x, &
                                            y, incy, stride_y, A, lda, stride_A, batch_count) &
    bind(c, name='hipblasDsyr2StridedBatchedFortran')
    use iso_c_binding
    use hipblas_enums
    implicit none
    integer(kind(HIPBLAS_STATUS_SUCCESS)) :: hipblasDsyr2StridedBatchedFortran
    type(c_ptr), value :: handle
    integer(kind(HIPBLAS_FILL_MODE_FULL)), value :: uplo
    integer(c_int), value :: n
    type(c_ptr), value :: alpha
    type(c_ptr), value :: x
    integer(c_int), value :: incx
    integer(c_int64_t), value :: stride_x
    type(c_ptr), value :: y
    integer(c_int), value :: incy
    integer(c_int64_t), value :: stride_y
    type(c_ptr), value :: A
    integer(c_int), value :: lda
    integer(c_int64_t), value :: stride_A
    integer(c_int), value :: batch_count
            hipblasDsyr2StridedBatchedFortran = &
        hipblasDsyr2StridedBatched(handle, uplo, n, alpha, &
                                    x, incx, stride_x, y, incy, stride_y, A, lda, stride_A, batch_count)
end function hipblasDsyr2StridedBatchedFortran

function hipblasCsyr2StridedBatchedFortran(handle, uplo, n, alpha, x, incx, stride_x, &
                                            y, incy, stride_y, A, lda, stride_A, batch_count) &
    bind(c, name='hipblasCsyr2StridedBatchedFortran')
    use iso_c_binding
    use hipblas_enums
    implicit none
    integer(kind(HIPBLAS_STATUS_SUCCESS)) :: hipblasCsyr2StridedBatchedFortran
    type(c_ptr), value :: handle
    integer(kind(HIPBLAS_FILL_MODE_FULL)), value :: uplo
    integer(c_int), value :: n
    type(c_ptr), value :: alpha
    type(c_ptr), value :: x
    integer(c_int), value :: incx
    integer(c_int64_t), value :: stride_x
    type(c_ptr), value :: y
    integer(c_int), value :: incy
    integer(c_int64_t), value :: stride_y
    type(c_ptr), value :: A
    integer(c_int), value :: lda
    integer(c_int64_t), value :: stride_A
    integer(c_int), value :: batch_count
            hipblasCsyr2StridedBatchedFortran = &
        hipblasCsyr2StridedBatched(handle, uplo, n, alpha, &
                                    x, incx, stride_x, y, incy, stride_y, A, lda, stride_A, batch_count)
end function hipblasCsyr2StridedBatchedFortran

function hipblasZsyr2StridedBatchedFortran(handle, uplo, n, alpha, x, incx, stride_x, &
                                            y, incy, stride_y, A, lda, stride_A, batch_count) &
    bind(c, name='hipblasZsyr2StridedBatchedFortran')
    use iso_c_binding
    use hipblas_enums
    implicit none
    integer(kind(HIPBLAS_STATUS_SUCCESS)) :: hipblasZsyr2StridedBatchedFortran
    type(c_ptr), value :: handle
    integer(kind(HIPBLAS_FILL_MODE_FULL)), value :: uplo
    integer(c_int), value :: n
    type(c_ptr), value :: alpha
    type(c_ptr), value :: x
    integer(c_int), value :: incx
    integer(c_int64_t), value :: stride_x
    type(c_ptr), value :: y
    integer(c_int), value :: incy
    integer(c_int64_t), value :: stride_y
    type(c_ptr), value :: A
    integer(c_int), value :: lda
    integer(c_int64_t), value :: stride_A
    integer(c_int), value :: batch_count
            hipblasZsyr2StridedBatchedFortran = &
        hipblasZsyr2StridedBatched(handle, uplo, n, alpha, &
                                    x, incx, stride_x, y, incy, stride_y, A, lda, stride_A, batch_count)
end function hipblasZsyr2StridedBatchedFortran

! trsv
function hipblasStrsvFortran(handle, uplo, transA, diag, m, &
                                A, lda, x, incx) &
    bind(c, name='hipblasStrsvFortran')
    use iso_c_binding
    use hipblas_enums
    implicit none
    integer(kind(HIPBLAS_STATUS_SUCCESS)) :: hipblasStrsvFortran
    type(c_ptr), value :: handle
    integer(kind(HIPBLAS_FILL_MODE_FULL)), value :: uplo
    integer(kind(HIPBLAS_OP_N)), value :: transA
    integer(kind(HIPBLAS_DIAG_NON_UNIT)), value :: diag
    integer(c_int), value :: m
    type(c_ptr), value :: A
    integer(c_int), value :: lda
    type(c_ptr), value :: x
    integer(c_int), value :: incx
            hipblasStrsvFortran = &
        hipblasStrsv(handle, uplo, transA, diag, m, &
                        A, lda, x, incx)
end function hipblasStrsvFortran

function hipblasDtrsvFortran(handle, uplo, transA, diag, m, &
                                A, lda, x, incx) &
    bind(c, name='hipblasDtrsvFortran')
    use iso_c_binding
    use hipblas_enums
    implicit none
    integer(kind(HIPBLAS_STATUS_SUCCESS)) :: hipblasDtrsvFortran
    type(c_ptr), value :: handle
    integer(kind(HIPBLAS_FILL_MODE_FULL)), value :: uplo
    integer(kind(HIPBLAS_OP_N)), value :: transA
    integer(kind(HIPBLAS_DIAG_NON_UNIT)), value :: diag
    integer(c_int), value :: m
    type(c_ptr), value :: A
    integer(c_int), value :: lda
    type(c_ptr), value :: x
    integer(c_int), value :: incx
            hipblasDtrsvFortran = &
        hipblasDtrsv(handle, uplo, transA, diag, m, &
                        A, lda, x, incx)
end function hipblasDtrsvFortran

function hipblasCtrsvFortran(handle, uplo, transA, diag, m, &
                                A, lda, x, incx) &
    bind(c, name='hipblasCtrsvFortran')
    use iso_c_binding
    use hipblas_enums
    implicit none
    integer(kind(HIPBLAS_STATUS_SUCCESS)) :: hipblasCtrsvFortran
    type(c_ptr), value :: handle
    integer(kind(HIPBLAS_FILL_MODE_FULL)), value :: uplo
    integer(kind(HIPBLAS_OP_N)), value :: transA
    integer(kind(HIPBLAS_DIAG_NON_UNIT)), value :: diag
    integer(c_int), value :: m
    type(c_ptr), value :: A
    integer(c_int), value :: lda
    type(c_ptr), value :: x
    integer(c_int), value :: incx
            hipblasCtrsvFortran = &
        hipblasCtrsv(handle, uplo, transA, diag, m, &
                        A, lda, x, incx)
end function hipblasCtrsvFortran

function hipblasZtrsvFortran(handle, uplo, transA, diag, m, &
                                A, lda, x, incx) &
    bind(c, name='hipblasZtrsvFortran')
    use iso_c_binding
    use hipblas_enums
    implicit none
    integer(kind(HIPBLAS_STATUS_SUCCESS)) :: hipblasZtrsvFortran
    type(c_ptr), value :: handle
    integer(kind(HIPBLAS_FILL_MODE_FULL)), value :: uplo
    integer(kind(HIPBLAS_OP_N)), value :: transA
    integer(kind(HIPBLAS_DIAG_NON_UNIT)), value :: diag
    integer(c_int), value :: m
    type(c_ptr), value :: A
    integer(c_int), value :: lda
    type(c_ptr), value :: x
    integer(c_int), value :: incx
            hipblasZtrsvFortran = &
        hipblasZtrsv(handle, uplo, transA, diag, m, &
                        A, lda, x, incx)
end function hipblasZtrsvFortran

! trsvBatched
function hipblasStrsvBatchedFortran(handle, uplo, transA, diag, m, &
                                    A, lda, x, incx, batch_count) &
    bind(c, name='hipblasStrsvBatchedFortran')
    use iso_c_binding
    use hipblas_enums
    implicit none
    integer(kind(HIPBLAS_STATUS_SUCCESS)) :: hipblasStrsvBatchedFortran
    type(c_ptr), value :: handle
    integer(kind(HIPBLAS_FILL_MODE_FULL)), value :: uplo
    integer(kind(HIPBLAS_OP_N)), value :: transA
    integer(kind(HIPBLAS_DIAG_NON_UNIT)), value :: diag
    integer(c_int), value :: m
    type(c_ptr), value :: A
    integer(c_int), value :: lda
    type(c_ptr), value :: x
    integer(c_int), value :: incx
    integer(c_int), value :: batch_count
            hipblasStrsvBatchedFortran = &
        hipblasStrsvBatched(handle, uplo, transA, diag, m, &
                            A, lda, x, incx, batch_count)
end function hipblasStrsvBatchedFortran

function hipblasDtrsvBatchedFortran(handle, uplo, transA, diag, m, &
                                    A, lda, x, incx, batch_count) &
    bind(c, name='hipblasDtrsvBatchedFortran')
    use iso_c_binding
    use hipblas_enums
    implicit none
    integer(kind(HIPBLAS_STATUS_SUCCESS)) :: hipblasDtrsvBatchedFortran
    type(c_ptr), value :: handle
    integer(kind(HIPBLAS_FILL_MODE_FULL)), value :: uplo
    integer(kind(HIPBLAS_OP_N)), value :: transA
    integer(kind(HIPBLAS_DIAG_NON_UNIT)), value :: diag
    integer(c_int), value :: m
    type(c_ptr), value :: A
    integer(c_int), value :: lda
    type(c_ptr), value :: x
    integer(c_int), value :: incx
    integer(c_int), value :: batch_count
            hipblasDtrsvBatchedFortran = &
        hipblasDtrsvBatched(handle, uplo, transA, diag, m, &
                            A, lda, x, incx, batch_count)
end function hipblasDtrsvBatchedFortran

function hipblasCtrsvBatchedFortran(handle, uplo, transA, diag, m, &
                                    A, lda, x, incx, batch_count) &
    bind(c, name='hipblasCtrsvBatchedFortran')
    use iso_c_binding
    use hipblas_enums
    implicit none
    integer(kind(HIPBLAS_STATUS_SUCCESS)) :: hipblasCtrsvBatchedFortran
    type(c_ptr), value :: handle
    integer(kind(HIPBLAS_FILL_MODE_FULL)), value :: uplo
    integer(kind(HIPBLAS_OP_N)), value :: transA
    integer(kind(HIPBLAS_DIAG_NON_UNIT)), value :: diag
    integer(c_int), value :: m
    type(c_ptr), value :: A
    integer(c_int), value :: lda
    type(c_ptr), value :: x
    integer(c_int), value :: incx
    integer(c_int), value :: batch_count
            hipblasCtrsvBatchedFortran = &
        hipblasCtrsvBatched(handle, uplo, transA, diag, m, &
                            A, lda, x, incx, batch_count)
end function hipblasCtrsvBatchedFortran

function hipblasZtrsvBatchedFortran(handle, uplo, transA, diag, m, &
                                    A, lda, x, incx, batch_count) &
    bind(c, name='hipblasZtrsvBatchedFortran')
    use iso_c_binding
    use hipblas_enums
    implicit none
    integer(kind(HIPBLAS_STATUS_SUCCESS)) :: hipblasZtrsvBatchedFortran
    type(c_ptr), value :: handle
    integer(kind(HIPBLAS_FILL_MODE_FULL)), value :: uplo
    integer(kind(HIPBLAS_OP_N)), value :: transA
    integer(kind(HIPBLAS_DIAG_NON_UNIT)), value :: diag
    integer(c_int), value :: m
    type(c_ptr), value :: A
    integer(c_int), value :: lda
    type(c_ptr), value :: x
    integer(c_int), value :: incx
    integer(c_int), value :: batch_count
            hipblasZtrsvBatchedFortran = &
        hipblasZtrsvBatched(handle, uplo, transA, diag, m, &
                            A, lda, x, incx, batch_count)
end function hipblasZtrsvBatchedFortran

! trsvStridedBatched
function hipblasStrsvStridedBatchedFortran(handle, uplo, transA, diag, m, &
                                            A, lda, stride_A, x, incx, stride_x, batch_count) &
    bind(c, name='hipblasStrsvStridedBatchedFortran')
    use iso_c_binding
    use hipblas_enums
    implicit none
    integer(kind(HIPBLAS_STATUS_SUCCESS)) :: hipblasStrsvStridedBatchedFortran
    type(c_ptr), value :: handle
    integer(kind(HIPBLAS_FILL_MODE_FULL)), value :: uplo
    integer(kind(HIPBLAS_OP_N)), value :: transA
    integer(kind(HIPBLAS_DIAG_NON_UNIT)), value :: diag
    integer(c_int), value :: m
    type(c_ptr), value :: A
    integer(c_int), value :: lda
    integer(c_int64_t), value :: stride_A
    type(c_ptr), value :: x
    integer(c_int), value :: incx
    integer(c_int64_t), value :: stride_x
    integer(c_int), value :: batch_count
            hipblasStrsvStridedBatchedFortran = &
        hipblasStrsvStridedBatched(handle, uplo, transA, diag, m, &
                                    A, lda, stride_A, x, incx, stride_x, batch_count)
end function hipblasStrsvStridedBatchedFortran

function hipblasDtrsvStridedBatchedFortran(handle, uplo, transA, diag, m, &
                                            A, lda, stride_A, x, incx, stride_x, batch_count) &
    bind(c, name='hipblasDtrsvStridedBatchedFortran')
    use iso_c_binding
    use hipblas_enums
    implicit none
    integer(kind(HIPBLAS_STATUS_SUCCESS)) :: hipblasDtrsvStridedBatchedFortran
    type(c_ptr), value :: handle
    integer(kind(HIPBLAS_FILL_MODE_FULL)), value :: uplo
    integer(kind(HIPBLAS_OP_N)), value :: transA
    integer(kind(HIPBLAS_DIAG_NON_UNIT)), value :: diag
    integer(c_int), value :: m
    type(c_ptr), value :: A
    integer(c_int), value :: lda
    integer(c_int64_t), value :: stride_A
    type(c_ptr), value :: x
    integer(c_int), value :: incx
    integer(c_int64_t), value :: stride_x
    integer(c_int), value :: batch_count
            hipblasDtrsvStridedBatchedFortran = &
        hipblasDtrsvStridedBatched(handle, uplo, transA, diag, m, &
                                    A, lda, stride_A, x, incx, stride_x, batch_count)
end function hipblasDtrsvStridedBatchedFortran

function hipblasCtrsvStridedBatchedFortran(handle, uplo, transA, diag, m, &
                                            A, lda, stride_A, x, incx, stride_x, batch_count) &
    bind(c, name='hipblasCtrsvStridedBatchedFortran')
    use iso_c_binding
    use hipblas_enums
    implicit none
    integer(kind(HIPBLAS_STATUS_SUCCESS)) :: hipblasCtrsvStridedBatchedFortran
    type(c_ptr), value :: handle
    integer(kind(HIPBLAS_FILL_MODE_FULL)), value :: uplo
    integer(kind(HIPBLAS_OP_N)), value :: transA
    integer(kind(HIPBLAS_DIAG_NON_UNIT)), value :: diag
    integer(c_int), value :: m
    type(c_ptr), value :: A
    integer(c_int), value :: lda
    integer(c_int64_t), value :: stride_A
    type(c_ptr), value :: x
    integer(c_int), value :: incx
    integer(c_int64_t), value :: stride_x
    integer(c_int), value :: batch_count
            hipblasCtrsvStridedBatchedFortran = &
        hipblasCtrsvStridedBatched(handle, uplo, transA, diag, m, &
                                    A, lda, stride_A, x, incx, stride_x, batch_count)
end function hipblasCtrsvStridedBatchedFortran

function hipblasZtrsvStridedBatchedFortran(handle, uplo, transA, diag, m, &
                                            A, lda, stride_A, x, incx, stride_x, batch_count) &
    bind(c, name='hipblasZtrsvStridedBatchedFortran')
    use iso_c_binding
    use hipblas_enums
    implicit none
    integer(kind(HIPBLAS_STATUS_SUCCESS)) :: hipblasZtrsvStridedBatchedFortran
    type(c_ptr), value :: handle
    integer(kind(HIPBLAS_FILL_MODE_FULL)), value :: uplo
    integer(kind(HIPBLAS_OP_N)), value :: transA
    integer(kind(HIPBLAS_DIAG_NON_UNIT)), value :: diag
    integer(c_int), value :: m
    type(c_ptr), value :: A
    integer(c_int), value :: lda
    integer(c_int64_t), value :: stride_A
    type(c_ptr), value :: x
    integer(c_int), value :: incx
    integer(c_int64_t), value :: stride_x
    integer(c_int), value :: batch_count
            hipblasZtrsvStridedBatchedFortran = &
        hipblasZtrsvStridedBatched(handle, uplo, transA, diag, m, &
                                    A, lda, stride_A, x, incx, stride_x, batch_count)
end function hipblasZtrsvStridedBatchedFortran

!--------!
! blas 3 !
!--------!

! hemm
function hipblasChemmFortran(handle, side, uplo, n, k, alpha, &
                                A, lda, B, ldb, beta, C, ldc) &
    bind(c, name='hipblasChemmFortran')
    use iso_c_binding
    use hipblas_enums
    implicit none
    integer(kind(HIPBLAS_STATUS_SUCCESS)) :: hipblasChemmFortran
    type(c_ptr), value :: handle
    integer(kind(HIPBLAS_SIDE_LEFT)), value :: side
    integer(kind(HIPBLAS_FILL_MODE_FULL)), value :: uplo
    integer(c_int), value :: n
    integer(c_int), value :: k
    type(c_ptr), value :: alpha
    type(c_ptr), value :: A
    integer(c_int), value :: lda
    type(c_ptr), value :: B
    integer(c_int), value :: ldb
    type(c_ptr), value :: beta
    type(c_ptr), value :: C
    integer(c_int), value :: ldc
            hipblasChemmFortran = &
        hipblasChemm(handle, side, uplo, n, k, alpha, &
                        A, lda, B, ldb, beta, C, ldc)
end function hipblasChemmFortran

function hipblasZhemmFortran(handle, side, uplo, n, k, alpha, &
                                A, lda, B, ldb, beta, C, ldc) &
    bind(c, name='hipblasZhemmFortran')
    use iso_c_binding
    use hipblas_enums
    implicit none
    integer(kind(HIPBLAS_STATUS_SUCCESS)) :: hipblasZhemmFortran
    type(c_ptr), value :: handle
    integer(kind(HIPBLAS_SIDE_LEFT)), value :: side
    integer(kind(HIPBLAS_FILL_MODE_FULL)), value :: uplo
    integer(c_int), value :: n
    integer(c_int), value :: k
    type(c_ptr), value :: alpha
    type(c_ptr), value :: A
    integer(c_int), value :: lda
    type(c_ptr), value :: B
    integer(c_int), value :: ldb
    type(c_ptr), value :: beta
    type(c_ptr), value :: C
    integer(c_int), value :: ldc
            hipblasZhemmFortran = &
        hipblasZhemm(handle, side, uplo, n, k, alpha, &
                        A, lda, B, ldb, beta, C, ldc)
end function hipblasZhemmFortran

! hemmBatched
function hipblasChemmBatchedFortran(handle, side, uplo, n, k, alpha, &
                                    A, lda, B, ldb, beta, C, ldc, batch_count) &
    bind(c, name='hipblasChemmBatchedFortran')
    use iso_c_binding
    use hipblas_enums
    implicit none
    integer(kind(HIPBLAS_STATUS_SUCCESS)) :: hipblasChemmBatchedFortran
    type(c_ptr), value :: handle
    integer(kind(HIPBLAS_SIDE_LEFT)), value :: side
    integer(kind(HIPBLAS_FILL_MODE_FULL)), value :: uplo
    integer(c_int), value :: n
    integer(c_int), value :: k
    type(c_ptr), value :: alpha
    type(c_ptr), value :: A
    integer(c_int), value :: lda
    type(c_ptr), value :: B
    integer(c_int), value :: ldb
    type(c_ptr), value :: beta
    type(c_ptr), value :: C
    integer(c_int), value :: ldc
    integer(c_int), value :: batch_count
            hipblasChemmBatchedFortran = &
        hipblasChemmBatched(handle, side, uplo, n, k, alpha, &
                            A, lda, B, ldb, beta, C, ldc, batch_count)
end function hipblasChemmBatchedFortran

function hipblasZhemmBatchedFortran(handle, side, uplo, n, k, alpha, &
                                    A, lda, B, ldb, beta, C, ldc, batch_count) &
    bind(c, name='hipblasZhemmBatchedFortran')
    use iso_c_binding
    use hipblas_enums
    implicit none
    integer(kind(HIPBLAS_STATUS_SUCCESS)) :: hipblasZhemmBatchedFortran
    type(c_ptr), value :: handle
    integer(kind(HIPBLAS_SIDE_LEFT)), value :: side
    integer(kind(HIPBLAS_FILL_MODE_FULL)), value :: uplo
    integer(c_int), value :: n
    integer(c_int), value :: k
    type(c_ptr), value :: alpha
    type(c_ptr), value :: A
    integer(c_int), value :: lda
    type(c_ptr), value :: B
    integer(c_int), value :: ldb
    type(c_ptr), value :: beta
    type(c_ptr), value :: C
    integer(c_int), value :: ldc
    integer(c_int), value :: batch_count
            hipblasZhemmBatchedFortran = &
        hipblasZhemmBatched(handle, side, uplo, n, k, alpha, &
                            A, lda, B, ldb, beta, C, ldc, batch_count)
end function hipblasZhemmBatchedFortran

! hemmStridedBatched
function hipblasChemmStridedBatchedFortran(handle, side, uplo, n, k, alpha, &
                                            A, lda, stride_A, B, ldb, stride_B, beta, C, ldc, stride_C, batch_count) &
    bind(c, name='hipblasChemmStridedBatchedFortran')
    use iso_c_binding
    use hipblas_enums
    implicit none
    integer(kind(HIPBLAS_STATUS_SUCCESS)) :: hipblasChemmStridedBatchedFortran
    type(c_ptr), value :: handle
    integer(kind(HIPBLAS_SIDE_LEFT)), value :: side
    integer(kind(HIPBLAS_FILL_MODE_FULL)), value :: uplo
    integer(c_int), value :: n
    integer(c_int), value :: k
    type(c_ptr), value :: alpha
    type(c_ptr), value :: A
    integer(c_int), value :: lda
    integer(c_int64_t), value :: stride_A
    type(c_ptr), value :: B
    integer(c_int), value :: ldb
    integer(c_int64_t), value :: stride_B
    type(c_ptr), value :: beta
    type(c_ptr), value :: C
    integer(c_int), value :: ldc
    integer(c_int64_t), value :: stride_C
    integer(c_int), value :: batch_count
            hipblasChemmStridedBatchedFortran = &
        hipblasChemmStridedBatched(handle, side, uplo, n, k, alpha, &
                                    A, lda, stride_A, B, ldb, stride_B, beta, C, ldc, stride_C, batch_count)
end function hipblasChemmStridedBatchedFortran

function hipblasZhemmStridedBatchedFortran(handle, side, uplo, n, k, alpha, &
                                            A, lda, stride_A, B, ldb, stride_B, beta, C, ldc, stride_C, batch_count) &
    bind(c, name='hipblasZhemmStridedBatchedFortran')
    use iso_c_binding
    use hipblas_enums
    implicit none
    integer(kind(HIPBLAS_STATUS_SUCCESS)) :: hipblasZhemmStridedBatchedFortran
    type(c_ptr), value :: handle
    integer(kind(HIPBLAS_SIDE_LEFT)), value :: side
    integer(kind(HIPBLAS_FILL_MODE_FULL)), value :: uplo
    integer(c_int), value :: n
    integer(c_int), value :: k
    type(c_ptr), value :: alpha
    type(c_ptr), value :: A
    integer(c_int), value :: lda
    integer(c_int64_t), value :: stride_A
    type(c_ptr), value :: B
    integer(c_int), value :: ldb
    integer(c_int64_t), value :: stride_B
    type(c_ptr), value :: beta
    type(c_ptr), value :: C
    integer(c_int), value :: ldc
    integer(c_int64_t), value :: stride_C
    integer(c_int), value :: batch_count
            hipblasZhemmStridedBatchedFortran = &
        hipblasZhemmStridedBatched(handle, side, uplo, n, k, alpha, &
                                    A, lda, stride_A, B, ldb, stride_B, beta, C, ldc, stride_C, batch_count)
end function hipblasZhemmStridedBatchedFortran

! herk
function hipblasCherkFortran(handle, uplo, transA, n, k, alpha, &
                                A, lda, beta, C, ldc) &
    bind(c, name='hipblasCherkFortran')
    use iso_c_binding
    use hipblas_enums
    implicit none
    integer(kind(HIPBLAS_STATUS_SUCCESS)) :: hipblasCherkFortran
    type(c_ptr), value :: handle
    integer(kind(HIPBLAS_FILL_MODE_FULL)), value :: uplo
    integer(kind(HIPBLAS_OP_N)), value :: transA
    integer(c_int), value :: n
    integer(c_int), value :: k
    type(c_ptr), value :: alpha
    type(c_ptr), value :: A
    integer(c_int), value :: lda
    type(c_ptr), value :: beta
    type(c_ptr), value :: C
    integer(c_int), value :: ldc
            hipblasCherkFortran = &
        hipblasCherk(handle, uplo, transA, n, k, alpha, &
                        A, lda, beta, C, ldc)
end function hipblasCherkFortran

function hipblasZherkFortran(handle, uplo, transA, n, k, alpha, &
                                A, lda, beta, C, ldc) &
    bind(c, name='hipblasZherkFortran')
    use iso_c_binding
    use hipblas_enums
    implicit none
    integer(kind(HIPBLAS_STATUS_SUCCESS)) :: hipblasZherkFortran
    type(c_ptr), value :: handle
    integer(kind(HIPBLAS_FILL_MODE_FULL)), value :: uplo
    integer(kind(HIPBLAS_OP_N)), value :: transA
    integer(c_int), value :: n
    integer(c_int), value :: k
    type(c_ptr), value :: alpha
    type(c_ptr), value :: A
    integer(c_int), value :: lda
    type(c_ptr), value :: beta
    type(c_ptr), value :: C
    integer(c_int), value :: ldc
            hipblasZherkFortran = &
        hipblasZherk(handle, uplo, transA, n, k, alpha, &
                        A, lda, beta, C, ldc)
end function hipblasZherkFortran

! herkBatched
function hipblasCherkBatchedFortran(handle, uplo, transA, n, k, alpha, &
                                    A, lda, beta, C, ldc, batch_count) &
    bind(c, name='hipblasCherkBatchedFortran')
    use iso_c_binding
    use hipblas_enums
    implicit none
    integer(kind(HIPBLAS_STATUS_SUCCESS)) :: hipblasCherkBatchedFortran
    type(c_ptr), value :: handle
    integer(kind(HIPBLAS_FILL_MODE_FULL)), value :: uplo
    integer(kind(HIPBLAS_OP_N)), value :: transA
    integer(c_int), value :: n
    integer(c_int), value :: k
    type(c_ptr), value :: alpha
    type(c_ptr), value :: A
    integer(c_int), value :: lda
    type(c_ptr), value :: beta
    type(c_ptr), value :: C
    integer(c_int), value :: ldc
    integer(c_int), value :: batch_count
            hipblasCherkBatchedFortran = &
        hipblasCherkBatched(handle, uplo, transA, n, k, alpha, &
                            A, lda, beta, C, ldc, batch_count)
end function hipblasCherkBatchedFortran

function hipblasZherkBatchedFortran(handle, uplo, transA, n, k, alpha, &
                                    A, lda, beta, C, ldc, batch_count) &
    bind(c, name='hipblasZherkBatchedFortran')
    use iso_c_binding
    use hipblas_enums
    implicit none
    integer(kind(HIPBLAS_STATUS_SUCCESS)) :: hipblasZherkBatchedFortran
    type(c_ptr), value :: handle
    integer(kind(HIPBLAS_FILL_MODE_FULL)), value :: uplo
    integer(kind(HIPBLAS_OP_N)), value :: transA
    integer(c_int), value :: n
    integer(c_int), value :: k
    type(c_ptr), value :: alpha
    type(c_ptr), value :: A
    integer(c_int), value :: lda
    type(c_ptr), value :: beta
    type(c_ptr), value :: C
    integer(c_int), value :: ldc
    integer(c_int), value :: batch_count
            hipblasZherkBatchedFortran = &
        hipblasZherkBatched(handle, uplo, transA, n, k, alpha, &
                            A, lda, beta, C, ldc, batch_count)
end function hipblasZherkBatchedFortran

! herkStridedBatched
function hipblasCherkStridedBatchedFortran(handle, uplo, transA, n, k, alpha, &
                                            A, lda, stride_A, beta, C, ldc, stride_C, batch_count) &
    bind(c, name='hipblasCherkStridedBatchedFortran')
    use iso_c_binding
    use hipblas_enums
    implicit none
    integer(kind(HIPBLAS_STATUS_SUCCESS)) :: hipblasCherkStridedBatchedFortran
    type(c_ptr), value :: handle
    integer(kind(HIPBLAS_FILL_MODE_FULL)), value :: uplo
    integer(kind(HIPBLAS_OP_N)), value :: transA
    integer(c_int), value :: n
    integer(c_int), value :: k
    type(c_ptr), value :: alpha
    type(c_ptr), value :: A
    integer(c_int), value :: lda
    integer(c_int64_t), value :: stride_A
    type(c_ptr), value :: beta
    type(c_ptr), value :: C
    integer(c_int), value :: ldc
    integer(c_int64_t), value :: stride_C
    integer(c_int), value :: batch_count
            hipblasCherkStridedBatchedFortran = &
        hipblasCherkStridedBatched(handle, uplo, transA, n, k, alpha, &
                                    A, lda, stride_A, beta, C, ldc, stride_C, batch_count)
end function hipblasCherkStridedBatchedFortran

function hipblasZherkStridedBatchedFortran(handle, uplo, transA, n, k, alpha, &
                                            A, lda, stride_A, beta, C, ldc, stride_C, batch_count) &
    bind(c, name='hipblasZherkStridedBatchedFortran')
    use iso_c_binding
    use hipblas_enums
    implicit none
    integer(kind(HIPBLAS_STATUS_SUCCESS)) :: hipblasZherkStridedBatchedFortran
    type(c_ptr), value :: handle
    integer(kind(HIPBLAS_FILL_MODE_FULL)), value :: uplo
    integer(kind(HIPBLAS_OP_N)), value :: transA
    integer(c_int), value :: n
    integer(c_int), value :: k
    type(c_ptr), value :: alpha
    type(c_ptr), value :: A
    integer(c_int), value :: lda
    integer(c_int64_t), value :: stride_A
    type(c_ptr), value :: beta
    type(c_ptr), value :: C
    integer(c_int), value :: ldc
    integer(c_int64_t), value :: stride_C
    integer(c_int), value :: batch_count
            hipblasZherkStridedBatchedFortran = &
        hipblasZherkStridedBatched(handle, uplo, transA, n, k, alpha, &
                                    A, lda, stride_A, beta, C, ldc, stride_C, batch_count)
end function hipblasZherkStridedBatchedFortran

! her2k
function hipblasCher2kFortran(handle, uplo, transA, n, k, alpha, &
                                A, lda, B, ldb, beta, C, ldc) &
    bind(c, name='hipblasCher2kFortran')
    use iso_c_binding
    use hipblas_enums
    implicit none
    integer(kind(HIPBLAS_STATUS_SUCCESS)) :: hipblasCher2kFortran
    type(c_ptr), value :: handle
    integer(kind(HIPBLAS_FILL_MODE_FULL)), value :: uplo
    integer(kind(HIPBLAS_OP_N)), value :: transA
    integer(c_int), value :: n
    integer(c_int), value :: k
    type(c_ptr), value :: alpha
    type(c_ptr), value :: A
    integer(c_int), value :: lda
    type(c_ptr), value :: B
    integer(c_int), value :: ldb
    type(c_ptr), value :: beta
    type(c_ptr), value :: C
    integer(c_int), value :: ldc
            hipblasCher2kFortran = &
        hipblasCher2k(handle, uplo, transA, n, k, alpha, &
                        A, lda, B, ldb, beta, C, ldc)
end function hipblasCher2kFortran

function hipblasZher2kFortran(handle, uplo, transA, n, k, alpha, &
                                A, lda, B, ldb, beta, C, ldc) &
    bind(c, name='hipblasZher2kFortran')
    use iso_c_binding
    use hipblas_enums
    implicit none
    integer(kind(HIPBLAS_STATUS_SUCCESS)) :: hipblasZher2kFortran
    type(c_ptr), value :: handle
    integer(kind(HIPBLAS_FILL_MODE_FULL)), value :: uplo
    integer(kind(HIPBLAS_OP_N)), value :: transA
    integer(c_int), value :: n
    integer(c_int), value :: k
    type(c_ptr), value :: alpha
    type(c_ptr), value :: A
    integer(c_int), value :: lda
    type(c_ptr), value :: B
    integer(c_int), value :: ldb
    type(c_ptr), value :: beta
    type(c_ptr), value :: C
    integer(c_int), value :: ldc
            hipblasZher2kFortran = &
        hipblasZher2k(handle, uplo, transA, n, k, alpha, &
                        A, lda, B, ldb, beta, C, ldc)
end function hipblasZher2kFortran

! her2kBatched
function hipblasCher2kBatchedFortran(handle, uplo, transA, n, k, alpha, &
                                        A, lda, B, ldb, beta, C, ldc, batch_count) &
    bind(c, name='hipblasCher2kBatchedFortran')
    use iso_c_binding
    use hipblas_enums
    implicit none
    integer(kind(HIPBLAS_STATUS_SUCCESS)) :: hipblasCher2kBatchedFortran
    type(c_ptr), value :: handle
    integer(kind(HIPBLAS_FILL_MODE_FULL)), value :: uplo
    integer(kind(HIPBLAS_OP_N)), value :: transA
    integer(c_int), value :: n
    integer(c_int), value :: k
    type(c_ptr), value :: alpha
    type(c_ptr), value :: A
    integer(c_int), value :: lda
    type(c_ptr), value :: B
    integer(c_int), value :: ldb
    type(c_ptr), value :: beta
    type(c_ptr), value :: C
    integer(c_int), value :: ldc
    integer(c_int), value :: batch_count
            hipblasCher2kBatchedFortran = &
        hipblasCher2kBatched(handle, uplo, transA, n, k, alpha, &
                                A, lda, B, ldb, beta, C, ldc, batch_count)
end function hipblasCher2kBatchedFortran

function hipblasZher2kBatchedFortran(handle, uplo, transA, n, k, alpha, &
                                        A, lda, B, ldb, beta, C, ldc, batch_count) &
    bind(c, name='hipblasZher2kBatchedFortran')
    use iso_c_binding
    use hipblas_enums
    implicit none
    integer(kind(HIPBLAS_STATUS_SUCCESS)) :: hipblasZher2kBatchedFortran
    type(c_ptr), value :: handle
    integer(kind(HIPBLAS_FILL_MODE_FULL)), value :: uplo
    integer(kind(HIPBLAS_OP_N)), value :: transA
    integer(c_int), value :: n
    integer(c_int), value :: k
    type(c_ptr), value :: alpha
    type(c_ptr), value :: A
    integer(c_int), value :: lda
    type(c_ptr), value :: B
    integer(c_int), value :: ldb
    type(c_ptr), value :: beta
    type(c_ptr), value :: C
    integer(c_int), value :: ldc
    integer(c_int), value :: batch_count
            hipblasZher2kBatchedFortran = &
        hipblasZher2kBatched(handle, uplo, transA, n, k, alpha, &
                                A, lda, B, ldb, beta, C, ldc, batch_count)
end function hipblasZher2kBatchedFortran

! her2kStridedBatched
function hipblasCher2kStridedBatchedFortran(handle, uplo, transA, n, k, alpha, &
                                            A, lda, stride_A, B, ldb, stride_B, beta, C, ldc, stride_C, batch_count) &
    bind(c, name='hipblasCher2kStridedBatchedFortran')
    use iso_c_binding
    use hipblas_enums
    implicit none
    integer(kind(HIPBLAS_STATUS_SUCCESS)) :: hipblasCher2kStridedBatchedFortran
    type(c_ptr), value :: handle
    integer(kind(HIPBLAS_FILL_MODE_FULL)), value :: uplo
    integer(kind(HIPBLAS_OP_N)), value :: transA
    integer(c_int), value :: n
    integer(c_int), value :: k
    type(c_ptr), value :: alpha
    type(c_ptr), value :: A
    integer(c_int), value :: lda
    integer(c_int64_t), value :: stride_A
    type(c_ptr), value :: B
    integer(c_int), value :: ldb
    integer(c_int64_t), value :: stride_B
    type(c_ptr), value :: beta
    type(c_ptr), value :: C
    integer(c_int), value :: ldc
    integer(c_int64_t), value :: stride_C
    integer(c_int), value :: batch_count
            hipblasCher2kStridedBatchedFortran = &
        hipblasCher2kStridedBatched(handle, uplo, transA, n, k, alpha, &
                                    A, lda, stride_A, B, ldb, stride_B, beta, C, ldc, stride_C, batch_count)
end function hipblasCher2kStridedBatchedFortran

function hipblasZher2kStridedBatchedFortran(handle, uplo, transA, n, k, alpha, &
                                            A, lda, stride_A, B, ldb, stride_B, beta, C, ldc, stride_C, batch_count) &
    bind(c, name='hipblasZher2kStridedBatchedFortran')
    use iso_c_binding
    use hipblas_enums
    implicit none
    integer(kind(HIPBLAS_STATUS_SUCCESS)) :: hipblasZher2kStridedBatchedFortran
    type(c_ptr), value :: handle
    integer(kind(HIPBLAS_FILL_MODE_FULL)), value :: uplo
    integer(kind(HIPBLAS_OP_N)), value :: transA
    integer(c_int), value :: n
    integer(c_int), value :: k
    type(c_ptr), value :: alpha
    type(c_ptr), value :: A
    integer(c_int), value :: lda
    integer(c_int64_t), value :: stride_A
    type(c_ptr), value :: B
    integer(c_int), value :: ldb
    integer(c_int64_t), value :: stride_B
    type(c_ptr), value :: beta
    type(c_ptr), value :: C
    integer(c_int), value :: ldc
    integer(c_int64_t), value :: stride_C
    integer(c_int), value :: batch_count
            hipblasZher2kStridedBatchedFortran = &
        hipblasZher2kStridedBatched(handle, uplo, transA, n, k, alpha, &
                                    A, lda, stride_A, B, ldb, stride_B, beta, C, ldc, stride_C, batch_count)
end function hipblasZher2kStridedBatchedFortran

! herkx
function hipblasCherkxFortran(handle, uplo, transA, n, k, alpha, &
                                A, lda, B, ldb, beta, C, ldc) &
    bind(c, name='hipblasCherkxFortran')
    use iso_c_binding
    use hipblas_enums
    implicit none
    integer(kind(HIPBLAS_STATUS_SUCCESS)) :: hipblasCherkxFortran
    type(c_ptr), value :: handle
    integer(kind(HIPBLAS_FILL_MODE_FULL)), value :: uplo
    integer(kind(HIPBLAS_OP_N)), value :: transA
    integer(c_int), value :: n
    integer(c_int), value :: k
    type(c_ptr), value :: alpha
    type(c_ptr), value :: A
    integer(c_int), value :: lda
    type(c_ptr), value :: B
    integer(c_int), value :: ldb
    type(c_ptr), value :: beta
    type(c_ptr), value :: C
    integer(c_int), value :: ldc
            hipblasCherkxFortran = &
        hipblasCherkx(handle, uplo, transA, n, k, alpha, &
                        A, lda, B, ldb, beta, C, ldc)
end function hipblasCherkxFortran

function hipblasZherkxFortran(handle, uplo, transA, n, k, alpha, &
                                A, lda, B, ldb, beta, C, ldc) &
    bind(c, name='hipblasZherkxFortran')
    use iso_c_binding
    use hipblas_enums
    implicit none
    integer(kind(HIPBLAS_STATUS_SUCCESS)) :: hipblasZherkxFortran
    type(c_ptr), value :: handle
    integer(kind(HIPBLAS_FILL_MODE_FULL)), value :: uplo
    integer(kind(HIPBLAS_OP_N)), value :: transA
    integer(c_int), value :: n
    integer(c_int), value :: k
    type(c_ptr), value :: alpha
    type(c_ptr), value :: A
    integer(c_int), value :: lda
    type(c_ptr), value :: B
    integer(c_int), value :: ldb
    type(c_ptr), value :: beta
    type(c_ptr), value :: C
    integer(c_int), value :: ldc
            hipblasZherkxFortran = &
        hipblasZherkx(handle, uplo, transA, n, k, alpha, &
                        A, lda, B, ldb, beta, C, ldc)
end function hipblasZherkxFortran

! herkxBatched
function hipblasCherkxBatchedFortran(handle, uplo, transA, n, k, alpha, &
                                        A, lda, B, ldb, beta, C, ldc, batch_count) &
    bind(c, name='hipblasCherkxBatchedFortran')
    use iso_c_binding
    use hipblas_enums
    implicit none
    integer(kind(HIPBLAS_STATUS_SUCCESS)) :: hipblasCherkxBatchedFortran
    type(c_ptr), value :: handle
    integer(kind(HIPBLAS_FILL_MODE_FULL)), value :: uplo
    integer(kind(HIPBLAS_OP_N)), value :: transA
    integer(c_int), value :: n
    integer(c_int), value :: k
    type(c_ptr), value :: alpha
    type(c_ptr), value :: A
    integer(c_int), value :: lda
    type(c_ptr), value :: B
    integer(c_int), value :: ldb
    type(c_ptr), value :: beta
    type(c_ptr), value :: C
    integer(c_int), value :: ldc
    integer(c_int), value :: batch_count
            hipblasCherkxBatchedFortran = &
        hipblasCherkxBatched(handle, uplo, transA, n, k, alpha, &
                                A, lda, B, ldb, beta, C, ldc, batch_count)
end function hipblasCherkxBatchedFortran

function hipblasZherkxBatchedFortran(handle, uplo, transA, n, k, alpha, &
                                        A, lda, B, ldb, beta, C, ldc, batch_count) &
    bind(c, name='hipblasZherkxBatchedFortran')
    use iso_c_binding
    use hipblas_enums
    implicit none
    integer(kind(HIPBLAS_STATUS_SUCCESS)) :: hipblasZherkxBatchedFortran
    type(c_ptr), value :: handle
    integer(kind(HIPBLAS_FILL_MODE_FULL)), value :: uplo
    integer(kind(HIPBLAS_OP_N)), value :: transA
    integer(c_int), value :: n
    integer(c_int), value :: k
    type(c_ptr), value :: alpha
    type(c_ptr), value :: A
    integer(c_int), value :: lda
    type(c_ptr), value :: B
    integer(c_int), value :: ldb
    type(c_ptr), value :: beta
    type(c_ptr), value :: C
    integer(c_int), value :: ldc
    integer(c_int), value :: batch_count
            hipblasZherkxBatchedFortran = &
        hipblasZherkxBatched(handle, uplo, transA, n, k, alpha, &
                                A, lda, B, ldb, beta, C, ldc, batch_count)
end function hipblasZherkxBatchedFortran

! herkxStridedBatched
function hipblasCherkxStridedBatchedFortran(handle, uplo, transA, n, k, alpha, &
                                            A, lda, stride_A, B, ldb, stride_B, beta, C, ldc, stride_C, batch_count) &
    bind(c, name='hipblasCherkxStridedBatchedFortran')
    use iso_c_binding
    use hipblas_enums
    implicit none
    integer(kind(HIPBLAS_STATUS_SUCCESS)) :: hipblasCherkxStridedBatchedFortran
    type(c_ptr), value :: handle
    integer(kind(HIPBLAS_FILL_MODE_FULL)), value :: uplo
    integer(kind(HIPBLAS_OP_N)), value :: transA
    integer(c_int), value :: n
    integer(c_int), value :: k
    type(c_ptr), value :: alpha
    type(c_ptr), value :: A
    integer(c_int), value :: lda
    integer(c_int64_t), value :: stride_A
    type(c_ptr), value :: B
    integer(c_int), value :: ldb
    integer(c_int64_t), value :: stride_B
    type(c_ptr), value :: beta
    type(c_ptr), value :: C
    integer(c_int), value :: ldc
    integer(c_int64_t), value :: stride_C
    integer(c_int), value :: batch_count
            hipblasCherkxStridedBatchedFortran = &
        hipblasCherkxStridedBatched(handle, uplo, transA, n, k, alpha, &
                                    A, lda, stride_A, B, ldb, stride_B, beta, C, ldc, stride_C, batch_count)
end function hipblasCherkxStridedBatchedFortran

function hipblasZherkxStridedBatchedFortran(handle, uplo, transA, n, k, alpha, &
                                            A, lda, stride_A, B, ldb, stride_B, beta, C, ldc, stride_C, batch_count) &
    bind(c, name='hipblasZherkxStridedBatchedFortran')
    use iso_c_binding
    use hipblas_enums
    implicit none
    integer(kind(HIPBLAS_STATUS_SUCCESS)) :: hipblasZherkxStridedBatchedFortran
    type(c_ptr), value :: handle
    integer(kind(HIPBLAS_FILL_MODE_FULL)), value :: uplo
    integer(kind(HIPBLAS_OP_N)), value :: transA
    integer(c_int), value :: n
    integer(c_int), value :: k
    type(c_ptr), value :: alpha
    type(c_ptr), value :: A
    integer(c_int), value :: lda
    integer(c_int64_t), value :: stride_A
    type(c_ptr), value :: B
    integer(c_int), value :: ldb
    integer(c_int64_t), value :: stride_B
    type(c_ptr), value :: beta
    type(c_ptr), value :: C
    integer(c_int), value :: ldc
    integer(c_int64_t), value :: stride_C
    integer(c_int), value :: batch_count
            hipblasZherkxStridedBatchedFortran = &
        hipblasZherkxStridedBatched(handle, uplo, transA, n, k, alpha, &
                                    A, lda, stride_A, B, ldb, stride_B, beta, C, ldc, stride_C, batch_count)
end function hipblasZherkxStridedBatchedFortran

! symm
function hipblasSsymmFortran(handle, side, uplo, m, n, alpha, &
                                A, lda, B, ldb, beta, C, ldc) &
    bind(c, name='hipblasSsymmFortran')
    use iso_c_binding
    use hipblas_enums
    implicit none
    integer(kind(HIPBLAS_STATUS_SUCCESS)) :: hipblasSsymmFortran
    type(c_ptr), value :: handle
    integer(kind(HIPBLAS_SIDE_LEFT)), value :: side
    integer(kind(HIPBLAS_FILL_MODE_FULL)), value :: uplo
    integer(c_int), value :: m
    integer(c_int), value :: n
    type(c_ptr), value :: alpha
    type(c_ptr), value :: A
    integer(c_int), value :: lda
    type(c_ptr), value :: B
    integer(c_int), value :: ldb
    type(c_ptr), value :: beta
    type(c_ptr), value :: C
    integer(c_int), value :: ldc
            hipblasSsymmFortran = &
        hipblasSsymm(handle, side, uplo, m, n, alpha, &
                        A, lda, B, ldb, beta, C, ldc)
end function hipblasSsymmFortran

function hipblasDsymmFortran(handle, side, uplo, m, n, alpha, &
                                A, lda, B, ldb, beta, C, ldc) &
    bind(c, name='hipblasDsymmFortran')
    use iso_c_binding
    use hipblas_enums
    implicit none
    integer(kind(HIPBLAS_STATUS_SUCCESS)) :: hipblasDsymmFortran
    type(c_ptr), value :: handle
    integer(kind(HIPBLAS_SIDE_LEFT)), value :: side
    integer(kind(HIPBLAS_FILL_MODE_FULL)), value :: uplo
    integer(c_int), value :: m
    integer(c_int), value :: n
    type(c_ptr), value :: alpha
    type(c_ptr), value :: A
    integer(c_int), value :: lda
    type(c_ptr), value :: B
    integer(c_int), value :: ldb
    type(c_ptr), value :: beta
    type(c_ptr), value :: C
    integer(c_int), value :: ldc
            hipblasDsymmFortran = &
        hipblasDsymm(handle, side, uplo, m, n, alpha, &
                        A, lda, B, ldb, beta, C, ldc)
end function hipblasDsymmFortran

function hipblasCsymmFortran(handle, side, uplo, m, n, alpha, &
                                A, lda, B, ldb, beta, C, ldc) &
    bind(c, name='hipblasCsymmFortran')
    use iso_c_binding
    use hipblas_enums
    implicit none
    integer(kind(HIPBLAS_STATUS_SUCCESS)) :: hipblasCsymmFortran
    type(c_ptr), value :: handle
    integer(kind(HIPBLAS_SIDE_LEFT)), value :: side
    integer(kind(HIPBLAS_FILL_MODE_FULL)), value :: uplo
    integer(c_int), value :: m
    integer(c_int), value :: n
    type(c_ptr), value :: alpha
    type(c_ptr), value :: A
    integer(c_int), value :: lda
    type(c_ptr), value :: B
    integer(c_int), value :: ldb
    type(c_ptr), value :: beta
    type(c_ptr), value :: C
    integer(c_int), value :: ldc
            hipblasCsymmFortran = &
        hipblasCsymm(handle, side, uplo, m, n, alpha, &
                        A, lda, B, ldb, beta, C, ldc)
end function hipblasCsymmFortran

function hipblasZsymmFortran(handle, side, uplo, m, n, alpha, &
                                A, lda, B, ldb, beta, C, ldc) &
    bind(c, name='hipblasZsymmFortran')
    use iso_c_binding
    use hipblas_enums
    implicit none
    integer(kind(HIPBLAS_STATUS_SUCCESS)) :: hipblasZsymmFortran
    type(c_ptr), value :: handle
    integer(kind(HIPBLAS_SIDE_LEFT)), value :: side
    integer(kind(HIPBLAS_FILL_MODE_FULL)), value :: uplo
    integer(c_int), value :: m
    integer(c_int), value :: n
    type(c_ptr), value :: alpha
    type(c_ptr), value :: A
    integer(c_int), value :: lda
    type(c_ptr), value :: B
    integer(c_int), value :: ldb
    type(c_ptr), value :: beta
    type(c_ptr), value :: C
    integer(c_int), value :: ldc
            hipblasZsymmFortran = &
        hipblasZsymm(handle, side, uplo, m, n, alpha, &
                        A, lda, B, ldb, beta, C, ldc)
end function hipblasZsymmFortran

! symmBatched
function hipblasSsymmBatchedFortran(handle, side, uplo, m, n, alpha, &
                                    A, lda, B, ldb, beta, C, ldc, batch_count) &
    bind(c, name='hipblasSsymmBatchedFortran')
    use iso_c_binding
    use hipblas_enums
    implicit none
    integer(kind(HIPBLAS_STATUS_SUCCESS)) :: hipblasSsymmBatchedFortran
    type(c_ptr), value :: handle
    integer(kind(HIPBLAS_SIDE_LEFT)), value :: side
    integer(kind(HIPBLAS_FILL_MODE_FULL)), value :: uplo
    integer(c_int), value :: m
    integer(c_int), value :: n
    type(c_ptr), value :: alpha
    type(c_ptr), value :: A
    integer(c_int), value :: lda
    type(c_ptr), value :: B
    integer(c_int), value :: ldb
    type(c_ptr), value :: beta
    type(c_ptr), value :: C
    integer(c_int), value :: ldc
    integer(c_int), value :: batch_count
            hipblasSsymmBatchedFortran = &
        hipblasSsymmBatched(handle, side, uplo, m, n, alpha, &
                            A, lda, B, ldb, beta, C, ldc, batch_count)
end function hipblasSsymmBatchedFortran

function hipblasDsymmBatchedFortran(handle, side, uplo, m, n, alpha, &
                                    A, lda, B, ldb, beta, C, ldc, batch_count) &
    bind(c, name='hipblasDsymmBatchedFortran')
    use iso_c_binding
    use hipblas_enums
    implicit none
    integer(kind(HIPBLAS_STATUS_SUCCESS)) :: hipblasDsymmBatchedFortran
    type(c_ptr), value :: handle
    integer(kind(HIPBLAS_SIDE_LEFT)), value :: side
    integer(kind(HIPBLAS_FILL_MODE_FULL)), value :: uplo
    integer(c_int), value :: m
    integer(c_int), value :: n
    type(c_ptr), value :: alpha
    type(c_ptr), value :: A
    integer(c_int), value :: lda
    type(c_ptr), value :: B
    integer(c_int), value :: ldb
    type(c_ptr), value :: beta
    type(c_ptr), value :: C
    integer(c_int), value :: ldc
    integer(c_int), value :: batch_count
            hipblasDsymmBatchedFortran = &
        hipblasDsymmBatched(handle, side, uplo, m, n, alpha, &
                            A, lda, B, ldb, beta, C, ldc, batch_count)
end function hipblasDsymmBatchedFortran

function hipblasCsymmBatchedFortran(handle, side, uplo, m, n, alpha, &
                                    A, lda, B, ldb, beta, C, ldc, batch_count) &
    bind(c, name='hipblasCsymmBatchedFortran')
    use iso_c_binding
    use hipblas_enums
    implicit none
    integer(kind(HIPBLAS_STATUS_SUCCESS)) :: hipblasCsymmBatchedFortran
    type(c_ptr), value :: handle
    integer(kind(HIPBLAS_SIDE_LEFT)), value :: side
    integer(kind(HIPBLAS_FILL_MODE_FULL)), value :: uplo
    integer(c_int), value :: m
    integer(c_int), value :: n
    type(c_ptr), value :: alpha
    type(c_ptr), value :: A
    integer(c_int), value :: lda
    type(c_ptr), value :: B
    integer(c_int), value :: ldb
    type(c_ptr), value :: beta
    type(c_ptr), value :: C
    integer(c_int), value :: ldc
    integer(c_int), value :: batch_count
            hipblasCsymmBatchedFortran = &
        hipblasCsymmBatched(handle, side, uplo, m, n, alpha, &
                            A, lda, B, ldb, beta, C, ldc, batch_count)
end function hipblasCsymmBatchedFortran

function hipblasZsymmBatchedFortran(handle, side, uplo, m, n, alpha, &
                                    A, lda, B, ldb, beta, C, ldc, batch_count) &
    bind(c, name='hipblasZsymmBatchedFortran')
    use iso_c_binding
    use hipblas_enums
    implicit none
    integer(kind(HIPBLAS_STATUS_SUCCESS)) :: hipblasZsymmBatchedFortran
    type(c_ptr), value :: handle
    integer(kind(HIPBLAS_SIDE_LEFT)), value :: side
    integer(kind(HIPBLAS_FILL_MODE_FULL)), value :: uplo
    integer(c_int), value :: m
    integer(c_int), value :: n
    type(c_ptr), value :: alpha
    type(c_ptr), value :: A
    integer(c_int), value :: lda
    type(c_ptr), value :: B
    integer(c_int), value :: ldb
    type(c_ptr), value :: beta
    type(c_ptr), value :: C
    integer(c_int), value :: ldc
    integer(c_int), value :: batch_count
            hipblasZsymmBatchedFortran = &
        hipblasZsymmBatched(handle, side, uplo, m, n, alpha, &
                            A, lda, B, ldb, beta, C, ldc, batch_count)
end function hipblasZsymmBatchedFortran

! symmStridedBatched
function hipblasSsymmStridedBatchedFortran(handle, side, uplo, m, n, alpha, &
                                            A, lda, stride_A, B, ldb, stride_B, beta, C, ldc, stride_C, batch_count) &
    bind(c, name='hipblasSsymmStridedBatchedFortran')
    use iso_c_binding
    use hipblas_enums
    implicit none
    integer(kind(HIPBLAS_STATUS_SUCCESS)) :: hipblasSsymmStridedBatchedFortran
    type(c_ptr), value :: handle
    integer(kind(HIPBLAS_SIDE_LEFT)), value :: side
    integer(kind(HIPBLAS_FILL_MODE_FULL)), value :: uplo
    integer(c_int), value :: m
    integer(c_int), value :: n
    type(c_ptr), value :: alpha
    type(c_ptr), value :: A
    integer(c_int), value :: lda
    integer(c_int64_t), value :: stride_A
    type(c_ptr), value :: B
    integer(c_int), value :: ldb
    integer(c_int64_t), value :: stride_B
    type(c_ptr), value :: beta
    type(c_ptr), value :: C
    integer(c_int), value :: ldc
    integer(c_int64_t), value :: stride_C
    integer(c_int), value :: batch_count
            hipblasSsymmStridedBatchedFortran = &
        hipblasSsymmStridedBatched(handle, side, uplo, m, n, alpha, &
                                    A, lda, stride_A, B, ldb, stride_B, beta, C, ldc, stride_C, batch_count)
end function hipblasSsymmStridedBatchedFortran

function hipblasDsymmStridedBatchedFortran(handle, side, uplo, m, n, alpha, &
                                            A, lda, stride_A, B, ldb, stride_B, beta, C, ldc, stride_C, batch_count) &
    bind(c, name='hipblasDsymmStridedBatchedFortran')
    use iso_c_binding
    use hipblas_enums
    implicit none
    integer(kind(HIPBLAS_STATUS_SUCCESS)) :: hipblasDsymmStridedBatchedFortran
    type(c_ptr), value :: handle
    integer(kind(HIPBLAS_SIDE_LEFT)), value :: side
    integer(kind(HIPBLAS_FILL_MODE_FULL)), value :: uplo
    integer(c_int), value :: m
    integer(c_int), value :: n
    type(c_ptr), value :: alpha
    type(c_ptr), value :: A
    integer(c_int), value :: lda
    integer(c_int64_t), value :: stride_A
    type(c_ptr), value :: B
    integer(c_int), value :: ldb
    integer(c_int64_t), value :: stride_B
    type(c_ptr), value :: beta
    type(c_ptr), value :: C
    integer(c_int), value :: ldc
    integer(c_int64_t), value :: stride_C
    integer(c_int), value :: batch_count
            hipblasDsymmStridedBatchedFortran = &
        hipblasDsymmStridedBatched(handle, side, uplo, m, n, alpha, &
                                    A, lda, stride_A, B, ldb, stride_B, beta, C, ldc, stride_C, batch_count)
end function hipblasDsymmStridedBatchedFortran

function hipblasCsymmStridedBatchedFortran(handle, side, uplo, m, n, alpha, &
                                            A, lda, stride_A, B, ldb, stride_B, beta, C, ldc, stride_C, batch_count) &
    bind(c, name='hipblasCsymmStridedBatchedFortran')
    use iso_c_binding
    use hipblas_enums
    implicit none
    integer(kind(HIPBLAS_STATUS_SUCCESS)) :: hipblasCsymmStridedBatchedFortran
    type(c_ptr), value :: handle
    integer(kind(HIPBLAS_SIDE_LEFT)), value :: side
    integer(kind(HIPBLAS_FILL_MODE_FULL)), value :: uplo
    integer(c_int), value :: m
    integer(c_int), value :: n
    type(c_ptr), value :: alpha
    type(c_ptr), value :: A
    integer(c_int), value :: lda
    integer(c_int64_t), value :: stride_A
    type(c_ptr), value :: B
    integer(c_int), value :: ldb
    integer(c_int64_t), value :: stride_B
    type(c_ptr), value :: beta
    type(c_ptr), value :: C
    integer(c_int), value :: ldc
    integer(c_int64_t), value :: stride_C
    integer(c_int), value :: batch_count
            hipblasCsymmStridedBatchedFortran = &
        hipblasCsymmStridedBatched(handle, side, uplo, m, n, alpha, &
                                    A, lda, stride_A, B, ldb, stride_B, beta, C, ldc, stride_C, batch_count)
end function hipblasCsymmStridedBatchedFortran

function hipblasZsymmStridedBatchedFortran(handle, side, uplo, m, n, alpha, &
                                            A, lda, stride_A, B, ldb, stride_B, beta, C, ldc, stride_C, batch_count) &
    bind(c, name='hipblasZsymmStridedBatchedFortran')
    use iso_c_binding
    use hipblas_enums
    implicit none
    integer(kind(HIPBLAS_STATUS_SUCCESS)) :: hipblasZsymmStridedBatchedFortran
    type(c_ptr), value :: handle
    integer(kind(HIPBLAS_SIDE_LEFT)), value :: side
    integer(kind(HIPBLAS_FILL_MODE_FULL)), value :: uplo
    integer(c_int), value :: m
    integer(c_int), value :: n
    type(c_ptr), value :: alpha
    type(c_ptr), value :: A
    integer(c_int), value :: lda
    integer(c_int64_t), value :: stride_A
    type(c_ptr), value :: B
    integer(c_int), value :: ldb
    integer(c_int64_t), value :: stride_B
    type(c_ptr), value :: beta
    type(c_ptr), value :: C
    integer(c_int), value :: ldc
    integer(c_int64_t), value :: stride_C
    integer(c_int), value :: batch_count
            hipblasZsymmStridedBatchedFortran = &
        hipblasZsymmStridedBatched(handle, side, uplo, m, n, alpha, &
                                    A, lda, stride_A, B, ldb, stride_B, beta, C, ldc, stride_C, batch_count)
end function hipblasZsymmStridedBatchedFortran

! syrk
function hipblasSsyrkFortran(handle, uplo, transA, n, k, alpha, &
                                A, lda, beta, C, ldc) &
    bind(c, name='hipblasSsyrkFortran')
    use iso_c_binding
    use hipblas_enums
    implicit none
    integer(kind(HIPBLAS_STATUS_SUCCESS)) :: hipblasSsyrkFortran
    type(c_ptr), value :: handle
    integer(kind(HIPBLAS_FILL_MODE_FULL)), value :: uplo
    integer(kind(HIPBLAS_OP_N)), value :: transA
    integer(c_int), value :: n
    integer(c_int), value :: k
    type(c_ptr), value :: alpha
    type(c_ptr), value :: A
    integer(c_int), value :: lda
    type(c_ptr), value :: beta
    type(c_ptr), value :: C
    integer(c_int), value :: ldc
            hipblasSsyrkFortran = &
        hipblasSsyrk(handle, uplo, transA, n, k, alpha, &
                        A, lda, beta, C, ldc)
end function hipblasSsyrkFortran

function hipblasDsyrkFortran(handle, uplo, transA, n, k, alpha, &
                                A, lda, beta, C, ldc) &
    bind(c, name='hipblasDsyrkFortran')
    use iso_c_binding
    use hipblas_enums
    implicit none
    integer(kind(HIPBLAS_STATUS_SUCCESS)) :: hipblasDsyrkFortran
    type(c_ptr), value :: handle
    integer(kind(HIPBLAS_FILL_MODE_FULL)), value :: uplo
    integer(kind(HIPBLAS_OP_N)), value :: transA
    integer(c_int), value :: n
    integer(c_int), value :: k
    type(c_ptr), value :: alpha
    type(c_ptr), value :: A
    integer(c_int), value :: lda
    type(c_ptr), value :: beta
    type(c_ptr), value :: C
    integer(c_int), value :: ldc
            hipblasDsyrkFortran = &
        hipblasDsyrk(handle, uplo, transA, n, k, alpha, &
                        A, lda, beta, C, ldc)
end function hipblasDsyrkFortran

function hipblasCsyrkFortran(handle, uplo, transA, n, k, alpha, &
                                A, lda, beta, C, ldc) &
    bind(c, name='hipblasCsyrkFortran')
    use iso_c_binding
    use hipblas_enums
    implicit none
    integer(kind(HIPBLAS_STATUS_SUCCESS)) :: hipblasCsyrkFortran
    type(c_ptr), value :: handle
    integer(kind(HIPBLAS_FILL_MODE_FULL)), value :: uplo
    integer(kind(HIPBLAS_OP_N)), value :: transA
    integer(c_int), value :: n
    integer(c_int), value :: k
    type(c_ptr), value :: alpha
    type(c_ptr), value :: A
    integer(c_int), value :: lda
    type(c_ptr), value :: beta
    type(c_ptr), value :: C
    integer(c_int), value :: ldc
            hipblasCsyrkFortran = &
        hipblasCsyrk(handle, uplo, transA, n, k, alpha, &
                        A, lda, beta, C, ldc)
end function hipblasCsyrkFortran

function hipblasZsyrkFortran(handle, uplo, transA, n, k, alpha, &
                                A, lda, beta, C, ldc) &
    bind(c, name='hipblasZsyrkFortran')
    use iso_c_binding
    use hipblas_enums
    implicit none
    integer(kind(HIPBLAS_STATUS_SUCCESS)) :: hipblasZsyrkFortran
    type(c_ptr), value :: handle
    integer(kind(HIPBLAS_FILL_MODE_FULL)), value :: uplo
    integer(kind(HIPBLAS_OP_N)), value :: transA
    integer(c_int), value :: n
    integer(c_int), value :: k
    type(c_ptr), value :: alpha
    type(c_ptr), value :: A
    integer(c_int), value :: lda
    type(c_ptr), value :: beta
    type(c_ptr), value :: C
    integer(c_int), value :: ldc
            hipblasZsyrkFortran = &
        hipblasZsyrk(handle, uplo, transA, n, k, alpha, &
                        A, lda, beta, C, ldc)
end function hipblasZsyrkFortran

! syrkBatched
function hipblasSsyrkBatchedFortran(handle, uplo, transA, n, k, alpha, &
                                    A, lda, beta, C, ldc, batch_count) &
    bind(c, name='hipblasSsyrkBatchedFortran')
    use iso_c_binding
    use hipblas_enums
    implicit none
    integer(kind(HIPBLAS_STATUS_SUCCESS)) :: hipblasSsyrkBatchedFortran
    type(c_ptr), value :: handle
    integer(kind(HIPBLAS_FILL_MODE_FULL)), value :: uplo
    integer(kind(HIPBLAS_OP_N)), value :: transA
    integer(c_int), value :: n
    integer(c_int), value :: k
    type(c_ptr), value :: alpha
    type(c_ptr), value :: A
    integer(c_int), value :: lda
    type(c_ptr), value :: beta
    type(c_ptr), value :: C
    integer(c_int), value :: ldc
    integer(c_int), value :: batch_count
            hipblasSsyrkBatchedFortran = &
        hipblasSsyrkBatched(handle, uplo, transA, n, k, alpha, &
                            A, lda, beta, C, ldc, batch_count)
end function hipblasSsyrkBatchedFortran

function hipblasDsyrkBatchedFortran(handle, uplo, transA, n, k, alpha, &
                                    A, lda, beta, C, ldc, batch_count) &
    bind(c, name='hipblasDsyrkBatchedFortran')
    use iso_c_binding
    use hipblas_enums
    implicit none
    integer(kind(HIPBLAS_STATUS_SUCCESS)) :: hipblasDsyrkBatchedFortran
    type(c_ptr), value :: handle
    integer(kind(HIPBLAS_FILL_MODE_FULL)), value :: uplo
    integer(kind(HIPBLAS_OP_N)), value :: transA
    integer(c_int), value :: n
    integer(c_int), value :: k
    type(c_ptr), value :: alpha
    type(c_ptr), value :: A
    integer(c_int), value :: lda
    type(c_ptr), value :: beta
    type(c_ptr), value :: C
    integer(c_int), value :: ldc
    integer(c_int), value :: batch_count
            hipblasDsyrkBatchedFortran = &
        hipblasDsyrkBatched(handle, uplo, transA, n, k, alpha, &
                            A, lda, beta, C, ldc, batch_count)
end function hipblasDsyrkBatchedFortran

function hipblasCsyrkBatchedFortran(handle, uplo, transA, n, k, alpha, &
                                    A, lda, beta, C, ldc, batch_count) &
    bind(c, name='hipblasCsyrkBatchedFortran')
    use iso_c_binding
    use hipblas_enums
    implicit none
    integer(kind(HIPBLAS_STATUS_SUCCESS)) :: hipblasCsyrkBatchedFortran
    type(c_ptr), value :: handle
    integer(kind(HIPBLAS_FILL_MODE_FULL)), value :: uplo
    integer(kind(HIPBLAS_OP_N)), value :: transA
    integer(c_int), value :: n
    integer(c_int), value :: k
    type(c_ptr), value :: alpha
    type(c_ptr), value :: A
    integer(c_int), value :: lda
    type(c_ptr), value :: beta
    type(c_ptr), value :: C
    integer(c_int), value :: ldc
    integer(c_int), value :: batch_count
            hipblasCsyrkBatchedFortran = &
        hipblasCsyrkBatched(handle, uplo, transA, n, k, alpha, &
                            A, lda, beta, C, ldc, batch_count)
end function hipblasCsyrkBatchedFortran

function hipblasZsyrkBatchedFortran(handle, uplo, transA, n, k, alpha, &
                                    A, lda, beta, C, ldc, batch_count) &
    bind(c, name='hipblasZsyrkBatchedFortran')
    use iso_c_binding
    use hipblas_enums
    implicit none
    integer(kind(HIPBLAS_STATUS_SUCCESS)) :: hipblasZsyrkBatchedFortran
    type(c_ptr), value :: handle
    integer(kind(HIPBLAS_FILL_MODE_FULL)), value :: uplo
    integer(kind(HIPBLAS_OP_N)), value :: transA
    integer(c_int), value :: n
    integer(c_int), value :: k
    type(c_ptr), value :: alpha
    type(c_ptr), value :: A
    integer(c_int), value :: lda
    type(c_ptr), value :: beta
    type(c_ptr), value :: C
    integer(c_int), value :: ldc
    integer(c_int), value :: batch_count
            hipblasZsyrkBatchedFortran = &
        hipblasZsyrkBatched(handle, uplo, transA, n, k, alpha, &
                            A, lda, beta, C, ldc, batch_count)
end function hipblasZsyrkBatchedFortran

! syrkStridedBatched
function hipblasSsyrkStridedBatchedFortran(handle, uplo, transA, n, k, alpha, &
                                            A, lda, stride_A, beta, C, ldc, stride_C, batch_count) &
    bind(c, name='hipblasSsyrkStridedBatchedFortran')
    use iso_c_binding
    use hipblas_enums
    implicit none
    integer(kind(HIPBLAS_STATUS_SUCCESS)) :: hipblasSsyrkStridedBatchedFortran
    type(c_ptr), value :: handle
    integer(kind(HIPBLAS_FILL_MODE_FULL)), value :: uplo
    integer(kind(HIPBLAS_OP_N)), value :: transA
    integer(c_int), value :: n
    integer(c_int), value :: k
    type(c_ptr), value :: alpha
    type(c_ptr), value :: A
    integer(c_int), value :: lda
    integer(c_int64_t), value :: stride_A
    type(c_ptr), value :: beta
    type(c_ptr), value :: C
    integer(c_int), value :: ldc
    integer(c_int64_t), value :: stride_C
    integer(c_int), value :: batch_count
            hipblasSsyrkStridedBatchedFortran = &
        hipblasSsyrkStridedBatched(handle, uplo, transA, n, k, alpha, &
                                    A, lda, stride_A, beta, C, ldc, stride_C, batch_count)
end function hipblasSsyrkStridedBatchedFortran

function hipblasDsyrkStridedBatchedFortran(handle, uplo, transA, n, k, alpha, &
                                            A, lda, stride_A, beta, C, ldc, stride_C, batch_count) &
    bind(c, name='hipblasDsyrkStridedBatchedFortran')
    use iso_c_binding
    use hipblas_enums
    implicit none
    integer(kind(HIPBLAS_STATUS_SUCCESS)) :: hipblasDsyrkStridedBatchedFortran
    type(c_ptr), value :: handle
    integer(kind(HIPBLAS_FILL_MODE_FULL)), value :: uplo
    integer(kind(HIPBLAS_OP_N)), value :: transA
    integer(c_int), value :: n
    integer(c_int), value :: k
    type(c_ptr), value :: alpha
    type(c_ptr), value :: A
    integer(c_int), value :: lda
    integer(c_int64_t), value :: stride_A
    type(c_ptr), value :: beta
    type(c_ptr), value :: C
    integer(c_int), value :: ldc
    integer(c_int64_t), value :: stride_C
    integer(c_int), value :: batch_count
            hipblasDsyrkStridedBatchedFortran = &
        hipblasDsyrkStridedBatched(handle, uplo, transA, n, k, alpha, &
                                    A, lda, stride_A, beta, C, ldc, stride_C, batch_count)
end function hipblasDsyrkStridedBatchedFortran

function hipblasCsyrkStridedBatchedFortran(handle, uplo, transA, n, k, alpha, &
                                            A, lda, stride_A, beta, C, ldc, stride_C, batch_count) &
    bind(c, name='hipblasCsyrkStridedBatchedFortran')
    use iso_c_binding
    use hipblas_enums
    implicit none
    integer(kind(HIPBLAS_STATUS_SUCCESS)) :: hipblasCsyrkStridedBatchedFortran
    type(c_ptr), value :: handle
    integer(kind(HIPBLAS_FILL_MODE_FULL)), value :: uplo
    integer(kind(HIPBLAS_OP_N)), value :: transA
    integer(c_int), value :: n
    integer(c_int), value :: k
    type(c_ptr), value :: alpha
    type(c_ptr), value :: A
    integer(c_int), value :: lda
    integer(c_int64_t), value :: stride_A
    type(c_ptr), value :: beta
    type(c_ptr), value :: C
    integer(c_int), value :: ldc
    integer(c_int64_t), value :: stride_C
    integer(c_int), value :: batch_count
            hipblasCsyrkStridedBatchedFortran = &
        hipblasCsyrkStridedBatched(handle, uplo, transA, n, k, alpha, &
                                    A, lda, stride_A, beta, C, ldc, stride_C, batch_count)
end function hipblasCsyrkStridedBatchedFortran

function hipblasZsyrkStridedBatchedFortran(handle, uplo, transA, n, k, alpha, &
                                            A, lda, stride_A, beta, C, ldc, stride_C, batch_count) &
    bind(c, name='hipblasZsyrkStridedBatchedFortran')
    use iso_c_binding
    use hipblas_enums
    implicit none
    integer(kind(HIPBLAS_STATUS_SUCCESS)) :: hipblasZsyrkStridedBatchedFortran
    type(c_ptr), value :: handle
    integer(kind(HIPBLAS_FILL_MODE_FULL)), value :: uplo
    integer(kind(HIPBLAS_OP_N)), value :: transA
    integer(c_int), value :: n
    integer(c_int), value :: k
    type(c_ptr), value :: alpha
    type(c_ptr), value :: A
    integer(c_int), value :: lda
    integer(c_int64_t), value :: stride_A
    type(c_ptr), value :: beta
    type(c_ptr), value :: C
    integer(c_int), value :: ldc
    integer(c_int64_t), value :: stride_C
    integer(c_int), value :: batch_count
            hipblasZsyrkStridedBatchedFortran = &
        hipblasZsyrkStridedBatched(handle, uplo, transA, n, k, alpha, &
                                    A, lda, stride_A, beta, C, ldc, stride_C, batch_count)
end function hipblasZsyrkStridedBatchedFortran

! syr2k
function hipblasSsyr2kFortran(handle, uplo, transA, n, k, alpha, &
                                A, lda, B, ldb, beta, C, ldc) &
    bind(c, name='hipblasSsyr2kFortran')
    use iso_c_binding
    use hipblas_enums
    implicit none
    integer(kind(HIPBLAS_STATUS_SUCCESS)) :: hipblasSsyr2kFortran
    type(c_ptr), value :: handle
    integer(kind(HIPBLAS_FILL_MODE_FULL)), value :: uplo
    integer(kind(HIPBLAS_OP_N)), value :: transA
    integer(c_int), value :: n
    integer(c_int), value :: k
    type(c_ptr), value :: alpha
    type(c_ptr), value :: A
    integer(c_int), value :: lda
    type(c_ptr), value :: B
    integer(c_int), value :: ldb
    type(c_ptr), value :: beta
    type(c_ptr), value :: C
    integer(c_int), value :: ldc
            hipblasSsyr2kFortran = &
        hipblasSsyr2k(handle, uplo, transA, n, k, alpha, &
                        A, lda, B, ldb, beta, C, ldc)
end function hipblasSsyr2kFortran

function hipblasDsyr2kFortran(handle, uplo, transA, n, k, alpha, &
                                A, lda, B, ldb, beta, C, ldc) &
    bind(c, name='hipblasDsyr2kFortran')
    use iso_c_binding
    use hipblas_enums
    implicit none
    integer(kind(HIPBLAS_STATUS_SUCCESS)) :: hipblasDsyr2kFortran
    type(c_ptr), value :: handle
    integer(kind(HIPBLAS_FILL_MODE_FULL)), value :: uplo
    integer(kind(HIPBLAS_OP_N)), value :: transA
    integer(c_int), value :: n
    integer(c_int), value :: k
    type(c_ptr), value :: alpha
    type(c_ptr), value :: A
    integer(c_int), value :: lda
    type(c_ptr), value :: B
    integer(c_int), value :: ldb
    type(c_ptr), value :: beta
    type(c_ptr), value :: C
    integer(c_int), value :: ldc
            hipblasDsyr2kFortran = &
        hipblasDsyr2k(handle, uplo, transA, n, k, alpha, &
                        A, lda, B, ldb, beta, C, ldc)
end function hipblasDsyr2kFortran

function hipblasCsyr2kFortran(handle, uplo, transA, n, k, alpha, &
                                A, lda, B, ldb, beta, C, ldc) &
    bind(c, name='hipblasCsyr2kFortran')
    use iso_c_binding
    use hipblas_enums
    implicit none
    integer(kind(HIPBLAS_STATUS_SUCCESS)) :: hipblasCsyr2kFortran
    type(c_ptr), value :: handle
    integer(kind(HIPBLAS_FILL_MODE_FULL)), value :: uplo
    integer(kind(HIPBLAS_OP_N)), value :: transA
    integer(c_int), value :: n
    integer(c_int), value :: k
    type(c_ptr), value :: alpha
    type(c_ptr), value :: A
    integer(c_int), value :: lda
    type(c_ptr), value :: B
    integer(c_int), value :: ldb
    type(c_ptr), value :: beta
    type(c_ptr), value :: C
    integer(c_int), value :: ldc
            hipblasCsyr2kFortran = &
        hipblasCsyr2k(handle, uplo, transA, n, k, alpha, &
                        A, lda, B, ldb, beta, C, ldc)
end function hipblasCsyr2kFortran

function hipblasZsyr2kFortran(handle, uplo, transA, n, k, alpha, &
                                A, lda, B, ldb, beta, C, ldc) &
    bind(c, name='hipblasZsyr2kFortran')
    use iso_c_binding
    use hipblas_enums
    implicit none
    integer(kind(HIPBLAS_STATUS_SUCCESS)) :: hipblasZsyr2kFortran
    type(c_ptr), value :: handle
    integer(kind(HIPBLAS_FILL_MODE_FULL)), value :: uplo
    integer(kind(HIPBLAS_OP_N)), value :: transA
    integer(c_int), value :: n
    integer(c_int), value :: k
    type(c_ptr), value :: alpha
    type(c_ptr), value :: A
    integer(c_int), value :: lda
    type(c_ptr), value :: B
    integer(c_int), value :: ldb
    type(c_ptr), value :: beta
    type(c_ptr), value :: C
    integer(c_int), value :: ldc
            hipblasZsyr2kFortran = &
        hipblasZsyr2k(handle, uplo, transA, n, k, alpha, &
                        A, lda, B, ldb, beta, C, ldc)
end function hipblasZsyr2kFortran

! syr2kBatched
function hipblasSsyr2kBatchedFortran(handle, uplo, transA, n, k, alpha, &
                                        A, lda, B, ldb, beta, C, ldc, batch_count) &
    bind(c, name='hipblasSsyr2kBatchedFortran')
    use iso_c_binding
    use hipblas_enums
    implicit none
    integer(kind(HIPBLAS_STATUS_SUCCESS)) :: hipblasSsyr2kBatchedFortran
    type(c_ptr), value :: handle
    integer(kind(HIPBLAS_FILL_MODE_FULL)), value :: uplo
    integer(kind(HIPBLAS_OP_N)), value :: transA
    integer(c_int), value :: n
    integer(c_int), value :: k
    type(c_ptr), value :: alpha
    type(c_ptr), value :: A
    integer(c_int), value :: lda
    type(c_ptr), value :: B
    integer(c_int), value :: ldb
    type(c_ptr), value :: beta
    type(c_ptr), value :: C
    integer(c_int), value :: ldc
    integer(c_int), value :: batch_count
            hipblasSsyr2kBatchedFortran = &
        hipblasSsyr2kBatched(handle, uplo, transA, n, k, alpha, &
                                A, lda, B, ldb, beta, C, ldc, batch_count)
end function hipblasSsyr2kBatchedFortran

function hipblasDsyr2kBatchedFortran(handle, uplo, transA, n, k, alpha, &
                                        A, lda, B, ldb, beta, C, ldc, batch_count) &
    bind(c, name='hipblasDsyr2kBatchedFortran')
    use iso_c_binding
    use hipblas_enums
    implicit none
    integer(kind(HIPBLAS_STATUS_SUCCESS)) :: hipblasDsyr2kBatchedFortran
    type(c_ptr), value :: handle
    integer(kind(HIPBLAS_FILL_MODE_FULL)), value :: uplo
    integer(kind(HIPBLAS_OP_N)), value :: transA
    integer(c_int), value :: n
    integer(c_int), value :: k
    type(c_ptr), value :: alpha
    type(c_ptr), value :: A
    integer(c_int), value :: lda
    type(c_ptr), value :: B
    integer(c_int), value :: ldb
    type(c_ptr), value :: beta
    type(c_ptr), value :: C
    integer(c_int), value :: ldc
    integer(c_int), value :: batch_count
            hipblasDsyr2kBatchedFortran = &
        hipblasDsyr2kBatched(handle, uplo, transA, n, k, alpha, &
                                A, lda, B, ldb, beta, C, ldc, batch_count)
end function hipblasDsyr2kBatchedFortran

function hipblasCsyr2kBatchedFortran(handle, uplo, transA, n, k, alpha, &
                                        A, lda, B, ldb, beta, C, ldc, batch_count) &
    bind(c, name='hipblasCsyr2kBatchedFortran')
    use iso_c_binding
    use hipblas_enums
    implicit none
    integer(kind(HIPBLAS_STATUS_SUCCESS)) :: hipblasCsyr2kBatchedFortran
    type(c_ptr), value :: handle
    integer(kind(HIPBLAS_FILL_MODE_FULL)), value :: uplo
    integer(kind(HIPBLAS_OP_N)), value :: transA
    integer(c_int), value :: n
    integer(c_int), value :: k
    type(c_ptr), value :: alpha
    type(c_ptr), value :: A
    integer(c_int), value :: lda
    type(c_ptr), value :: B
    integer(c_int), value :: ldb
    type(c_ptr), value :: beta
    type(c_ptr), value :: C
    integer(c_int), value :: ldc
    integer(c_int), value :: batch_count
            hipblasCsyr2kBatchedFortran = &
        hipblasCsyr2kBatched(handle, uplo, transA, n, k, alpha, &
                                A, lda, B, ldb, beta, C, ldc, batch_count)
end function hipblasCsyr2kBatchedFortran

function hipblasZsyr2kBatchedFortran(handle, uplo, transA, n, k, alpha, &
                                        A, lda, B, ldb, beta, C, ldc, batch_count) &
    bind(c, name='hipblasZsyr2kBatchedFortran')
    use iso_c_binding
    use hipblas_enums
    implicit none
    integer(kind(HIPBLAS_STATUS_SUCCESS)) :: hipblasZsyr2kBatchedFortran
    type(c_ptr), value :: handle
    integer(kind(HIPBLAS_FILL_MODE_FULL)), value :: uplo
    integer(kind(HIPBLAS_OP_N)), value :: transA
    integer(c_int), value :: n
    integer(c_int), value :: k
    type(c_ptr), value :: alpha
    type(c_ptr), value :: A
    integer(c_int), value :: lda
    type(c_ptr), value :: B
    integer(c_int), value :: ldb
    type(c_ptr), value :: beta
    type(c_ptr), value :: C
    integer(c_int), value :: ldc
    integer(c_int), value :: batch_count
            hipblasZsyr2kBatchedFortran = &
        hipblasZsyr2kBatched(handle, uplo, transA, n, k, alpha, &
                                A, lda, B, ldb, beta, C, ldc, batch_count)
end function hipblasZsyr2kBatchedFortran

! syr2kStridedBatched
function hipblasSsyr2kStridedBatchedFortran(handle, uplo, transA, n, k, alpha, &
                                            A, lda, stride_A, B, ldb, stride_B, beta, C, ldc, stride_C, batch_count) &
    bind(c, name='hipblasSsyr2kStridedBatchedFortran')
    use iso_c_binding
    use hipblas_enums
    implicit none
    integer(kind(HIPBLAS_STATUS_SUCCESS)) :: hipblasSsyr2kStridedBatchedFortran
    type(c_ptr), value :: handle
    integer(kind(HIPBLAS_FILL_MODE_FULL)), value :: uplo
    integer(kind(HIPBLAS_OP_N)), value :: transA
    integer(c_int), value :: n
    integer(c_int), value :: k
    type(c_ptr), value :: alpha
    type(c_ptr), value :: A
    integer(c_int), value :: lda
    integer(c_int64_t), value :: stride_A
    type(c_ptr), value :: B
    integer(c_int), value :: ldb
    integer(c_int64_t), value :: stride_B
    type(c_ptr), value :: beta
    type(c_ptr), value :: C
    integer(c_int), value :: ldc
    integer(c_int64_t), value :: stride_C
    integer(c_int), value :: batch_count
            hipblasSsyr2kStridedBatchedFortran = &
        hipblasSsyr2kStridedBatched(handle, uplo, transA, n, k, alpha, &
                                    A, lda, stride_A, B, ldb, stride_B, beta, C, ldc, stride_C, batch_count)
end function hipblasSsyr2kStridedBatchedFortran

function hipblasDsyr2kStridedBatchedFortran(handle, uplo, transA, n, k, alpha, &
                                            A, lda, stride_A, B, ldb, stride_B, beta, C, ldc, stride_C, batch_count) &
    bind(c, name='hipblasDsyr2kStridedBatchedFortran')
    use iso_c_binding
    use hipblas_enums
    implicit none
    integer(kind(HIPBLAS_STATUS_SUCCESS)) :: hipblasDsyr2kStridedBatchedFortran
    type(c_ptr), value :: handle
    integer(kind(HIPBLAS_FILL_MODE_FULL)), value :: uplo
    integer(kind(HIPBLAS_OP_N)), value :: transA
    integer(c_int), value :: n
    integer(c_int), value :: k
    type(c_ptr), value :: alpha
    type(c_ptr), value :: A
    integer(c_int), value :: lda
    integer(c_int64_t), value :: stride_A
    type(c_ptr), value :: B
    integer(c_int), value :: ldb
    integer(c_int64_t), value :: stride_B
    type(c_ptr), value :: beta
    type(c_ptr), value :: C
    integer(c_int), value :: ldc
    integer(c_int64_t), value :: stride_C
    integer(c_int), value :: batch_count
            hipblasDsyr2kStridedBatchedFortran = &
        hipblasDsyr2kStridedBatched(handle, uplo, transA, n, k, alpha, &
                                    A, lda, stride_A, B, ldb, stride_B, beta, C, ldc, stride_C, batch_count)
end function hipblasDsyr2kStridedBatchedFortran

function hipblasCsyr2kStridedBatchedFortran(handle, uplo, transA, n, k, alpha, &
                                            A, lda, stride_A, B, ldb, stride_B, beta, C, ldc, stride_C, batch_count) &
    bind(c, name='hipblasCsyr2kStridedBatchedFortran')
    use iso_c_binding
    use hipblas_enums
    implicit none
    integer(kind(HIPBLAS_STATUS_SUCCESS)) :: hipblasCsyr2kStridedBatchedFortran
    type(c_ptr), value :: handle
    integer(kind(HIPBLAS_FILL_MODE_FULL)), value :: uplo
    integer(kind(HIPBLAS_OP_N)), value :: transA
    integer(c_int), value :: n
    integer(c_int), value :: k
    type(c_ptr), value :: alpha
    type(c_ptr), value :: A
    integer(c_int), value :: lda
    integer(c_int64_t), value :: stride_A
    type(c_ptr), value :: B
    integer(c_int), value :: ldb
    integer(c_int64_t), value :: stride_B
    type(c_ptr), value :: beta
    type(c_ptr), value :: C
    integer(c_int), value :: ldc
    integer(c_int64_t), value :: stride_C
    integer(c_int), value :: batch_count
            hipblasCsyr2kStridedBatchedFortran = &
        hipblasCsyr2kStridedBatched(handle, uplo, transA, n, k, alpha, &
                                    A, lda, stride_A, B, ldb, stride_B, beta, C, ldc, stride_C, batch_count)
end function hipblasCsyr2kStridedBatchedFortran

function hipblasZsyr2kStridedBatchedFortran(handle, uplo, transA, n, k, alpha, &
                                            A, lda, stride_A, B, ldb, stride_B, beta, C, ldc, stride_C, batch_count) &
    bind(c, name='hipblasZsyr2kStridedBatchedFortran')
    use iso_c_binding
    use hipblas_enums
    implicit none
    integer(kind(HIPBLAS_STATUS_SUCCESS)) :: hipblasZsyr2kStridedBatchedFortran
    type(c_ptr), value :: handle
    integer(kind(HIPBLAS_FILL_MODE_FULL)), value :: uplo
    integer(kind(HIPBLAS_OP_N)), value :: transA
    integer(c_int), value :: n
    integer(c_int), value :: k
    type(c_ptr), value :: alpha
    type(c_ptr), value :: A
    integer(c_int), value :: lda
    integer(c_int64_t), value :: stride_A
    type(c_ptr), value :: B
    integer(c_int), value :: ldb
    integer(c_int64_t), value :: stride_B
    type(c_ptr), value :: beta
    type(c_ptr), value :: C
    integer(c_int), value :: ldc
    integer(c_int64_t), value :: stride_C
    integer(c_int), value :: batch_count
            hipblasZsyr2kStridedBatchedFortran = &
        hipblasZsyr2kStridedBatched(handle, uplo, transA, n, k, alpha, &
                                    A, lda, stride_A, B, ldb, stride_B, beta, C, ldc, stride_C, batch_count)
end function hipblasZsyr2kStridedBatchedFortran

! syrkx
function hipblasSsyrkxFortran(handle, uplo, transA, n, k, alpha, &
                                A, lda, B, ldb, beta, C, ldc) &
    bind(c, name='hipblasSsyrkxFortran')
    use iso_c_binding
    use hipblas_enums
    implicit none
    integer(kind(HIPBLAS_STATUS_SUCCESS)) :: hipblasSsyrkxFortran
    type(c_ptr), value :: handle
    integer(kind(HIPBLAS_FILL_MODE_FULL)), value :: uplo
    integer(kind(HIPBLAS_OP_N)), value :: transA
    integer(c_int), value :: n
    integer(c_int), value :: k
    type(c_ptr), value :: alpha
    type(c_ptr), value :: A
    integer(c_int), value :: lda
    type(c_ptr), value :: B
    integer(c_int), value :: ldb
    type(c_ptr), value :: beta
    type(c_ptr), value :: C
    integer(c_int), value :: ldc
            hipblasSsyrkxFortran = &
        hipblasSsyrkx(handle, uplo, transA, n, k, alpha, &
                        A, lda, B, ldb, beta, C, ldc)
end function hipblasSsyrkxFortran

function hipblasDsyrkxFortran(handle, uplo, transA, n, k, alpha, &
                                A, lda, B, ldb, beta, C, ldc) &
    bind(c, name='hipblasDsyrkxFortran')
    use iso_c_binding
    use hipblas_enums
    implicit none
    integer(kind(HIPBLAS_STATUS_SUCCESS)) :: hipblasDsyrkxFortran
    type(c_ptr), value :: handle
    integer(kind(HIPBLAS_FILL_MODE_FULL)), value :: uplo
    integer(kind(HIPBLAS_OP_N)), value :: transA
    integer(c_int), value :: n
    integer(c_int), value :: k
    type(c_ptr), value :: alpha
    type(c_ptr), value :: A
    integer(c_int), value :: lda
    type(c_ptr), value :: B
    integer(c_int), value :: ldb
    type(c_ptr), value :: beta
    type(c_ptr), value :: C
    integer(c_int), value :: ldc
            hipblasDsyrkxFortran = &
        hipblasDsyrkx(handle, uplo, transA, n, k, alpha, &
                        A, lda, B, ldb, beta, C, ldc)
end function hipblasDsyrkxFortran

function hipblasCsyrkxFortran(handle, uplo, transA, n, k, alpha, &
                                A, lda, B, ldb, beta, C, ldc) &
    bind(c, name='hipblasCsyrkxFortran')
    use iso_c_binding
    use hipblas_enums
    implicit none
    integer(kind(HIPBLAS_STATUS_SUCCESS)) :: hipblasCsyrkxFortran
    type(c_ptr), value :: handle
    integer(kind(HIPBLAS_FILL_MODE_FULL)), value :: uplo
    integer(kind(HIPBLAS_OP_N)), value :: transA
    integer(c_int), value :: n
    integer(c_int), value :: k
    type(c_ptr), value :: alpha
    type(c_ptr), value :: A
    integer(c_int), value :: lda
    type(c_ptr), value :: B
    integer(c_int), value :: ldb
    type(c_ptr), value :: beta
    type(c_ptr), value :: C
    integer(c_int), value :: ldc
            hipblasCsyrkxFortran = &
        hipblasCsyrkx(handle, uplo, transA, n, k, alpha, &
                        A, lda, B, ldb, beta, C, ldc)
end function hipblasCsyrkxFortran

function hipblasZsyrkxFortran(handle, uplo, transA, n, k, alpha, &
                                A, lda, B, ldb, beta, C, ldc) &
    bind(c, name='hipblasZsyrkxFortran')
    use iso_c_binding
    use hipblas_enums
    implicit none
    integer(kind(HIPBLAS_STATUS_SUCCESS)) :: hipblasZsyrkxFortran
    type(c_ptr), value :: handle
    integer(kind(HIPBLAS_FILL_MODE_FULL)), value :: uplo
    integer(kind(HIPBLAS_OP_N)), value :: transA
    integer(c_int), value :: n
    integer(c_int), value :: k
    type(c_ptr), value :: alpha
    type(c_ptr), value :: A
    integer(c_int), value :: lda
    type(c_ptr), value :: B
    integer(c_int), value :: ldb
    type(c_ptr), value :: beta
    type(c_ptr), value :: C
    integer(c_int), value :: ldc
            hipblasZsyrkxFortran = &
        hipblasZsyrkx(handle, uplo, transA, n, k, alpha, &
                        A, lda, B, ldb, beta, C, ldc)
end function hipblasZsyrkxFortran

! syrkxBatched
function hipblasSsyrkxBatchedFortran(handle, uplo, transA, n, k, alpha, &
                                        A, lda, B, ldb, beta, C, ldc, batch_count) &
    bind(c, name='hipblasSsyrkxBatchedFortran')
    use iso_c_binding
    use hipblas_enums
    implicit none
    integer(kind(HIPBLAS_STATUS_SUCCESS)) :: hipblasSsyrkxBatchedFortran
    type(c_ptr), value :: handle
    integer(kind(HIPBLAS_FILL_MODE_FULL)), value :: uplo
    integer(kind(HIPBLAS_OP_N)), value :: transA
    integer(c_int), value :: n
    integer(c_int), value :: k
    type(c_ptr), value :: alpha
    type(c_ptr), value :: A
    integer(c_int), value :: lda
    type(c_ptr), value :: B
    integer(c_int), value :: ldb
    type(c_ptr), value :: beta
    type(c_ptr), value :: C
    integer(c_int), value :: ldc
    integer(c_int), value :: batch_count
            hipblasSsyrkxBatchedFortran = &
        hipblasSsyrkxBatched(handle, uplo, transA, n, k, alpha, &
                                A, lda, B, ldb, beta, C, ldc, batch_count)
end function hipblasSsyrkxBatchedFortran

function hipblasDsyrkxBatchedFortran(handle, uplo, transA, n, k, alpha, &
                                        A, lda, B, ldb, beta, C, ldc, batch_count) &
    bind(c, name='hipblasDsyrkxBatchedFortran')
    use iso_c_binding
    use hipblas_enums
    implicit none
    integer(kind(HIPBLAS_STATUS_SUCCESS)) :: hipblasDsyrkxBatchedFortran
    type(c_ptr), value :: handle
    integer(kind(HIPBLAS_FILL_MODE_FULL)), value :: uplo
    integer(kind(HIPBLAS_OP_N)), value :: transA
    integer(c_int), value :: n
    integer(c_int), value :: k
    type(c_ptr), value :: alpha
    type(c_ptr), value :: A
    integer(c_int), value :: lda
    type(c_ptr), value :: B
    integer(c_int), value :: ldb
    type(c_ptr), value :: beta
    type(c_ptr), value :: C
    integer(c_int), value :: ldc
    integer(c_int), value :: batch_count
            hipblasDsyrkxBatchedFortran = &
        hipblasDsyrkxBatched(handle, uplo, transA, n, k, alpha, &
                                A, lda, B, ldb, beta, C, ldc, batch_count)
end function hipblasDsyrkxBatchedFortran

function hipblasCsyrkxBatchedFortran(handle, uplo, transA, n, k, alpha, &
                                        A, lda, B, ldb, beta, C, ldc, batch_count) &
    bind(c, name='hipblasCsyrkxBatchedFortran')
    use iso_c_binding
    use hipblas_enums
    implicit none
    integer(kind(HIPBLAS_STATUS_SUCCESS)) :: hipblasCsyrkxBatchedFortran
    type(c_ptr), value :: handle
    integer(kind(HIPBLAS_FILL_MODE_FULL)), value :: uplo
    integer(kind(HIPBLAS_OP_N)), value :: transA
    integer(c_int), value :: n
    integer(c_int), value :: k
    type(c_ptr), value :: alpha
    type(c_ptr), value :: A
    integer(c_int), value :: lda
    type(c_ptr), value :: B
    integer(c_int), value :: ldb
    type(c_ptr), value :: beta
    type(c_ptr), value :: C
    integer(c_int), value :: ldc
    integer(c_int), value :: batch_count
            hipblasCsyrkxBatchedFortran = &
        hipblasCsyrkxBatched(handle, uplo, transA, n, k, alpha, &
                                A, lda, B, ldb, beta, C, ldc, batch_count)
end function hipblasCsyrkxBatchedFortran

function hipblasZsyrkxBatchedFortran(handle, uplo, transA, n, k, alpha, &
                                        A, lda, B, ldb, beta, C, ldc, batch_count) &
    bind(c, name='hipblasZsyrkxBatchedFortran')
    use iso_c_binding
    use hipblas_enums
    implicit none
    integer(kind(HIPBLAS_STATUS_SUCCESS)) :: hipblasZsyrkxBatchedFortran
    type(c_ptr), value :: handle
    integer(kind(HIPBLAS_FILL_MODE_FULL)), value :: uplo
    integer(kind(HIPBLAS_OP_N)), value :: transA
    integer(c_int), value :: n
    integer(c_int), value :: k
    type(c_ptr), value :: alpha
    type(c_ptr), value :: A
    integer(c_int), value :: lda
    type(c_ptr), value :: B
    integer(c_int), value :: ldb
    type(c_ptr), value :: beta
    type(c_ptr), value :: C
    integer(c_int), value :: ldc
    integer(c_int), value :: batch_count
            hipblasZsyrkxBatchedFortran = &
        hipblasZsyrkxBatched(handle, uplo, transA, n, k, alpha, &
                                A, lda, B, ldb, beta, C, ldc, batch_count)
end function hipblasZsyrkxBatchedFortran

! syrkxStridedBatched
function hipblasSsyrkxStridedBatchedFortran(handle, uplo, transA, n, k, alpha, &
                                            A, lda, stride_A, B, ldb, stride_B, beta, C, ldc, stride_C, batch_count) &
    bind(c, name='hipblasSsyrkxStridedBatchedFortran')
    use iso_c_binding
    use hipblas_enums
    implicit none
    integer(kind(HIPBLAS_STATUS_SUCCESS)) :: hipblasSsyrkxStridedBatchedFortran
    type(c_ptr), value :: handle
    integer(kind(HIPBLAS_FILL_MODE_FULL)), value :: uplo
    integer(kind(HIPBLAS_OP_N)), value :: transA
    integer(c_int), value :: n
    integer(c_int), value :: k
    type(c_ptr), value :: alpha
    type(c_ptr), value :: A
    integer(c_int), value :: lda
    integer(c_int64_t), value :: stride_A
    type(c_ptr), value :: B
    integer(c_int), value :: ldb
    integer(c_int64_t), value :: stride_B
    type(c_ptr), value :: beta
    type(c_ptr), value :: C
    integer(c_int), value :: ldc
    integer(c_int64_t), value :: stride_C
    integer(c_int), value :: batch_count
            hipblasSsyrkxStridedBatchedFortran = &
        hipblasSsyrkxStridedBatched(handle, uplo, transA, n, k, alpha, &
                                    A, lda, stride_A, B, ldb, stride_B, beta, C, ldc, stride_C, batch_count)
end function hipblasSsyrkxStridedBatchedFortran

function hipblasDsyrkxStridedBatchedFortran(handle, uplo, transA, n, k, alpha, &
                                            A, lda, stride_A, B, ldb, stride_B, beta, C, ldc, stride_C, batch_count) &
    bind(c, name='hipblasDsyrkxStridedBatchedFortran')
    use iso_c_binding
    use hipblas_enums
    implicit none
    integer(kind(HIPBLAS_STATUS_SUCCESS)) :: hipblasDsyrkxStridedBatchedFortran
    type(c_ptr), value :: handle
    integer(kind(HIPBLAS_FILL_MODE_FULL)), value :: uplo
    integer(kind(HIPBLAS_OP_N)), value :: transA
    integer(c_int), value :: n
    integer(c_int), value :: k
    type(c_ptr), value :: alpha
    type(c_ptr), value :: A
    integer(c_int), value :: lda
    integer(c_int64_t), value :: stride_A
    type(c_ptr), value :: B
    integer(c_int), value :: ldb
    integer(c_int64_t), value :: stride_B
    type(c_ptr), value :: beta
    type(c_ptr), value :: C
    integer(c_int), value :: ldc
    integer(c_int64_t), value :: stride_C
    integer(c_int), value :: batch_count
            hipblasDsyrkxStridedBatchedFortran = &
        hipblasDsyrkxStridedBatched(handle, uplo, transA, n, k, alpha, &
                                    A, lda, stride_A, B, ldb, stride_B, beta, C, ldc, stride_C, batch_count)
end function hipblasDsyrkxStridedBatchedFortran

function hipblasCsyrkxStridedBatchedFortran(handle, uplo, transA, n, k, alpha, &
                                            A, lda, stride_A, B, ldb, stride_B, beta, C, ldc, stride_C, batch_count) &
    bind(c, name='hipblasCsyrkxStridedBatchedFortran')
    use iso_c_binding
    use hipblas_enums
    implicit none
    integer(kind(HIPBLAS_STATUS_SUCCESS)) :: hipblasCsyrkxStridedBatchedFortran
    type(c_ptr), value :: handle
    integer(kind(HIPBLAS_FILL_MODE_FULL)), value :: uplo
    integer(kind(HIPBLAS_OP_N)), value :: transA
    integer(c_int), value :: n
    integer(c_int), value :: k
    type(c_ptr), value :: alpha
    type(c_ptr), value :: A
    integer(c_int), value :: lda
    integer(c_int64_t), value :: stride_A
    type(c_ptr), value :: B
    integer(c_int), value :: ldb
    integer(c_int64_t), value :: stride_B
    type(c_ptr), value :: beta
    type(c_ptr), value :: C
    integer(c_int), value :: ldc
    integer(c_int64_t), value :: stride_C
    integer(c_int), value :: batch_count
            hipblasCsyrkxStridedBatchedFortran = &
        hipblasCsyrkxStridedBatched(handle, uplo, transA, n, k, alpha, &
                                    A, lda, stride_A, B, ldb, stride_B, beta, C, ldc, stride_C, batch_count)
end function hipblasCsyrkxStridedBatchedFortran

function hipblasZsyrkxStridedBatchedFortran(handle, uplo, transA, n, k, alpha, &
                                            A, lda, stride_A, B, ldb, stride_B, beta, C, ldc, stride_C, batch_count) &
    bind(c, name='hipblasZsyrkxStridedBatchedFortran')
    use iso_c_binding
    use hipblas_enums
    implicit none
    integer(kind(HIPBLAS_STATUS_SUCCESS)) :: hipblasZsyrkxStridedBatchedFortran
    type(c_ptr), value :: handle
    integer(kind(HIPBLAS_FILL_MODE_FULL)), value :: uplo
    integer(kind(HIPBLAS_OP_N)), value :: transA
    integer(c_int), value :: n
    integer(c_int), value :: k
    type(c_ptr), value :: alpha
    type(c_ptr), value :: A
    integer(c_int), value :: lda
    integer(c_int64_t), value :: stride_A
    type(c_ptr), value :: B
    integer(c_int), value :: ldb
    integer(c_int64_t), value :: stride_B
    type(c_ptr), value :: beta
    type(c_ptr), value :: C
    integer(c_int), value :: ldc
    integer(c_int64_t), value :: stride_C
    integer(c_int), value :: batch_count
            hipblasZsyrkxStridedBatchedFortran = &
        hipblasZsyrkxStridedBatched(handle, uplo, transA, n, k, alpha, &
                                    A, lda, stride_A, B, ldb, stride_B, beta, C, ldc, stride_C, batch_count)
end function hipblasZsyrkxStridedBatchedFortran

! trmm
function hipblasStrmmFortran(handle, side, uplo, transA, diag, m, n, alpha, &
                                A, lda, B, ldb, C, ldc) &
    bind(c, name='hipblasStrmmFortran')
    use iso_c_binding
    use hipblas_enums
    implicit none
    integer(kind(HIPBLAS_STATUS_SUCCESS)) :: hipblasStrmmFortran
    type(c_ptr), value :: handle
    integer(kind(HIPBLAS_SIDE_LEFT)), value :: side
    integer(kind(HIPBLAS_FILL_MODE_FULL)), value :: uplo
    integer(kind(HIPBLAS_OP_N)), value :: transA
    integer(kind(HIPBLAS_DIAG_UNIT)), value :: diag
    integer(c_int), value :: m
    integer(c_int), value :: n
    type(c_ptr), value :: alpha
    type(c_ptr), value :: A
    integer(c_int), value :: lda
    type(c_ptr), value :: B
    integer(c_int), value :: ldb
    type(c_ptr), value :: C
    integer(c_int), value :: ldc
            hipblasStrmmFortran = &
        hipblasStrmm(handle, side, uplo, transA, diag, m, n, alpha, &
                        A, lda, B, ldb, C, ldc)
end function hipblasStrmmFortran

function hipblasDtrmmFortran(handle, side, uplo, transA, diag, m, n, alpha, &
                                A, lda, B, ldb, C, ldc) &
    bind(c, name='hipblasDtrmmFortran')
    use iso_c_binding
    use hipblas_enums
    implicit none
    integer(kind(HIPBLAS_STATUS_SUCCESS)) :: hipblasDtrmmFortran
    type(c_ptr), value :: handle
    integer(kind(HIPBLAS_SIDE_LEFT)), value :: side
    integer(kind(HIPBLAS_FILL_MODE_FULL)), value :: uplo
    integer(kind(HIPBLAS_OP_N)), value :: transA
    integer(kind(HIPBLAS_DIAG_UNIT)), value :: diag
    integer(c_int), value :: m
    integer(c_int), value :: n
    type(c_ptr), value :: alpha
    type(c_ptr), value :: A
    integer(c_int), value :: lda
    type(c_ptr), value :: B
    integer(c_int), value :: ldb
    type(c_ptr), value :: C
    integer(c_int), value :: ldc
            hipblasDtrmmFortran = &
        hipblasDtrmm(handle, side, uplo, transA, diag, m, n, alpha, &
                        A, lda, B, ldb, C, ldc)
end function hipblasDtrmmFortran

function hipblasCtrmmFortran(handle, side, uplo, transA, diag, m, n, alpha, &
                                A, lda, B, ldb, C, ldc) &
    bind(c, name='hipblasCtrmmFortran')
    use iso_c_binding
    use hipblas_enums
    implicit none
    integer(kind(HIPBLAS_STATUS_SUCCESS)) :: hipblasCtrmmFortran
    type(c_ptr), value :: handle
    integer(kind(HIPBLAS_SIDE_LEFT)), value :: side
    integer(kind(HIPBLAS_FILL_MODE_FULL)), value :: uplo
    integer(kind(HIPBLAS_OP_N)), value :: transA
    integer(kind(HIPBLAS_DIAG_UNIT)), value :: diag
    integer(c_int), value :: m
    integer(c_int), value :: n
    type(c_ptr), value :: alpha
    type(c_ptr), value :: A
    integer(c_int), value :: lda
    type(c_ptr), value :: B
    integer(c_int), value :: ldb
    type(c_ptr), value :: C
    integer(c_int), value :: ldc
            hipblasCtrmmFortran = &
        hipblasCtrmm(handle, side, uplo, transA, diag, m, n, alpha, &
                        A, lda, B, ldb, C, ldc)
end function hipblasCtrmmFortran

function hipblasZtrmmFortran(handle, side, uplo, transA, diag, m, n, alpha, &
                                A, lda, B, ldb, C, ldc) &
    bind(c, name='hipblasZtrmmFortran')
    use iso_c_binding
    use hipblas_enums
    implicit none
    integer(kind(HIPBLAS_STATUS_SUCCESS)) :: hipblasZtrmmFortran
    type(c_ptr), value :: handle
    integer(kind(HIPBLAS_SIDE_LEFT)), value :: side
    integer(kind(HIPBLAS_FILL_MODE_FULL)), value :: uplo
    integer(kind(HIPBLAS_OP_N)), value :: transA
    integer(kind(HIPBLAS_DIAG_UNIT)), value :: diag
    integer(c_int), value :: m
    integer(c_int), value :: n
    type(c_ptr), value :: alpha
    type(c_ptr), value :: A
    integer(c_int), value :: lda
    type(c_ptr), value :: B
    integer(c_int), value :: ldb
    type(c_ptr), value :: C
    integer(c_int), value :: ldc
            hipblasZtrmmFortran = &
        hipblasZtrmm(handle, side, uplo, transA, diag, m, n, alpha, &
                        A, lda, B, ldb, C, ldc)
end function hipblasZtrmmFortran

! trmmBatched
function hipblasStrmmBatchedFortran(handle, side, uplo, transA, diag, m, n, alpha, &
                                    A, lda, B, ldb, C, ldc, batch_count) &
    bind(c, name='hipblasStrmmBatchedFortran')
    use iso_c_binding
    use hipblas_enums
    implicit none
    integer(kind(HIPBLAS_STATUS_SUCCESS)) :: hipblasStrmmBatchedFortran
    type(c_ptr), value :: handle
    integer(kind(HIPBLAS_SIDE_LEFT)), value :: side
    integer(kind(HIPBLAS_FILL_MODE_FULL)), value :: uplo
    integer(kind(HIPBLAS_OP_N)), value :: transA
    integer(kind(HIPBLAS_DIAG_UNIT)), value :: diag
    integer(c_int), value :: m
    integer(c_int), value :: n
    type(c_ptr), value :: alpha
    type(c_ptr), value :: A
    integer(c_int), value :: lda
    type(c_ptr), value :: B
    integer(c_int), value :: ldb
    type(c_ptr), value :: C
    integer(c_int), value :: ldc
    integer(c_int), value :: batch_count
            hipblasStrmmBatchedFortran = &
        hipblasStrmmBatched(handle, side, uplo, transA, diag, m, n, alpha, &
                            A, lda, B, ldb, C, ldc, batch_count)
end function hipblasStrmmBatchedFortran

function hipblasDtrmmBatchedFortran(handle, side, uplo, transA, diag, m, n, alpha, &
                                    A, lda, B, ldb, C, ldc, batch_count) &
    bind(c, name='hipblasDtrmmBatchedFortran')
    use iso_c_binding
    use hipblas_enums
    implicit none
    integer(kind(HIPBLAS_STATUS_SUCCESS)) :: hipblasDtrmmBatchedFortran
    type(c_ptr), value :: handle
    integer(kind(HIPBLAS_SIDE_LEFT)), value :: side
    integer(kind(HIPBLAS_FILL_MODE_FULL)), value :: uplo
    integer(kind(HIPBLAS_OP_N)), value :: transA
    integer(kind(HIPBLAS_DIAG_UNIT)), value :: diag
    integer(c_int), value :: m
    integer(c_int), value :: n
    type(c_ptr), value :: alpha
    type(c_ptr), value :: A
    integer(c_int), value :: lda
    type(c_ptr), value :: B
    integer(c_int), value :: ldb
    type(c_ptr), value :: C
    integer(c_int), value :: ldc
    integer(c_int), value :: batch_count
            hipblasDtrmmBatchedFortran = &
        hipblasDtrmmBatched(handle, side, uplo, transA, diag, m, n, alpha, &
                            A, lda, B, ldb, C, ldc, batch_count)
end function hipblasDtrmmBatchedFortran

function hipblasCtrmmBatchedFortran(handle, side, uplo, transA, diag, m, n, alpha, &
                                    A, lda, B, ldb, C, ldc, batch_count) &
    bind(c, name='hipblasCtrmmBatchedFortran')
    use iso_c_binding
    use hipblas_enums
    implicit none
    integer(kind(HIPBLAS_STATUS_SUCCESS)) :: hipblasCtrmmBatchedFortran
    type(c_ptr), value :: handle
    integer(kind(HIPBLAS_SIDE_LEFT)), value :: side
    integer(kind(HIPBLAS_FILL_MODE_FULL)), value :: uplo
    integer(kind(HIPBLAS_OP_N)), value :: transA
    integer(kind(HIPBLAS_DIAG_UNIT)), value :: diag
    integer(c_int), value :: m
    integer(c_int), value :: n
    type(c_ptr), value :: alpha
    type(c_ptr), value :: A
    integer(c_int), value :: lda
    type(c_ptr), value :: B
    integer(c_int), value :: ldb
    type(c_ptr), value :: C
    integer(c_int), value :: ldc
    integer(c_int), value :: batch_count
            hipblasCtrmmBatchedFortran = &
        hipblasCtrmmBatched(handle, side, uplo, transA, diag, m, n, alpha, &
                            A, lda, B, ldb, C, ldc, batch_count)
end function hipblasCtrmmBatchedFortran

function hipblasZtrmmBatchedFortran(handle, side, uplo, transA, diag, m, n, alpha, &
                                    A, lda, B, ldb, C, ldc, batch_count) &
    bind(c, name='hipblasZtrmmBatchedFortran')
    use iso_c_binding
    use hipblas_enums
    implicit none
    integer(kind(HIPBLAS_STATUS_SUCCESS)) :: hipblasZtrmmBatchedFortran
    type(c_ptr), value :: handle
    integer(kind(HIPBLAS_SIDE_LEFT)), value :: side
    integer(kind(HIPBLAS_FILL_MODE_FULL)), value :: uplo
    integer(kind(HIPBLAS_OP_N)), value :: transA
    integer(kind(HIPBLAS_DIAG_UNIT)), value :: diag
    integer(c_int), value :: m
    integer(c_int), value :: n
    type(c_ptr), value :: alpha
    type(c_ptr), value :: A
    integer(c_int), value :: lda
    type(c_ptr), value :: B
    integer(c_int), value :: ldb
    type(c_ptr), value :: C
    integer(c_int), value :: ldc
    integer(c_int), value :: batch_count
            hipblasZtrmmBatchedFortran = &
        hipblasZtrmmBatched(handle, side, uplo, transA, diag, m, n, alpha, &
                            A, lda, B, ldb, C, ldc, batch_count)
end function hipblasZtrmmBatchedFortran

! trmmStridedBatched
function hipblasStrmmStridedBatchedFortran(handle, side, uplo, transA, diag, m, n, alpha, &
                                            A, lda, stride_A, B, ldb, stride_B, C, ldc, stride_C, batch_count) &
    bind(c, name='hipblasStrmmStridedBatchedFortran')
    use iso_c_binding
    use hipblas_enums
    implicit none
    integer(kind(HIPBLAS_STATUS_SUCCESS)) :: hipblasStrmmStridedBatchedFortran
    type(c_ptr), value :: handle
    integer(kind(HIPBLAS_SIDE_LEFT)), value :: side
    integer(kind(HIPBLAS_FILL_MODE_FULL)), value :: uplo
    integer(kind(HIPBLAS_OP_N)), value :: transA
    integer(kind(HIPBLAS_DIAG_UNIT)), value :: diag
    integer(c_int), value :: m
    integer(c_int), value :: n
    type(c_ptr), value :: alpha
    type(c_ptr), value :: A
    integer(c_int), value :: lda
    integer(c_int64_t), value :: stride_A
    type(c_ptr), value :: B
    integer(c_int), value :: ldb
    integer(c_int64_t), value :: stride_B
    type(c_ptr), value :: C
    integer(c_int), value :: ldc
    integer(c_int64_t), value :: stride_C
    integer(c_int), value :: batch_count
            hipblasStrmmStridedBatchedFortran = &
        hipblasStrmmStridedBatched(handle, side, uplo, transA, diag, m, n, alpha, &
                                    A, lda, stride_A, B, ldb, stride_B, C, ldc, stride_C, batch_count)
end function hipblasStrmmStridedBatchedFortran

function hipblasDtrmmStridedBatchedFortran(handle, side, uplo, transA, diag, m, n, alpha, &
                                            A, lda, stride_A, B, ldb, stride_B, C, ldc, stride_C, batch_count) &
    bind(c, name='hipblasDtrmmStridedBatchedFortran')
    use iso_c_binding
    use hipblas_enums
    implicit none
    integer(kind(HIPBLAS_STATUS_SUCCESS)) :: hipblasDtrmmStridedBatchedFortran
    type(c_ptr), value :: handle
    integer(kind(HIPBLAS_SIDE_LEFT)), value :: side
    integer(kind(HIPBLAS_FILL_MODE_FULL)), value :: uplo
    integer(kind(HIPBLAS_OP_N)), value :: transA
    integer(kind(HIPBLAS_DIAG_UNIT)), value :: diag
    integer(c_int), value :: m
    integer(c_int), value :: n
    type(c_ptr), value :: alpha
    type(c_ptr), value :: A
    integer(c_int), value :: lda
    integer(c_int64_t), value :: stride_A
    type(c_ptr), value :: B
    integer(c_int), value :: ldb
    integer(c_int64_t), value :: stride_B
    type(c_ptr), value :: C
    integer(c_int), value :: ldc
    integer(c_int64_t), value :: stride_C
    integer(c_int), value :: batch_count
            hipblasDtrmmStridedBatchedFortran = &
        hipblasDtrmmStridedBatched(handle, side, uplo, transA, diag, m, n, alpha, &
                                    A, lda, stride_A, B, ldb, stride_B, C, ldc, stride_C, batch_count)
end function hipblasDtrmmStridedBatchedFortran

function hipblasCtrmmStridedBatchedFortran(handle, side, uplo, transA, diag, m, n, alpha, &
                                            A, lda, stride_A, B, ldb, stride_B, C, ldc, stride_C, batch_count) &
    bind(c, name='hipblasCtrmmStridedBatchedFortran')
    use iso_c_binding
    use hipblas_enums
    implicit none
    integer(kind(HIPBLAS_STATUS_SUCCESS)) :: hipblasCtrmmStridedBatchedFortran
    type(c_ptr), value :: handle
    integer(kind(HIPBLAS_SIDE_LEFT)), value :: side
    integer(kind(HIPBLAS_FILL_MODE_FULL)), value :: uplo
    integer(kind(HIPBLAS_OP_N)), value :: transA
    integer(kind(HIPBLAS_DIAG_UNIT)), value :: diag
    integer(c_int), value :: m
    integer(c_int), value :: n
    type(c_ptr), value :: alpha
    type(c_ptr), value :: A
    integer(c_int), value :: lda
    integer(c_int64_t), value :: stride_A
    type(c_ptr), value :: B
    integer(c_int), value :: ldb
    integer(c_int64_t), value :: stride_B
    type(c_ptr), value :: C
    integer(c_int), value :: ldc
    integer(c_int64_t), value :: stride_C
    integer(c_int), value :: batch_count
            hipblasCtrmmStridedBatchedFortran = &
        hipblasCtrmmStridedBatched(handle, side, uplo, transA, diag, m, n, alpha, &
                                    A, lda, stride_A, B, ldb, stride_B, C, ldc, stride_C, batch_count)
end function hipblasCtrmmStridedBatchedFortran

function hipblasZtrmmStridedBatchedFortran(handle, side, uplo, transA, diag, m, n, alpha, &
                                            A, lda, stride_A, B, ldb, stride_B, C, ldc, stride_C, batch_count) &
    bind(c, name='hipblasZtrmmStridedBatchedFortran')
    use iso_c_binding
    use hipblas_enums
    implicit none
    integer(kind(HIPBLAS_STATUS_SUCCESS)) :: hipblasZtrmmStridedBatchedFortran
    type(c_ptr), value :: handle
    integer(kind(HIPBLAS_SIDE_LEFT)), value :: side
    integer(kind(HIPBLAS_FILL_MODE_FULL)), value :: uplo
    integer(kind(HIPBLAS_OP_N)), value :: transA
    integer(kind(HIPBLAS_DIAG_UNIT)), value :: diag
    integer(c_int), value :: m
    integer(c_int), value :: n
    type(c_ptr), value :: alpha
    type(c_ptr), value :: A
    integer(c_int), value :: lda
    integer(c_int64_t), value :: stride_A
    type(c_ptr), value :: B
    integer(c_int), value :: ldb
    integer(c_int64_t), value :: stride_B
    type(c_ptr), value :: C
    integer(c_int), value :: ldc
    integer(c_int64_t), value :: stride_C
    integer(c_int), value :: batch_count
            hipblasZtrmmStridedBatchedFortran = &
        hipblasZtrmmStridedBatched(handle, side, uplo, transA, diag, m, n, alpha, &
                                    A, lda, stride_A, B, ldb, stride_B, C, ldc, stride_C, batch_count)
end function hipblasZtrmmStridedBatchedFortran

! trtri
function hipblasStrtriFortran(handle, uplo, diag, n, &
                                A, lda, invA, ldinvA) &
    bind(c, name='hipblasStrtriFortran')
    use iso_c_binding
    use hipblas_enums
    implicit none
    integer(kind(HIPBLAS_STATUS_SUCCESS)) :: hipblasStrtriFortran
    type(c_ptr), value :: handle
    integer(kind(HIPBLAS_FILL_MODE_FULL)), value :: uplo
    integer(kind(HIPBLAS_DIAG_UNIT)), value :: diag
    integer(c_int), value :: n
    type(c_ptr), value :: A
    integer(c_int), value :: lda
    type(c_ptr), value :: invA
    integer(c_int), value :: ldinvA
            hipblasStrtriFortran = &
        hipblasStrtri(handle, uplo, diag, n, &
                        A, lda, invA, ldinvA)
end function hipblasStrtriFortran

function hipblasDtrtriFortran(handle, uplo, diag, n, &
                                A, lda, invA, ldinvA) &
    bind(c, name='hipblasDtrtriFortran')
    use iso_c_binding
    use hipblas_enums
    implicit none
    integer(kind(HIPBLAS_STATUS_SUCCESS)) :: hipblasDtrtriFortran
    type(c_ptr), value :: handle
    integer(kind(HIPBLAS_FILL_MODE_FULL)), value :: uplo
    integer(kind(HIPBLAS_DIAG_UNIT)), value :: diag
    integer(c_int), value :: n
    type(c_ptr), value :: A
    integer(c_int), value :: lda
    type(c_ptr), value :: invA
    integer(c_int), value :: ldinvA
            hipblasDtrtriFortran = &
        hipblasDtrtri(handle, uplo, diag, n, &
                        A, lda, invA, ldinvA)
end function hipblasDtrtriFortran

function hipblasCtrtriFortran(handle, uplo, diag, n, &
                                A, lda, invA, ldinvA) &
    bind(c, name='hipblasCtrtriFortran')
    use iso_c_binding
    use hipblas_enums
    implicit none
    integer(kind(HIPBLAS_STATUS_SUCCESS)) :: hipblasCtrtriFortran
    type(c_ptr), value :: handle
    integer(kind(HIPBLAS_FILL_MODE_FULL)), value :: uplo
    integer(kind(HIPBLAS_DIAG_UNIT)), value :: diag
    integer(c_int), value :: n
    type(c_ptr), value :: A
    integer(c_int), value :: lda
    type(c_ptr), value :: invA
    integer(c_int), value :: ldinvA
            hipblasCtrtriFortran = &
        hipblasCtrtri(handle, uplo, diag, n, &
                        A, lda, invA, ldinvA)
end function hipblasCtrtriFortran

function hipblasZtrtriFortran(handle, uplo, diag, n, &
                                A, lda, invA, ldinvA) &
    bind(c, name='hipblasZtrtriFortran')
    use iso_c_binding
    use hipblas_enums
    implicit none
    integer(kind(HIPBLAS_STATUS_SUCCESS)) :: hipblasZtrtriFortran
    type(c_ptr), value :: handle
    integer(kind(HIPBLAS_FILL_MODE_FULL)), value :: uplo
    integer(kind(HIPBLAS_DIAG_UNIT)), value :: diag
    integer(c_int), value :: n
    type(c_ptr), value :: A
    integer(c_int), value :: lda
    type(c_ptr), value :: invA
    integer(c_int), value :: ldinvA
            hipblasZtrtriFortran = &
        hipblasZtrtri(handle, uplo, diag, n, &
                        A, lda, invA, ldinvA)
end function hipblasZtrtriFortran

! trtriBatched
function hipblasStrtriBatchedFortran(handle, uplo, diag, n, &
                                        A, lda, invA, ldinvA, batch_count) &
    bind(c, name='hipblasStrtriBatchedFortran')
    use iso_c_binding
    use hipblas_enums
    implicit none
    integer(kind(HIPBLAS_STATUS_SUCCESS)) :: hipblasStrtriBatchedFortran
    type(c_ptr), value :: handle
    integer(kind(HIPBLAS_FILL_MODE_FULL)), value :: uplo
    integer(kind(HIPBLAS_DIAG_UNIT)), value :: diag
    integer(c_int), value :: n
    type(c_ptr), value :: A
    integer(c_int), value :: lda
    type(c_ptr), value :: invA
    integer(c_int), value :: ldinvA
    integer(c_int), value :: batch_count
            hipblasStrtriBatchedFortran = &
        hipblasStrtriBatched(handle, uplo, diag, n, &
                                A, lda, invA, ldinvA, batch_count)
end function hipblasStrtriBatchedFortran

function hipblasDtrtriBatchedFortran(handle, uplo, diag, n, &
                                        A, lda, invA, ldinvA, batch_count) &
    bind(c, name='hipblasDtrtriBatchedFortran')
    use iso_c_binding
    use hipblas_enums
    implicit none
    integer(kind(HIPBLAS_STATUS_SUCCESS)) :: hipblasDtrtriBatchedFortran
    type(c_ptr), value :: handle
    integer(kind(HIPBLAS_FILL_MODE_FULL)), value :: uplo
    integer(kind(HIPBLAS_DIAG_UNIT)), value :: diag
    integer(c_int), value :: n
    type(c_ptr), value :: A
    integer(c_int), value :: lda
    type(c_ptr), value :: invA
    integer(c_int), value :: ldinvA
    integer(c_int), value :: batch_count
            hipblasDtrtriBatchedFortran = &
        hipblasDtrtriBatched(handle, uplo, diag, n, &
                                A, lda, invA, ldinvA, batch_count)
end function hipblasDtrtriBatchedFortran

function hipblasCtrtriBatchedFortran(handle, uplo, diag, n, &
                                        A, lda, invA, ldinvA, batch_count) &
    bind(c, name='hipblasCtrtriBatchedFortran')
    use iso_c_binding
    use hipblas_enums
    implicit none
    integer(kind(HIPBLAS_STATUS_SUCCESS)) :: hipblasCtrtriBatchedFortran
    type(c_ptr), value :: handle
    integer(kind(HIPBLAS_FILL_MODE_FULL)), value :: uplo
    integer(kind(HIPBLAS_DIAG_UNIT)), value :: diag
    integer(c_int), value :: n
    type(c_ptr), value :: A
    integer(c_int), value :: lda
    type(c_ptr), value :: invA
    integer(c_int), value :: ldinvA
    integer(c_int), value :: batch_count
            hipblasCtrtriBatchedFortran = &
        hipblasCtrtriBatched(handle, uplo, diag, n, &
                                A, lda, invA, ldinvA, batch_count)
end function hipblasCtrtriBatchedFortran

function hipblasZtrtriBatchedFortran(handle, uplo, diag, n, &
                                        A, lda, invA, ldinvA, batch_count) &
    bind(c, name='hipblasZtrtriBatchedFortran')
    use iso_c_binding
    use hipblas_enums
    implicit none
    integer(kind(HIPBLAS_STATUS_SUCCESS)) :: hipblasZtrtriBatchedFortran
    type(c_ptr), value :: handle
    integer(kind(HIPBLAS_FILL_MODE_FULL)), value :: uplo
    integer(kind(HIPBLAS_DIAG_UNIT)), value :: diag
    integer(c_int), value :: n
    type(c_ptr), value :: A
    integer(c_int), value :: lda
    type(c_ptr), value :: invA
    integer(c_int), value :: ldinvA
    integer(c_int), value :: batch_count
            hipblasZtrtriBatchedFortran = &
        hipblasZtrtriBatched(handle, uplo, diag, n, &
                                A, lda, invA, ldinvA, batch_count)
end function hipblasZtrtriBatchedFortran

! trtriStridedBatched
function hipblasStrtriStridedBatchedFortran(handle, uplo, diag, n, &
                                            A, lda, stride_A, invA, ldinvA, stride_invA, batch_count) &
    bind(c, name='hipblasStrtriStridedBatchedFortran')
    use iso_c_binding
    use hipblas_enums
    implicit none
    integer(kind(HIPBLAS_STATUS_SUCCESS)) :: hipblasStrtriStridedBatchedFortran
    type(c_ptr), value :: handle
    integer(kind(HIPBLAS_FILL_MODE_FULL)), value :: uplo
    integer(kind(HIPBLAS_DIAG_UNIT)), value :: diag
    integer(c_int), value :: n
    type(c_ptr), value :: A
    integer(c_int), value :: lda
    integer(c_int64_t), value :: stride_A
    type(c_ptr), value :: invA
    integer(c_int), value :: ldinvA
    integer(c_int64_t), value :: stride_invA
    integer(c_int), value :: batch_count
            hipblasStrtriStridedBatchedFortran = &
        hipblasStrtriStridedBatched(handle, uplo, diag, n, &
                                    A, lda, stride_A, invA, ldinvA, stride_invA, batch_count)
end function hipblasStrtriStridedBatchedFortran

function hipblasDtrtriStridedBatchedFortran(handle, uplo, diag, n, &
                                            A, lda, stride_A, invA, ldinvA, stride_invA, batch_count) &
    bind(c, name='hipblasDtrtriStridedBatchedFortran')
    use iso_c_binding
    use hipblas_enums
    implicit none
    integer(kind(HIPBLAS_STATUS_SUCCESS)) :: hipblasDtrtriStridedBatchedFortran
    type(c_ptr), value :: handle
    integer(kind(HIPBLAS_FILL_MODE_FULL)), value :: uplo
    integer(kind(HIPBLAS_DIAG_UNIT)), value :: diag
    integer(c_int), value :: n
    type(c_ptr), value :: A
    integer(c_int), value :: lda
    integer(c_int64_t), value :: stride_A
    type(c_ptr), value :: invA
    integer(c_int), value :: ldinvA
    integer(c_int64_t), value :: stride_invA
    integer(c_int), value :: batch_count
            hipblasDtrtriStridedBatchedFortran = &
        hipblasDtrtriStridedBatched(handle, uplo, diag, n, &
                                    A, lda, stride_A, invA, ldinvA, stride_invA, batch_count)
end function hipblasDtrtriStridedBatchedFortran

function hipblasCtrtriStridedBatchedFortran(handle, uplo, diag, n, &
                                            A, lda, stride_A, invA, ldinvA, stride_invA, batch_count) &
    bind(c, name='hipblasCtrtriStridedBatchedFortran')
    use iso_c_binding
    use hipblas_enums
    implicit none
    integer(kind(HIPBLAS_STATUS_SUCCESS)) :: hipblasCtrtriStridedBatchedFortran
    type(c_ptr), value :: handle
    integer(kind(HIPBLAS_FILL_MODE_FULL)), value :: uplo
    integer(kind(HIPBLAS_DIAG_UNIT)), value :: diag
    integer(c_int), value :: n
    type(c_ptr), value :: A
    integer(c_int), value :: lda
    integer(c_int64_t), value :: stride_A
    type(c_ptr), value :: invA
    integer(c_int), value :: ldinvA
    integer(c_int64_t), value :: stride_invA
    integer(c_int), value :: batch_count
            hipblasCtrtriStridedBatchedFortran = &
        hipblasCtrtriStridedBatched(handle, uplo, diag, n, &
                                    A, lda, stride_A, invA, ldinvA, stride_invA, batch_count)
end function hipblasCtrtriStridedBatchedFortran

function hipblasZtrtriStridedBatchedFortran(handle, uplo, diag, n, &
                                            A, lda, stride_A, invA, ldinvA, stride_invA, batch_count) &
    bind(c, name='hipblasZtrtriStridedBatchedFortran')
    use iso_c_binding
    use hipblas_enums
    implicit none
    integer(kind(HIPBLAS_STATUS_SUCCESS)) :: hipblasZtrtriStridedBatchedFortran
    type(c_ptr), value :: handle
    integer(kind(HIPBLAS_FILL_MODE_FULL)), value :: uplo
    integer(kind(HIPBLAS_DIAG_UNIT)), value :: diag
    integer(c_int), value :: n
    type(c_ptr), value :: A
    integer(c_int), value :: lda
    integer(c_int64_t), value :: stride_A
    type(c_ptr), value :: invA
    integer(c_int), value :: ldinvA
    integer(c_int64_t), value :: stride_invA
    integer(c_int), value :: batch_count
            hipblasZtrtriStridedBatchedFortran = &
        hipblasZtrtriStridedBatched(handle, uplo, diag, n, &
                                    A, lda, stride_A, invA, ldinvA, stride_invA, batch_count)
end function hipblasZtrtriStridedBatchedFortran

! trsm
function hipblasStrsmFortran(handle, side, uplo, transA, diag, m, n, alpha, &
                                A, lda, B, ldb) &
    bind(c, name='hipblasStrsmFortran')
    use iso_c_binding
    use hipblas_enums
    implicit none
    integer(kind(HIPBLAS_STATUS_SUCCESS)) :: hipblasStrsmFortran
    type(c_ptr), value :: handle
    integer(kind(HIPBLAS_SIDE_LEFT)), value :: side
    integer(kind(HIPBLAS_FILL_MODE_FULL)), value :: uplo
    integer(kind(HIPBLAS_OP_N)), value :: transA
    integer(kind(HIPBLAS_DIAG_UNIT)), value :: diag
    integer(c_int), value :: m
    integer(c_int), value :: n
    type(c_ptr), value :: alpha
    type(c_ptr), value :: A
    integer(c_int), value :: lda
    type(c_ptr), value :: B
    integer(c_int), value :: ldb
            hipblasStrsmFortran = &
        hipblasStrsm(handle, side, uplo, transA, diag, m, n, alpha, &
                        A, lda, B, ldb)
end function hipblasStrsmFortran

function hipblasDtrsmFortran(handle, side, uplo, transA, diag, m, n, alpha, &
                                A, lda, B, ldb) &
    bind(c, name='hipblasDtrsmFortran')
    use iso_c_binding
    use hipblas_enums
    implicit none
    integer(kind(HIPBLAS_STATUS_SUCCESS)) :: hipblasDtrsmFortran
    type(c_ptr), value :: handle
    integer(kind(HIPBLAS_SIDE_LEFT)), value :: side
    integer(kind(HIPBLAS_FILL_MODE_FULL)), value :: uplo
    integer(kind(HIPBLAS_OP_N)), value :: transA
    integer(kind(HIPBLAS_DIAG_UNIT)), value :: diag
    integer(c_int), value :: m
    integer(c_int), value :: n
    type(c_ptr), value :: alpha
    type(c_ptr), value :: A
    integer(c_int), value :: lda
    type(c_ptr), value :: B
    integer(c_int), value :: ldb
            hipblasDtrsmFortran = &
        hipblasDtrsm(handle, side, uplo, transA, diag, m, n, alpha, &
                        A, lda, B, ldb)
end function hipblasDtrsmFortran

function hipblasCtrsmFortran(handle, side, uplo, transA, diag, m, n, alpha, &
                                A, lda, B, ldb) &
    bind(c, name='hipblasCtrsmFortran')
    use iso_c_binding
    use hipblas_enums
    implicit none
    integer(kind(HIPBLAS_STATUS_SUCCESS)) :: hipblasCtrsmFortran
    type(c_ptr), value :: handle
    integer(kind(HIPBLAS_SIDE_LEFT)), value :: side
    integer(kind(HIPBLAS_FILL_MODE_FULL)), value :: uplo
    integer(kind(HIPBLAS_OP_N)), value :: transA
    integer(kind(HIPBLAS_DIAG_UNIT)), value :: diag
    integer(c_int), value :: m
    integer(c_int), value :: n
    type(c_ptr), value :: alpha
    type(c_ptr), value :: A
    integer(c_int), value :: lda
    type(c_ptr), value :: B
    integer(c_int), value :: ldb
            hipblasCtrsmFortran = &
        hipblasCtrsm(handle, side, uplo, transA, diag, m, n, alpha, &
                        A, lda, B, ldb)
end function hipblasCtrsmFortran

function hipblasZtrsmFortran(handle, side, uplo, transA, diag, m, n, alpha, &
                                A, lda, B, ldb) &
    bind(c, name='hipblasZtrsmFortran')
    use iso_c_binding
    use hipblas_enums
    implicit none
    integer(kind(HIPBLAS_STATUS_SUCCESS)) :: hipblasZtrsmFortran
    type(c_ptr), value :: handle
    integer(kind(HIPBLAS_SIDE_LEFT)), value :: side
    integer(kind(HIPBLAS_FILL_MODE_FULL)), value :: uplo
    integer(kind(HIPBLAS_OP_N)), value :: transA
    integer(kind(HIPBLAS_DIAG_UNIT)), value :: diag
    integer(c_int), value :: m
    integer(c_int), value :: n
    type(c_ptr), value :: alpha
    type(c_ptr), value :: A
    integer(c_int), value :: lda
    type(c_ptr), value :: B
    integer(c_int), value :: ldb
            hipblasZtrsmFortran = &
        hipblasZtrsm(handle, side, uplo, transA, diag, m, n, alpha, &
                        A, lda, B, ldb)
end function hipblasZtrsmFortran

! trsmBatched
function hipblasStrsmBatchedFortran(handle, side, uplo, transA, diag, m, n, alpha, &
                                    A, lda, B, ldb, batch_count) &
    bind(c, name='hipblasStrsmBatchedFortran')
    use iso_c_binding
    use hipblas_enums
    implicit none
    integer(kind(HIPBLAS_STATUS_SUCCESS)) :: hipblasStrsmBatchedFortran
    type(c_ptr), value :: handle
    integer(kind(HIPBLAS_SIDE_LEFT)), value :: side
    integer(kind(HIPBLAS_FILL_MODE_FULL)), value :: uplo
    integer(kind(HIPBLAS_OP_N)), value :: transA
    integer(kind(HIPBLAS_DIAG_UNIT)), value :: diag
    integer(c_int), value :: m
    integer(c_int), value :: n
    type(c_ptr), value :: alpha
    type(c_ptr), value :: A
    integer(c_int), value :: lda
    type(c_ptr), value :: B
    integer(c_int), value :: ldb
    integer(c_int), value :: batch_count
            hipblasStrsmBatchedFortran = &
        hipblasStrsmBatched(handle, side, uplo, transA, diag, m, n, alpha, &
                            A, lda, B, ldb, batch_count)
end function hipblasStrsmBatchedFortran

function hipblasDtrsmBatchedFortran(handle, side, uplo, transA, diag, m, n, alpha, &
                                    A, lda, B, ldb, batch_count) &
    bind(c, name='hipblasDtrsmBatchedFortran')
    use iso_c_binding
    use hipblas_enums
    implicit none
    integer(kind(HIPBLAS_STATUS_SUCCESS)) :: hipblasDtrsmBatchedFortran
    type(c_ptr), value :: handle
    integer(kind(HIPBLAS_SIDE_LEFT)), value :: side
    integer(kind(HIPBLAS_FILL_MODE_FULL)), value :: uplo
    integer(kind(HIPBLAS_OP_N)), value :: transA
    integer(kind(HIPBLAS_DIAG_UNIT)), value :: diag
    integer(c_int), value :: m
    integer(c_int), value :: n
    type(c_ptr), value :: alpha
    type(c_ptr), value :: A
    integer(c_int), value :: lda
    type(c_ptr), value :: B
    integer(c_int), value :: ldb
    integer(c_int), value :: batch_count
            hipblasDtrsmBatchedFortran = &
        hipblasDtrsmBatched(handle, side, uplo, transA, diag, m, n, alpha, &
                            A, lda, B, ldb, batch_count)
end function hipblasDtrsmBatchedFortran

function hipblasCtrsmBatchedFortran(handle, side, uplo, transA, diag, m, n, alpha, &
                                    A, lda, B, ldb, batch_count) &
    bind(c, name='hipblasCtrsmBatchedFortran')
    use iso_c_binding
    use hipblas_enums
    implicit none
    integer(kind(HIPBLAS_STATUS_SUCCESS)) :: hipblasCtrsmBatchedFortran
    type(c_ptr), value :: handle
    integer(kind(HIPBLAS_SIDE_LEFT)), value :: side
    integer(kind(HIPBLAS_FILL_MODE_FULL)), value :: uplo
    integer(kind(HIPBLAS_OP_N)), value :: transA
    integer(kind(HIPBLAS_DIAG_UNIT)), value :: diag
    integer(c_int), value :: m
    integer(c_int), value :: n
    type(c_ptr), value :: alpha
    type(c_ptr), value :: A
    integer(c_int), value :: lda
    type(c_ptr), value :: B
    integer(c_int), value :: ldb
    integer(c_int), value :: batch_count
            hipblasCtrsmBatchedFortran = &
        hipblasCtrsmBatched(handle, side, uplo, transA, diag, m, n, alpha, &
                            A, lda, B, ldb, batch_count)
end function hipblasCtrsmBatchedFortran

function hipblasZtrsmBatchedFortran(handle, side, uplo, transA, diag, m, n, alpha, &
                                    A, lda, B, ldb, batch_count) &
    bind(c, name='hipblasZtrsmBatchedFortran')
    use iso_c_binding
    use hipblas_enums
    implicit none
    integer(kind(HIPBLAS_STATUS_SUCCESS)) :: hipblasZtrsmBatchedFortran
    type(c_ptr), value :: handle
    integer(kind(HIPBLAS_SIDE_LEFT)), value :: side
    integer(kind(HIPBLAS_FILL_MODE_FULL)), value :: uplo
    integer(kind(HIPBLAS_OP_N)), value :: transA
    integer(kind(HIPBLAS_DIAG_UNIT)), value :: diag
    integer(c_int), value :: m
    integer(c_int), value :: n
    type(c_ptr), value :: alpha
    type(c_ptr), value :: A
    integer(c_int), value :: lda
    type(c_ptr), value :: B
    integer(c_int), value :: ldb
    integer(c_int), value :: batch_count
            hipblasZtrsmBatchedFortran = &
        hipblasZtrsmBatched(handle, side, uplo, transA, diag, m, n, alpha, &
                            A, lda, B, ldb, batch_count)
end function hipblasZtrsmBatchedFortran

! trsmStridedBatched
function hipblasStrsmStridedBatchedFortran(handle, side, uplo, transA, diag, m, n, alpha, &
                                            A, lda, stride_A, B, ldb, stride_B, batch_count) &
    bind(c, name='hipblasStrsmStridedBatchedFortran')
    use iso_c_binding
    use hipblas_enums
    implicit none
    integer(kind(HIPBLAS_STATUS_SUCCESS)) :: hipblasStrsmStridedBatchedFortran
    type(c_ptr), value :: handle
    integer(kind(HIPBLAS_SIDE_LEFT)), value :: side
    integer(kind(HIPBLAS_FILL_MODE_FULL)), value :: uplo
    integer(kind(HIPBLAS_OP_N)), value :: transA
    integer(kind(HIPBLAS_DIAG_UNIT)), value :: diag
    integer(c_int), value :: m
    integer(c_int), value :: n
    type(c_ptr), value :: alpha
    type(c_ptr), value :: A
    integer(c_int), value :: lda
    integer(c_int64_t), value :: stride_A
    type(c_ptr), value :: B
    integer(c_int), value :: ldb
    integer(c_int64_t), value :: stride_B
    integer(c_int), value :: batch_count
            hipblasStrsmStridedBatchedFortran = &
        hipblasStrsmStridedBatched(handle, side, uplo, transA, diag, m, n, alpha, &
                                    A, lda, stride_A, B, ldb, stride_B, batch_count)
end function hipblasStrsmStridedBatchedFortran

function hipblasDtrsmStridedBatchedFortran(handle, side, uplo, transA, diag, m, n, alpha, &
                                            A, lda, stride_A, B, ldb, stride_B, batch_count) &
    bind(c, name='hipblasDtrsmStridedBatchedFortran')
    use iso_c_binding
    use hipblas_enums
    implicit none
    integer(kind(HIPBLAS_STATUS_SUCCESS)) :: hipblasDtrsmStridedBatchedFortran
    type(c_ptr), value :: handle
    integer(kind(HIPBLAS_SIDE_LEFT)), value :: side
    integer(kind(HIPBLAS_FILL_MODE_FULL)), value :: uplo
    integer(kind(HIPBLAS_OP_N)), value :: transA
    integer(kind(HIPBLAS_DIAG_UNIT)), value :: diag
    integer(c_int), value :: m
    integer(c_int), value :: n
    type(c_ptr), value :: alpha
    type(c_ptr), value :: A
    integer(c_int), value :: lda
    integer(c_int64_t), value :: stride_A
    type(c_ptr), value :: B
    integer(c_int), value :: ldb
    integer(c_int64_t), value :: stride_B
    integer(c_int), value :: batch_count
            hipblasDtrsmStridedBatchedFortran = &
        hipblasDtrsmStridedBatched(handle, side, uplo, transA, diag, m, n, alpha, &
                                    A, lda, stride_A, B, ldb, stride_B, batch_count)
end function hipblasDtrsmStridedBatchedFortran

function hipblasCtrsmStridedBatchedFortran(handle, side, uplo, transA, diag, m, n, alpha, &
                                            A, lda, stride_A, B, ldb, stride_B, batch_count) &
    bind(c, name='hipblasCtrsmStridedBatchedFortran')
    use iso_c_binding
    use hipblas_enums
    implicit none
    integer(kind(HIPBLAS_STATUS_SUCCESS)) :: hipblasCtrsmStridedBatchedFortran
    type(c_ptr), value :: handle
    integer(kind(HIPBLAS_SIDE_LEFT)), value :: side
    integer(kind(HIPBLAS_FILL_MODE_FULL)), value :: uplo
    integer(kind(HIPBLAS_OP_N)), value :: transA
    integer(kind(HIPBLAS_DIAG_UNIT)), value :: diag
    integer(c_int), value :: m
    integer(c_int), value :: n
    type(c_ptr), value :: alpha
    type(c_ptr), value :: A
    integer(c_int), value :: lda
    integer(c_int64_t), value :: stride_A
    type(c_ptr), value :: B
    integer(c_int), value :: ldb
    integer(c_int64_t), value :: stride_B
    integer(c_int), value :: batch_count
            hipblasCtrsmStridedBatchedFortran = &
        hipblasCtrsmStridedBatched(handle, side, uplo, transA, diag, m, n, alpha, &
                                    A, lda, stride_A, B, ldb, stride_B, batch_count)
end function hipblasCtrsmStridedBatchedFortran

function hipblasZtrsmStridedBatchedFortran(handle, side, uplo, transA, diag, m, n, alpha, &
                                            A, lda, stride_A, B, ldb, stride_B, batch_count) &
    bind(c, name='hipblasZtrsmStridedBatchedFortran')
    use iso_c_binding
    use hipblas_enums
    implicit none
    integer(kind(HIPBLAS_STATUS_SUCCESS)) :: hipblasZtrsmStridedBatchedFortran
    type(c_ptr), value :: handle
    integer(kind(HIPBLAS_SIDE_LEFT)), value :: side
    integer(kind(HIPBLAS_FILL_MODE_FULL)), value :: uplo
    integer(kind(HIPBLAS_OP_N)), value :: transA
    integer(kind(HIPBLAS_DIAG_UNIT)), value :: diag
    integer(c_int), value :: m
    integer(c_int), value :: n
    type(c_ptr), value :: alpha
    type(c_ptr), value :: A
    integer(c_int), value :: lda
    integer(c_int64_t), value :: stride_A
    type(c_ptr), value :: B
    integer(c_int), value :: ldb
    integer(c_int64_t), value :: stride_B
    integer(c_int), value :: batch_count
            hipblasZtrsmStridedBatchedFortran = &
        hipblasZtrsmStridedBatched(handle, side, uplo, transA, diag, m, n, alpha, &
                                    A, lda, stride_A, B, ldb, stride_B, batch_count)
end function hipblasZtrsmStridedBatchedFortran

! gemm
function hipblasHgemmFortran(handle, transA, transB, m, n, k, alpha, &
                                A, lda, B, ldb, beta, C, ldc) &
    bind(c, name='hipblasHgemmFortran')
    use iso_c_binding
    use hipblas_enums
    implicit none
    integer(kind(HIPBLAS_STATUS_SUCCESS)) :: hipblasHgemmFortran
    type(c_ptr), value :: handle
    integer(kind(HIPBLAS_OP_N)), value :: transA
    integer(kind(HIPBLAS_OP_N)), value :: transB
    integer(c_int), value :: m
    integer(c_int), value :: n
    integer(c_int), value :: k
    type(c_ptr), value :: alpha
    type(c_ptr), value :: A
    integer(c_int), value :: lda
    type(c_ptr), value :: B
    integer(c_int), value :: ldb
    type(c_ptr), value :: beta
    type(c_ptr), value :: C
    integer(c_int), value :: ldc
            hipblasHgemmFortran = &
        hipblasHgemm(handle, transA, transB, m, n, k, alpha, &
                        A, lda, B, ldb, beta, C, ldc)
end function hipblasHgemmFortran

function hipblasSgemmFortran(handle, transA, transB, m, n, k, alpha, &
                                A, lda, B, ldb, beta, C, ldc) &
    bind(c, name='hipblasSgemmFortran')
    use iso_c_binding
    use hipblas_enums
    implicit none
    integer(kind(HIPBLAS_STATUS_SUCCESS)) :: hipblasSgemmFortran
    type(c_ptr), value :: handle
    integer(kind(HIPBLAS_OP_N)), value :: transA
    integer(kind(HIPBLAS_OP_N)), value :: transB
    integer(c_int), value :: m
    integer(c_int), value :: n
    integer(c_int), value :: k
    type(c_ptr), value :: alpha
    type(c_ptr), value :: A
    integer(c_int), value :: lda
    type(c_ptr), value :: B
    integer(c_int), value :: ldb
    type(c_ptr), value :: beta
    type(c_ptr), value :: C
    integer(c_int), value :: ldc
            hipblasSgemmFortran = &
        hipblasSgemm(handle, transA, transB, m, n, k, alpha, &
                        A, lda, B, ldb, beta, C, ldc)
end function hipblasSgemmFortran

function hipblasDgemmFortran(handle, transA, transB, m, n, k, alpha, &
                                A, lda, B, ldb, beta, C, ldc) &
    bind(c, name='hipblasDgemmFortran')
    use iso_c_binding
    use hipblas_enums
    implicit none
    integer(kind(HIPBLAS_STATUS_SUCCESS)) :: hipblasDgemmFortran
    type(c_ptr), value :: handle
    integer(kind(HIPBLAS_OP_N)), value :: transA
    integer(kind(HIPBLAS_OP_N)), value :: transB
    integer(c_int), value :: m
    integer(c_int), value :: n
    integer(c_int), value :: k
    type(c_ptr), value :: alpha
    type(c_ptr), value :: A
    integer(c_int), value :: lda
    type(c_ptr), value :: B
    integer(c_int), value :: ldb
    type(c_ptr), value :: beta
    type(c_ptr), value :: C
    integer(c_int), value :: ldc
            hipblasDgemmFortran = &
        hipblasDgemm(handle, transA, transB, m, n, k, alpha, &
                        A, lda, B, ldb, beta, C, ldc)
end function hipblasDgemmFortran

function hipblasCgemmFortran(handle, transA, transB, m, n, k, alpha, &
                                A, lda, B, ldb, beta, C, ldc) &
    bind(c, name='hipblasCgemmFortran')
    use iso_c_binding
    use hipblas_enums
    implicit none
    integer(kind(HIPBLAS_STATUS_SUCCESS)) :: hipblasCgemmFortran
    type(c_ptr), value :: handle
    integer(kind(HIPBLAS_OP_N)), value :: transA
    integer(kind(HIPBLAS_OP_N)), value :: transB
    integer(c_int), value :: m
    integer(c_int), value :: n
    integer(c_int), value :: k
    type(c_ptr), value :: alpha
    type(c_ptr), value :: A
    integer(c_int), value :: lda
    type(c_ptr), value :: B
    integer(c_int), value :: ldb
    type(c_ptr), value :: beta
    type(c_ptr), value :: C
    integer(c_int), value :: ldc
            hipblasCgemmFortran = &
        hipblasCgemm(handle, transA, transB, m, n, k, alpha, &
                        A, lda, B, ldb, beta, C, ldc)
end function hipblasCgemmFortran

function hipblasZgemmFortran(handle, transA, transB, m, n, k, alpha, &
                                A, lda, B, ldb, beta, C, ldc) &
    bind(c, name='hipblasZgemmFortran')
    use iso_c_binding
    use hipblas_enums
    implicit none
    integer(kind(HIPBLAS_STATUS_SUCCESS)) :: hipblasZgemmFortran
    type(c_ptr), value :: handle
    integer(kind(HIPBLAS_OP_N)), value :: transA
    integer(kind(HIPBLAS_OP_N)), value :: transB
    integer(c_int), value :: m
    integer(c_int), value :: n
    integer(c_int), value :: k
    type(c_ptr), value :: alpha
    type(c_ptr), value :: A
    integer(c_int), value :: lda
    type(c_ptr), value :: B
    integer(c_int), value :: ldb
    type(c_ptr), value :: beta
    type(c_ptr), value :: C
    integer(c_int), value :: ldc
            hipblasZgemmFortran = &
        hipblasZgemm(handle, transA, transB, m, n, k, alpha, &
                        A, lda, B, ldb, beta, C, ldc)
end function hipblasZgemmFortran

! gemmBatched
function hipblasHgemmBatchedFortran(handle, transA, transB, m, n, k, alpha, &
                                    A, lda, B, ldb, beta, C, ldc, batch_count) &
    bind(c, name='hipblasHgemmBatchedFortran')
    use iso_c_binding
    use hipblas_enums
    implicit none
    integer(kind(HIPBLAS_STATUS_SUCCESS)) :: hipblasHgemmBatchedFortran
    type(c_ptr), value :: handle
    integer(kind(HIPBLAS_OP_N)), value :: transA
    integer(kind(HIPBLAS_OP_N)), value :: transB
    integer(c_int), value :: m
    integer(c_int), value :: n
    integer(c_int), value :: k
    type(c_ptr), value :: alpha
    type(c_ptr), value :: A
    integer(c_int), value :: lda
    type(c_ptr), value :: B
    integer(c_int), value :: ldb
    type(c_ptr), value :: beta
    type(c_ptr), value :: C
    integer(c_int), value :: ldc
    integer(c_int), value :: batch_count
            hipblasHgemmBatchedFortran = &
        hipblasHgemmBatched(handle, transA, transB, m, n, k, alpha, &
                            A, lda, B, ldb, beta, C, ldc, batch_count)
end function hipblasHgemmBatchedFortran

function hipblasSgemmBatchedFortran(handle, transA, transB, m, n, k, alpha, &
                                    A, lda, B, ldb, beta, C, ldc, batch_count) &
    bind(c, name='hipblasSgemmBatchedFortran')
    use iso_c_binding
    use hipblas_enums
    implicit none
    integer(kind(HIPBLAS_STATUS_SUCCESS)) :: hipblasSgemmBatchedFortran
    type(c_ptr), value :: handle
    integer(kind(HIPBLAS_OP_N)), value :: transA
    integer(kind(HIPBLAS_OP_N)), value :: transB
    integer(c_int), value :: m
    integer(c_int), value :: n
    integer(c_int), value :: k
    type(c_ptr), value :: alpha
    type(c_ptr), value :: A
    integer(c_int), value :: lda
    type(c_ptr), value :: B
    integer(c_int), value :: ldb
    type(c_ptr), value :: beta
    type(c_ptr), value :: C
    integer(c_int), value :: ldc
    integer(c_int), value :: batch_count
            hipblasSgemmBatchedFortran = &
        hipblasSgemmBatched(handle, transA, transB, m, n, k, alpha, &
                            A, lda, B, ldb, beta, C, ldc, batch_count)
end function hipblasSgemmBatchedFortran

function hipblasDgemmBatchedFortran(handle, transA, transB, m, n, k, alpha, &
                                    A, lda, B, ldb, beta, C, ldc, batch_count) &
    bind(c, name='hipblasDgemmBatchedFortran')
    use iso_c_binding
    use hipblas_enums
    implicit none
    integer(kind(HIPBLAS_STATUS_SUCCESS)) :: hipblasDgemmBatchedFortran
    type(c_ptr), value :: handle
    integer(kind(HIPBLAS_OP_N)), value :: transA
    integer(kind(HIPBLAS_OP_N)), value :: transB
    integer(c_int), value :: m
    integer(c_int), value :: n
    integer(c_int), value :: k
    type(c_ptr), value :: alpha
    type(c_ptr), value :: A
    integer(c_int), value :: lda
    type(c_ptr), value :: B
    integer(c_int), value :: ldb
    type(c_ptr), value :: beta
    type(c_ptr), value :: C
    integer(c_int), value :: ldc
    integer(c_int), value :: batch_count
            hipblasDgemmBatchedFortran = &
        hipblasDgemmBatched(handle, transA, transB, m, n, k, alpha, &
                            A, lda, B, ldb, beta, C, ldc, batch_count)
end function hipblasDgemmBatchedFortran

function hipblasCgemmBatchedFortran(handle, transA, transB, m, n, k, alpha, &
                                    A, lda, B, ldb, beta, C, ldc, batch_count) &
    bind(c, name='hipblasCgemmBatchedFortran')
    use iso_c_binding
    use hipblas_enums
    implicit none
    integer(kind(HIPBLAS_STATUS_SUCCESS)) :: hipblasCgemmBatchedFortran
    type(c_ptr), value :: handle
    integer(kind(HIPBLAS_OP_N)), value :: transA
    integer(kind(HIPBLAS_OP_N)), value :: transB
    integer(c_int), value :: m
    integer(c_int), value :: n
    integer(c_int), value :: k
    type(c_ptr), value :: alpha
    type(c_ptr), value :: A
    integer(c_int), value :: lda
    type(c_ptr), value :: B
    integer(c_int), value :: ldb
    type(c_ptr), value :: beta
    type(c_ptr), value :: C
    integer(c_int), value :: ldc
    integer(c_int), value :: batch_count
            hipblasCgemmBatchedFortran = &
        hipblasCgemmBatched(handle, transA, transB, m, n, k, alpha, &
                            A, lda, B, ldb, beta, C, ldc, batch_count)
end function hipblasCgemmBatchedFortran

function hipblasZgemmBatchedFortran(handle, transA, transB, m, n, k, alpha, &
                                    A, lda, B, ldb, beta, C, ldc, batch_count) &
    bind(c, name='hipblasZgemmBatchedFortran')
    use iso_c_binding
    use hipblas_enums
    implicit none
    integer(kind(HIPBLAS_STATUS_SUCCESS)) :: hipblasZgemmBatchedFortran
    type(c_ptr), value :: handle
    integer(kind(HIPBLAS_OP_N)), value :: transA
    integer(kind(HIPBLAS_OP_N)), value :: transB
    integer(c_int), value :: m
    integer(c_int), value :: n
    integer(c_int), value :: k
    type(c_ptr), value :: alpha
    type(c_ptr), value :: A
    integer(c_int), value :: lda
    type(c_ptr), value :: B
    integer(c_int), value :: ldb
    type(c_ptr), value :: beta
    type(c_ptr), value :: C
    integer(c_int), value :: ldc
    integer(c_int), value :: batch_count
            hipblasZgemmBatchedFortran = &
        hipblasZgemmBatched(handle, transA, transB, m, n, k, alpha, &
                            A, lda, B, ldb, beta, C, ldc, batch_count)
end function hipblasZgemmBatchedFortran

! gemmStridedBatched
function hipblasHgemmStridedBatchedFortran(handle, transA, transB, m, n, k, alpha, &
                                            A, lda, stride_A, B, ldb, stride_B, beta, C, ldc, stride_C, batch_count) &
    bind(c, name='hipblasHgemmStridedBatchedFortran')
    use iso_c_binding
    use hipblas_enums
    implicit none
    integer(kind(HIPBLAS_STATUS_SUCCESS)) :: hipblasHgemmStridedBatchedFortran
    type(c_ptr), value :: handle
    integer(kind(HIPBLAS_OP_N)), value :: transA
    integer(kind(HIPBLAS_OP_N)), value :: transB
    integer(c_int), value :: m
    integer(c_int), value :: n
    integer(c_int), value :: k
    type(c_ptr), value :: alpha
    type(c_ptr), value :: A
    integer(c_int), value :: lda
    integer(c_int64_t), value :: stride_A
    type(c_ptr), value :: B
    integer(c_int), value :: ldb
    integer(c_int64_t), value :: stride_B
    type(c_ptr), value :: beta
    type(c_ptr), value :: C
    integer(c_int), value :: ldc
    integer(c_int64_t), value :: stride_C
    integer(c_int), value :: batch_count
            hipblasHgemmStridedBatchedFortran = &
        hipblasHgemmStridedBatched(handle, transA, transB, m, n, k, alpha, &
                                    A, lda, stride_A, B, ldb, stride_B, beta, C, ldc, stride_C, batch_count)
end function hipblasHgemmStridedBatchedFortran

function hipblasSgemmStridedBatchedFortran(handle, transA, transB, m, n, k, alpha, &
                                            A, lda, stride_A, B, ldb, stride_B, beta, C, ldc, stride_C, batch_count) &
    bind(c, name='hipblasSgemmStridedBatchedFortran')
    use iso_c_binding
    use hipblas_enums
    implicit none
    integer(kind(HIPBLAS_STATUS_SUCCESS)) :: hipblasSgemmStridedBatchedFortran
    type(c_ptr), value :: handle
    integer(kind(HIPBLAS_OP_N)), value :: transA
    integer(kind(HIPBLAS_OP_N)), value :: transB
    integer(c_int), value :: m
    integer(c_int), value :: n
    integer(c_int), value :: k
    type(c_ptr), value :: alpha
    type(c_ptr), value :: A
    integer(c_int), value :: lda
    integer(c_int64_t), value :: stride_A
    type(c_ptr), value :: B
    integer(c_int), value :: ldb
    integer(c_int64_t), value :: stride_B
    type(c_ptr), value :: beta
    type(c_ptr), value :: C
    integer(c_int), value :: ldc
    integer(c_int64_t), value :: stride_C
    integer(c_int), value :: batch_count
            hipblasSgemmStridedBatchedFortran = &
        hipblasSgemmStridedBatched(handle, transA, transB, m, n, k, alpha, &
                                    A, lda, stride_A, B, ldb, stride_B, beta, C, ldc, stride_C, batch_count)
end function hipblasSgemmStridedBatchedFortran

function hipblasDgemmStridedBatchedFortran(handle, transA, transB, m, n, k, alpha, &
                                            A, lda, stride_A, B, ldb, stride_B, beta, C, ldc, stride_C, batch_count) &
    bind(c, name='hipblasDgemmStridedBatchedFortran')
    use iso_c_binding
    use hipblas_enums
    implicit none
    integer(kind(HIPBLAS_STATUS_SUCCESS)) :: hipblasDgemmStridedBatchedFortran
    type(c_ptr), value :: handle
    integer(kind(HIPBLAS_OP_N)), value :: transA
    integer(kind(HIPBLAS_OP_N)), value :: transB
    integer(c_int), value :: m
    integer(c_int), value :: n
    integer(c_int), value :: k
    type(c_ptr), value :: alpha
    type(c_ptr), value :: A
    integer(c_int), value :: lda
    integer(c_int64_t), value :: stride_A
    type(c_ptr), value :: B
    integer(c_int), value :: ldb
    integer(c_int64_t), value :: stride_B
    type(c_ptr), value :: beta
    type(c_ptr), value :: C
    integer(c_int), value :: ldc
    integer(c_int64_t), value :: stride_C
    integer(c_int), value :: batch_count
            hipblasDgemmStridedBatchedFortran = &
        hipblasDgemmStridedBatched(handle, transA, transB, m, n, k, alpha, &
                                    A, lda, stride_A, B, ldb, stride_B, beta, C, ldc, stride_C, batch_count)
end function hipblasDgemmStridedBatchedFortran

function hipblasCgemmStridedBatchedFortran(handle, transA, transB, m, n, k, alpha, &
                                            A, lda, stride_A, B, ldb, stride_B, beta, C, ldc, stride_C, batch_count) &
    bind(c, name='hipblasCgemmStridedBatchedFortran')
    use iso_c_binding
    use hipblas_enums
    implicit none
    integer(kind(HIPBLAS_STATUS_SUCCESS)) :: hipblasCgemmStridedBatchedFortran
    type(c_ptr), value :: handle
    integer(kind(HIPBLAS_OP_N)), value :: transA
    integer(kind(HIPBLAS_OP_N)), value :: transB
    integer(c_int), value :: m
    integer(c_int), value :: n
    integer(c_int), value :: k
    type(c_ptr), value :: alpha
    type(c_ptr), value :: A
    integer(c_int), value :: lda
    integer(c_int64_t), value :: stride_A
    type(c_ptr), value :: B
    integer(c_int), value :: ldb
    integer(c_int64_t), value :: stride_B
    type(c_ptr), value :: beta
    type(c_ptr), value :: C
    integer(c_int), value :: ldc
    integer(c_int64_t), value :: stride_C
    integer(c_int), value :: batch_count
            hipblasCgemmStridedBatchedFortran = &
        hipblasCgemmStridedBatched(handle, transA, transB, m, n, k, alpha, &
                                    A, lda, stride_A, B, ldb, stride_B, beta, C, ldc, stride_C, batch_count)
end function hipblasCgemmStridedBatchedFortran

function hipblasZgemmStridedBatchedFortran(handle, transA, transB, m, n, k, alpha, &
                                            A, lda, stride_A, B, ldb, stride_B, beta, C, ldc, stride_C, batch_count) &
    bind(c, name='hipblasZgemmStridedBatchedFortran')
    use iso_c_binding
    use hipblas_enums
    implicit none
    integer(kind(HIPBLAS_STATUS_SUCCESS)) :: hipblasZgemmStridedBatchedFortran
    type(c_ptr), value :: handle
    integer(kind(HIPBLAS_OP_N)), value :: transA
    integer(kind(HIPBLAS_OP_N)), value :: transB
    integer(c_int), value :: m
    integer(c_int), value :: n
    integer(c_int), value :: k
    type(c_ptr), value :: alpha
    type(c_ptr), value :: A
    integer(c_int), value :: lda
    integer(c_int64_t), value :: stride_A
    type(c_ptr), value :: B
    integer(c_int), value :: ldb
    integer(c_int64_t), value :: stride_B
    type(c_ptr), value :: beta
    type(c_ptr), value :: C
    integer(c_int), value :: ldc
    integer(c_int64_t), value :: stride_C
    integer(c_int), value :: batch_count
            hipblasZgemmStridedBatchedFortran = &
        hipblasZgemmStridedBatched(handle, transA, transB, m, n, k, alpha, &
                                    A, lda, stride_A, B, ldb, stride_B, beta, C, ldc, stride_C, batch_count)
end function hipblasZgemmStridedBatchedFortran

! dgmm
function hipblasSdgmmFortran(handle, side, m, n, &
                                A, lda, x, incx, C, ldc) &
    bind(c, name='hipblasSdgmmFortran')
    use iso_c_binding
    use hipblas_enums
    implicit none
    integer(kind(HIPBLAS_STATUS_SUCCESS)) :: hipblasSdgmmFortran
    type(c_ptr), value :: handle
    integer(kind(HIPBLAS_SIDE_LEFT)), value :: side
    integer(c_int), value :: m
    integer(c_int), value :: n
    type(c_ptr), value :: A
    integer(c_int), value :: lda
    type(c_ptr), value :: x
    integer(c_int), value :: incx
    type(c_ptr), value :: C
    integer(c_int), value :: ldc
            hipblasSdgmmFortran = &
        hipblasSdgmm(handle, side, m, n, &
                        A, lda, x, incx, C, ldc)
end function hipblasSdgmmFortran

function hipblasDdgmmFortran(handle, side, m, n, &
                                A, lda, x, incx, C, ldc) &
    bind(c, name='hipblasDdgmmFortran')
    use iso_c_binding
    use hipblas_enums
    implicit none
    integer(kind(HIPBLAS_STATUS_SUCCESS)) :: hipblasDdgmmFortran
    type(c_ptr), value :: handle
    integer(kind(HIPBLAS_SIDE_LEFT)), value :: side
    integer(c_int), value :: m
    integer(c_int), value :: n
    type(c_ptr), value :: A
    integer(c_int), value :: lda
    type(c_ptr), value :: x
    integer(c_int), value :: incx
    type(c_ptr), value :: C
    integer(c_int), value :: ldc
            hipblasDdgmmFortran = &
        hipblasDdgmm(handle, side, m, n, &
                        A, lda, x, incx, C, ldc)
end function hipblasDdgmmFortran

function hipblasCdgmmFortran(handle, side, m, n, &
                                A, lda, x, incx, C, ldc) &
    bind(c, name='hipblasCdgmmFortran')
    use iso_c_binding
    use hipblas_enums
    implicit none
    integer(kind(HIPBLAS_STATUS_SUCCESS)) :: hipblasCdgmmFortran
    type(c_ptr), value :: handle
    integer(kind(HIPBLAS_SIDE_LEFT)), value :: side
    integer(c_int), value :: m
    integer(c_int), value :: n
    type(c_ptr), value :: A
    integer(c_int), value :: lda
    type(c_ptr), value :: x
    integer(c_int), value :: incx
    type(c_ptr), value :: C
    integer(c_int), value :: ldc
            hipblasCdgmmFortran = &
        hipblasCdgmm(handle, side, m, n, &
                        A, lda, x, incx, C, ldc)
end function hipblasCdgmmFortran

function hipblasZdgmmFortran(handle, side, m, n, &
                                A, lda, x, incx, C, ldc) &
    bind(c, name='hipblasZdgmmFortran')
    use iso_c_binding
    use hipblas_enums
    implicit none
    integer(kind(HIPBLAS_STATUS_SUCCESS)) :: hipblasZdgmmFortran
    type(c_ptr), value :: handle
    integer(kind(HIPBLAS_SIDE_LEFT)), value :: side
    integer(c_int), value :: m
    integer(c_int), value :: n
    type(c_ptr), value :: A
    integer(c_int), value :: lda
    type(c_ptr), value :: x
    integer(c_int), value :: incx
    type(c_ptr), value :: C
    integer(c_int), value :: ldc
            hipblasZdgmmFortran = &
        hipblasZdgmm(handle, side, m, n, &
                        A, lda, x, incx, C, ldc)
end function hipblasZdgmmFortran

! dgmmBatched
function hipblasSdgmmBatchedFortran(handle, side, m, n, &
                                    A, lda, x, incx, C, ldc, batch_count) &
    bind(c, name='hipblasSdgmmBatchedFortran')
    use iso_c_binding
    use hipblas_enums
    implicit none
    integer(kind(HIPBLAS_STATUS_SUCCESS)) :: hipblasSdgmmBatchedFortran
    type(c_ptr), value :: handle
    integer(kind(HIPBLAS_SIDE_LEFT)), value :: side
    integer(c_int), value :: m
    integer(c_int), value :: n
    type(c_ptr), value :: A
    integer(c_int), value :: lda
    type(c_ptr), value :: x
    integer(c_int), value :: incx
    type(c_ptr), value :: C
    integer(c_int), value :: ldc
    integer(c_int), value :: batch_count
            hipblasSdgmmBatchedFortran = &
        hipblasSdgmmBatched(handle, side, m, n, &
                            A, lda, x, incx, C, ldc, batch_count)
end function hipblasSdgmmBatchedFortran

function hipblasDdgmmBatchedFortran(handle, side, m, n, &
                                    A, lda, x, incx, C, ldc, batch_count) &
    bind(c, name='hipblasDdgmmBatchedFortran')
    use iso_c_binding
    use hipblas_enums
    implicit none
    integer(kind(HIPBLAS_STATUS_SUCCESS)) :: hipblasDdgmmBatchedFortran
    type(c_ptr), value :: handle
    integer(kind(HIPBLAS_SIDE_LEFT)), value :: side
    integer(c_int), value :: m
    integer(c_int), value :: n
    type(c_ptr), value :: A
    integer(c_int), value :: lda
    type(c_ptr), value :: x
    integer(c_int), value :: incx
    type(c_ptr), value :: C
    integer(c_int), value :: ldc
    integer(c_int), value :: batch_count
            hipblasDdgmmBatchedFortran = &
        hipblasDdgmmBatched(handle, side, m, n, &
                            A, lda, x, incx, C, ldc, batch_count)
end function hipblasDdgmmBatchedFortran

function hipblasCdgmmBatchedFortran(handle, side, m, n, &
                                    A, lda, x, incx, C, ldc, batch_count) &
    bind(c, name='hipblasCdgmmBatchedFortran')
    use iso_c_binding
    use hipblas_enums
    implicit none
    integer(kind(HIPBLAS_STATUS_SUCCESS)) :: hipblasCdgmmBatchedFortran
    type(c_ptr), value :: handle
    integer(kind(HIPBLAS_SIDE_LEFT)), value :: side
    integer(c_int), value :: m
    integer(c_int), value :: n
    type(c_ptr), value :: A
    integer(c_int), value :: lda
    type(c_ptr), value :: x
    integer(c_int), value :: incx
    type(c_ptr), value :: C
    integer(c_int), value :: ldc
    integer(c_int), value :: batch_count
            hipblasCdgmmBatchedFortran = &
        hipblasCdgmmBatched(handle, side, m, n, &
                            A, lda, x, incx, C, ldc, batch_count)
end function hipblasCdgmmBatchedFortran

function hipblasZdgmmBatchedFortran(handle, side, m, n, &
                                    A, lda, x, incx, C, ldc, batch_count) &
    bind(c, name='hipblasZdgmmBatchedFortran')
    use iso_c_binding
    use hipblas_enums
    implicit none
    integer(kind(HIPBLAS_STATUS_SUCCESS)) :: hipblasZdgmmBatchedFortran
    type(c_ptr), value :: handle
    integer(kind(HIPBLAS_SIDE_LEFT)), value :: side
    integer(c_int), value :: m
    integer(c_int), value :: n
    type(c_ptr), value :: A
    integer(c_int), value :: lda
    type(c_ptr), value :: x
    integer(c_int), value :: incx
    type(c_ptr), value :: C
    integer(c_int), value :: ldc
    integer(c_int), value :: batch_count
            hipblasZdgmmBatchedFortran = &
        hipblasZdgmmBatched(handle, side, m, n, &
                            A, lda, x, incx, C, ldc, batch_count)
end function hipblasZdgmmBatchedFortran

! dgmmStridedBatched
function hipblasSdgmmStridedBatchedFortran(handle, side, m, n, &
                                            A, lda, stride_A, x, incx, stride_x, C, ldc, stride_C, batch_count) &
    bind(c, name='hipblasSdgmmStridedBatchedFortran')
    use iso_c_binding
    use hipblas_enums
    implicit none
    integer(kind(HIPBLAS_STATUS_SUCCESS)) :: hipblasSdgmmStridedBatchedFortran
    type(c_ptr), value :: handle
    integer(kind(HIPBLAS_SIDE_LEFT)), value :: side
    integer(c_int), value :: m
    integer(c_int), value :: n
    type(c_ptr), value :: A
    integer(c_int), value :: lda
    integer(c_int64_t), value :: stride_A
    type(c_ptr), value :: x
    integer(c_int), value :: incx
    integer(c_int64_t), value :: stride_x
    type(c_ptr), value :: C
    integer(c_int), value :: ldc
    integer(c_int64_t), value :: stride_C
    integer(c_int), value :: batch_count
            hipblasSdgmmStridedBatchedFortran = &
        hipblasSdgmmStridedBatched(handle, side, m, n, &
                                    A, lda, stride_A, x, incx, stride_x, C, ldc, stride_C, batch_count)
end function hipblasSdgmmStridedBatchedFortran

function hipblasDdgmmStridedBatchedFortran(handle, side, m, n, &
                                            A, lda, stride_A, x, incx, stride_x, C, ldc, stride_C, batch_count) &
    bind(c, name='hipblasDdgmmStridedBatchedFortran')
    use iso_c_binding
    use hipblas_enums
    implicit none
    integer(kind(HIPBLAS_STATUS_SUCCESS)) :: hipblasDdgmmStridedBatchedFortran
    type(c_ptr), value :: handle
    integer(kind(HIPBLAS_SIDE_LEFT)), value :: side
    integer(c_int), value :: m
    integer(c_int), value :: n
    type(c_ptr), value :: A
    integer(c_int), value :: lda
    integer(c_int64_t), value :: stride_A
    type(c_ptr), value :: x
    integer(c_int), value :: incx
    integer(c_int64_t), value :: stride_x
    type(c_ptr), value :: C
    integer(c_int), value :: ldc
    integer(c_int64_t), value :: stride_C
    integer(c_int), value :: batch_count
            hipblasDdgmmStridedBatchedFortran = &
        hipblasDdgmmStridedBatched(handle, side, m, n, &
                                    A, lda, stride_A, x, incx, stride_x, C, ldc, stride_C, batch_count)
end function hipblasDdgmmStridedBatchedFortran

function hipblasCdgmmStridedBatchedFortran(handle, side, m, n, &
                                            A, lda, stride_A, x, incx, stride_x, C, ldc, stride_C, batch_count) &
    bind(c, name='hipblasCdgmmStridedBatchedFortran')
    use iso_c_binding
    use hipblas_enums
    implicit none
    integer(kind(HIPBLAS_STATUS_SUCCESS)) :: hipblasCdgmmStridedBatchedFortran
    type(c_ptr), value :: handle
    integer(kind(HIPBLAS_SIDE_LEFT)), value :: side
    integer(c_int), value :: m
    integer(c_int), value :: n
    type(c_ptr), value :: A
    integer(c_int), value :: lda
    integer(c_int64_t), value :: stride_A
    type(c_ptr), value :: x
    integer(c_int), value :: incx
    integer(c_int64_t), value :: stride_x
    type(c_ptr), value :: C
    integer(c_int), value :: ldc
    integer(c_int64_t), value :: stride_C
    integer(c_int), value :: batch_count
            hipblasCdgmmStridedBatchedFortran = &
        hipblasCdgmmStridedBatched(handle, side, m, n, &
                                    A, lda, stride_A, x, incx, stride_x, C, ldc, stride_C, batch_count)
end function hipblasCdgmmStridedBatchedFortran

function hipblasZdgmmStridedBatchedFortran(handle, side, m, n, &
                                            A, lda, stride_A, x, incx, stride_x, C, ldc, stride_C, batch_count) &
    bind(c, name='hipblasZdgmmStridedBatchedFortran')
    use iso_c_binding
    use hipblas_enums
    implicit none
    integer(kind(HIPBLAS_STATUS_SUCCESS)) :: hipblasZdgmmStridedBatchedFortran
    type(c_ptr), value :: handle
    integer(kind(HIPBLAS_SIDE_LEFT)), value :: side
    integer(c_int), value :: m
    integer(c_int), value :: n
    type(c_ptr), value :: A
    integer(c_int), value :: lda
    integer(c_int64_t), value :: stride_A
    type(c_ptr), value :: x
    integer(c_int), value :: incx
    integer(c_int64_t), value :: stride_x
    type(c_ptr), value :: C
    integer(c_int), value :: ldc
    integer(c_int64_t), value :: stride_C
    integer(c_int), value :: batch_count
            hipblasZdgmmStridedBatchedFortran = &
        hipblasZdgmmStridedBatched(handle, side, m, n, &
                                    A, lda, stride_A, x, incx, stride_x, C, ldc, stride_C, batch_count)
end function hipblasZdgmmStridedBatchedFortran

! geam
function hipblasSgeamFortran(handle, transA, transB, m, n, alpha, &
                                A, lda, beta, B, ldb, C, ldc) &
    bind(c, name='hipblasSgeamFortran')
    use iso_c_binding
    use hipblas_enums
    implicit none
    integer(kind(HIPBLAS_STATUS_SUCCESS)) :: hipblasSgeamFortran
    type(c_ptr), value :: handle
    integer(kind(HIPBLAS_OP_N)), value :: transA
    integer(kind(HIPBLAS_OP_N)), value :: transB
    integer(c_int), value :: m
    integer(c_int), value :: n
    type(c_ptr), value :: alpha
    type(c_ptr), value :: A
    integer(c_int), value :: lda
    type(c_ptr), value :: beta
    type(c_ptr), value :: B
    integer(c_int), value :: ldb
    type(c_ptr), value :: C
    integer(c_int), value :: ldc
            hipblasSgeamFortran = &
        hipblasSgeam(handle, transA, transB, m, n, alpha, &
                        A, lda, beta, B, ldb, C, ldc)
end function hipblasSgeamFortran

function hipblasDgeamFortran(handle, transA, transB, m, n, alpha, &
                                A, lda, beta, B, ldb, C, ldc) &
    bind(c, name='hipblasDgeamFortran')
    use iso_c_binding
    use hipblas_enums
    implicit none
    integer(kind(HIPBLAS_STATUS_SUCCESS)) :: hipblasDgeamFortran
    type(c_ptr), value :: handle
    integer(kind(HIPBLAS_OP_N)), value :: transA
    integer(kind(HIPBLAS_OP_N)), value :: transB
    integer(c_int), value :: m
    integer(c_int), value :: n
    type(c_ptr), value :: alpha
    type(c_ptr), value :: A
    integer(c_int), value :: lda
    type(c_ptr), value :: beta
    type(c_ptr), value :: B
    integer(c_int), value :: ldb
    type(c_ptr), value :: C
    integer(c_int), value :: ldc
            hipblasDgeamFortran = &
        hipblasDgeam(handle, transA, transB, m, n, alpha, &
                        A, lda, beta, B, ldb, C, ldc)
end function hipblasDgeamFortran

function hipblasCgeamFortran(handle, transA, transB, m, n, alpha, &
                                A, lda, beta, B, ldb, C, ldc) &
    bind(c, name='hipblasCgeamFortran')
    use iso_c_binding
    use hipblas_enums
    implicit none
    integer(kind(HIPBLAS_STATUS_SUCCESS)) :: hipblasCgeamFortran
    type(c_ptr), value :: handle
    integer(kind(HIPBLAS_OP_N)), value :: transA
    integer(kind(HIPBLAS_OP_N)), value :: transB
    integer(c_int), value :: m
    integer(c_int), value :: n
    type(c_ptr), value :: alpha
    type(c_ptr), value :: A
    integer(c_int), value :: lda
    type(c_ptr), value :: beta
    type(c_ptr), value :: B
    integer(c_int), value :: ldb
    type(c_ptr), value :: C
    integer(c_int), value :: ldc
            hipblasCgeamFortran = &
        hipblasCgeam(handle, transA, transB, m, n, alpha, &
                        A, lda, beta, B, ldb, C, ldc)
end function hipblasCgeamFortran

function hipblasZgeamFortran(handle, transA, transB, m, n, alpha, &
                                A, lda, beta, B, ldb, C, ldc) &
    bind(c, name='hipblasZgeamFortran')
    use iso_c_binding
    use hipblas_enums
    implicit none
    integer(kind(HIPBLAS_STATUS_SUCCESS)) :: hipblasZgeamFortran
    type(c_ptr), value :: handle
    integer(kind(HIPBLAS_OP_N)), value :: transA
    integer(kind(HIPBLAS_OP_N)), value :: transB
    integer(c_int), value :: m
    integer(c_int), value :: n
    type(c_ptr), value :: alpha
    type(c_ptr), value :: A
    integer(c_int), value :: lda
    type(c_ptr), value :: beta
    type(c_ptr), value :: B
    integer(c_int), value :: ldb
    type(c_ptr), value :: C
    integer(c_int), value :: ldc
            hipblasZgeamFortran = &
        hipblasZgeam(handle, transA, transB, m, n, alpha, &
                        A, lda, beta, B, ldb, C, ldc)
end function hipblasZgeamFortran

! geamBatched
function hipblasSgeamBatchedFortran(handle, transA, transB, m, n, alpha, &
                                    A, lda, beta, B, ldb, C, ldc, batch_count) &
    bind(c, name='hipblasSgeamBatchedFortran')
    use iso_c_binding
    use hipblas_enums
    implicit none
    integer(kind(HIPBLAS_STATUS_SUCCESS)) :: hipblasSgeamBatchedFortran
    type(c_ptr), value :: handle
    integer(kind(HIPBLAS_OP_N)), value :: transA
    integer(kind(HIPBLAS_OP_N)), value :: transB
    integer(c_int), value :: m
    integer(c_int), value :: n
    type(c_ptr), value :: alpha
    type(c_ptr), value :: A
    integer(c_int), value :: lda
    type(c_ptr), value :: beta
    type(c_ptr), value :: B
    integer(c_int), value :: ldb
    type(c_ptr), value :: C
    integer(c_int), value :: ldc
    integer(c_int), value :: batch_count
            hipblasSgeamBatchedFortran = &
        hipblasSgeamBatched(handle, transA, transB, m, n, alpha, &
                            A, lda, beta, B, ldb, C, ldc, batch_count)
end function hipblasSgeamBatchedFortran

function hipblasDgeamBatchedFortran(handle, transA, transB, m, n, alpha, &
                                    A, lda, beta, B, ldb, C, ldc, batch_count) &
    bind(c, name='hipblasDgeamBatchedFortran')
    use iso_c_binding
    use hipblas_enums
    implicit none
    integer(kind(HIPBLAS_STATUS_SUCCESS)) :: hipblasDgeamBatchedFortran
    type(c_ptr), value :: handle
    integer(kind(HIPBLAS_OP_N)), value :: transA
    integer(kind(HIPBLAS_OP_N)), value :: transB
    integer(c_int), value :: m
    integer(c_int), value :: n
    type(c_ptr), value :: alpha
    type(c_ptr), value :: A
    integer(c_int), value :: lda
    type(c_ptr), value :: beta
    type(c_ptr), value :: B
    integer(c_int), value :: ldb
    type(c_ptr), value :: C
    integer(c_int), value :: ldc
    integer(c_int), value :: batch_count
            hipblasDgeamBatchedFortran = &
        hipblasDgeamBatched(handle, transA, transB, m, n, alpha, &
                            A, lda, beta, B, ldb, C, ldc, batch_count)
end function hipblasDgeamBatchedFortran

function hipblasCgeamBatchedFortran(handle, transA, transB, m, n, alpha, &
                                    A, lda, beta, B, ldb, C, ldc, batch_count) &
    bind(c, name='hipblasCgeamBatchedFortran')
    use iso_c_binding
    use hipblas_enums
    implicit none
    integer(kind(HIPBLAS_STATUS_SUCCESS)) :: hipblasCgeamBatchedFortran
    type(c_ptr), value :: handle
    integer(kind(HIPBLAS_OP_N)), value :: transA
    integer(kind(HIPBLAS_OP_N)), value :: transB
    integer(c_int), value :: m
    integer(c_int), value :: n
    type(c_ptr), value :: alpha
    type(c_ptr), value :: A
    integer(c_int), value :: lda
    type(c_ptr), value :: beta
    type(c_ptr), value :: B
    integer(c_int), value :: ldb
    type(c_ptr), value :: C
    integer(c_int), value :: ldc
    integer(c_int), value :: batch_count
            hipblasCgeamBatchedFortran = &
        hipblasCgeamBatched(handle, transA, transB, m, n, alpha, &
                            A, lda, beta, B, ldb, C, ldc, batch_count)
end function hipblasCgeamBatchedFortran

function hipblasZgeamBatchedFortran(handle, transA, transB, m, n, alpha, &
                                    A, lda, beta, B, ldb, C, ldc, batch_count) &
    bind(c, name='hipblasZgeamBatchedFortran')
    use iso_c_binding
    use hipblas_enums
    implicit none
    integer(kind(HIPBLAS_STATUS_SUCCESS)) :: hipblasZgeamBatchedFortran
    type(c_ptr), value :: handle
    integer(kind(HIPBLAS_OP_N)), value :: transA
    integer(kind(HIPBLAS_OP_N)), value :: transB
    integer(c_int), value :: m
    integer(c_int), value :: n
    type(c_ptr), value :: alpha
    type(c_ptr), value :: A
    integer(c_int), value :: lda
    type(c_ptr), value :: beta
    type(c_ptr), value :: B
    integer(c_int), value :: ldb
    type(c_ptr), value :: C
    integer(c_int), value :: ldc
    integer(c_int), value :: batch_count
            hipblasZgeamBatchedFortran = &
        hipblasZgeamBatched(handle, transA, transB, m, n, alpha, &
                            A, lda, beta, B, ldb, C, ldc, batch_count)
end function hipblasZgeamBatchedFortran

! geamStridedBatched
function hipblasSgeamStridedBatchedFortran(handle, transA, transB, m, n, alpha, &
                                            A, lda, stride_A, beta, B, ldb, stride_B, C, ldc, stride_C, batch_count) &
    bind(c, name='hipblasSgeamStridedBatchedFortran')
    use iso_c_binding
    use hipblas_enums
    implicit none
    integer(kind(HIPBLAS_STATUS_SUCCESS)) :: hipblasSgeamStridedBatchedFortran
    type(c_ptr), value :: handle
    integer(kind(HIPBLAS_OP_N)), value :: transA
    integer(kind(HIPBLAS_OP_N)), value :: transB
    integer(c_int), value :: m
    integer(c_int), value :: n
    type(c_ptr), value :: alpha
    type(c_ptr), value :: A
    integer(c_int), value :: lda
    integer(c_int64_t), value :: stride_A
    type(c_ptr), value :: beta
    type(c_ptr), value :: B
    integer(c_int), value :: ldb
    integer(c_int64_t), value :: stride_B
    type(c_ptr), value :: C
    integer(c_int), value :: ldc
    integer(c_int64_t), value :: stride_C
    integer(c_int), value :: batch_count
            hipblasSgeamStridedBatchedFortran = &
        hipblasSgeamStridedBatched(handle, transA, transB, m, n, alpha, &
                                    A, lda, stride_A, beta, B, ldb, stride_B, C, ldc, stride_C, batch_count)
end function hipblasSgeamStridedBatchedFortran

function hipblasDgeamStridedBatchedFortran(handle, transA, transB, m, n, alpha, &
                                            A, lda, stride_A, beta, B, ldb, stride_B, C, ldc, stride_C, batch_count) &
    bind(c, name='hipblasDgeamStridedBatchedFortran')
    use iso_c_binding
    use hipblas_enums
    implicit none
    integer(kind(HIPBLAS_STATUS_SUCCESS)) :: hipblasDgeamStridedBatchedFortran
    type(c_ptr), value :: handle
    integer(kind(HIPBLAS_OP_N)), value :: transA
    integer(kind(HIPBLAS_OP_N)), value :: transB
    integer(c_int), value :: m
    integer(c_int), value :: n
    type(c_ptr), value :: alpha
    type(c_ptr), value :: A
    integer(c_int), value :: lda
    integer(c_int64_t), value :: stride_A
    type(c_ptr), value :: beta
    type(c_ptr), value :: B
    integer(c_int), value :: ldb
    integer(c_int64_t), value :: stride_B
    type(c_ptr), value :: C
    integer(c_int), value :: ldc
    integer(c_int64_t), value :: stride_C
    integer(c_int), value :: batch_count
            hipblasDgeamStridedBatchedFortran = &
        hipblasDgeamStridedBatched(handle, transA, transB, m, n, alpha, &
                                    A, lda, stride_A, beta, B, ldb, stride_B, C, ldc, stride_C, batch_count)
end function hipblasDgeamStridedBatchedFortran

function hipblasCgeamStridedBatchedFortran(handle, transA, transB, m, n, alpha, &
                                            A, lda, stride_A, beta, B, ldb, stride_B, C, ldc, stride_C, batch_count) &
    bind(c, name='hipblasCgeamStridedBatchedFortran')
    use iso_c_binding
    use hipblas_enums
    implicit none
    integer(kind(HIPBLAS_STATUS_SUCCESS)) :: hipblasCgeamStridedBatchedFortran
    type(c_ptr), value :: handle
    integer(kind(HIPBLAS_OP_N)), value :: transA
    integer(kind(HIPBLAS_OP_N)), value :: transB
    integer(c_int), value :: m
    integer(c_int), value :: n
    type(c_ptr), value :: alpha
    type(c_ptr), value :: A
    integer(c_int), value :: lda
    integer(c_int64_t), value :: stride_A
    type(c_ptr), value :: beta
    type(c_ptr), value :: B
    integer(c_int), value :: ldb
    integer(c_int64_t), value :: stride_B
    type(c_ptr), value :: C
    integer(c_int), value :: ldc
    integer(c_int64_t), value :: stride_C
    integer(c_int), value :: batch_count
            hipblasCgeamStridedBatchedFortran = &
        hipblasCgeamStridedBatched(handle, transA, transB, m, n, alpha, &
                                    A, lda, stride_A, beta, B, ldb, stride_B, C, ldc, stride_C, batch_count)
end function hipblasCgeamStridedBatchedFortran

function hipblasZgeamStridedBatchedFortran(handle, transA, transB, m, n, alpha, &
                                            A, lda, stride_A, beta, B, ldb, stride_B, C, ldc, stride_C, batch_count) &
    bind(c, name='hipblasZgeamStridedBatchedFortran')
    use iso_c_binding
    use hipblas_enums
    implicit none
    integer(kind(HIPBLAS_STATUS_SUCCESS)) :: hipblasZgeamStridedBatchedFortran
    type(c_ptr), value :: handle
    integer(kind(HIPBLAS_OP_N)), value :: transA
    integer(kind(HIPBLAS_OP_N)), value :: transB
    integer(c_int), value :: m
    integer(c_int), value :: n
    type(c_ptr), value :: alpha
    type(c_ptr), value :: A
    integer(c_int), value :: lda
    integer(c_int64_t), value :: stride_A
    type(c_ptr), value :: beta
    type(c_ptr), value :: B
    integer(c_int), value :: ldb
    integer(c_int64_t), value :: stride_B
    type(c_ptr), value :: C
    integer(c_int), value :: ldc
    integer(c_int64_t), value :: stride_C
    integer(c_int), value :: batch_count
            hipblasZgeamStridedBatchedFortran = &
        hipblasZgeamStridedBatched(handle, transA, transB, m, n, alpha, &
                                    A, lda, stride_A, beta, B, ldb, stride_B, C, ldc, stride_C, batch_count)
end function hipblasZgeamStridedBatchedFortran

!-----------------!
! blas Extensions !
!-----------------!

! gemmEx
function hipblasGemmExFortran(handle, transA, transB, m, n, k, alpha, a, a_type, lda, &
                                b, b_type, ldb, beta, c, c_type, ldc, &
                                compute_type, algo) &
    bind(c, name='hipblasGemmExFortran')
    use iso_c_binding
    use hipblas_enums
    implicit none
    integer(kind(HIPBLAS_STATUS_SUCCESS)) :: hipblasGemmExFortran
    type(c_ptr), value :: handle
    integer(kind(HIPBLAS_OP_N)), value :: transA
    integer(kind(HIPBLAS_OP_N)), value :: transB
    integer(c_int), value :: m
    integer(c_int), value :: n
    integer(c_int), value :: k
    type(c_ptr), value :: alpha
    type(c_ptr), value :: a
    integer(kind(HIP_R_16F)), value :: a_type
    integer(c_int), value :: lda
    type(c_ptr), value :: b
    integer(kind(HIP_R_16F)), value :: b_type
    integer(c_int), value :: ldb
    type(c_ptr), value :: beta
    type(c_ptr), value :: c
    integer(kind(HIP_R_16F)), value :: c_type
    integer(c_int), value :: ldc
    integer(kind(HIPBLAS_COMPUTE_16F)), value :: compute_type
    integer(kind(HIPBLAS_GEMM_DEFAULT)), value :: algo
            hipblasGemmExFortran = &
        hipblasGemmEx(handle, transA, transB, m, n, k, alpha, &
                        a, a_type, lda, b, b_type, ldb, beta, c, c_type, ldc, &
                        compute_type, algo)
end function hipblasGemmExFortran

function hipblasGemmBatchedExFortran(handle, transA, transB, m, n, k, alpha, a, a_type, lda, &
                                        b, b_type, ldb, beta, c, c_type, ldc, &
                                        batch_count, compute_type, algo) &
    bind(c, name='hipblasGemmBatchedExFortran')
    use iso_c_binding
    use hipblas_enums
    implicit none
    integer(kind(HIPBLAS_STATUS_SUCCESS)) :: hipblasGemmBatchedExFortran
    type(c_ptr), value :: handle
    integer(kind(HIPBLAS_OP_N)), value :: transA
    integer(kind(HIPBLAS_OP_N)), value :: transB
    integer(c_int), value :: m
    integer(c_int), value :: n
    integer(c_int), value :: k
    type(c_ptr), value :: alpha
    type(c_ptr), value :: a
    integer(kind(HIP_R_16F)), value :: a_type
    integer(c_int), value :: lda
    type(c_ptr), value :: b
    integer(kind(HIP_R_16F)), value :: b_type
    integer(c_int), value :: ldb
    type(c_ptr), value :: beta
    type(c_ptr), value :: c
    integer(kind(HIP_R_16F)), value :: c_type
    integer(c_int), value :: ldc
    integer(c_int), value :: batch_count
    integer(kind(HIPBLAS_COMPUTE_16F)), value :: compute_type
    integer(kind(HIPBLAS_GEMM_DEFAULT)), value :: algo
            hipblasGemmBatchedExFortran = &
        hipblasGemmBatchedEx(handle, transA, transB, m, n, k, alpha, &
                                a, a_type, lda, b, b_type, ldb, beta, c, c_type, ldc, &
                                batch_count, compute_type, algo)
end function hipblasGemmBatchedExFortran

function hipblasGemmStridedBatchedExFortran(handle, transA, transB, m, n, k, alpha, a, a_type, lda, stride_a, &
                                            b, b_type, ldb, stride_b, beta, c, c_type, ldc, stride_c, &
                                            batch_count, compute_type, algo) &
    bind(c, name='hipblasGemmStridedBatchedExFortran')
    use iso_c_binding
    use hipblas_enums
    implicit none
    integer(kind(HIPBLAS_STATUS_SUCCESS)) :: hipblasGemmStridedBatchedExFortran
    type(c_ptr), value :: handle
    integer(kind(HIPBLAS_OP_N)), value :: transA
    integer(kind(HIPBLAS_OP_N)), value :: transB
    integer(c_int), value :: m
    integer(c_int), value :: n
    integer(c_int), value :: k
    type(c_ptr), value :: alpha
    type(c_ptr), value :: a
    integer(kind(HIP_R_16F)), value :: a_type
    integer(c_int), value :: lda
    integer(c_int64_t), value :: stride_a
    type(c_ptr), value :: b
    integer(kind(HIP_R_16F)), value :: b_type
    integer(c_int), value :: ldb
    integer(c_int64_t), value :: stride_b
    type(c_ptr), value :: beta
    type(c_ptr), value :: c
    integer(kind(HIP_R_16F)), value :: c_type
    integer(c_int), value :: ldc
    integer(c_int64_t), value :: stride_c
    integer(c_int), value :: batch_count
    integer(kind(HIPBLAS_COMPUTE_16F)), value :: compute_type
    integer(kind(HIPBLAS_GEMM_DEFAULT)), value :: algo
            hipblasGemmStridedBatchedExFortran = &
        hipblasGemmStridedBatchedEx(handle, transA, transB, m, n, k, alpha, &
                                    a, a_type, lda, stride_a, b, b_type, ldb, stride_b, beta, c, c_type, ldc, stride_c, &
                                    batch_count, compute_type, algo)
end function hipblasGemmStridedBatchedExFortran

function hipblasGemmExWithFlagsFortran(handle, transA, transB, m, n, k, alpha, a, a_type, lda, &
                                b, b_type, ldb, beta, c, c_type, ldc, &
                                compute_type, algo, flags) &
    bind(c, name='hipblasGemmExWithFlagsFortran')
    use iso_c_binding
    use hipblas_enums
    implicit none
    integer(kind(HIPBLAS_STATUS_SUCCESS)) :: hipblasGemmExWithFlagsFortran
    type(c_ptr), value :: handle
    integer(kind(HIPBLAS_OP_N)), value :: transA
    integer(kind(HIPBLAS_OP_N)), value :: transB
    integer(c_int), value :: m
    integer(c_int), value :: n
    integer(c_int), value :: k
    type(c_ptr), value :: alpha
    type(c_ptr), value :: a
    integer(kind(HIP_R_16F)), value :: a_type
    integer(c_int), value :: lda
    type(c_ptr), value :: b
    integer(kind(HIP_R_16F)), value :: b_type
    integer(c_int), value :: ldb
    type(c_ptr), value :: beta
    type(c_ptr), value :: c
    integer(kind(HIP_R_16F)), value :: c_type
    integer(c_int), value :: ldc
    integer(kind(HIPBLAS_COMPUTE_16F)), value :: compute_type
    integer(kind(HIPBLAS_GEMM_DEFAULT)), value :: algo
    integer(kind(HIPBLAS_GEMM_FLAGS_NONE)), value :: flags
            hipblasGemmExWithFlagsFortran = &
        hipblasGemmExWithFlags(handle, transA, transB, m, n, k, alpha, &
                        a, a_type, lda, b, b_type, ldb, beta, c, c_type, ldc, &
                        compute_type, algo, flags)
end function hipblasGemmExWithFlagsFortran

function hipblasGemmBatchedExWithFlagsFortran(handle, transA, transB, m, n, k, alpha, a, a_type, lda, &
                                        b, b_type, ldb, beta, c, c_type, ldc, &
                                        batch_count, compute_type, algo, flags) &
    bind(c, name='hipblasGemmBatchedExWithFlagsFortran')
    use iso_c_binding
    use hipblas_enums
    implicit none
    integer(kind(HIPBLAS_STATUS_SUCCESS)) :: hipblasGemmBatchedExWithFlagsFortran
    type(c_ptr), value :: handle
    integer(kind(HIPBLAS_OP_N)), value :: transA
    integer(kind(HIPBLAS_OP_N)), value :: transB
    integer(c_int), value :: m
    integer(c_int), value :: n
    integer(c_int), value :: k
    type(c_ptr), value :: alpha
    type(c_ptr), value :: a
    integer(kind(HIP_R_16F)), value :: a_type
    integer(c_int), value :: lda
    type(c_ptr), value :: b
    integer(kind(HIP_R_16F)), value :: b_type
    integer(c_int), value :: ldb
    type(c_ptr), value :: beta
    type(c_ptr), value :: c
    integer(kind(HIP_R_16F)), value :: c_type
    integer(c_int), value :: ldc
    integer(c_int), value :: batch_count
    integer(kind(HIPBLAS_COMPUTE_16F)), value :: compute_type
    integer(kind(HIPBLAS_GEMM_DEFAULT)), value :: algo
    integer(kind(HIPBLAS_GEMM_FLAGS_NONE)), value :: flags
            hipblasGemmBatchedExWithFlagsFortran = &
        hipblasGemmBatchedExWithFlags(handle, transA, transB, m, n, k, alpha, &
                                a, a_type, lda, b, b_type, ldb, beta, c, c_type, ldc, &
                                batch_count, compute_type, algo, flags)
end function hipblasGemmBatchedExWithFlagsFortran

function hipblasGemmStridedBatchedExWithFlagsFortran(handle, transA, transB, m, n, k, alpha, a, a_type, lda, stride_a, &
                                            b, b_type, ldb, stride_b, beta, c, c_type, ldc, stride_c, &
                                            batch_count, compute_type, algo, flags) &
    bind(c, name='hipblasGemmStridedBatchedExWithFlagsFortran')
    use iso_c_binding
    use hipblas_enums
    implicit none
    integer(kind(HIPBLAS_STATUS_SUCCESS)) :: hipblasGemmStridedBatchedExWithFlagsFortran
    type(c_ptr), value :: handle
    integer(kind(HIPBLAS_OP_N)), value :: transA
    integer(kind(HIPBLAS_OP_N)), value :: transB
    integer(c_int), value :: m
    integer(c_int), value :: n
    integer(c_int), value :: k
    type(c_ptr), value :: alpha
    type(c_ptr), value :: a
    integer(kind(HIP_R_16F)), value :: a_type
    integer(c_int), value :: lda
    integer(c_int64_t), value :: stride_a
    type(c_ptr), value :: b
    integer(kind(HIP_R_16F)), value :: b_type
    integer(c_int), value :: ldb
    integer(c_int64_t), value :: stride_b
    type(c_ptr), value :: beta
    type(c_ptr), value :: c
    integer(kind(HIP_R_16F)), value :: c_type
    integer(c_int), value :: ldc
    integer(c_int64_t), value :: stride_c
    integer(c_int), value :: batch_count
    integer(kind(HIPBLAS_COMPUTE_16F)), value :: compute_type
    integer(kind(HIPBLAS_GEMM_DEFAULT)), value :: algo
    integer(kind(HIPBLAS_GEMM_FLAGS_NONE)), value :: flags
            hipblasGemmStridedBatchedExWithFlagsFortran = &
        hipblasGemmStridedBatchedExWithFlags(handle, transA, transB, m, n, k, alpha, &
                                    a, a_type, lda, stride_a, b, b_type, ldb, stride_b, beta, c, c_type, ldc, stride_c, &
                                    batch_count, compute_type, algo, flags)
end function hipblasGemmStridedBatchedExWithFlagsFortran

! syrkEx
function hipblasSyrkExFortran(handle, uplo, transA, n, k, alpha, a, a_type, lda, &
                                beta, c, c_type, ldc) &
    bind(c, name='hipblasSyrkExFortran')
    use iso_c_binding
    use hipblas_enums
    implicit none
    integer(kind(HIPBLAS_STATUS_SUCCESS)) :: hipblasSyrkExFortran
    type(c_ptr), value :: handle
    integer(kind(HIPBLAS_FILL_MODE_UPPER)), value :: uplo
    integer(kind(HIPBLAS_OP_N)), value :: transA
    integer(c_int), value :: n
    integer(c_int), value :: k
    type(c_ptr), value :: alpha
    type(c_ptr), value :: a
    integer(kind(HIP_R_16F)), value :: a_type
    integer(c_int), value :: lda
    type(c_ptr), value :: beta
    type(c_ptr), value :: c
    integer(kind(HIP_R_16F)), value :: c_type
    integer(c_int), value :: ldc
            hipblasSyrkExFortran = &
        hipblasSyrkEx(handle, uplo, transA, n, k, alpha, &
                        a, a_type, lda, beta, c, c_type, ldc)
end function hipblasSyrkExFortran

! herkEx
function hipblasHerkExFortran(handle, uplo, transA, n, k, alpha, a, a_type, lda, &
                                beta, c, c_type, ldc, compute_type) &
    bind(c, name='hipblasHerkExFortran')
    use iso_c_binding
    use hipblas_enums
    implicit none
    integer(kind(HIPBLAS_STATUS_SUCCESS)) :: hipblasHerkExFortran
    type(c_ptr), value :: handle
    integer(kind(HIPBLAS_FILL_MODE_UPPER)), value :: uplo
    integer(kind(HIPBLAS_OP_N)), value :: transA
    integer(c_int), value :: n
    integer(c_int), value :: k
    type(c_ptr), value :: alpha
    type(c_ptr), value :: a
    integer(kind(HIP_R_16F)), value :: a_type
    integer(c_int), value :: lda
    type(c_ptr), value :: beta
    type(c_ptr), value :: c
    integer(kind(HIP_R_16F)), value :: c_type
    integer(c_int), value :: ldc
    integer(kind(HIP_R_16F)), value :: compute_type
        hipblasHerkExFortran = &
        hipblasHerkEx(handle, uplo, transA, n, k, alpha, &
                        a, a_type, lda, beta, c, c_type, ldc, compute_type)
end function hipblasHerkExFortran

! trsmEx
function hipblasTrsmExFortran(handle, side, uplo, transA, diag, m, n, alpha, A, lda, &
                                B, ldb, invA, invA_size, compute_type) &
    bind(c, name='hipblasTrsmExFortran')
    use iso_c_binding
    use hipblas_enums
    implicit none
    integer(kind(HIPBLAS_STATUS_SUCCESS)) :: hipblasTrsmExFortran
    type(c_ptr), value :: handle
    integer(kind(HIPBLAS_SIDE_LEFT)), value :: side
    integer(kind(HIPBLAS_FILL_MODE_UPPER)), value :: uplo
    integer(kind(HIPBLAS_OP_N)), value :: transA
    integer(kind(HIPBLAS_DIAG_UNIT)), value :: diag
    integer(c_int), value :: m
    integer(c_int), value :: n
    type(c_ptr), value :: alpha
    type(c_ptr), value :: A
    integer(c_int), value :: lda
    type(c_ptr), value :: B
    integer(c_int), value :: ldb
    type(c_ptr), value :: invA
    integer(c_int), value :: invA_size
    integer(kind(HIP_R_16F)), value :: compute_type
            hipblasTrsmExFortran = &
        hipblasTrsmEx(handle, side, uplo, transA, diag, m, n, alpha, &
                        A, lda, B, ldb, invA, invA_size, compute_type)
end function hipblasTrsmExFortran

function hipblasTrsmBatchedExFortran(handle, side, uplo, transA, diag, m, n, alpha, A, lda, &
                                        B, ldb, batch_count, invA, invA_size, compute_type) &
    bind(c, name='hipblasTrsmBatchedExFortran')
    use iso_c_binding
    use hipblas_enums
    implicit none
    integer(kind(HIPBLAS_STATUS_SUCCESS)) :: hipblasTrsmBatchedExFortran
    type(c_ptr), value :: handle
    integer(kind(HIPBLAS_SIDE_LEFT)), value :: side
    integer(kind(HIPBLAS_FILL_MODE_UPPER)), value :: uplo
    integer(kind(HIPBLAS_OP_N)), value :: transA
    integer(kind(HIPBLAS_DIAG_UNIT)), value :: diag
    integer(c_int), value :: m
    integer(c_int), value :: n
    type(c_ptr), value :: alpha
    type(c_ptr), value :: A
    integer(c_int), value :: lda
    type(c_ptr), value :: B
    integer(c_int), value :: ldb
    integer(c_int), value :: batch_count
    type(c_ptr), value :: invA
    integer(c_int), value :: invA_size
    integer(kind(HIP_R_16F)), value :: compute_type
            hipblasTrsmBatchedExFortran = &
        hipblasTrsmBatchedEx(handle, side, uplo, transA, diag, m, n, alpha, &
                                A, lda, B, ldb, batch_count, invA, invA_size, compute_type)
end function hipblasTrsmBatchedExFortran

function hipblasTrsmStridedBatchedExFortran(handle, side, uplo, transA, diag, m, n, alpha, A, lda, stride_A, &
                                            B, ldb, stride_B, batch_count, invA, invA_size, stride_invA, compute_type) &
    bind(c, name='hipblasTrsmStridedBatchedExFortran')
    use iso_c_binding
    use hipblas_enums
    implicit none
    integer(kind(HIPBLAS_STATUS_SUCCESS)) :: hipblasTrsmStridedBatchedExFortran
    type(c_ptr), value :: handle
    integer(kind(HIPBLAS_SIDE_LEFT)), value :: side
    integer(kind(HIPBLAS_FILL_MODE_UPPER)), value :: uplo
    integer(kind(HIPBLAS_OP_N)), value :: transA
    integer(kind(HIPBLAS_DIAG_UNIT)), value :: diag
    integer(c_int), value :: m
    integer(c_int), value :: n
    type(c_ptr), value :: alpha
    type(c_ptr), value :: A
    integer(c_int), value :: lda
    integer(c_int64_t), value :: stride_A
    type(c_ptr), value :: B
    integer(c_int), value :: ldb
    integer(c_int64_t), value :: stride_B
    integer(c_int), value :: batch_count
    type(c_ptr), value :: invA
    integer(c_int), value :: invA_size
    integer(c_int64_t), value :: stride_invA
    integer(kind(HIP_R_16F)), value :: compute_type
            hipblasTrsmStridedBatchedExFortran = &
        hipblasTrsmStridedBatchedEx(handle, side, uplo, transA, diag, m, n, alpha, &
                                    A, lda, stride_A, B, ldb, stride_B, batch_count, invA, invA_size, stride_invA, compute_type)
end function hipblasTrsmStridedBatchedExFortran

! AxpyEx
function hipblasAxpyExFortran(handle, n, alpha, alphaType, x, xType, incx, y, yType, incy, executionType) &
    bind(c, name='hipblasAxpyExFortran')
    use iso_c_binding
    use hipblas_enums
    implicit none
    integer(kind(HIPBLAS_STATUS_SUCCESS)) :: hipblasAxpyExFortran
    type(c_ptr), value :: handle
    integer(c_int), value :: n
    type(c_ptr), value :: alpha
    integer(kind(HIP_R_16F)), value :: alphaType
    type(c_ptr), value :: x
    integer(kind(HIP_R_16F)), value :: xType
    integer(c_int), value :: incx
    type(c_ptr), value :: y
    integer(kind(HIP_R_16F)), value :: yType
    integer(c_int), value :: incy
    integer(kind(HIP_R_16F)), value :: executionType
            hipblasAxpyExFortran = &
        hipblasAxpyEx(handle, n, alpha, alphaType, x, xType, incx, y, yType, incy, executionType)
    return
end function hipblasAxpyExFortran

function hipblasAxpyBatchedExFortran(handle, n, alpha, alphaType, x, xType, incx, y, yType, incy, batch_count, executionType) &
    bind(c, name='hipblasAxpyBatchedExFortran')
    use iso_c_binding
    use hipblas_enums
    implicit none
    integer(kind(HIPBLAS_STATUS_SUCCESS)) :: hipblasAxpyBatchedExFortran
    type(c_ptr), value :: handle
    integer(c_int), value :: n
    type(c_ptr), value :: alpha
    integer(kind(HIP_R_16F)), value :: alphaType
    type(c_ptr), value :: x
    integer(kind(HIP_R_16F)), value :: xType
    integer(c_int), value :: incx
    type(c_ptr), value :: y
    integer(kind(HIP_R_16F)), value :: yType
    integer(c_int), value :: incy
    integer(c_int), value :: batch_count
    integer(kind(HIP_R_16F)), value :: executionType
            hipblasAxpyBatchedExFortran = &
        hipblasAxpyBatchedEx(handle, n, alpha, alphaType, x, xType, incx, y, yType, incy, batch_count, executionType)
    return
end function hipblasAxpyBatchedExFortran

function hipblasAxpyStridedBatchedExFortran(handle, n, alpha, alphaType, x, xType, incx, stridex, &
                                            y, yType, incy, stridey, batch_count, executionType) &
    bind(c, name='hipblasAxpyStridedBatchedExFortran')
    use iso_c_binding
    use hipblas_enums
    implicit none
    integer(kind(HIPBLAS_STATUS_SUCCESS)) :: hipblasAxpyStridedBatchedExFortran
    type(c_ptr), value :: handle
    integer(c_int), value :: n
    type(c_ptr), value :: alpha
    integer(kind(HIP_R_16F)), value :: alphaType
    type(c_ptr), value :: x
    integer(kind(HIP_R_16F)), value :: xType
    integer(c_int), value :: incx
    integer(c_int64_t), value :: stridex
    type(c_ptr), value :: y
    integer(kind(HIP_R_16F)), value :: yType
    integer(c_int), value :: incy
    integer(c_int64_t), value :: stridey
    integer(c_int), value :: batch_count
    integer(kind(HIP_R_16F)), value :: executionType
            hipblasAxpyStridedBatchedExFortran = &
        hipblasAxpyStridedBatchedEx(handle, n, alpha, alphaType, x, xType, incx, stridex, &
                                    y, yType, incy, stridey, batch_count, executionType)
    return
end function hipblasAxpyStridedBatchedExFortran

! DotEx
function hipblasDotExFortran(handle, n, x, xType, incx, y, yType, incy, result, &
                                resultType, executionType) &
    bind(c, name='hipblasDotExFortran')
    use iso_c_binding
    use hipblas_enums
    implicit none
    integer(kind(HIPBLAS_STATUS_SUCCESS)) :: hipblasDotExFortran
    type(c_ptr), value :: handle
    integer(c_int), value :: n
    type(c_ptr), value :: x
    integer(kind(HIP_R_16F)), value :: xType
    integer(c_int), value :: incx
    type(c_ptr), value :: y
    integer(kind(HIP_R_16F)), value :: yType
    integer(c_int), value :: incy
    type(c_ptr), value :: result
    integer(kind(HIP_R_16F)), value :: resultType
    integer(kind(HIP_R_16F)), value :: executionType
            hipblasDotExFortran = &
        hipblasDotEx(handle, n, x, xType, incx, y, yType, incy, result, resultType, executionType)
    return
end function hipblasDotExFortran

function hipblasDotcExFortran(handle, n, x, xType, incx, y, yType, incy, result, &
                                resultType, executionType) &
    bind(c, name='hipblasDotcExFortran')
    use iso_c_binding
    use hipblas_enums
    implicit none
    integer(kind(HIPBLAS_STATUS_SUCCESS)) :: hipblasDotcExFortran
    type(c_ptr), value :: handle
    integer(c_int), value :: n
    type(c_ptr), value :: x
    integer(kind(HIP_R_16F)), value :: xType
    integer(c_int), value :: incx
    type(c_ptr), value :: y
    integer(kind(HIP_R_16F)), value :: yType
    integer(c_int), value :: incy
    type(c_ptr), value :: result
    integer(kind(HIP_R_16F)), value :: resultType
    integer(kind(HIP_R_16F)), value :: executionType
            hipblasDotcExFortran = &
        hipblasDotcEx(handle, n, x, xType, incx, y, yType, incy, result, resultType, executionType)
    return
end function hipblasDotcExFortran

function hipblasDotBatchedExFortran(handle, n, x, xType, incx, y, yType, incy, batch_count, result, &
                                    resultType, executionType) &
    bind(c, name='hipblasDotBatchedExFortran')
    use iso_c_binding
    use hipblas_enums
    implicit none
    integer(kind(HIPBLAS_STATUS_SUCCESS)) :: hipblasDotBatchedExFortran
    type(c_ptr), value :: handle
    integer(c_int), value :: n
    type(c_ptr), value :: x
    integer(kind(HIP_R_16F)), value :: xType
    integer(c_int), value :: incx
    type(c_ptr), value :: y
    integer(kind(HIP_R_16F)), value :: yType
    integer(c_int), value :: incy
    integer(c_int), value :: batch_count
    type(c_ptr), value :: result
    integer(kind(HIP_R_16F)), value :: resultType
    integer(kind(HIP_R_16F)), value :: executionType
            hipblasDotBatchedExFortran = &
        hipblasDotBatchedEx(handle, n, x, xType, incx, y, yType, incy, batch_count, result, resultType, executionType)
    return
end function hipblasDotBatchedExFortran

function hipblasDotcBatchedExFortran(handle, n, x, xType, incx, y, yType, incy, batch_count, result, &
                                        resultType, executionType) &
    bind(c, name='hipblasDotcBatchedExFortran')
    use iso_c_binding
    use hipblas_enums
    implicit none
    integer(kind(HIPBLAS_STATUS_SUCCESS)) :: hipblasDotcBatchedExFortran
    type(c_ptr), value :: handle
    integer(c_int), value :: n
    type(c_ptr), value :: x
    integer(kind(HIP_R_16F)), value :: xType
    integer(c_int), value :: incx
    type(c_ptr), value :: y
    integer(kind(HIP_R_16F)), value :: yType
    integer(c_int), value :: incy
    integer(c_int), value :: batch_count
    type(c_ptr), value :: result
    integer(kind(HIP_R_16F)), value :: resultType
    integer(kind(HIP_R_16F)), value :: executionType
            hipblasDotcBatchedExFortran = &
        hipblasDotcBatchedEx(handle, n, x, xType, incx, y, yType, incy, batch_count, result, resultType, executionType)
    return
end function hipblasDotcBatchedExFortran

function hipblasDotStridedBatchedExFortran(handle, n, x, xType, incx, stridex, &
                                            y, yType, incy, stridey, batch_count, result, resultType, executionType) &
    bind(c, name='hipblasDotStridedBatchedExFortran')
    use iso_c_binding
    use hipblas_enums
    implicit none
    integer(kind(HIPBLAS_STATUS_SUCCESS)) :: hipblasDotStridedBatchedExFortran
    type(c_ptr), value :: handle
    integer(c_int), value :: n
    type(c_ptr), value :: x
    integer(kind(HIP_R_16F)), value :: xType
    integer(c_int), value :: incx
    integer(c_int64_t), value :: stridex
    type(c_ptr), value :: y
    integer(kind(HIP_R_16F)), value :: yType
    integer(c_int), value :: incy
    integer(c_int64_t), value :: stridey
    integer(c_int), value :: batch_count
    type(c_ptr), value :: result
    integer(kind(HIP_R_16F)), value :: resultType
    integer(kind(HIP_R_16F)), value :: executionType
            hipblasDotStridedBatchedExFortran = &
        hipblasDotStridedBatchedEx(handle, n, x, xType, incx, stridex, &
                                    y, yType, incy, stridey, batch_count, result, resultType, executionType)
    return
end function hipblasDotStridedBatchedExFortran

function hipblasDotcStridedBatchedExFortran(handle, n, x, xType, incx, stridex, &
                                            y, yType, incy, stridey, batch_count, result, resultType, executionType) &
    bind(c, name='hipblasDotcStridedBatchedExFortran')
    use iso_c_binding
    use hipblas_enums
    implicit none
    integer(kind(HIPBLAS_STATUS_SUCCESS)) :: hipblasDotcStridedBatchedExFortran
    type(c_ptr), value :: handle
    integer(c_int), value :: n
    type(c_ptr), value :: x
    integer(kind(HIP_R_16F)), value :: xType
    integer(c_int), value :: incx
    integer(c_int64_t), value :: stridex
    type(c_ptr), value :: y
    integer(kind(HIP_R_16F)), value :: yType
    integer(c_int), value :: incy
    integer(c_int64_t), value :: stridey
    integer(c_int), value :: batch_count
    type(c_ptr), value :: result
    integer(kind(HIP_R_16F)), value :: resultType
    integer(kind(HIP_R_16F)), value :: executionType
            hipblasDotcStridedBatchedExFortran = &
        hipblasDotcStridedBatchedEx(handle, n, x, xType, incx, stridex, &
                                    y, yType, incy, stridey, batch_count, result, resultType, executionType)
    return
end function hipblasDotcStridedBatchedExFortran

! Nrm2Ex
function hipblasNrm2ExFortran(handle, n, x, xType, incx, result, resultType, executionType) &
    bind(c, name='hipblasNrm2ExFortran')
    use iso_c_binding
    use hipblas_enums
    implicit none
    integer(kind(HIPBLAS_STATUS_SUCCESS)) :: hipblasNrm2ExFortran
    type(c_ptr), value :: handle
    integer(c_int), value :: n
    type(c_ptr), value :: x
    integer(kind(HIP_R_16F)), value :: xType
    integer(c_int), value :: incx
    type(c_ptr), value :: result
    integer(kind(HIP_R_16F)), value :: resultType
    integer(kind(HIP_R_16F)), value :: executionType
            hipblasNrm2ExFortran = &
        hipblasNrm2Ex(handle, n, x, xType, incx, result, resultType, executionType)
    return
end function hipblasNrm2ExFortran

function hipblasNrm2BatchedExFortran(handle, n, x, xType, incx, batch_count, result, resultType, executionType) &
    bind(c, name='hipblasNrm2BatchedExFortran')
    use iso_c_binding
    use hipblas_enums
    implicit none
    integer(kind(HIPBLAS_STATUS_SUCCESS)) :: hipblasNrm2BatchedExFortran
    type(c_ptr), value :: handle
    integer(c_int), value :: n
    type(c_ptr), value :: x
    integer(kind(HIP_R_16F)), value :: xType
    integer(c_int), value :: incx
    integer(c_int), value :: batch_count
    type(c_ptr), value :: result
    integer(kind(HIP_R_16F)), value :: resultType
    integer(kind(HIP_R_16F)), value :: executionType
            hipblasNrm2BatchedExFortran = &
        hipblasNrm2BatchedEx(handle, n, x, xType, incx, batch_count, result, resultType, executionType)
    return
end function hipblasNrm2BatchedExFortran

function hipblasNrm2StridedBatchedExFortran(handle, n, x, xType, incx, stridex, &
                                            batch_count, result, resultType, executionType) &
    bind(c, name='hipblasNrm2StridedBatchedExFortran')
    use iso_c_binding
    use hipblas_enums
    implicit none
    integer(kind(HIPBLAS_STATUS_SUCCESS)) :: hipblasNrm2StridedBatchedExFortran
    type(c_ptr), value :: handle
    integer(c_int), value :: n
    type(c_ptr), value :: x
    integer(kind(HIP_R_16F)), value :: xType
    integer(c_int), value :: incx
    integer(c_int64_t), value :: stridex
    integer(c_int), value :: batch_count
    type(c_ptr), value :: result
    integer(kind(HIP_R_16F)), value :: resultType
    integer(kind(HIP_R_16F)), value :: executionType
            hipblasNrm2StridedBatchedExFortran = &
        hipblasNrm2StridedBatchedEx(handle, n, x, xType, incx, stridex, &
                                    batch_count, result, resultType, executionType)
    return
end function hipblasNrm2StridedBatchedExFortran

! RotEx
function hipblasRotExFortran(handle, n, x, xType, incx, y, yType, incy, c, s, &
                                csType, executionType) &
    bind(c, name='hipblasRotExFortran')
    use iso_c_binding
    use hipblas_enums
    implicit none
    integer(kind(HIPBLAS_STATUS_SUCCESS)) :: hipblasRotExFortran
    type(c_ptr), value :: handle
    integer(c_int), value :: n
    type(c_ptr), value :: x
    integer(kind(HIP_R_16F)), value :: xType
    integer(c_int), value :: incx
    type(c_ptr), value :: y
    integer(kind(HIP_R_16F)), value :: yType
    integer(c_int), value :: incy
    type(c_ptr), value :: c
    type(c_ptr), value :: s
    integer(kind(HIP_R_16F)), value :: csType
    integer(kind(HIP_R_16F)), value :: executionType
            hipblasRotExFortran = &
        hipblasRotEx(handle, n, x, xType, incx, y, yType, incy, c, s, csType, executionType)
    return
end function hipblasRotExFortran

function hipblasRotBatchedExFortran(handle, n, x, xType, incx, y, yType, incy, c, s, &
                                    csType, batch_count, executionType) &
    bind(c, name='hipblasRotBatchedExFortran')
    use iso_c_binding
    use hipblas_enums
    implicit none
    integer(kind(HIPBLAS_STATUS_SUCCESS)) :: hipblasRotBatchedExFortran
    type(c_ptr), value :: handle
    integer(c_int), value :: n
    type(c_ptr), value :: x
    integer(kind(HIP_R_16F)), value :: xType
    integer(c_int), value :: incx
    type(c_ptr), value :: y
    integer(kind(HIP_R_16F)), value :: yType
    integer(c_int), value :: incy
    type(c_ptr), value :: c
    type(c_ptr), value :: s
    integer(kind(HIP_R_16F)), value :: csType
    integer(c_int), value :: batch_count
    integer(kind(HIP_R_16F)), value :: executionType
            hipblasRotBatchedExFortran = &
        hipblasRotBatchedEx(handle, n, x, xType, incx, y, yType, incy, c, s, csType, batch_count, executionType)
    return
end function hipblasRotBatchedExFortran

function hipblasRotStridedBatchedExFortran(handle, n, x, xType, incx, stridex, &
                                            y, yType, incy, stridey, c, s, csType, batch_count, executionType) &
    bind(c, name='hipblasRotStridedBatchedExFortran')
    use iso_c_binding
    use hipblas_enums
    implicit none
    integer(kind(HIPBLAS_STATUS_SUCCESS)) :: hipblasRotStridedBatchedExFortran
    type(c_ptr), value :: handle
    integer(c_int), value :: n
    type(c_ptr), value :: x
    integer(kind(HIP_R_16F)), value :: xType
    integer(c_int), value :: incx
    integer(c_int64_t), value :: stridex
    type(c_ptr), value :: y
    integer(kind(HIP_R_16F)), value :: yType
    integer(c_int), value :: incy
    integer(c_int64_t), value :: stridey
    type(c_ptr), value :: c
    type(c_ptr), value :: s
    integer(kind(HIP_R_16F)), value :: csType
    integer(c_int), value :: batch_count
    integer(kind(HIP_R_16F)), value :: executionType
            hipblasRotStridedBatchedExFortran = &
        hipblasRotStridedBatchedEx(handle, n, x, xType, incx, stridex, &
                                    y, yType, incy, stridey, c, s, csType, batch_count, executionType)
    return
end function hipblasRotStridedBatchedExFortran

! ScalEx
function hipblasScalExFortran(handle, n, alpha, alphaType, x, xType, incx, executionType) &
    bind(c, name='hipblasScalExFortran')
    use iso_c_binding
    use hipblas_enums
    implicit none
    integer(kind(HIPBLAS_STATUS_SUCCESS)) :: hipblasScalExFortran
    type(c_ptr), value :: handle
    integer(c_int), value :: n
    type(c_ptr), value :: alpha
    integer(kind(HIP_R_16F)), value :: alphaType
    type(c_ptr), value :: x
    integer(kind(HIP_R_16F)), value :: xType
    integer(c_int), value :: incx
    integer(kind(HIP_R_16F)), value :: executionType
            hipblasScalExFortran = &
        hipblasScalEx(handle, n, alpha, alphaType, x, xType, incx, executionType)
    return
end function hipblasScalExFortran

function hipblasScalBatchedExFortran(handle, n, alpha, alphaType, x, xType, incx, batch_count, executionType) &
    bind(c, name='hipblasScalBatchedExFortran')
    use iso_c_binding
    use hipblas_enums
    implicit none
    integer(kind(HIPBLAS_STATUS_SUCCESS)) :: hipblasScalBatchedExFortran
    type(c_ptr), value :: handle
    integer(c_int), value :: n
    type(c_ptr), value :: alpha
    integer(kind(HIP_R_16F)), value :: alphaType
    type(c_ptr), value :: x
    integer(kind(HIP_R_16F)), value :: xType
    integer(c_int), value :: incx
    integer(c_int), value :: batch_count
    integer(kind(HIP_R_16F)), value :: executionType
            hipblasScalBatchedExFortran = &
        hipblasScalBatchedEx(handle, n, alpha, alphaType, x, xType, incx, batch_count, executionType)
    return
end function hipblasScalBatchedExFortran

function hipblasScalStridedBatchedExFortran(handle, n, alpha, alphaType, x, xType, incx, stridex, &
                                            batch_count, executionType) &
    bind(c, name='hipblasScalStridedBatchedExFortran')
    use iso_c_binding
    use hipblas_enums
    implicit none
    integer(kind(HIPBLAS_STATUS_SUCCESS)) :: hipblasScalStridedBatchedExFortran
    type(c_ptr), value :: handle
    integer(c_int), value :: n
    type(c_ptr), value :: alpha
    integer(kind(HIP_R_16F)), value :: alphaType
    type(c_ptr), value :: x
    integer(kind(HIP_R_16F)), value :: xType
    integer(c_int), value :: incx
    integer(c_int64_t), value :: stridex
    integer(c_int), value :: batch_count
    integer(kind(HIP_R_16F)), value :: executionType
            hipblasScalStridedBatchedExFortran = &
        hipblasScalStridedBatchedEx(handle, n, alpha, alphaType, x, xType, incx, stridex, &
                                    batch_count, executionType)
    return
end function hipblasScalStridedBatchedExFortran

!     ! CsyrkEx
!     function hipblasCsyrkExFortran(handle, uplo, trans, n, k, alpha, A, Atype, lda, beta, C, Ctype, ldc) &
!         bind(c, name = 'hipblasCsyrkExFortran')
!         use iso_c_binding
!         use hipblas_enums
!         implicit none
!         integer(kind(HIPBLAS_STATUS_SUCCESS)) :: hipblasCsyrkExFortran
!         type(c_ptr), value :: handle
!         integer(kind(HIPBLAS_FILL_MODE_FULL)), value :: uplo
!         integer(kind(HIPBLAS_OP_N)), value :: trans
!         integer(c_int), value :: n
!         integer(c_int), value :: k
!         type(c_ptr), value :: alpha
!         type(c_ptr), value :: A
!         integer(kind(HIP_R_16F)), value :: Atype
!         integer(c_int), value :: lda
!         type(c_ptr), value :: beta
!         type(c_ptr), value :: C
!         integer(kind(HIP_R_16F)), value :: Ctype
!         integer(c_int), value :: ldc
!         !         hipblasCsyrkExFortran = &
!         hipblasCsyrkEx(handle, uplo, trans, n, k, alpha, A, Atype, lda, beta, C, Ctype, ldc)
!     end function hipblasCsyrkExFortran
!     ! CherkEx
!     function hipblasCherkExFortran(handle, uplo, trans, n, k, alpha, A, Atype, lda, beta, C, Ctype, ldc) &
!         bind(c, name = 'hipblasCherkExFortran')
!         use iso_c_binding
!         use hipblas_enums
!         implicit none
!         integer(kind(HIPBLAS_STATUS_SUCCESS)) :: hipblasCherkExFortran
!         type(c_ptr), value :: handle
!         integer(kind(HIPBLAS_FILL_MODE_FULL)), value :: uplo
!         integer(kind(HIPBLAS_OP_N)), value :: trans
!         integer(c_int), value :: n
!         integer(c_int), value :: k
!         type(c_ptr), value :: alpha
!         type(c_ptr), value :: A
!         integer(kind(HIP_R_16F)), value :: Atype
!         integer(c_int), value :: lda
!         type(c_ptr), value :: beta
!         type(c_ptr), value :: C
!         integer(kind(HIP_R_16F)), value :: Ctype
!         integer(c_int), value :: ldc
!         !         hipblasCherkExFortran = &
!         hipblasCherkEx(handle, uplo, trans, n, k, alpha, A, Atype, lda, beta, C, Ctype, ldc)
!     end function hipblasCherkExFortran
