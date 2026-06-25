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

subroutine HIP_CHECK(stat)
    use iso_c_binding

    implicit none

    integer(c_int) :: stat

    if(stat /= 0) then
        write(*,*) 'Error: hip error'
        stop
    end if
end subroutine HIP_CHECK

subroutine HIPBLAS_CHECK(stat)
    use iso_c_binding

    implicit none

    integer(c_int) :: stat

    if(stat /= 0) then
        write(*,*) 'Error: hipblas error'
        stop
    endif
end subroutine HIPBLAS_CHECK


program example_fortran_gemm_ex
    use iso_c_binding
    use hipblas
    use hipblas_enums

    implicit none

    ! TODO: hip workaround until plugin is ready.
    interface
        function hipMalloc(ptr, size) &
#ifdef __HIP_PLATFORM_NVCC__
                bind(c, name = 'cudaMalloc')
#else
                bind(c, name = 'hipMalloc')
#endif
            use iso_c_binding
            implicit none
            integer :: hipMalloc
            type(c_ptr) :: ptr
            integer(c_size_t), value :: size
        end function hipMalloc

        function hipFree(ptr) &
#ifdef __HIP_PLATFORM_NVCC__
                bind(c, name = 'cudaFree')
#else
                bind(c, name = 'hipFree')
#endif
            use iso_c_binding
            implicit none
            integer :: hipFree
            type(c_ptr), value :: ptr
        end function hipFree

        function hipMemcpy(dst, src, size, kind) &
#ifdef __HIP_PLATFORM_NVCC__
                bind(c, name = 'cudaMemcpy')
#else
                bind(c, name = 'hipMemcpy')
#endif
            use iso_c_binding
            implicit none
            integer :: hipMemcpy
            type(c_ptr), value :: dst
            type(c_ptr), intent(in), value :: src
            integer(c_size_t), value :: size
            integer(c_int), value :: kind
        end function hipMemcpy

        function hipDeviceSynchronize() &
#ifdef __HIP_PLATFORM_NVCC__
                bind(c, name = 'cudaDeviceSynchronize')
#else
                bind(c, name = 'hipDeviceSynchronize')
#endif
            use iso_c_binding
            implicit none
            integer :: hipDeviceSynchronize
        end function hipDeviceSynchronize

        function hipDeviceReset() &
#ifdef __HIP_PLATFORM_NVCC__
                bind(c, name = 'cudaDeviceReset')
#else
                bind(c, name = 'hipDeviceReset')
#endif
            use iso_c_binding
            implicit none
            integer :: hipDeviceReset
        end function hipDeviceReset
    end interface
    ! TODO end


    integer tbegin(8)
    integer tend(8)
    real(8) timing
    logical :: failure_in_gemm = .FALSE.
    real(c_float) :: res

    integer(c_int) :: n = 1024
    integer(c_int) :: m = 1024
    integer(c_int) :: k = 1024
    integer(c_int) :: lda, ldb, ldc, size_A, size_B, size_C

    integer(kind(HIPBLAS_OP_N)), parameter :: transA = HIPBLAS_OP_N
    integer(kind(HIPBLAS_OP_N)), parameter :: transB = HIPBLAS_OP_N

    integer(kind(HIP_R_32F)), parameter :: aType = HIP_R_32F
    integer(kind(HIP_R_32F)), parameter :: bType = HIP_R_32F
    integer(kind(HIP_R_32F)), parameter :: cType = HIP_R_32F
    integer(kind(HIP_R_32F)), parameter :: computeType = HIPBLAS_COMPUTE_32F
    integer(kind(HIPBLAS_GEMM_DEFAULT)), parameter :: algo = HIPBLAS_GEMM_DEFAULT

    real(c_float), target :: alpha = 2
    real(c_float), target :: beta = 1

    real(4), dimension(:), allocatable, target :: hA
    real(4), dimension(:), allocatable, target :: hB
    real(4), dimension(:), allocatable, target :: hC
    type(c_ptr), target :: dA
    type(c_ptr), target :: dB
    type(c_ptr), target :: dC

    real :: gpu_time_used = 0.0

    integer(c_int) :: i, element

    ! Create hipBLAS handle
    type(c_ptr), target :: handle
    call HIPBLAS_CHECK(hipblasCreate(c_loc(handle)))

    ! transA = transB = N
    lda = m
    ldb = k
    ldc = m
    size_A = lda * k
    size_B = ldb * n
    size_C = ldc * n

    ! Allocate host-side memory
    ! transA = transB = N
    allocate(hA(size_A))
    allocate(hB(size_B))
    allocate(hC(size_C))

    ! Allocate device-side memory
    call HIP_CHECK(hipMalloc(dA, int(size_A, c_size_t) * 4))
    call HIP_CHECK(hipMalloc(dB, int(size_B, c_size_t) * 4))
    call HIP_CHECK(hipMalloc(dC, int(size_C, c_size_t) * 4))

    ! Initialize host memory
    ! Using constant matrices so result is easy to check
    do i = 1, size_A
        hA(i) = 2
    end do
    do i = 1, size_B
        hB(i) = 3
    end do
    do i = 1, size_C
        hC(i) = 4
    end do

    res = alpha * 2 * 3 * k + beta * 4

    ! Copy memory from host to device
    call HIP_CHECK(hipMemcpy(dA, c_loc(hA), int(size_A, c_size_t) * 4, 1))
    call HIP_CHECK(hipMemcpy(dB, c_loc(hB), int(size_B, c_size_t) * 4, 1))
    call HIP_CHECK(hipMemcpy(dC, c_loc(hC), int(size_C, c_size_t) * 4, 1))

    ! Begin time
    call date_and_time(values = tbegin)

    ! Call hipblasGemmEx
    call HIPBLAS_CHECK(hipblasSetPointerMode(handle, HIPBLAS_POINTER_MODE_HOST))
    call HIPBLAS_CHECK(hipblasGemmEx(handle, transA, transB, m, n, k,&
                                     c_loc(alpha), dA, aType, lda, dB,&
                                     bType, ldb, c_loc(beta), dC, Ctype,&
                                     ldc, computeType, algo))
    call HIP_CHECK(hipDeviceSynchronize())

    ! Stop time
    call date_and_time(values = tend)

    ! Copy output from device to host
    call HIP_CHECK(hipMemcpy(c_loc(hC), dC, int(size_C, c_size_t) * 4, 2))

    do element = 1, size_C
        if(res .ne. hC(element)) then
            failure_in_gemm = .true.
            write(*,*) '[hipblasGemmEx] ERROR: ', res, '!=', hC(element)
        end if
    end do

    ! Calculate time
    tbegin = tend - tbegin
    timing = (0.001d0 * tbegin(8) + tbegin(7) + 60d0 * tbegin(6) + 3600d0 * tbegin(5)) / 200d0 * 1000d0
    write(*,fmt='(A,F0.2,A)') '[hipblasGemmEx] took ', timing, ' msec'

    if(failure_in_gemm) then
        write(*,*) 'GEMMEX TEST FAIL'
    else
        write(*,*) 'GEMMEX TEST PASS'
    end if

    ! Cleanup
    call HIP_CHECK(hipFree(dA))
    call HIP_CHECK(hipFree(dB))
    call HIP_CHECK(hipFree(dC))
    deallocate(hA, hB, hC)
    call HIPBLAS_CHECK(hipblasDestroy(handle))
    call HIP_CHECK(hipDeviceReset())

end program example_fortran_gemm_ex
