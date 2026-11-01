# CMake generated Testfile for ROCshmem functional tests (install-time)
# Source directory: /__w/TheRock/TheRock/rocm-systems/projects/rocshmem/tests/functional_tests
# Build directory: /__w/TheRock/TheRock/build/comm-libs/rocshmem/build/tests/functional_tests
# Working directory: <install>/bin/rocshmem/tests/functional
#
# This file registers functional tests using installed binaries and scripts.
# Test commands use relative paths from the working directory:
#   - Wrapper script: ../../../../share/rocshmem/test_wrapper.sh
#   - Test executable: ../../../rocshmem_functional_tests  (installed to bin/)
#   - MPI paths are absolute (found at build configure time)
#
# Note: Test definitions are generated during CMake configuration

add_test(put_n2_w1_z1_1048576B_uuid "env" "ROCSHMEM_SLR_NP=2" "ROCSHMEM_MAX_NUM_CONTEXTS=1" "ROCSHMEM_HEAP_SIZE=6442450944" "ROCSHMEM_TEST_UUID=1" "../../../../share/rocshmem/test_wrapper.sh" "put_n2_w1_z1_1048576B_uuid" "../../../rocshmem_functional_tests" "-a" "2" "-w" "1" "-z" "1" "-s" "1048576" "-localbuftype" "heap")
set_tests_properties(put_n2_w1_z1_1048576B_uuid PROPERTIES
    TIMEOUT 300
    SKIP_RETURN_CODE 125
    LABELS "functional;backend_all;gpu_all;variant_uuid;launcher_slr;RMA"
    PROCESSORS 2
    WILL_FAIL FALSE
    FAIL_REGULAR_EXPRESSION "FAILED")

add_test(put_n2_w1_z1024_512B_uuid "env" "ROCSHMEM_SLR_NP=2" "ROCSHMEM_MAX_NUM_CONTEXTS=1" "ROCSHMEM_HEAP_SIZE=6442450944" "ROCSHMEM_TEST_UUID=1" "../../../../share/rocshmem/test_wrapper.sh" "put_n2_w1_z1024_512B_uuid" "../../../rocshmem_functional_tests" "-a" "2" "-w" "1" "-z" "1024" "-s" "512" "-localbuftype" "heap")
set_tests_properties(put_n2_w1_z1024_512B_uuid PROPERTIES
    TIMEOUT 300
    SKIP_RETURN_CODE 125
    LABELS "functional;backend_all;gpu_all;variant_uuid;launcher_slr;RMA"
    PROCESSORS 2
    WILL_FAIL FALSE
    FAIL_REGULAR_EXPRESSION "FAILED")

add_test(put_n2_w8_z1_1048576B_uuid "env" "ROCSHMEM_SLR_NP=2" "ROCSHMEM_MAX_NUM_CONTEXTS=8" "ROCSHMEM_HEAP_SIZE=6442450944" "ROCSHMEM_TEST_UUID=1" "../../../../share/rocshmem/test_wrapper.sh" "put_n2_w8_z1_1048576B_uuid" "../../../rocshmem_functional_tests" "-a" "2" "-w" "8" "-z" "1" "-s" "1048576" "-localbuftype" "heap")
set_tests_properties(put_n2_w8_z1_1048576B_uuid PROPERTIES
    TIMEOUT 300
    SKIP_RETURN_CODE 125
    LABELS "functional;backend_all;gpu_all;variant_uuid;launcher_slr;RMA"
    PROCESSORS 2
    WILL_FAIL FALSE
    FAIL_REGULAR_EXPRESSION "FAILED")

add_test(put_n2_w16_z128_8B_uuid "env" "ROCSHMEM_SLR_NP=2" "ROCSHMEM_MAX_NUM_CONTEXTS=16" "ROCSHMEM_HEAP_SIZE=6442450944" "ROCSHMEM_TEST_UUID=1" "../../../../share/rocshmem/test_wrapper.sh" "put_n2_w16_z128_8B_uuid" "../../../rocshmem_functional_tests" "-a" "2" "-w" "16" "-z" "128" "-s" "8" "-localbuftype" "heap")
set_tests_properties(put_n2_w16_z128_8B_uuid PROPERTIES
    TIMEOUT 300
    SKIP_RETURN_CODE 125
    LABELS "functional;backend_all;gpu_all;variant_uuid;launcher_slr;RMA"
    PROCESSORS 2
    WILL_FAIL FALSE
    FAIL_REGULAR_EXPRESSION "FAILED")

add_test(put_n2_w32_z256_512B_uuid "env" "ROCSHMEM_SLR_NP=2" "ROCSHMEM_MAX_NUM_CONTEXTS=32" "ROCSHMEM_HEAP_SIZE=6442450944" "ROCSHMEM_TEST_UUID=1" "../../../../share/rocshmem/test_wrapper.sh" "put_n2_w32_z256_512B_uuid" "../../../rocshmem_functional_tests" "-a" "2" "-w" "32" "-z" "256" "-s" "512" "-localbuftype" "heap")
set_tests_properties(put_n2_w32_z256_512B_uuid PROPERTIES
    TIMEOUT 300
    SKIP_RETURN_CODE 125
    LABELS "functional;backend_all;gpu_all;variant_uuid;launcher_slr;RMA"
    PROCESSORS 2
    WILL_FAIL FALSE
    FAIL_REGULAR_EXPRESSION "FAILED")

add_test(put_n2_w64_z1024_8B_uuid "env" "ROCSHMEM_SLR_NP=2" "ROCSHMEM_MAX_NUM_CONTEXTS=64" "ROCSHMEM_HEAP_SIZE=6442450944" "ROCSHMEM_TEST_UUID=1" "../../../../share/rocshmem/test_wrapper.sh" "put_n2_w64_z1024_8B_uuid" "../../../rocshmem_functional_tests" "-a" "2" "-w" "64" "-z" "1024" "-s" "8" "-localbuftype" "heap")
set_tests_properties(put_n2_w64_z1024_8B_uuid PROPERTIES
    TIMEOUT 300
    SKIP_RETURN_CODE 125
    LABELS "functional;backend_all;gpu_all;variant_uuid;launcher_slr;RMA"
    PROCESSORS 2
    WILL_FAIL FALSE
    FAIL_REGULAR_EXPRESSION "FAILED")

add_test(defaultctxput_n2_w4_z128_1024B_uuid "env" "ROCSHMEM_SLR_NP=2" "ROCSHMEM_MAX_NUM_CONTEXTS=4" "ROCSHMEM_HEAP_SIZE=6442450944" "ROCSHMEM_TEST_UUID=1" "../../../../share/rocshmem/test_wrapper.sh" "defaultctxput_n2_w4_z128_1024B_uuid" "../../../rocshmem_functional_tests" "-a" "61" "-w" "4" "-z" "128" "-s" "1024" "-localbuftype" "heap")
set_tests_properties(defaultctxput_n2_w4_z128_1024B_uuid PROPERTIES
    TIMEOUT 300
    SKIP_RETURN_CODE 125
    LABELS "functional;backend_all;gpu_all;variant_uuid;launcher_slr;RMA"
    PROCESSORS 2
    WILL_FAIL FALSE
    FAIL_REGULAR_EXPRESSION "FAILED")

add_test(teamctxput_n2_w4_z128_1024B_uuid "env" "ROCSHMEM_SLR_NP=2" "ROCSHMEM_MAX_NUM_CONTEXTS=4" "ROCSHMEM_HEAP_SIZE=6442450944" "ROCSHMEM_TEST_UUID=1" "../../../../share/rocshmem/test_wrapper.sh" "teamctxput_n2_w4_z128_1024B_uuid" "../../../rocshmem_functional_tests" "-a" "36" "-w" "4" "-z" "128" "-s" "1024" "-localbuftype" "heap")
set_tests_properties(teamctxput_n2_w4_z128_1024B_uuid PROPERTIES
    TIMEOUT 300
    SKIP_RETURN_CODE 125
    LABELS "functional;backend_all;gpu_all;variant_uuid;launcher_slr;RMA"
    PROCESSORS 2
    WILL_FAIL FALSE
    FAIL_REGULAR_EXPRESSION "FAILED")

add_test(teamctxput_n2_w16_z256_1024B_uuid "env" "ROCSHMEM_SLR_NP=2" "ROCSHMEM_MAX_NUM_CONTEXTS=16" "ROCSHMEM_HEAP_SIZE=6442450944" "ROCSHMEM_TEST_UUID=1" "../../../../share/rocshmem/test_wrapper.sh" "teamctxput_n2_w16_z256_1024B_uuid" "../../../rocshmem_functional_tests" "-a" "36" "-w" "16" "-z" "256" "-s" "1024" "-localbuftype" "heap")
set_tests_properties(teamctxput_n2_w16_z256_1024B_uuid PROPERTIES
    TIMEOUT 300
    SKIP_RETURN_CODE 125
    LABELS "functional;backend_all;gpu_all;variant_uuid;launcher_slr;RMA"
    PROCESSORS 2
    WILL_FAIL FALSE
    FAIL_REGULAR_EXPRESSION "FAILED")

add_test(wgput_n2_w1_z64_1048576B_uuid "env" "ROCSHMEM_SLR_NP=2" "ROCSHMEM_MAX_NUM_CONTEXTS=1" "ROCSHMEM_HEAP_SIZE=6442450944" "ROCSHMEM_TEST_UUID=1" "../../../../share/rocshmem/test_wrapper.sh" "wgput_n2_w1_z64_1048576B_uuid" "../../../rocshmem_functional_tests" "-a" "26" "-w" "1" "-z" "64" "-s" "1048576" "-localbuftype" "heap")
set_tests_properties(wgput_n2_w1_z64_1048576B_uuid PROPERTIES
    TIMEOUT 300
    SKIP_RETURN_CODE 125
    LABELS "functional;backend_all;gpu_all;variant_uuid;launcher_slr;RMA"
    PROCESSORS 2
    WILL_FAIL FALSE
    FAIL_REGULAR_EXPRESSION "FAILED")

add_test(wgput_n2_w2_z64_1048576B_uuid "env" "ROCSHMEM_SLR_NP=2" "ROCSHMEM_MAX_NUM_CONTEXTS=2" "ROCSHMEM_HEAP_SIZE=6442450944" "ROCSHMEM_TEST_UUID=1" "../../../../share/rocshmem/test_wrapper.sh" "wgput_n2_w2_z64_1048576B_uuid" "../../../rocshmem_functional_tests" "-a" "26" "-w" "2" "-z" "64" "-s" "1048576" "-localbuftype" "heap")
set_tests_properties(wgput_n2_w2_z64_1048576B_uuid PROPERTIES
    TIMEOUT 300
    SKIP_RETURN_CODE 125
    LABELS "functional;backend_all;gpu_all;variant_uuid;launcher_slr;RMA"
    PROCESSORS 2
    WILL_FAIL FALSE
    FAIL_REGULAR_EXPRESSION "FAILED")

add_test(wgput_n2_w16_z64_8B_uuid "env" "ROCSHMEM_SLR_NP=2" "ROCSHMEM_MAX_NUM_CONTEXTS=16" "ROCSHMEM_HEAP_SIZE=6442450944" "ROCSHMEM_TEST_UUID=1" "../../../../share/rocshmem/test_wrapper.sh" "wgput_n2_w16_z64_8B_uuid" "../../../rocshmem_functional_tests" "-a" "26" "-w" "16" "-z" "64" "-s" "8" "-localbuftype" "heap")
set_tests_properties(wgput_n2_w16_z64_8B_uuid PROPERTIES
    TIMEOUT 300
    SKIP_RETURN_CODE 125
    LABELS "functional;backend_all;gpu_all;variant_uuid;launcher_slr;RMA"
    PROCESSORS 2
    WILL_FAIL FALSE
    FAIL_REGULAR_EXPRESSION "FAILED")

add_test(waveput_n2_w1_z64_1048576B_uuid "env" "ROCSHMEM_SLR_NP=2" "ROCSHMEM_MAX_NUM_CONTEXTS=1" "ROCSHMEM_HEAP_SIZE=6442450944" "ROCSHMEM_TEST_UUID=1" "../../../../share/rocshmem/test_wrapper.sh" "waveput_n2_w1_z64_1048576B_uuid" "../../../rocshmem_functional_tests" "-a" "30" "-w" "1" "-z" "64" "-s" "1048576" "-localbuftype" "heap")
set_tests_properties(waveput_n2_w1_z64_1048576B_uuid PROPERTIES
    TIMEOUT 300
    SKIP_RETURN_CODE 125
    LABELS "functional;backend_all;gpu_all;variant_uuid;launcher_slr;RMA"
    PROCESSORS 2
    WILL_FAIL FALSE
    FAIL_REGULAR_EXPRESSION "FAILED")

add_test(waveput_n2_w2_z64_1048576B_uuid "env" "ROCSHMEM_SLR_NP=2" "ROCSHMEM_MAX_NUM_CONTEXTS=2" "ROCSHMEM_HEAP_SIZE=6442450944" "ROCSHMEM_TEST_UUID=1" "../../../../share/rocshmem/test_wrapper.sh" "waveput_n2_w2_z64_1048576B_uuid" "../../../rocshmem_functional_tests" "-a" "30" "-w" "2" "-z" "64" "-s" "1048576" "-localbuftype" "heap")
set_tests_properties(waveput_n2_w2_z64_1048576B_uuid PROPERTIES
    TIMEOUT 300
    SKIP_RETURN_CODE 125
    LABELS "functional;backend_all;gpu_all;variant_uuid;launcher_slr;RMA"
    PROCESSORS 2
    WILL_FAIL FALSE
    FAIL_REGULAR_EXPRESSION "FAILED")

add_test(waveput_n2_w2_z128_1048576B_uuid "env" "ROCSHMEM_SLR_NP=2" "ROCSHMEM_MAX_NUM_CONTEXTS=2" "ROCSHMEM_HEAP_SIZE=6442450944" "ROCSHMEM_TEST_UUID=1" "../../../../share/rocshmem/test_wrapper.sh" "waveput_n2_w2_z128_1048576B_uuid" "../../../rocshmem_functional_tests" "-a" "30" "-w" "2" "-z" "128" "-s" "1048576" "-localbuftype" "heap")
set_tests_properties(waveput_n2_w2_z128_1048576B_uuid PROPERTIES
    TIMEOUT 300
    SKIP_RETURN_CODE 125
    LABELS "functional;backend_all;gpu_all;variant_uuid;launcher_slr;RMA"
    PROCESSORS 2
    WILL_FAIL FALSE
    FAIL_REGULAR_EXPRESSION "FAILED")

add_test(waveput_n2_w16_z128_8B_uuid "env" "ROCSHMEM_SLR_NP=2" "ROCSHMEM_MAX_NUM_CONTEXTS=16" "ROCSHMEM_HEAP_SIZE=6442450944" "ROCSHMEM_TEST_UUID=1" "../../../../share/rocshmem/test_wrapper.sh" "waveput_n2_w16_z128_8B_uuid" "../../../rocshmem_functional_tests" "-a" "30" "-w" "16" "-z" "128" "-s" "8" "-localbuftype" "heap")
set_tests_properties(waveput_n2_w16_z128_8B_uuid PROPERTIES
    TIMEOUT 300
    SKIP_RETURN_CODE 125
    LABELS "functional;backend_all;gpu_all;variant_uuid;launcher_slr;RMA"
    PROCESSORS 2
    WILL_FAIL FALSE
    FAIL_REGULAR_EXPRESSION "FAILED")

add_test(p_n2_w1_z1_128B_uuid "env" "ROCSHMEM_SLR_NP=2" "ROCSHMEM_MAX_NUM_CONTEXTS=1" "ROCSHMEM_HEAP_SIZE=6442450944" "ROCSHMEM_TEST_UUID=1" "../../../../share/rocshmem/test_wrapper.sh" "p_n2_w1_z1_128B_uuid" "../../../rocshmem_functional_tests" "-a" "22" "-w" "1" "-z" "1" "-s" "128" "-localbuftype" "heap")
set_tests_properties(p_n2_w1_z1_128B_uuid PROPERTIES
    TIMEOUT 300
    SKIP_RETURN_CODE 125
    LABELS "functional;backend_all;gpu_all;variant_uuid;launcher_slr;RMA"
    PROCESSORS 2
    WILL_FAIL FALSE
    FAIL_REGULAR_EXPRESSION "FAILED")

add_test(p_n2_w1_z1024_2B_uuid "env" "ROCSHMEM_SLR_NP=2" "ROCSHMEM_MAX_NUM_CONTEXTS=1" "ROCSHMEM_HEAP_SIZE=6442450944" "ROCSHMEM_TEST_UUID=1" "../../../../share/rocshmem/test_wrapper.sh" "p_n2_w1_z1024_2B_uuid" "../../../rocshmem_functional_tests" "-a" "22" "-w" "1" "-z" "1024" "-s" "2" "-localbuftype" "heap")
set_tests_properties(p_n2_w1_z1024_2B_uuid PROPERTIES
    TIMEOUT 300
    SKIP_RETURN_CODE 125
    LABELS "functional;backend_all;gpu_all;variant_uuid;launcher_slr;RMA"
    PROCESSORS 2
    WILL_FAIL FALSE
    FAIL_REGULAR_EXPRESSION "FAILED")

add_test(p_n2_w8_z1_32B_uuid "env" "ROCSHMEM_SLR_NP=2" "ROCSHMEM_MAX_NUM_CONTEXTS=8" "ROCSHMEM_HEAP_SIZE=6442450944" "ROCSHMEM_TEST_UUID=1" "../../../../share/rocshmem/test_wrapper.sh" "p_n2_w8_z1_32B_uuid" "../../../rocshmem_functional_tests" "-a" "22" "-w" "8" "-z" "1" "-s" "32" "-localbuftype" "heap")
set_tests_properties(p_n2_w8_z1_32B_uuid PROPERTIES
    TIMEOUT 300
    SKIP_RETURN_CODE 125
    LABELS "functional;backend_all;gpu_all;variant_uuid;launcher_slr;RMA"
    PROCESSORS 2
    WILL_FAIL FALSE
    FAIL_REGULAR_EXPRESSION "FAILED")

add_test(p_n2_w16_z128_4B_uuid "env" "ROCSHMEM_SLR_NP=2" "ROCSHMEM_MAX_NUM_CONTEXTS=16" "ROCSHMEM_HEAP_SIZE=6442450944" "ROCSHMEM_TEST_UUID=1" "../../../../share/rocshmem/test_wrapper.sh" "p_n2_w16_z128_4B_uuid" "../../../rocshmem_functional_tests" "-a" "22" "-w" "16" "-z" "128" "-s" "4" "-localbuftype" "heap")
set_tests_properties(p_n2_w16_z128_4B_uuid PROPERTIES
    TIMEOUT 300
    SKIP_RETURN_CODE 125
    LABELS "functional;backend_all;gpu_all;variant_uuid;launcher_slr;RMA"
    PROCESSORS 2
    WILL_FAIL FALSE
    FAIL_REGULAR_EXPRESSION "FAILED")

add_test(putnbi_n2_w1_z1_1048576B_uuid "env" "ROCSHMEM_SLR_NP=2" "ROCSHMEM_MAX_NUM_CONTEXTS=1" "ROCSHMEM_HEAP_SIZE=6442450944" "ROCSHMEM_TEST_UUID=1" "../../../../share/rocshmem/test_wrapper.sh" "putnbi_n2_w1_z1_1048576B_uuid" "../../../rocshmem_functional_tests" "-a" "3" "-w" "1" "-z" "1" "-s" "1048576" "-localbuftype" "heap")
set_tests_properties(putnbi_n2_w1_z1_1048576B_uuid PROPERTIES
    TIMEOUT 300
    SKIP_RETURN_CODE 125
    LABELS "functional;backend_all;gpu_all;variant_uuid;launcher_slr;RMA"
    PROCESSORS 2
    WILL_FAIL FALSE
    FAIL_REGULAR_EXPRESSION "FAILED")

add_test(putnbi_n2_w1_z1024_512B_uuid "env" "ROCSHMEM_SLR_NP=2" "ROCSHMEM_MAX_NUM_CONTEXTS=1" "ROCSHMEM_HEAP_SIZE=6442450944" "ROCSHMEM_TEST_UUID=1" "../../../../share/rocshmem/test_wrapper.sh" "putnbi_n2_w1_z1024_512B_uuid" "../../../rocshmem_functional_tests" "-a" "3" "-w" "1" "-z" "1024" "-s" "512" "-localbuftype" "heap")
set_tests_properties(putnbi_n2_w1_z1024_512B_uuid PROPERTIES
    TIMEOUT 300
    SKIP_RETURN_CODE 125
    LABELS "functional;backend_all;gpu_all;variant_uuid;launcher_slr;RMA"
    PROCESSORS 2
    WILL_FAIL FALSE
    FAIL_REGULAR_EXPRESSION "FAILED")

add_test(putnbi_n2_w8_z1_1048576B_uuid "env" "ROCSHMEM_SLR_NP=2" "ROCSHMEM_MAX_NUM_CONTEXTS=8" "ROCSHMEM_HEAP_SIZE=6442450944" "ROCSHMEM_TEST_UUID=1" "../../../../share/rocshmem/test_wrapper.sh" "putnbi_n2_w8_z1_1048576B_uuid" "../../../rocshmem_functional_tests" "-a" "3" "-w" "8" "-z" "1" "-s" "1048576" "-localbuftype" "heap")
set_tests_properties(putnbi_n2_w8_z1_1048576B_uuid PROPERTIES
    TIMEOUT 300
    SKIP_RETURN_CODE 125
    LABELS "functional;backend_all;gpu_all;variant_uuid;launcher_slr;RMA"
    PROCESSORS 2
    WILL_FAIL FALSE
    FAIL_REGULAR_EXPRESSION "FAILED")

add_test(putnbi_n2_w16_z128_8B_uuid "env" "ROCSHMEM_SLR_NP=2" "ROCSHMEM_MAX_NUM_CONTEXTS=16" "ROCSHMEM_HEAP_SIZE=6442450944" "ROCSHMEM_TEST_UUID=1" "../../../../share/rocshmem/test_wrapper.sh" "putnbi_n2_w16_z128_8B_uuid" "../../../rocshmem_functional_tests" "-a" "3" "-w" "16" "-z" "128" "-s" "8" "-localbuftype" "heap")
set_tests_properties(putnbi_n2_w16_z128_8B_uuid PROPERTIES
    TIMEOUT 300
    SKIP_RETURN_CODE 125
    LABELS "functional;backend_all;gpu_all;variant_uuid;launcher_slr;RMA"
    PROCESSORS 2
    WILL_FAIL FALSE
    FAIL_REGULAR_EXPRESSION "FAILED")

add_test(putnbi_n2_w32_z256_512B_uuid "env" "ROCSHMEM_SLR_NP=2" "ROCSHMEM_MAX_NUM_CONTEXTS=32" "ROCSHMEM_HEAP_SIZE=6442450944" "ROCSHMEM_TEST_UUID=1" "../../../../share/rocshmem/test_wrapper.sh" "putnbi_n2_w32_z256_512B_uuid" "../../../rocshmem_functional_tests" "-a" "3" "-w" "32" "-z" "256" "-s" "512" "-localbuftype" "heap")
set_tests_properties(putnbi_n2_w32_z256_512B_uuid PROPERTIES
    TIMEOUT 300
    SKIP_RETURN_CODE 125
    LABELS "functional;backend_all;gpu_all;variant_uuid;launcher_slr;RMA"
    PROCESSORS 2
    WILL_FAIL FALSE
    FAIL_REGULAR_EXPRESSION "FAILED")

add_test(putnbi_n2_w64_z1024_8B_uuid "env" "ROCSHMEM_SLR_NP=2" "ROCSHMEM_MAX_NUM_CONTEXTS=64" "ROCSHMEM_HEAP_SIZE=6442450944" "ROCSHMEM_TEST_UUID=1" "../../../../share/rocshmem/test_wrapper.sh" "putnbi_n2_w64_z1024_8B_uuid" "../../../rocshmem_functional_tests" "-a" "3" "-w" "64" "-z" "1024" "-s" "8" "-localbuftype" "heap")
set_tests_properties(putnbi_n2_w64_z1024_8B_uuid PROPERTIES
    TIMEOUT 300
    SKIP_RETURN_CODE 125
    LABELS "functional;backend_all;gpu_all;variant_uuid;launcher_slr;RMA"
    PROCESSORS 2
    WILL_FAIL FALSE
    FAIL_REGULAR_EXPRESSION "FAILED")

add_test(defaultctxputnbi_n2_w4_z128_1024B_uuid "env" "ROCSHMEM_SLR_NP=2" "ROCSHMEM_MAX_NUM_CONTEXTS=4" "ROCSHMEM_HEAP_SIZE=6442450944" "ROCSHMEM_TEST_UUID=1" "../../../../share/rocshmem/test_wrapper.sh" "defaultctxputnbi_n2_w4_z128_1024B_uuid" "../../../rocshmem_functional_tests" "-a" "62" "-w" "4" "-z" "128" "-s" "1024" "-localbuftype" "heap")
set_tests_properties(defaultctxputnbi_n2_w4_z128_1024B_uuid PROPERTIES
    TIMEOUT 300
    SKIP_RETURN_CODE 125
    LABELS "functional;backend_all;gpu_all;variant_uuid;launcher_slr;RMA"
    PROCESSORS 2
    WILL_FAIL FALSE
    FAIL_REGULAR_EXPRESSION "FAILED")

add_test(teamctxputnbi_n2_w4_z128_1024B_uuid "env" "ROCSHMEM_SLR_NP=2" "ROCSHMEM_MAX_NUM_CONTEXTS=4" "ROCSHMEM_HEAP_SIZE=6442450944" "ROCSHMEM_TEST_UUID=1" "../../../../share/rocshmem/test_wrapper.sh" "teamctxputnbi_n2_w4_z128_1024B_uuid" "../../../rocshmem_functional_tests" "-a" "37" "-w" "4" "-z" "128" "-s" "1024" "-localbuftype" "heap")
set_tests_properties(teamctxputnbi_n2_w4_z128_1024B_uuid PROPERTIES
    TIMEOUT 300
    SKIP_RETURN_CODE 125
    LABELS "functional;backend_all;gpu_all;variant_uuid;launcher_slr;RMA"
    PROCESSORS 2
    WILL_FAIL FALSE
    FAIL_REGULAR_EXPRESSION "FAILED")

add_test(teamctxputnbi_n2_w16_z256_1024B_uuid "env" "ROCSHMEM_SLR_NP=2" "ROCSHMEM_MAX_NUM_CONTEXTS=16" "ROCSHMEM_HEAP_SIZE=6442450944" "ROCSHMEM_TEST_UUID=1" "../../../../share/rocshmem/test_wrapper.sh" "teamctxputnbi_n2_w16_z256_1024B_uuid" "../../../rocshmem_functional_tests" "-a" "37" "-w" "16" "-z" "256" "-s" "1024" "-localbuftype" "heap")
set_tests_properties(teamctxputnbi_n2_w16_z256_1024B_uuid PROPERTIES
    TIMEOUT 300
    SKIP_RETURN_CODE 125
    LABELS "functional;backend_all;gpu_all;variant_uuid;launcher_slr;RMA"
    PROCESSORS 2
    WILL_FAIL FALSE
    FAIL_REGULAR_EXPRESSION "FAILED")

add_test(wgputnbi_n2_w1_z64_1048576B_uuid "env" "ROCSHMEM_SLR_NP=2" "ROCSHMEM_MAX_NUM_CONTEXTS=1" "ROCSHMEM_HEAP_SIZE=6442450944" "ROCSHMEM_TEST_UUID=1" "../../../../share/rocshmem/test_wrapper.sh" "wgputnbi_n2_w1_z64_1048576B_uuid" "../../../rocshmem_functional_tests" "-a" "27" "-w" "1" "-z" "64" "-s" "1048576" "-localbuftype" "heap")
set_tests_properties(wgputnbi_n2_w1_z64_1048576B_uuid PROPERTIES
    TIMEOUT 300
    SKIP_RETURN_CODE 125
    LABELS "functional;backend_all;gpu_all;variant_uuid;launcher_slr;RMA"
    PROCESSORS 2
    WILL_FAIL FALSE
    FAIL_REGULAR_EXPRESSION "FAILED")

add_test(wgputnbi_n2_w2_z64_1048576B_uuid "env" "ROCSHMEM_SLR_NP=2" "ROCSHMEM_MAX_NUM_CONTEXTS=2" "ROCSHMEM_HEAP_SIZE=6442450944" "ROCSHMEM_TEST_UUID=1" "../../../../share/rocshmem/test_wrapper.sh" "wgputnbi_n2_w2_z64_1048576B_uuid" "../../../rocshmem_functional_tests" "-a" "27" "-w" "2" "-z" "64" "-s" "1048576" "-localbuftype" "heap")
set_tests_properties(wgputnbi_n2_w2_z64_1048576B_uuid PROPERTIES
    TIMEOUT 300
    SKIP_RETURN_CODE 125
    LABELS "functional;backend_all;gpu_all;variant_uuid;launcher_slr;RMA"
    PROCESSORS 2
    WILL_FAIL FALSE
    FAIL_REGULAR_EXPRESSION "FAILED")

add_test(wgputnbi_n2_w16_z64_8B_uuid "env" "ROCSHMEM_SLR_NP=2" "ROCSHMEM_MAX_NUM_CONTEXTS=16" "ROCSHMEM_HEAP_SIZE=6442450944" "ROCSHMEM_TEST_UUID=1" "../../../../share/rocshmem/test_wrapper.sh" "wgputnbi_n2_w16_z64_8B_uuid" "../../../rocshmem_functional_tests" "-a" "27" "-w" "16" "-z" "64" "-s" "8" "-localbuftype" "heap")
set_tests_properties(wgputnbi_n2_w16_z64_8B_uuid PROPERTIES
    TIMEOUT 300
    SKIP_RETURN_CODE 125
    LABELS "functional;backend_all;gpu_all;variant_uuid;launcher_slr;RMA"
    PROCESSORS 2
    WILL_FAIL FALSE
    FAIL_REGULAR_EXPRESSION "FAILED")

add_test(waveputnbi_n2_w1_z64_1048576B_uuid "env" "ROCSHMEM_SLR_NP=2" "ROCSHMEM_MAX_NUM_CONTEXTS=1" "ROCSHMEM_HEAP_SIZE=6442450944" "ROCSHMEM_TEST_UUID=1" "../../../../share/rocshmem/test_wrapper.sh" "waveputnbi_n2_w1_z64_1048576B_uuid" "../../../rocshmem_functional_tests" "-a" "31" "-w" "1" "-z" "64" "-s" "1048576" "-localbuftype" "heap")
set_tests_properties(waveputnbi_n2_w1_z64_1048576B_uuid PROPERTIES
    TIMEOUT 300
    SKIP_RETURN_CODE 125
    LABELS "functional;backend_all;gpu_all;variant_uuid;launcher_slr;RMA"
    PROCESSORS 2
    WILL_FAIL FALSE
    FAIL_REGULAR_EXPRESSION "FAILED")

add_test(waveputnbi_n2_w2_z64_1048576B_uuid "env" "ROCSHMEM_SLR_NP=2" "ROCSHMEM_MAX_NUM_CONTEXTS=2" "ROCSHMEM_HEAP_SIZE=6442450944" "ROCSHMEM_TEST_UUID=1" "../../../../share/rocshmem/test_wrapper.sh" "waveputnbi_n2_w2_z64_1048576B_uuid" "../../../rocshmem_functional_tests" "-a" "31" "-w" "2" "-z" "64" "-s" "1048576" "-localbuftype" "heap")
set_tests_properties(waveputnbi_n2_w2_z64_1048576B_uuid PROPERTIES
    TIMEOUT 300
    SKIP_RETURN_CODE 125
    LABELS "functional;backend_all;gpu_all;variant_uuid;launcher_slr;RMA"
    PROCESSORS 2
    WILL_FAIL FALSE
    FAIL_REGULAR_EXPRESSION "FAILED")

add_test(waveputnbi_n2_w2_z128_1048576B_uuid "env" "ROCSHMEM_SLR_NP=2" "ROCSHMEM_MAX_NUM_CONTEXTS=2" "ROCSHMEM_HEAP_SIZE=6442450944" "ROCSHMEM_TEST_UUID=1" "../../../../share/rocshmem/test_wrapper.sh" "waveputnbi_n2_w2_z128_1048576B_uuid" "../../../rocshmem_functional_tests" "-a" "31" "-w" "2" "-z" "128" "-s" "1048576" "-localbuftype" "heap")
set_tests_properties(waveputnbi_n2_w2_z128_1048576B_uuid PROPERTIES
    TIMEOUT 300
    SKIP_RETURN_CODE 125
    LABELS "functional;backend_all;gpu_all;variant_uuid;launcher_slr;RMA"
    PROCESSORS 2
    WILL_FAIL FALSE
    FAIL_REGULAR_EXPRESSION "FAILED")

add_test(waveputnbi_n2_w16_z128_8B_uuid "env" "ROCSHMEM_SLR_NP=2" "ROCSHMEM_MAX_NUM_CONTEXTS=16" "ROCSHMEM_HEAP_SIZE=6442450944" "ROCSHMEM_TEST_UUID=1" "../../../../share/rocshmem/test_wrapper.sh" "waveputnbi_n2_w16_z128_8B_uuid" "../../../rocshmem_functional_tests" "-a" "31" "-w" "16" "-z" "128" "-s" "8" "-localbuftype" "heap")
set_tests_properties(waveputnbi_n2_w16_z128_8B_uuid PROPERTIES
    TIMEOUT 300
    SKIP_RETURN_CODE 125
    LABELS "functional;backend_all;gpu_all;variant_uuid;launcher_slr;RMA"
    PROCESSORS 2
    WILL_FAIL FALSE
    FAIL_REGULAR_EXPRESSION "FAILED")

add_test(get_n2_w1_z1_1048576B_uuid "env" "ROCSHMEM_SLR_NP=2" "ROCSHMEM_MAX_NUM_CONTEXTS=1" "ROCSHMEM_HEAP_SIZE=6442450944" "ROCSHMEM_TEST_UUID=1" "../../../../share/rocshmem/test_wrapper.sh" "get_n2_w1_z1_1048576B_uuid" "../../../rocshmem_functional_tests" "-a" "0" "-w" "1" "-z" "1" "-s" "1048576" "-localbuftype" "heap")
set_tests_properties(get_n2_w1_z1_1048576B_uuid PROPERTIES
    TIMEOUT 300
    SKIP_RETURN_CODE 125
    LABELS "functional;backend_ipc;backend_gda;gpu_all;variant_uuid;launcher_slr;RMA"
    PROCESSORS 2
    WILL_FAIL FALSE
    FAIL_REGULAR_EXPRESSION "FAILED")

add_test(get_n2_w1_z1024_512B_uuid "env" "ROCSHMEM_SLR_NP=2" "ROCSHMEM_MAX_NUM_CONTEXTS=1" "ROCSHMEM_HEAP_SIZE=6442450944" "ROCSHMEM_TEST_UUID=1" "../../../../share/rocshmem/test_wrapper.sh" "get_n2_w1_z1024_512B_uuid" "../../../rocshmem_functional_tests" "-a" "0" "-w" "1" "-z" "1024" "-s" "512" "-localbuftype" "heap")
set_tests_properties(get_n2_w1_z1024_512B_uuid PROPERTIES
    TIMEOUT 300
    SKIP_RETURN_CODE 125
    LABELS "functional;backend_ipc;backend_gda;gpu_all;variant_uuid;launcher_slr;RMA"
    PROCESSORS 2
    WILL_FAIL FALSE
    FAIL_REGULAR_EXPRESSION "FAILED")

add_test(get_n2_w8_z1_1048576B_uuid "env" "ROCSHMEM_SLR_NP=2" "ROCSHMEM_MAX_NUM_CONTEXTS=8" "ROCSHMEM_HEAP_SIZE=6442450944" "ROCSHMEM_TEST_UUID=1" "../../../../share/rocshmem/test_wrapper.sh" "get_n2_w8_z1_1048576B_uuid" "../../../rocshmem_functional_tests" "-a" "0" "-w" "8" "-z" "1" "-s" "1048576" "-localbuftype" "heap")
set_tests_properties(get_n2_w8_z1_1048576B_uuid PROPERTIES
    TIMEOUT 300
    SKIP_RETURN_CODE 125
    LABELS "functional;backend_ipc;backend_gda;gpu_all;variant_uuid;launcher_slr;RMA"
    PROCESSORS 2
    WILL_FAIL FALSE
    FAIL_REGULAR_EXPRESSION "FAILED")

add_test(get_n2_w16_z128_8B_uuid "env" "ROCSHMEM_SLR_NP=2" "ROCSHMEM_MAX_NUM_CONTEXTS=16" "ROCSHMEM_HEAP_SIZE=6442450944" "ROCSHMEM_TEST_UUID=1" "../../../../share/rocshmem/test_wrapper.sh" "get_n2_w16_z128_8B_uuid" "../../../rocshmem_functional_tests" "-a" "0" "-w" "16" "-z" "128" "-s" "8" "-localbuftype" "heap")
set_tests_properties(get_n2_w16_z128_8B_uuid PROPERTIES
    TIMEOUT 300
    SKIP_RETURN_CODE 125
    LABELS "functional;backend_ipc;backend_gda;gpu_all;variant_uuid;launcher_slr;RMA"
    PROCESSORS 2
    WILL_FAIL FALSE
    FAIL_REGULAR_EXPRESSION "FAILED")

add_test(get_n2_w32_z256_512B_uuid "env" "ROCSHMEM_SLR_NP=2" "ROCSHMEM_MAX_NUM_CONTEXTS=32" "ROCSHMEM_HEAP_SIZE=6442450944" "ROCSHMEM_TEST_UUID=1" "../../../../share/rocshmem/test_wrapper.sh" "get_n2_w32_z256_512B_uuid" "../../../rocshmem_functional_tests" "-a" "0" "-w" "32" "-z" "256" "-s" "512" "-localbuftype" "heap")
set_tests_properties(get_n2_w32_z256_512B_uuid PROPERTIES
    TIMEOUT 300
    SKIP_RETURN_CODE 125
    LABELS "functional;backend_ipc;backend_gda;gpu_all;variant_uuid;launcher_slr;RMA"
    PROCESSORS 2
    WILL_FAIL FALSE
    FAIL_REGULAR_EXPRESSION "FAILED")

add_test(get_n2_w64_z1024_8B_uuid "env" "ROCSHMEM_SLR_NP=2" "ROCSHMEM_MAX_NUM_CONTEXTS=64" "ROCSHMEM_HEAP_SIZE=6442450944" "ROCSHMEM_TEST_UUID=1" "../../../../share/rocshmem/test_wrapper.sh" "get_n2_w64_z1024_8B_uuid" "../../../rocshmem_functional_tests" "-a" "0" "-w" "64" "-z" "1024" "-s" "8" "-localbuftype" "heap")
set_tests_properties(get_n2_w64_z1024_8B_uuid PROPERTIES
    TIMEOUT 300
    SKIP_RETURN_CODE 125
    LABELS "functional;backend_ipc;backend_gda;gpu_all;variant_uuid;launcher_slr;RMA"
    PROCESSORS 2
    WILL_FAIL FALSE
    FAIL_REGULAR_EXPRESSION "FAILED")

add_test(defaultctxget_n2_w4_z128_1024B_uuid "env" "ROCSHMEM_SLR_NP=2" "ROCSHMEM_MAX_NUM_CONTEXTS=4" "ROCSHMEM_HEAP_SIZE=6442450944" "ROCSHMEM_TEST_UUID=1" "../../../../share/rocshmem/test_wrapper.sh" "defaultctxget_n2_w4_z128_1024B_uuid" "../../../rocshmem_functional_tests" "-a" "59" "-w" "4" "-z" "128" "-s" "1024" "-localbuftype" "heap")
set_tests_properties(defaultctxget_n2_w4_z128_1024B_uuid PROPERTIES
    TIMEOUT 300
    SKIP_RETURN_CODE 125
    LABELS "functional;backend_ipc;backend_gda;gpu_all;variant_uuid;launcher_slr;RMA"
    PROCESSORS 2
    WILL_FAIL FALSE
    FAIL_REGULAR_EXPRESSION "FAILED")

add_test(teamctxget_n2_w4_z128_1024B_uuid "env" "ROCSHMEM_SLR_NP=2" "ROCSHMEM_MAX_NUM_CONTEXTS=4" "ROCSHMEM_HEAP_SIZE=6442450944" "ROCSHMEM_TEST_UUID=1" "../../../../share/rocshmem/test_wrapper.sh" "teamctxget_n2_w4_z128_1024B_uuid" "../../../rocshmem_functional_tests" "-a" "34" "-w" "4" "-z" "128" "-s" "1024" "-localbuftype" "heap")
set_tests_properties(teamctxget_n2_w4_z128_1024B_uuid PROPERTIES
    TIMEOUT 300
    SKIP_RETURN_CODE 125
    LABELS "functional;backend_ipc;backend_gda;gpu_all;variant_uuid;launcher_slr;RMA"
    PROCESSORS 2
    WILL_FAIL FALSE
    FAIL_REGULAR_EXPRESSION "FAILED")

add_test(teamctxget_n2_w16_z256_1024B_uuid "env" "ROCSHMEM_SLR_NP=2" "ROCSHMEM_MAX_NUM_CONTEXTS=16" "ROCSHMEM_HEAP_SIZE=6442450944" "ROCSHMEM_TEST_UUID=1" "../../../../share/rocshmem/test_wrapper.sh" "teamctxget_n2_w16_z256_1024B_uuid" "../../../rocshmem_functional_tests" "-a" "34" "-w" "16" "-z" "256" "-s" "1024" "-localbuftype" "heap")
set_tests_properties(teamctxget_n2_w16_z256_1024B_uuid PROPERTIES
    TIMEOUT 300
    SKIP_RETURN_CODE 125
    LABELS "functional;backend_ipc;backend_gda;gpu_all;variant_uuid;launcher_slr;RMA"
    PROCESSORS 2
    WILL_FAIL FALSE
    FAIL_REGULAR_EXPRESSION "FAILED")

add_test(wgget_n2_w1_z64_1048576B_uuid "env" "ROCSHMEM_SLR_NP=2" "ROCSHMEM_MAX_NUM_CONTEXTS=1" "ROCSHMEM_HEAP_SIZE=6442450944" "ROCSHMEM_TEST_UUID=1" "../../../../share/rocshmem/test_wrapper.sh" "wgget_n2_w1_z64_1048576B_uuid" "../../../rocshmem_functional_tests" "-a" "24" "-w" "1" "-z" "64" "-s" "1048576" "-localbuftype" "heap")
set_tests_properties(wgget_n2_w1_z64_1048576B_uuid PROPERTIES
    TIMEOUT 300
    SKIP_RETURN_CODE 125
    LABELS "functional;backend_ipc;backend_gda;gpu_all;variant_uuid;launcher_slr;RMA"
    PROCESSORS 2
    WILL_FAIL FALSE
    FAIL_REGULAR_EXPRESSION "FAILED")

add_test(wgget_n2_w2_z64_1048576B_uuid "env" "ROCSHMEM_SLR_NP=2" "ROCSHMEM_MAX_NUM_CONTEXTS=2" "ROCSHMEM_HEAP_SIZE=6442450944" "ROCSHMEM_TEST_UUID=1" "../../../../share/rocshmem/test_wrapper.sh" "wgget_n2_w2_z64_1048576B_uuid" "../../../rocshmem_functional_tests" "-a" "24" "-w" "2" "-z" "64" "-s" "1048576" "-localbuftype" "heap")
set_tests_properties(wgget_n2_w2_z64_1048576B_uuid PROPERTIES
    TIMEOUT 300
    SKIP_RETURN_CODE 125
    LABELS "functional;backend_ipc;backend_gda;gpu_all;variant_uuid;launcher_slr;RMA"
    PROCESSORS 2
    WILL_FAIL FALSE
    FAIL_REGULAR_EXPRESSION "FAILED")

add_test(wgget_n2_w16_z64_8B_uuid "env" "ROCSHMEM_SLR_NP=2" "ROCSHMEM_MAX_NUM_CONTEXTS=16" "ROCSHMEM_HEAP_SIZE=6442450944" "ROCSHMEM_TEST_UUID=1" "../../../../share/rocshmem/test_wrapper.sh" "wgget_n2_w16_z64_8B_uuid" "../../../rocshmem_functional_tests" "-a" "24" "-w" "16" "-z" "64" "-s" "8" "-localbuftype" "heap")
set_tests_properties(wgget_n2_w16_z64_8B_uuid PROPERTIES
    TIMEOUT 300
    SKIP_RETURN_CODE 125
    LABELS "functional;backend_ipc;backend_gda;gpu_all;variant_uuid;launcher_slr;RMA"
    PROCESSORS 2
    WILL_FAIL FALSE
    FAIL_REGULAR_EXPRESSION "FAILED")

add_test(waveget_n2_w1_z64_1048576B_uuid "env" "ROCSHMEM_SLR_NP=2" "ROCSHMEM_MAX_NUM_CONTEXTS=1" "ROCSHMEM_HEAP_SIZE=6442450944" "ROCSHMEM_TEST_UUID=1" "../../../../share/rocshmem/test_wrapper.sh" "waveget_n2_w1_z64_1048576B_uuid" "../../../rocshmem_functional_tests" "-a" "28" "-w" "1" "-z" "64" "-s" "1048576" "-localbuftype" "heap")
set_tests_properties(waveget_n2_w1_z64_1048576B_uuid PROPERTIES
    TIMEOUT 300
    SKIP_RETURN_CODE 125
    LABELS "functional;backend_ipc;backend_gda;gpu_all;variant_uuid;launcher_slr;RMA"
    PROCESSORS 2
    WILL_FAIL FALSE
    FAIL_REGULAR_EXPRESSION "FAILED")

add_test(waveget_n2_w2_z64_1048576B_uuid "env" "ROCSHMEM_SLR_NP=2" "ROCSHMEM_MAX_NUM_CONTEXTS=2" "ROCSHMEM_HEAP_SIZE=6442450944" "ROCSHMEM_TEST_UUID=1" "../../../../share/rocshmem/test_wrapper.sh" "waveget_n2_w2_z64_1048576B_uuid" "../../../rocshmem_functional_tests" "-a" "28" "-w" "2" "-z" "64" "-s" "1048576" "-localbuftype" "heap")
set_tests_properties(waveget_n2_w2_z64_1048576B_uuid PROPERTIES
    TIMEOUT 300
    SKIP_RETURN_CODE 125
    LABELS "functional;backend_ipc;backend_gda;gpu_all;variant_uuid;launcher_slr;RMA"
    PROCESSORS 2
    WILL_FAIL FALSE
    FAIL_REGULAR_EXPRESSION "FAILED")

add_test(waveget_n2_w2_z128_1048576B_uuid "env" "ROCSHMEM_SLR_NP=2" "ROCSHMEM_MAX_NUM_CONTEXTS=2" "ROCSHMEM_HEAP_SIZE=6442450944" "ROCSHMEM_TEST_UUID=1" "../../../../share/rocshmem/test_wrapper.sh" "waveget_n2_w2_z128_1048576B_uuid" "../../../rocshmem_functional_tests" "-a" "28" "-w" "2" "-z" "128" "-s" "1048576" "-localbuftype" "heap")
set_tests_properties(waveget_n2_w2_z128_1048576B_uuid PROPERTIES
    TIMEOUT 300
    SKIP_RETURN_CODE 125
    LABELS "functional;backend_ipc;backend_gda;gpu_all;variant_uuid;launcher_slr;RMA"
    PROCESSORS 2
    WILL_FAIL FALSE
    FAIL_REGULAR_EXPRESSION "FAILED")

add_test(waveget_n2_w16_z128_8B_uuid "env" "ROCSHMEM_SLR_NP=2" "ROCSHMEM_MAX_NUM_CONTEXTS=16" "ROCSHMEM_HEAP_SIZE=6442450944" "ROCSHMEM_TEST_UUID=1" "../../../../share/rocshmem/test_wrapper.sh" "waveget_n2_w16_z128_8B_uuid" "../../../rocshmem_functional_tests" "-a" "28" "-w" "16" "-z" "128" "-s" "8" "-localbuftype" "heap")
set_tests_properties(waveget_n2_w16_z128_8B_uuid PROPERTIES
    TIMEOUT 300
    SKIP_RETURN_CODE 125
    LABELS "functional;backend_ipc;backend_gda;gpu_all;variant_uuid;launcher_slr;RMA"
    PROCESSORS 2
    WILL_FAIL FALSE
    FAIL_REGULAR_EXPRESSION "FAILED")

add_test(g_n2_w1_z1_128B_uuid "env" "ROCSHMEM_SLR_NP=2" "ROCSHMEM_MAX_NUM_CONTEXTS=1" "ROCSHMEM_HEAP_SIZE=6442450944" "ROCSHMEM_TEST_UUID=1" "../../../../share/rocshmem/test_wrapper.sh" "g_n2_w1_z1_128B_uuid" "../../../rocshmem_functional_tests" "-a" "23" "-w" "1" "-z" "1" "-s" "128" "-localbuftype" "heap")
set_tests_properties(g_n2_w1_z1_128B_uuid PROPERTIES
    TIMEOUT 300
    SKIP_RETURN_CODE 125
    LABELS "functional;backend_ipc;gpu_all;variant_uuid;launcher_slr;RMA"
    PROCESSORS 2
    WILL_FAIL FALSE
    FAIL_REGULAR_EXPRESSION "FAILED")

add_test(g_n2_w1_z1024_1B_uuid "env" "ROCSHMEM_SLR_NP=2" "ROCSHMEM_MAX_NUM_CONTEXTS=1" "ROCSHMEM_HEAP_SIZE=6442450944" "ROCSHMEM_TEST_UUID=1" "../../../../share/rocshmem/test_wrapper.sh" "g_n2_w1_z1024_1B_uuid" "../../../rocshmem_functional_tests" "-a" "23" "-w" "1" "-z" "1024" "-s" "1" "-localbuftype" "heap")
set_tests_properties(g_n2_w1_z1024_1B_uuid PROPERTIES
    TIMEOUT 300
    SKIP_RETURN_CODE 125
    LABELS "functional;backend_ipc;gpu_all;variant_uuid;launcher_slr;RMA"
    PROCESSORS 2
    WILL_FAIL FALSE
    FAIL_REGULAR_EXPRESSION "FAILED")

add_test(g_n2_w8_z1_32B_uuid "env" "ROCSHMEM_SLR_NP=2" "ROCSHMEM_MAX_NUM_CONTEXTS=8" "ROCSHMEM_HEAP_SIZE=6442450944" "ROCSHMEM_TEST_UUID=1" "../../../../share/rocshmem/test_wrapper.sh" "g_n2_w8_z1_32B_uuid" "../../../rocshmem_functional_tests" "-a" "23" "-w" "8" "-z" "1" "-s" "32" "-localbuftype" "heap")
set_tests_properties(g_n2_w8_z1_32B_uuid PROPERTIES
    TIMEOUT 300
    SKIP_RETURN_CODE 125
    LABELS "functional;backend_ipc;gpu_all;variant_uuid;launcher_slr;RMA"
    PROCESSORS 2
    WILL_FAIL FALSE
    FAIL_REGULAR_EXPRESSION "FAILED")

add_test(g_n2_w16_z128_4B_uuid "env" "ROCSHMEM_SLR_NP=2" "ROCSHMEM_MAX_NUM_CONTEXTS=16" "ROCSHMEM_HEAP_SIZE=6442450944" "ROCSHMEM_TEST_UUID=1" "../../../../share/rocshmem/test_wrapper.sh" "g_n2_w16_z128_4B_uuid" "../../../rocshmem_functional_tests" "-a" "23" "-w" "16" "-z" "128" "-s" "4" "-localbuftype" "heap")
set_tests_properties(g_n2_w16_z128_4B_uuid PROPERTIES
    TIMEOUT 300
    SKIP_RETURN_CODE 125
    LABELS "functional;backend_ipc;gpu_all;variant_uuid;launcher_slr;RMA"
    PROCESSORS 2
    WILL_FAIL FALSE
    FAIL_REGULAR_EXPRESSION "FAILED")

add_test(getnbi_n2_w1_z1_1048576B_uuid "env" "ROCSHMEM_SLR_NP=2" "ROCSHMEM_MAX_NUM_CONTEXTS=1" "ROCSHMEM_HEAP_SIZE=6442450944" "ROCSHMEM_TEST_UUID=1" "../../../../share/rocshmem/test_wrapper.sh" "getnbi_n2_w1_z1_1048576B_uuid" "../../../rocshmem_functional_tests" "-a" "1" "-w" "1" "-z" "1" "-s" "1048576" "-localbuftype" "heap")
set_tests_properties(getnbi_n2_w1_z1_1048576B_uuid PROPERTIES
    TIMEOUT 300
    SKIP_RETURN_CODE 125
    LABELS "functional;backend_ipc;backend_gda;gpu_all;variant_uuid;launcher_slr;RMA"
    PROCESSORS 2
    WILL_FAIL FALSE
    FAIL_REGULAR_EXPRESSION "FAILED")

add_test(getnbi_n2_w1_z1024_512B_uuid "env" "ROCSHMEM_SLR_NP=2" "ROCSHMEM_MAX_NUM_CONTEXTS=1" "ROCSHMEM_HEAP_SIZE=6442450944" "ROCSHMEM_TEST_UUID=1" "../../../../share/rocshmem/test_wrapper.sh" "getnbi_n2_w1_z1024_512B_uuid" "../../../rocshmem_functional_tests" "-a" "1" "-w" "1" "-z" "1024" "-s" "512" "-localbuftype" "heap")
set_tests_properties(getnbi_n2_w1_z1024_512B_uuid PROPERTIES
    TIMEOUT 300
    SKIP_RETURN_CODE 125
    LABELS "functional;backend_ipc;backend_gda;gpu_all;variant_uuid;launcher_slr;RMA"
    PROCESSORS 2
    WILL_FAIL FALSE
    FAIL_REGULAR_EXPRESSION "FAILED")

add_test(getnbi_n2_w8_z1_1048576B_uuid "env" "ROCSHMEM_SLR_NP=2" "ROCSHMEM_MAX_NUM_CONTEXTS=8" "ROCSHMEM_HEAP_SIZE=6442450944" "ROCSHMEM_TEST_UUID=1" "../../../../share/rocshmem/test_wrapper.sh" "getnbi_n2_w8_z1_1048576B_uuid" "../../../rocshmem_functional_tests" "-a" "1" "-w" "8" "-z" "1" "-s" "1048576" "-localbuftype" "heap")
set_tests_properties(getnbi_n2_w8_z1_1048576B_uuid PROPERTIES
    TIMEOUT 300
    SKIP_RETURN_CODE 125
    LABELS "functional;backend_ipc;backend_gda;gpu_all;variant_uuid;launcher_slr;RMA"
    PROCESSORS 2
    WILL_FAIL FALSE
    FAIL_REGULAR_EXPRESSION "FAILED")

add_test(getnbi_n2_w16_z128_8B_uuid "env" "ROCSHMEM_SLR_NP=2" "ROCSHMEM_MAX_NUM_CONTEXTS=16" "ROCSHMEM_HEAP_SIZE=6442450944" "ROCSHMEM_TEST_UUID=1" "../../../../share/rocshmem/test_wrapper.sh" "getnbi_n2_w16_z128_8B_uuid" "../../../rocshmem_functional_tests" "-a" "1" "-w" "16" "-z" "128" "-s" "8" "-localbuftype" "heap")
set_tests_properties(getnbi_n2_w16_z128_8B_uuid PROPERTIES
    TIMEOUT 300
    SKIP_RETURN_CODE 125
    LABELS "functional;backend_ipc;backend_gda;gpu_all;variant_uuid;launcher_slr;RMA"
    PROCESSORS 2
    WILL_FAIL FALSE
    FAIL_REGULAR_EXPRESSION "FAILED")

add_test(getnbi_n2_w32_z256_512B_uuid "env" "ROCSHMEM_SLR_NP=2" "ROCSHMEM_MAX_NUM_CONTEXTS=32" "ROCSHMEM_HEAP_SIZE=6442450944" "ROCSHMEM_TEST_UUID=1" "../../../../share/rocshmem/test_wrapper.sh" "getnbi_n2_w32_z256_512B_uuid" "../../../rocshmem_functional_tests" "-a" "1" "-w" "32" "-z" "256" "-s" "512" "-localbuftype" "heap")
set_tests_properties(getnbi_n2_w32_z256_512B_uuid PROPERTIES
    TIMEOUT 300
    SKIP_RETURN_CODE 125
    LABELS "functional;backend_ipc;backend_gda;gpu_all;variant_uuid;launcher_slr;RMA"
    PROCESSORS 2
    WILL_FAIL FALSE
    FAIL_REGULAR_EXPRESSION "FAILED")

add_test(getnbi_n2_w64_z1024_8B_uuid "env" "ROCSHMEM_SLR_NP=2" "ROCSHMEM_MAX_NUM_CONTEXTS=64" "ROCSHMEM_HEAP_SIZE=6442450944" "ROCSHMEM_TEST_UUID=1" "../../../../share/rocshmem/test_wrapper.sh" "getnbi_n2_w64_z1024_8B_uuid" "../../../rocshmem_functional_tests" "-a" "1" "-w" "64" "-z" "1024" "-s" "8" "-localbuftype" "heap")
set_tests_properties(getnbi_n2_w64_z1024_8B_uuid PROPERTIES
    TIMEOUT 300
    SKIP_RETURN_CODE 125
    LABELS "functional;backend_ipc;backend_gda;gpu_all;variant_uuid;launcher_slr;RMA"
    PROCESSORS 2
    WILL_FAIL FALSE
    FAIL_REGULAR_EXPRESSION "FAILED")

add_test(defaultctxgetnbi_n2_w4_z128_1024B_uuid "env" "ROCSHMEM_SLR_NP=2" "ROCSHMEM_MAX_NUM_CONTEXTS=4" "ROCSHMEM_HEAP_SIZE=6442450944" "ROCSHMEM_TEST_UUID=1" "../../../../share/rocshmem/test_wrapper.sh" "defaultctxgetnbi_n2_w4_z128_1024B_uuid" "../../../rocshmem_functional_tests" "-a" "60" "-w" "4" "-z" "128" "-s" "1024" "-localbuftype" "heap")
set_tests_properties(defaultctxgetnbi_n2_w4_z128_1024B_uuid PROPERTIES
    TIMEOUT 300
    SKIP_RETURN_CODE 125
    LABELS "functional;backend_ipc;backend_gda;gpu_all;variant_uuid;launcher_slr;RMA"
    PROCESSORS 2
    WILL_FAIL FALSE
    FAIL_REGULAR_EXPRESSION "FAILED")

add_test(teamctxgetnbi_n2_w4_z128_1024B_uuid "env" "ROCSHMEM_SLR_NP=2" "ROCSHMEM_MAX_NUM_CONTEXTS=4" "ROCSHMEM_HEAP_SIZE=6442450944" "ROCSHMEM_TEST_UUID=1" "../../../../share/rocshmem/test_wrapper.sh" "teamctxgetnbi_n2_w4_z128_1024B_uuid" "../../../rocshmem_functional_tests" "-a" "35" "-w" "4" "-z" "128" "-s" "1024" "-localbuftype" "heap")
set_tests_properties(teamctxgetnbi_n2_w4_z128_1024B_uuid PROPERTIES
    TIMEOUT 300
    SKIP_RETURN_CODE 125
    LABELS "functional;backend_ipc;backend_gda;gpu_all;variant_uuid;launcher_slr;RMA"
    PROCESSORS 2
    WILL_FAIL FALSE
    FAIL_REGULAR_EXPRESSION "FAILED")

add_test(teamctxgetnbi_n2_w16_z256_1024B_uuid "env" "ROCSHMEM_SLR_NP=2" "ROCSHMEM_MAX_NUM_CONTEXTS=16" "ROCSHMEM_HEAP_SIZE=6442450944" "ROCSHMEM_TEST_UUID=1" "../../../../share/rocshmem/test_wrapper.sh" "teamctxgetnbi_n2_w16_z256_1024B_uuid" "../../../rocshmem_functional_tests" "-a" "35" "-w" "16" "-z" "256" "-s" "1024" "-localbuftype" "heap")
set_tests_properties(teamctxgetnbi_n2_w16_z256_1024B_uuid PROPERTIES
    TIMEOUT 300
    SKIP_RETURN_CODE 125
    LABELS "functional;backend_ipc;backend_gda;gpu_all;variant_uuid;launcher_slr;RMA"
    PROCESSORS 2
    WILL_FAIL FALSE
    FAIL_REGULAR_EXPRESSION "FAILED")

add_test(wggetnbi_n2_w1_z64_1048576B_uuid "env" "ROCSHMEM_SLR_NP=2" "ROCSHMEM_MAX_NUM_CONTEXTS=1" "ROCSHMEM_HEAP_SIZE=6442450944" "ROCSHMEM_TEST_UUID=1" "../../../../share/rocshmem/test_wrapper.sh" "wggetnbi_n2_w1_z64_1048576B_uuid" "../../../rocshmem_functional_tests" "-a" "25" "-w" "1" "-z" "64" "-s" "1048576" "-localbuftype" "heap")
set_tests_properties(wggetnbi_n2_w1_z64_1048576B_uuid PROPERTIES
    TIMEOUT 300
    SKIP_RETURN_CODE 125
    LABELS "functional;backend_ipc;backend_gda;gpu_all;variant_uuid;launcher_slr;RMA"
    PROCESSORS 2
    WILL_FAIL FALSE
    FAIL_REGULAR_EXPRESSION "FAILED")

add_test(wggetnbi_n2_w2_z64_1048576B_uuid "env" "ROCSHMEM_SLR_NP=2" "ROCSHMEM_MAX_NUM_CONTEXTS=2" "ROCSHMEM_HEAP_SIZE=6442450944" "ROCSHMEM_TEST_UUID=1" "../../../../share/rocshmem/test_wrapper.sh" "wggetnbi_n2_w2_z64_1048576B_uuid" "../../../rocshmem_functional_tests" "-a" "25" "-w" "2" "-z" "64" "-s" "1048576" "-localbuftype" "heap")
set_tests_properties(wggetnbi_n2_w2_z64_1048576B_uuid PROPERTIES
    TIMEOUT 300
    SKIP_RETURN_CODE 125
    LABELS "functional;backend_ipc;backend_gda;gpu_all;variant_uuid;launcher_slr;RMA"
    PROCESSORS 2
    WILL_FAIL FALSE
    FAIL_REGULAR_EXPRESSION "FAILED")

add_test(wggetnbi_n2_w16_z64_8B_uuid "env" "ROCSHMEM_SLR_NP=2" "ROCSHMEM_MAX_NUM_CONTEXTS=16" "ROCSHMEM_HEAP_SIZE=6442450944" "ROCSHMEM_TEST_UUID=1" "../../../../share/rocshmem/test_wrapper.sh" "wggetnbi_n2_w16_z64_8B_uuid" "../../../rocshmem_functional_tests" "-a" "25" "-w" "16" "-z" "64" "-s" "8" "-localbuftype" "heap")
set_tests_properties(wggetnbi_n2_w16_z64_8B_uuid PROPERTIES
    TIMEOUT 300
    SKIP_RETURN_CODE 125
    LABELS "functional;backend_ipc;backend_gda;gpu_all;variant_uuid;launcher_slr;RMA"
    PROCESSORS 2
    WILL_FAIL FALSE
    FAIL_REGULAR_EXPRESSION "FAILED")

add_test(wavegetnbi_n2_w1_z64_1048576B_uuid "env" "ROCSHMEM_SLR_NP=2" "ROCSHMEM_MAX_NUM_CONTEXTS=1" "ROCSHMEM_HEAP_SIZE=6442450944" "ROCSHMEM_TEST_UUID=1" "../../../../share/rocshmem/test_wrapper.sh" "wavegetnbi_n2_w1_z64_1048576B_uuid" "../../../rocshmem_functional_tests" "-a" "29" "-w" "1" "-z" "64" "-s" "1048576" "-localbuftype" "heap")
set_tests_properties(wavegetnbi_n2_w1_z64_1048576B_uuid PROPERTIES
    TIMEOUT 300
    SKIP_RETURN_CODE 125
    LABELS "functional;backend_ipc;backend_gda;gpu_all;variant_uuid;launcher_slr;RMA"
    PROCESSORS 2
    WILL_FAIL FALSE
    FAIL_REGULAR_EXPRESSION "FAILED")

add_test(wavegetnbi_n2_w2_z64_1048576B_uuid "env" "ROCSHMEM_SLR_NP=2" "ROCSHMEM_MAX_NUM_CONTEXTS=2" "ROCSHMEM_HEAP_SIZE=6442450944" "ROCSHMEM_TEST_UUID=1" "../../../../share/rocshmem/test_wrapper.sh" "wavegetnbi_n2_w2_z64_1048576B_uuid" "../../../rocshmem_functional_tests" "-a" "29" "-w" "2" "-z" "64" "-s" "1048576" "-localbuftype" "heap")
set_tests_properties(wavegetnbi_n2_w2_z64_1048576B_uuid PROPERTIES
    TIMEOUT 300
    SKIP_RETURN_CODE 125
    LABELS "functional;backend_ipc;backend_gda;gpu_all;variant_uuid;launcher_slr;RMA"
    PROCESSORS 2
    WILL_FAIL FALSE
    FAIL_REGULAR_EXPRESSION "FAILED")

add_test(wavegetnbi_n2_w2_z128_1048576B_uuid "env" "ROCSHMEM_SLR_NP=2" "ROCSHMEM_MAX_NUM_CONTEXTS=2" "ROCSHMEM_HEAP_SIZE=6442450944" "ROCSHMEM_TEST_UUID=1" "../../../../share/rocshmem/test_wrapper.sh" "wavegetnbi_n2_w2_z128_1048576B_uuid" "../../../rocshmem_functional_tests" "-a" "29" "-w" "2" "-z" "128" "-s" "1048576" "-localbuftype" "heap")
set_tests_properties(wavegetnbi_n2_w2_z128_1048576B_uuid PROPERTIES
    TIMEOUT 300
    SKIP_RETURN_CODE 125
    LABELS "functional;backend_ipc;backend_gda;gpu_all;variant_uuid;launcher_slr;RMA"
    PROCESSORS 2
    WILL_FAIL FALSE
    FAIL_REGULAR_EXPRESSION "FAILED")

add_test(wavegetnbi_n2_w16_z128_8B_uuid "env" "ROCSHMEM_SLR_NP=2" "ROCSHMEM_MAX_NUM_CONTEXTS=16" "ROCSHMEM_HEAP_SIZE=6442450944" "ROCSHMEM_TEST_UUID=1" "../../../../share/rocshmem/test_wrapper.sh" "wavegetnbi_n2_w16_z128_8B_uuid" "../../../rocshmem_functional_tests" "-a" "29" "-w" "16" "-z" "128" "-s" "8" "-localbuftype" "heap")
set_tests_properties(wavegetnbi_n2_w16_z128_8B_uuid PROPERTIES
    TIMEOUT 300
    SKIP_RETURN_CODE 125
    LABELS "functional;backend_ipc;backend_gda;gpu_all;variant_uuid;launcher_slr;RMA"
    PROCESSORS 2
    WILL_FAIL FALSE
    FAIL_REGULAR_EXPRESSION "FAILED")

add_test(amo_add_n2_w1_z1_uuid "env" "ROCSHMEM_SLR_NP=2" "ROCSHMEM_MAX_NUM_CONTEXTS=1" "ROCSHMEM_HEAP_SIZE=6442450944" "ROCSHMEM_TEST_UUID=1" "../../../../share/rocshmem/test_wrapper.sh" "amo_add_n2_w1_z1_uuid" "../../../rocshmem_functional_tests" "-a" "8" "-w" "1" "-z" "1" "-localbuftype" "heap")
set_tests_properties(amo_add_n2_w1_z1_uuid PROPERTIES
    TIMEOUT 300
    SKIP_RETURN_CODE 125
    LABELS "functional;backend_ipc;backend_gda;gpu_all;variant_uuid;launcher_slr;AMO"
    PROCESSORS 2
    WILL_FAIL FALSE
    FAIL_REGULAR_EXPRESSION "FAILED")

add_test(amo_add_n2_w1_z1024_uuid "env" "ROCSHMEM_SLR_NP=2" "ROCSHMEM_MAX_NUM_CONTEXTS=1" "ROCSHMEM_HEAP_SIZE=6442450944" "ROCSHMEM_TEST_UUID=1" "../../../../share/rocshmem/test_wrapper.sh" "amo_add_n2_w1_z1024_uuid" "../../../rocshmem_functional_tests" "-a" "8" "-w" "1" "-z" "1024" "-localbuftype" "heap")
set_tests_properties(amo_add_n2_w1_z1024_uuid PROPERTIES
    TIMEOUT 300
    SKIP_RETURN_CODE 125
    LABELS "functional;backend_ipc;backend_gda;gpu_all;variant_uuid;launcher_slr;AMO"
    PROCESSORS 2
    WILL_FAIL FALSE
    FAIL_REGULAR_EXPRESSION "FAILED")

add_test(amo_add_n2_w8_z1_uuid "env" "ROCSHMEM_SLR_NP=2" "ROCSHMEM_MAX_NUM_CONTEXTS=8" "ROCSHMEM_HEAP_SIZE=6442450944" "ROCSHMEM_TEST_UUID=1" "../../../../share/rocshmem/test_wrapper.sh" "amo_add_n2_w8_z1_uuid" "../../../rocshmem_functional_tests" "-a" "8" "-w" "8" "-z" "1" "-localbuftype" "heap")
set_tests_properties(amo_add_n2_w8_z1_uuid PROPERTIES
    TIMEOUT 300
    SKIP_RETURN_CODE 125
    LABELS "functional;backend_ipc;backend_gda;gpu_all;variant_uuid;launcher_slr;AMO"
    PROCESSORS 2
    WILL_FAIL FALSE
    FAIL_REGULAR_EXPRESSION "FAILED")

add_test(amo_add_n2_w32_z128_uuid "env" "ROCSHMEM_SLR_NP=2" "ROCSHMEM_MAX_NUM_CONTEXTS=32" "ROCSHMEM_HEAP_SIZE=6442450944" "ROCSHMEM_TEST_UUID=1" "../../../../share/rocshmem/test_wrapper.sh" "amo_add_n2_w32_z128_uuid" "../../../rocshmem_functional_tests" "-a" "8" "-w" "32" "-z" "128" "-localbuftype" "heap")
set_tests_properties(amo_add_n2_w32_z128_uuid PROPERTIES
    TIMEOUT 300
    SKIP_RETURN_CODE 125
    LABELS "functional;backend_ipc;backend_gda;gpu_all;variant_uuid;launcher_slr;AMO"
    PROCESSORS 2
    WILL_FAIL FALSE
    FAIL_REGULAR_EXPRESSION "FAILED")

add_test(amo_fadd_n2_w1_z1_uuid "env" "ROCSHMEM_SLR_NP=2" "ROCSHMEM_MAX_NUM_CONTEXTS=1" "ROCSHMEM_HEAP_SIZE=6442450944" "ROCSHMEM_TEST_UUID=1" "../../../../share/rocshmem/test_wrapper.sh" "amo_fadd_n2_w1_z1_uuid" "../../../rocshmem_functional_tests" "-a" "4" "-w" "1" "-z" "1" "-localbuftype" "heap")
set_tests_properties(amo_fadd_n2_w1_z1_uuid PROPERTIES
    TIMEOUT 300
    SKIP_RETURN_CODE 125
    LABELS "functional;backend_ipc;backend_gda;gpu_all;variant_uuid;launcher_slr;AMO"
    PROCESSORS 2
    WILL_FAIL FALSE
    FAIL_REGULAR_EXPRESSION "FAILED")

add_test(amo_fadd_n2_w1_z1024_uuid "env" "ROCSHMEM_SLR_NP=2" "ROCSHMEM_MAX_NUM_CONTEXTS=1" "ROCSHMEM_HEAP_SIZE=6442450944" "ROCSHMEM_TEST_UUID=1" "../../../../share/rocshmem/test_wrapper.sh" "amo_fadd_n2_w1_z1024_uuid" "../../../rocshmem_functional_tests" "-a" "4" "-w" "1" "-z" "1024" "-localbuftype" "heap")
set_tests_properties(amo_fadd_n2_w1_z1024_uuid PROPERTIES
    TIMEOUT 300
    SKIP_RETURN_CODE 125
    LABELS "functional;backend_ipc;backend_gda;gpu_all;variant_uuid;launcher_slr;AMO"
    PROCESSORS 2
    WILL_FAIL FALSE
    FAIL_REGULAR_EXPRESSION "FAILED")

add_test(amo_fadd_n2_w8_z1_uuid "env" "ROCSHMEM_SLR_NP=2" "ROCSHMEM_MAX_NUM_CONTEXTS=8" "ROCSHMEM_HEAP_SIZE=6442450944" "ROCSHMEM_TEST_UUID=1" "../../../../share/rocshmem/test_wrapper.sh" "amo_fadd_n2_w8_z1_uuid" "../../../rocshmem_functional_tests" "-a" "4" "-w" "8" "-z" "1" "-localbuftype" "heap")
set_tests_properties(amo_fadd_n2_w8_z1_uuid PROPERTIES
    TIMEOUT 300
    SKIP_RETURN_CODE 125
    LABELS "functional;backend_ipc;backend_gda;gpu_all;variant_uuid;launcher_slr;AMO"
    PROCESSORS 2
    WILL_FAIL FALSE
    FAIL_REGULAR_EXPRESSION "FAILED")

add_test(amo_fadd_n2_w32_z128_uuid "env" "ROCSHMEM_SLR_NP=2" "ROCSHMEM_MAX_NUM_CONTEXTS=32" "ROCSHMEM_HEAP_SIZE=6442450944" "ROCSHMEM_TEST_UUID=1" "../../../../share/rocshmem/test_wrapper.sh" "amo_fadd_n2_w32_z128_uuid" "../../../rocshmem_functional_tests" "-a" "4" "-w" "32" "-z" "128" "-localbuftype" "heap")
set_tests_properties(amo_fadd_n2_w32_z128_uuid PROPERTIES
    TIMEOUT 300
    SKIP_RETURN_CODE 125
    LABELS "functional;backend_ipc;backend_gda;gpu_all;variant_uuid;launcher_slr;AMO"
    PROCESSORS 2
    WILL_FAIL FALSE
    FAIL_REGULAR_EXPRESSION "FAILED")

add_test(amo_inc_n2_w1_z1_uuid "env" "ROCSHMEM_SLR_NP=2" "ROCSHMEM_MAX_NUM_CONTEXTS=1" "ROCSHMEM_HEAP_SIZE=6442450944" "ROCSHMEM_TEST_UUID=1" "../../../../share/rocshmem/test_wrapper.sh" "amo_inc_n2_w1_z1_uuid" "../../../rocshmem_functional_tests" "-a" "9" "-w" "1" "-z" "1" "-localbuftype" "heap")
set_tests_properties(amo_inc_n2_w1_z1_uuid PROPERTIES
    TIMEOUT 300
    SKIP_RETURN_CODE 125
    LABELS "functional;backend_ipc;backend_gda;gpu_all;variant_uuid;launcher_slr;AMO"
    PROCESSORS 2
    WILL_FAIL FALSE
    FAIL_REGULAR_EXPRESSION "FAILED")

add_test(amo_inc_n2_w1_z1024_uuid "env" "ROCSHMEM_SLR_NP=2" "ROCSHMEM_MAX_NUM_CONTEXTS=1" "ROCSHMEM_HEAP_SIZE=6442450944" "ROCSHMEM_TEST_UUID=1" "../../../../share/rocshmem/test_wrapper.sh" "amo_inc_n2_w1_z1024_uuid" "../../../rocshmem_functional_tests" "-a" "9" "-w" "1" "-z" "1024" "-localbuftype" "heap")
set_tests_properties(amo_inc_n2_w1_z1024_uuid PROPERTIES
    TIMEOUT 300
    SKIP_RETURN_CODE 125
    LABELS "functional;backend_ipc;backend_gda;gpu_all;variant_uuid;launcher_slr;AMO"
    PROCESSORS 2
    WILL_FAIL FALSE
    FAIL_REGULAR_EXPRESSION "FAILED")

add_test(amo_inc_n2_w8_z1_uuid "env" "ROCSHMEM_SLR_NP=2" "ROCSHMEM_MAX_NUM_CONTEXTS=8" "ROCSHMEM_HEAP_SIZE=6442450944" "ROCSHMEM_TEST_UUID=1" "../../../../share/rocshmem/test_wrapper.sh" "amo_inc_n2_w8_z1_uuid" "../../../rocshmem_functional_tests" "-a" "9" "-w" "8" "-z" "1" "-localbuftype" "heap")
set_tests_properties(amo_inc_n2_w8_z1_uuid PROPERTIES
    TIMEOUT 300
    SKIP_RETURN_CODE 125
    LABELS "functional;backend_ipc;backend_gda;gpu_all;variant_uuid;launcher_slr;AMO"
    PROCESSORS 2
    WILL_FAIL FALSE
    FAIL_REGULAR_EXPRESSION "FAILED")

add_test(amo_inc_n2_w32_z128_uuid "env" "ROCSHMEM_SLR_NP=2" "ROCSHMEM_MAX_NUM_CONTEXTS=32" "ROCSHMEM_HEAP_SIZE=6442450944" "ROCSHMEM_TEST_UUID=1" "../../../../share/rocshmem/test_wrapper.sh" "amo_inc_n2_w32_z128_uuid" "../../../rocshmem_functional_tests" "-a" "9" "-w" "32" "-z" "128" "-localbuftype" "heap")
set_tests_properties(amo_inc_n2_w32_z128_uuid PROPERTIES
    TIMEOUT 300
    SKIP_RETURN_CODE 125
    LABELS "functional;backend_ipc;backend_gda;gpu_all;variant_uuid;launcher_slr;AMO"
    PROCESSORS 2
    WILL_FAIL FALSE
    FAIL_REGULAR_EXPRESSION "FAILED")

add_test(amo_finc_n2_w1_z1_uuid "env" "ROCSHMEM_SLR_NP=2" "ROCSHMEM_MAX_NUM_CONTEXTS=1" "ROCSHMEM_HEAP_SIZE=6442450944" "ROCSHMEM_TEST_UUID=1" "../../../../share/rocshmem/test_wrapper.sh" "amo_finc_n2_w1_z1_uuid" "../../../rocshmem_functional_tests" "-a" "5" "-w" "1" "-z" "1" "-localbuftype" "heap")
set_tests_properties(amo_finc_n2_w1_z1_uuid PROPERTIES
    TIMEOUT 300
    SKIP_RETURN_CODE 125
    LABELS "functional;backend_ipc;backend_gda;gpu_all;variant_uuid;launcher_slr;AMO"
    PROCESSORS 2
    WILL_FAIL FALSE
    FAIL_REGULAR_EXPRESSION "FAILED")

add_test(amo_finc_n2_w1_z1024_uuid "env" "ROCSHMEM_SLR_NP=2" "ROCSHMEM_MAX_NUM_CONTEXTS=1" "ROCSHMEM_HEAP_SIZE=6442450944" "ROCSHMEM_TEST_UUID=1" "../../../../share/rocshmem/test_wrapper.sh" "amo_finc_n2_w1_z1024_uuid" "../../../rocshmem_functional_tests" "-a" "5" "-w" "1" "-z" "1024" "-localbuftype" "heap")
set_tests_properties(amo_finc_n2_w1_z1024_uuid PROPERTIES
    TIMEOUT 300
    SKIP_RETURN_CODE 125
    LABELS "functional;backend_ipc;backend_gda;gpu_all;variant_uuid;launcher_slr;AMO"
    PROCESSORS 2
    WILL_FAIL FALSE
    FAIL_REGULAR_EXPRESSION "FAILED")

add_test(amo_finc_n2_w8_z1_uuid "env" "ROCSHMEM_SLR_NP=2" "ROCSHMEM_MAX_NUM_CONTEXTS=8" "ROCSHMEM_HEAP_SIZE=6442450944" "ROCSHMEM_TEST_UUID=1" "../../../../share/rocshmem/test_wrapper.sh" "amo_finc_n2_w8_z1_uuid" "../../../rocshmem_functional_tests" "-a" "5" "-w" "8" "-z" "1" "-localbuftype" "heap")
set_tests_properties(amo_finc_n2_w8_z1_uuid PROPERTIES
    TIMEOUT 300
    SKIP_RETURN_CODE 125
    LABELS "functional;backend_ipc;backend_gda;gpu_all;variant_uuid;launcher_slr;AMO"
    PROCESSORS 2
    WILL_FAIL FALSE
    FAIL_REGULAR_EXPRESSION "FAILED")

add_test(amo_finc_n2_w32_z128_uuid "env" "ROCSHMEM_SLR_NP=2" "ROCSHMEM_MAX_NUM_CONTEXTS=32" "ROCSHMEM_HEAP_SIZE=6442450944" "ROCSHMEM_TEST_UUID=1" "../../../../share/rocshmem/test_wrapper.sh" "amo_finc_n2_w32_z128_uuid" "../../../rocshmem_functional_tests" "-a" "5" "-w" "32" "-z" "128" "-localbuftype" "heap")
set_tests_properties(amo_finc_n2_w32_z128_uuid PROPERTIES
    TIMEOUT 300
    SKIP_RETURN_CODE 125
    LABELS "functional;backend_ipc;backend_gda;gpu_all;variant_uuid;launcher_slr;AMO"
    PROCESSORS 2
    WILL_FAIL FALSE
    FAIL_REGULAR_EXPRESSION "FAILED")

add_test(amo_set_n2_w1_z1_uuid "env" "ROCSHMEM_SLR_NP=2" "ROCSHMEM_MAX_NUM_CONTEXTS=1" "ROCSHMEM_HEAP_SIZE=6442450944" "ROCSHMEM_TEST_UUID=1" "../../../../share/rocshmem/test_wrapper.sh" "amo_set_n2_w1_z1_uuid" "../../../rocshmem_functional_tests" "-a" "40" "-w" "1" "-z" "1" "-localbuftype" "heap")
set_tests_properties(amo_set_n2_w1_z1_uuid PROPERTIES
    TIMEOUT 300
    SKIP_RETURN_CODE 125
    LABELS "functional;backend_all;gpu_all;variant_uuid;launcher_slr;AMO"
    PROCESSORS 2
    WILL_FAIL FALSE
    FAIL_REGULAR_EXPRESSION "FAILED")

add_test(amo_set_n2_w8_z1_uuid "env" "ROCSHMEM_SLR_NP=2" "ROCSHMEM_MAX_NUM_CONTEXTS=8" "ROCSHMEM_HEAP_SIZE=6442450944" "ROCSHMEM_TEST_UUID=1" "../../../../share/rocshmem/test_wrapper.sh" "amo_set_n2_w8_z1_uuid" "../../../rocshmem_functional_tests" "-a" "40" "-w" "8" "-z" "1" "-localbuftype" "heap")
set_tests_properties(amo_set_n2_w8_z1_uuid PROPERTIES
    TIMEOUT 300
    SKIP_RETURN_CODE 125
    LABELS "functional;backend_all;gpu_all;variant_uuid;launcher_slr;AMO"
    PROCESSORS 2
    WILL_FAIL FALSE
    FAIL_REGULAR_EXPRESSION "FAILED")

add_test(amo_set_n2_w32_z1_uuid "env" "ROCSHMEM_SLR_NP=2" "ROCSHMEM_MAX_NUM_CONTEXTS=32" "ROCSHMEM_HEAP_SIZE=6442450944" "ROCSHMEM_TEST_UUID=1" "../../../../share/rocshmem/test_wrapper.sh" "amo_set_n2_w32_z1_uuid" "../../../rocshmem_functional_tests" "-a" "40" "-w" "32" "-z" "1" "-localbuftype" "heap")
set_tests_properties(amo_set_n2_w32_z1_uuid PROPERTIES
    TIMEOUT 300
    SKIP_RETURN_CODE 125
    LABELS "functional;backend_all;gpu_all;variant_uuid;launcher_slr;AMO"
    PROCESSORS 2
    WILL_FAIL FALSE
    FAIL_REGULAR_EXPRESSION "FAILED")

add_test(amo_fetch_n2_w1_z1_uuid "env" "ROCSHMEM_SLR_NP=2" "ROCSHMEM_MAX_NUM_CONTEXTS=1" "ROCSHMEM_HEAP_SIZE=6442450944" "ROCSHMEM_TEST_UUID=1" "../../../../share/rocshmem/test_wrapper.sh" "amo_fetch_n2_w1_z1_uuid" "../../../rocshmem_functional_tests" "-a" "6" "-w" "1" "-z" "1" "-localbuftype" "heap")
set_tests_properties(amo_fetch_n2_w1_z1_uuid PROPERTIES
    TIMEOUT 300
    SKIP_RETURN_CODE 125
    LABELS "functional;backend_all;gpu_all;variant_uuid;launcher_slr;AMO"
    PROCESSORS 2
    WILL_FAIL FALSE
    FAIL_REGULAR_EXPRESSION "FAILED")

add_test(amo_fetch_n2_w1_z1024_uuid "env" "ROCSHMEM_SLR_NP=2" "ROCSHMEM_MAX_NUM_CONTEXTS=1" "ROCSHMEM_HEAP_SIZE=6442450944" "ROCSHMEM_TEST_UUID=1" "../../../../share/rocshmem/test_wrapper.sh" "amo_fetch_n2_w1_z1024_uuid" "../../../rocshmem_functional_tests" "-a" "6" "-w" "1" "-z" "1024" "-localbuftype" "heap")
set_tests_properties(amo_fetch_n2_w1_z1024_uuid PROPERTIES
    TIMEOUT 300
    SKIP_RETURN_CODE 125
    LABELS "functional;backend_all;gpu_all;variant_uuid;launcher_slr;AMO"
    PROCESSORS 2
    WILL_FAIL FALSE
    FAIL_REGULAR_EXPRESSION "FAILED")

add_test(amo_fetch_n2_w8_z1_uuid "env" "ROCSHMEM_SLR_NP=2" "ROCSHMEM_MAX_NUM_CONTEXTS=8" "ROCSHMEM_HEAP_SIZE=6442450944" "ROCSHMEM_TEST_UUID=1" "../../../../share/rocshmem/test_wrapper.sh" "amo_fetch_n2_w8_z1_uuid" "../../../rocshmem_functional_tests" "-a" "6" "-w" "8" "-z" "1" "-localbuftype" "heap")
set_tests_properties(amo_fetch_n2_w8_z1_uuid PROPERTIES
    TIMEOUT 300
    SKIP_RETURN_CODE 125
    LABELS "functional;backend_all;gpu_all;variant_uuid;launcher_slr;AMO"
    PROCESSORS 2
    WILL_FAIL FALSE
    FAIL_REGULAR_EXPRESSION "FAILED")

add_test(amo_fetch_n2_w32_z128_uuid "env" "ROCSHMEM_SLR_NP=2" "ROCSHMEM_MAX_NUM_CONTEXTS=32" "ROCSHMEM_HEAP_SIZE=6442450944" "ROCSHMEM_TEST_UUID=1" "../../../../share/rocshmem/test_wrapper.sh" "amo_fetch_n2_w32_z128_uuid" "../../../rocshmem_functional_tests" "-a" "6" "-w" "32" "-z" "128" "-localbuftype" "heap")
set_tests_properties(amo_fetch_n2_w32_z128_uuid PROPERTIES
    TIMEOUT 300
    SKIP_RETURN_CODE 125
    LABELS "functional;backend_all;gpu_all;variant_uuid;launcher_slr;AMO"
    PROCESSORS 2
    WILL_FAIL FALSE
    FAIL_REGULAR_EXPRESSION "FAILED")

add_test(amo_fcswap_n2_w1_z1_uuid "env" "ROCSHMEM_SLR_NP=2" "ROCSHMEM_MAX_NUM_CONTEXTS=1" "ROCSHMEM_HEAP_SIZE=6442450944" "ROCSHMEM_TEST_UUID=1" "../../../../share/rocshmem/test_wrapper.sh" "amo_fcswap_n2_w1_z1_uuid" "../../../rocshmem_functional_tests" "-a" "7" "-w" "1" "-z" "1" "-localbuftype" "heap")
set_tests_properties(amo_fcswap_n2_w1_z1_uuid PROPERTIES
    TIMEOUT 300
    SKIP_RETURN_CODE 125
    LABELS "functional;backend_all;gpu_all;variant_uuid;launcher_slr;AMO"
    PROCESSORS 2
    WILL_FAIL FALSE
    FAIL_REGULAR_EXPRESSION "FAILED")

add_test(amo_fcswap_n2_w32_z1_uuid "env" "ROCSHMEM_SLR_NP=2" "ROCSHMEM_MAX_NUM_CONTEXTS=32" "ROCSHMEM_HEAP_SIZE=6442450944" "ROCSHMEM_TEST_UUID=1" "../../../../share/rocshmem/test_wrapper.sh" "amo_fcswap_n2_w32_z1_uuid" "../../../rocshmem_functional_tests" "-a" "7" "-w" "32" "-z" "1" "-localbuftype" "heap")
set_tests_properties(amo_fcswap_n2_w32_z1_uuid PROPERTIES
    TIMEOUT 300
    SKIP_RETURN_CODE 125
    LABELS "functional;backend_all;gpu_all;variant_uuid;launcher_slr;AMO"
    PROCESSORS 2
    WILL_FAIL FALSE
    FAIL_REGULAR_EXPRESSION "FAILED")

add_test(amo_fcswap_n2_w8_z1_uuid "env" "ROCSHMEM_SLR_NP=2" "ROCSHMEM_MAX_NUM_CONTEXTS=8" "ROCSHMEM_HEAP_SIZE=6442450944" "ROCSHMEM_TEST_UUID=1" "../../../../share/rocshmem/test_wrapper.sh" "amo_fcswap_n2_w8_z1_uuid" "../../../rocshmem_functional_tests" "-a" "7" "-w" "8" "-z" "1" "-localbuftype" "heap")
set_tests_properties(amo_fcswap_n2_w8_z1_uuid PROPERTIES
    TIMEOUT 300
    SKIP_RETURN_CODE 125
    LABELS "functional;backend_all;gpu_all;variant_uuid;launcher_slr;AMO"
    PROCESSORS 2
    WILL_FAIL FALSE
    FAIL_REGULAR_EXPRESSION "FAILED")

add_test(amo_and_n2_w1_z1_uuid "env" "ROCSHMEM_SLR_NP=2" "ROCSHMEM_MAX_NUM_CONTEXTS=1" "ROCSHMEM_HEAP_SIZE=6442450944" "ROCSHMEM_TEST_UUID=1" "../../../../share/rocshmem/test_wrapper.sh" "amo_and_n2_w1_z1_uuid" "../../../rocshmem_functional_tests" "-a" "45" "-w" "1" "-z" "1" "-localbuftype" "heap")
set_tests_properties(amo_and_n2_w1_z1_uuid PROPERTIES
    TIMEOUT 300
    SKIP_RETURN_CODE 125
    LABELS "functional;backend_all;gpu_all;variant_uuid;launcher_slr;AMO"
    PROCESSORS 2
    WILL_FAIL FALSE
    FAIL_REGULAR_EXPRESSION "FAILED")

add_test(amo_fetchand_n2_w1_z1_uuid "env" "ROCSHMEM_SLR_NP=2" "ROCSHMEM_MAX_NUM_CONTEXTS=1" "ROCSHMEM_HEAP_SIZE=6442450944" "ROCSHMEM_TEST_UUID=1" "../../../../share/rocshmem/test_wrapper.sh" "amo_fetchand_n2_w1_z1_uuid" "../../../rocshmem_functional_tests" "-a" "42" "-w" "1" "-z" "1" "-localbuftype" "heap")
set_tests_properties(amo_fetchand_n2_w1_z1_uuid PROPERTIES
    TIMEOUT 300
    SKIP_RETURN_CODE 125
    LABELS "functional;backend_all;gpu_all;variant_uuid;launcher_slr;AMO"
    PROCESSORS 2
    WILL_FAIL FALSE
    FAIL_REGULAR_EXPRESSION "FAILED")

add_test(amo_xor_n2_w1_z1_uuid "env" "ROCSHMEM_SLR_NP=2" "ROCSHMEM_MAX_NUM_CONTEXTS=1" "ROCSHMEM_HEAP_SIZE=6442450944" "ROCSHMEM_TEST_UUID=1" "../../../../share/rocshmem/test_wrapper.sh" "amo_xor_n2_w1_z1_uuid" "../../../rocshmem_functional_tests" "-a" "47" "-w" "1" "-z" "1" "-localbuftype" "heap")
set_tests_properties(amo_xor_n2_w1_z1_uuid PROPERTIES
    TIMEOUT 300
    SKIP_RETURN_CODE 125
    LABELS "functional;backend_all;gpu_all;variant_uuid;launcher_slr;AMO"
    PROCESSORS 2
    WILL_FAIL FALSE
    FAIL_REGULAR_EXPRESSION "FAILED")

add_test(putsignal_n2_w1_z1_1048576B_uuid "env" "ROCSHMEM_SLR_NP=2" "ROCSHMEM_MAX_NUM_CONTEXTS=1" "ROCSHMEM_HEAP_SIZE=6442450944" "ROCSHMEM_TEST_UUID=1" "../../../../share/rocshmem/test_wrapper.sh" "putsignal_n2_w1_z1_1048576B_uuid" "../../../rocshmem_functional_tests" "-a" "49" "-w" "1" "-z" "1" "-s" "1048576" "-localbuftype" "heap")
set_tests_properties(putsignal_n2_w1_z1_1048576B_uuid PROPERTIES
    TIMEOUT 300
    SKIP_RETURN_CODE 125
    LABELS "functional;backend_all;gpu_all;variant_uuid;launcher_slr;SIGOPS"
    PROCESSORS 2
    WILL_FAIL FALSE
    FAIL_REGULAR_EXPRESSION "FAILED")

add_test(putsignal_n2_w2_z32_1048576B_uuid "env" "ROCSHMEM_SLR_NP=2" "ROCSHMEM_MAX_NUM_CONTEXTS=2" "ROCSHMEM_HEAP_SIZE=6442450944" "ROCSHMEM_TEST_UUID=1" "../../../../share/rocshmem/test_wrapper.sh" "putsignal_n2_w2_z32_1048576B_uuid" "../../../rocshmem_functional_tests" "-a" "49" "-w" "2" "-z" "32" "-s" "1048576" "-localbuftype" "heap")
set_tests_properties(putsignal_n2_w2_z32_1048576B_uuid PROPERTIES
    TIMEOUT 300
    SKIP_RETURN_CODE 125
    LABELS "functional;backend_all;gpu_all;variant_uuid;launcher_slr;SIGOPS"
    PROCESSORS 2
    WILL_FAIL FALSE
    FAIL_REGULAR_EXPRESSION "FAILED")

add_test(wgputsignal_n2_w2_z32_1048576B_uuid "env" "ROCSHMEM_SLR_NP=2" "ROCSHMEM_MAX_NUM_CONTEXTS=2" "ROCSHMEM_HEAP_SIZE=6442450944" "ROCSHMEM_TEST_UUID=1" "../../../../share/rocshmem/test_wrapper.sh" "wgputsignal_n2_w2_z32_1048576B_uuid" "../../../rocshmem_functional_tests" "-a" "50" "-w" "2" "-z" "32" "-s" "1048576" "-localbuftype" "heap")
set_tests_properties(wgputsignal_n2_w2_z32_1048576B_uuid PROPERTIES
    TIMEOUT 300
    SKIP_RETURN_CODE 125
    LABELS "functional;backend_all;gpu_all;variant_uuid;launcher_slr;SIGOPS"
    PROCESSORS 2
    WILL_FAIL FALSE
    FAIL_REGULAR_EXPRESSION "FAILED")

add_test(waveputsignal_n2_w1_z32_1048576B_uuid "env" "ROCSHMEM_SLR_NP=2" "ROCSHMEM_MAX_NUM_CONTEXTS=1" "ROCSHMEM_HEAP_SIZE=6442450944" "ROCSHMEM_TEST_UUID=1" "../../../../share/rocshmem/test_wrapper.sh" "waveputsignal_n2_w1_z32_1048576B_uuid" "../../../rocshmem_functional_tests" "-a" "51" "-w" "1" "-z" "32" "-s" "1048576" "-localbuftype" "heap")
set_tests_properties(waveputsignal_n2_w1_z32_1048576B_uuid PROPERTIES
    TIMEOUT 300
    SKIP_RETURN_CODE 125
    LABELS "functional;backend_all;gpu_all;variant_uuid;launcher_slr;SIGOPS"
    PROCESSORS 2
    WILL_FAIL FALSE
    FAIL_REGULAR_EXPRESSION "FAILED")

add_test(waveputsignal_n2_w2_z64_1048576B_uuid "env" "ROCSHMEM_SLR_NP=2" "ROCSHMEM_MAX_NUM_CONTEXTS=2" "ROCSHMEM_HEAP_SIZE=6442450944" "ROCSHMEM_TEST_UUID=1" "../../../../share/rocshmem/test_wrapper.sh" "waveputsignal_n2_w2_z64_1048576B_uuid" "../../../rocshmem_functional_tests" "-a" "51" "-w" "2" "-z" "64" "-s" "1048576" "-localbuftype" "heap")
set_tests_properties(waveputsignal_n2_w2_z64_1048576B_uuid PROPERTIES
    TIMEOUT 300
    SKIP_RETURN_CODE 125
    LABELS "functional;backend_all;gpu_all;variant_uuid;launcher_slr;SIGOPS"
    PROCESSORS 2
    WILL_FAIL FALSE
    FAIL_REGULAR_EXPRESSION "FAILED")

add_test(putsignalnbi_n2_w1_z1_1048576B_uuid "env" "ROCSHMEM_SLR_NP=2" "ROCSHMEM_MAX_NUM_CONTEXTS=1" "ROCSHMEM_HEAP_SIZE=6442450944" "ROCSHMEM_TEST_UUID=1" "../../../../share/rocshmem/test_wrapper.sh" "putsignalnbi_n2_w1_z1_1048576B_uuid" "../../../rocshmem_functional_tests" "-a" "52" "-w" "1" "-z" "1" "-s" "1048576" "-localbuftype" "heap")
set_tests_properties(putsignalnbi_n2_w1_z1_1048576B_uuid PROPERTIES
    TIMEOUT 300
    SKIP_RETURN_CODE 125
    LABELS "functional;backend_all;gpu_all;variant_uuid;launcher_slr;SIGOPS"
    PROCESSORS 2
    WILL_FAIL FALSE
    FAIL_REGULAR_EXPRESSION "FAILED")

add_test(putsignalnbi_n2_w2_z32_1048576B_uuid "env" "ROCSHMEM_SLR_NP=2" "ROCSHMEM_MAX_NUM_CONTEXTS=2" "ROCSHMEM_HEAP_SIZE=6442450944" "ROCSHMEM_TEST_UUID=1" "../../../../share/rocshmem/test_wrapper.sh" "putsignalnbi_n2_w2_z32_1048576B_uuid" "../../../rocshmem_functional_tests" "-a" "52" "-w" "2" "-z" "32" "-s" "1048576" "-localbuftype" "heap")
set_tests_properties(putsignalnbi_n2_w2_z32_1048576B_uuid PROPERTIES
    TIMEOUT 300
    SKIP_RETURN_CODE 125
    LABELS "functional;backend_all;gpu_all;variant_uuid;launcher_slr;SIGOPS"
    PROCESSORS 2
    WILL_FAIL FALSE
    FAIL_REGULAR_EXPRESSION "FAILED")

add_test(wgputsignalnbi_n2_w2_z32_1048576B_uuid "env" "ROCSHMEM_SLR_NP=2" "ROCSHMEM_MAX_NUM_CONTEXTS=2" "ROCSHMEM_HEAP_SIZE=6442450944" "ROCSHMEM_TEST_UUID=1" "../../../../share/rocshmem/test_wrapper.sh" "wgputsignalnbi_n2_w2_z32_1048576B_uuid" "../../../rocshmem_functional_tests" "-a" "53" "-w" "2" "-z" "32" "-s" "1048576" "-localbuftype" "heap")
set_tests_properties(wgputsignalnbi_n2_w2_z32_1048576B_uuid PROPERTIES
    TIMEOUT 300
    SKIP_RETURN_CODE 125
    LABELS "functional;backend_all;gpu_all;variant_uuid;launcher_slr;SIGOPS"
    PROCESSORS 2
    WILL_FAIL FALSE
    FAIL_REGULAR_EXPRESSION "FAILED")

add_test(waveputsignalnbi_n2_w1_z32_1048576B_uuid "env" "ROCSHMEM_SLR_NP=2" "ROCSHMEM_MAX_NUM_CONTEXTS=1" "ROCSHMEM_HEAP_SIZE=6442450944" "ROCSHMEM_TEST_UUID=1" "../../../../share/rocshmem/test_wrapper.sh" "waveputsignalnbi_n2_w1_z32_1048576B_uuid" "../../../rocshmem_functional_tests" "-a" "54" "-w" "1" "-z" "32" "-s" "1048576" "-localbuftype" "heap")
set_tests_properties(waveputsignalnbi_n2_w1_z32_1048576B_uuid PROPERTIES
    TIMEOUT 300
    SKIP_RETURN_CODE 125
    LABELS "functional;backend_all;gpu_all;variant_uuid;launcher_slr;SIGOPS"
    PROCESSORS 2
    WILL_FAIL FALSE
    FAIL_REGULAR_EXPRESSION "FAILED")

add_test(waveputsignalnbi_n2_w2_z64_1048576B_uuid "env" "ROCSHMEM_SLR_NP=2" "ROCSHMEM_MAX_NUM_CONTEXTS=2" "ROCSHMEM_HEAP_SIZE=6442450944" "ROCSHMEM_TEST_UUID=1" "../../../../share/rocshmem/test_wrapper.sh" "waveputsignalnbi_n2_w2_z64_1048576B_uuid" "../../../rocshmem_functional_tests" "-a" "54" "-w" "2" "-z" "64" "-s" "1048576" "-localbuftype" "heap")
set_tests_properties(waveputsignalnbi_n2_w2_z64_1048576B_uuid PROPERTIES
    TIMEOUT 300
    SKIP_RETURN_CODE 125
    LABELS "functional;backend_all;gpu_all;variant_uuid;launcher_slr;SIGOPS"
    PROCESSORS 2
    WILL_FAIL FALSE
    FAIL_REGULAR_EXPRESSION "FAILED")

add_test(signalfetch_n2_w1_z1_uuid "env" "ROCSHMEM_SLR_NP=2" "ROCSHMEM_MAX_NUM_CONTEXTS=1" "ROCSHMEM_HEAP_SIZE=6442450944" "ROCSHMEM_TEST_UUID=1" "../../../../share/rocshmem/test_wrapper.sh" "signalfetch_n2_w1_z1_uuid" "../../../rocshmem_functional_tests" "-a" "55" "-w" "1" "-z" "1" "-localbuftype" "heap")
set_tests_properties(signalfetch_n2_w1_z1_uuid PROPERTIES
    TIMEOUT 300
    SKIP_RETURN_CODE 125
    LABELS "functional;backend_all;gpu_all;variant_uuid;launcher_slr;SIGOPS"
    PROCESSORS 2
    WILL_FAIL FALSE
    FAIL_REGULAR_EXPRESSION "FAILED")

add_test(wgsignalfetch_n2_w2_z32_uuid "env" "ROCSHMEM_SLR_NP=2" "ROCSHMEM_MAX_NUM_CONTEXTS=2" "ROCSHMEM_HEAP_SIZE=6442450944" "ROCSHMEM_TEST_UUID=1" "../../../../share/rocshmem/test_wrapper.sh" "wgsignalfetch_n2_w2_z32_uuid" "../../../rocshmem_functional_tests" "-a" "56" "-w" "2" "-z" "32" "-localbuftype" "heap")
set_tests_properties(wgsignalfetch_n2_w2_z32_uuid PROPERTIES
    TIMEOUT 300
    SKIP_RETURN_CODE 125
    LABELS "functional;backend_all;gpu_all;variant_uuid;launcher_slr;SIGOPS"
    PROCESSORS 2
    WILL_FAIL FALSE
    FAIL_REGULAR_EXPRESSION "FAILED")

add_test(wavesignalfetch_n2_w1_z32_uuid "env" "ROCSHMEM_SLR_NP=2" "ROCSHMEM_MAX_NUM_CONTEXTS=1" "ROCSHMEM_HEAP_SIZE=6442450944" "ROCSHMEM_TEST_UUID=1" "../../../../share/rocshmem/test_wrapper.sh" "wavesignalfetch_n2_w1_z32_uuid" "../../../rocshmem_functional_tests" "-a" "57" "-w" "1" "-z" "32" "-localbuftype" "heap")
set_tests_properties(wavesignalfetch_n2_w1_z32_uuid PROPERTIES
    TIMEOUT 300
    SKIP_RETURN_CODE 125
    LABELS "functional;backend_all;gpu_all;variant_uuid;launcher_slr;SIGOPS"
    PROCESSORS 2
    WILL_FAIL FALSE
    FAIL_REGULAR_EXPRESSION "FAILED")

add_test(wavesignalfetch_n2_w1_z64_uuid "env" "ROCSHMEM_SLR_NP=2" "ROCSHMEM_MAX_NUM_CONTEXTS=1" "ROCSHMEM_HEAP_SIZE=6442450944" "ROCSHMEM_TEST_UUID=1" "../../../../share/rocshmem/test_wrapper.sh" "wavesignalfetch_n2_w1_z64_uuid" "../../../rocshmem_functional_tests" "-a" "57" "-w" "1" "-z" "64" "-localbuftype" "heap")
set_tests_properties(wavesignalfetch_n2_w1_z64_uuid PROPERTIES
    TIMEOUT 300
    SKIP_RETURN_CODE 125
    LABELS "functional;backend_all;gpu_all;variant_uuid;launcher_slr;SIGOPS"
    PROCESSORS 2
    WILL_FAIL FALSE
    FAIL_REGULAR_EXPRESSION "FAILED")

add_test(signaladd_n2_w2_z32_uuid "env" "ROCSHMEM_SLR_NP=2" "ROCSHMEM_MAX_NUM_CONTEXTS=2" "ROCSHMEM_HEAP_SIZE=6442450944" "ROCSHMEM_TEST_UUID=1" "../../../../share/rocshmem/test_wrapper.sh" "signaladd_n2_w2_z32_uuid" "../../../rocshmem_functional_tests" "-a" "171" "-w" "2" "-z" "32" "-localbuftype" "heap")
set_tests_properties(signaladd_n2_w2_z32_uuid PROPERTIES
    TIMEOUT 300
    SKIP_RETURN_CODE 125
    LABELS "functional;backend_all;gpu_all;variant_uuid;launcher_slr;SIGOPS"
    PROCESSORS 2
    WILL_FAIL FALSE
    FAIL_REGULAR_EXPRESSION "FAILED")

add_test(signalset_n2_w2_z32_uuid "env" "ROCSHMEM_SLR_NP=2" "ROCSHMEM_MAX_NUM_CONTEXTS=2" "ROCSHMEM_HEAP_SIZE=6442450944" "ROCSHMEM_TEST_UUID=1" "../../../../share/rocshmem/test_wrapper.sh" "signalset_n2_w2_z32_uuid" "../../../rocshmem_functional_tests" "-a" "172" "-w" "2" "-z" "32" "-localbuftype" "heap")
set_tests_properties(signalset_n2_w2_z32_uuid PROPERTIES
    TIMEOUT 300
    SKIP_RETURN_CODE 125
    LABELS "functional;backend_all;gpu_all;variant_uuid;launcher_slr;SIGOPS"
    PROCESSORS 2
    WILL_FAIL FALSE
    FAIL_REGULAR_EXPRESSION "FAILED")

add_test(signalwaituntil_n2_w1_z1_uuid "env" "ROCSHMEM_SLR_NP=2" "ROCSHMEM_MAX_NUM_CONTEXTS=1" "ROCSHMEM_HEAP_SIZE=6442450944" "ROCSHMEM_TEST_UUID=1" "../../../../share/rocshmem/test_wrapper.sh" "signalwaituntil_n2_w1_z1_uuid" "../../../rocshmem_functional_tests" "-a" "173" "-w" "1" "-z" "1" "-localbuftype" "heap")
set_tests_properties(signalwaituntil_n2_w1_z1_uuid PROPERTIES
    TIMEOUT 300
    SKIP_RETURN_CODE 125
    LABELS "functional;backend_all;gpu_all;variant_uuid;launcher_slr;SIGOPS"
    PROCESSORS 2
    WILL_FAIL FALSE
    FAIL_REGULAR_EXPRESSION "FAILED")

add_test(init_n2_w1_z1_uuid "env" "ROCSHMEM_SLR_NP=2" "ROCSHMEM_MAX_NUM_CONTEXTS=1" "ROCSHMEM_HEAP_SIZE=6442450944" "ROCSHMEM_TEST_UUID=1" "../../../../share/rocshmem/test_wrapper.sh" "init_n2_w1_z1_uuid" "../../../rocshmem_functional_tests" "-a" "11" "-w" "1" "-z" "1" "-localbuftype" "heap")
set_tests_properties(init_n2_w1_z1_uuid PROPERTIES
    TIMEOUT 300
    SKIP_RETURN_CODE 125
    LABELS "functional;backend_all;gpu_all;variant_uuid;launcher_slr;COLLECTIVE"
    PROCESSORS 2
    WILL_FAIL FALSE
    FAIL_REGULAR_EXPRESSION "FAILED")

add_test(syncall_n2_w1_z1_uuid "env" "ROCSHMEM_SLR_NP=2" "ROCSHMEM_MAX_NUM_CONTEXTS=1" "ROCSHMEM_HEAP_SIZE=6442450944" "ROCSHMEM_TEST_UUID=1" "../../../../share/rocshmem/test_wrapper.sh" "syncall_n2_w1_z1_uuid" "../../../rocshmem_functional_tests" "-a" "15" "-w" "1" "-z" "1" "-localbuftype" "heap")
set_tests_properties(syncall_n2_w1_z1_uuid PROPERTIES
    TIMEOUT 300
    SKIP_RETURN_CODE 125
    LABELS "functional;backend_all;gpu_all;variant_uuid;launcher_slr;COLLECTIVE"
    PROCESSORS 2
    WILL_FAIL FALSE
    FAIL_REGULAR_EXPRESSION "FAILED")

add_test(syncall_n3_w1_z1_uuid "env" "ROCSHMEM_SLR_NP=3" "ROCSHMEM_MAX_NUM_CONTEXTS=1" "ROCSHMEM_HEAP_SIZE=6442450944" "ROCSHMEM_TEST_UUID=1" "../../../../share/rocshmem/test_wrapper.sh" "syncall_n3_w1_z1_uuid" "../../../rocshmem_functional_tests" "-a" "15" "-w" "1" "-z" "1" "-localbuftype" "heap")
set_tests_properties(syncall_n3_w1_z1_uuid PROPERTIES
    TIMEOUT 300
    SKIP_RETURN_CODE 125
    LABELS "functional;backend_all;gpu_all;variant_uuid;launcher_slr;COLLECTIVE"
    PROCESSORS 3
    WILL_FAIL FALSE
    FAIL_REGULAR_EXPRESSION "FAILED")

add_test(syncall_n5_w1_z1_uuid "env" "ROCSHMEM_SLR_NP=5" "ROCSHMEM_MAX_NUM_CONTEXTS=1" "ROCSHMEM_HEAP_SIZE=6442450944" "ROCSHMEM_TEST_UUID=1" "../../../../share/rocshmem/test_wrapper.sh" "syncall_n5_w1_z1_uuid" "../../../rocshmem_functional_tests" "-a" "15" "-w" "1" "-z" "1" "-localbuftype" "heap")
set_tests_properties(syncall_n5_w1_z1_uuid PROPERTIES
    TIMEOUT 300
    SKIP_RETURN_CODE 125
    LABELS "functional;backend_all;gpu_all;variant_uuid;launcher_slr;COLLECTIVE"
    PROCESSORS 5
    WILL_FAIL FALSE
    FAIL_REGULAR_EXPRESSION "FAILED")

add_test(wavesyncall_n2_w1_z1_uuid "env" "ROCSHMEM_SLR_NP=2" "ROCSHMEM_MAX_NUM_CONTEXTS=1" "ROCSHMEM_HEAP_SIZE=6442450944" "ROCSHMEM_TEST_UUID=1" "../../../../share/rocshmem/test_wrapper.sh" "wavesyncall_n2_w1_z1_uuid" "../../../rocshmem_functional_tests" "-a" "67" "-w" "1" "-z" "1" "-localbuftype" "heap")
set_tests_properties(wavesyncall_n2_w1_z1_uuid PROPERTIES
    TIMEOUT 300
    SKIP_RETURN_CODE 125
    LABELS "functional;backend_all;gpu_all;variant_uuid;launcher_slr;COLLECTIVE"
    PROCESSORS 2
    WILL_FAIL FALSE
    FAIL_REGULAR_EXPRESSION "FAILED")

add_test(wgsyncall_n2_w1_z1_uuid "env" "ROCSHMEM_SLR_NP=2" "ROCSHMEM_MAX_NUM_CONTEXTS=1" "ROCSHMEM_HEAP_SIZE=6442450944" "ROCSHMEM_TEST_UUID=1" "../../../../share/rocshmem/test_wrapper.sh" "wgsyncall_n2_w1_z1_uuid" "../../../rocshmem_functional_tests" "-a" "68" "-w" "1" "-z" "1" "-localbuftype" "heap")
set_tests_properties(wgsyncall_n2_w1_z1_uuid PROPERTIES
    TIMEOUT 300
    SKIP_RETURN_CODE 125
    LABELS "functional;backend_all;gpu_all;variant_uuid;launcher_slr;COLLECTIVE"
    PROCESSORS 2
    WILL_FAIL FALSE
    FAIL_REGULAR_EXPRESSION "FAILED")

add_test(teamsync_n2_w1_z1_uuid "env" "ROCSHMEM_SLR_NP=2" "ROCSHMEM_MAX_NUM_CONTEXTS=1" "ROCSHMEM_HEAP_SIZE=6442450944" "ROCSHMEM_TEST_UUID=1" "../../../../share/rocshmem/test_wrapper.sh" "teamsync_n2_w1_z1_uuid" "../../../rocshmem_functional_tests" "-a" "16" "-w" "1" "-z" "1" "-localbuftype" "heap")
set_tests_properties(teamsync_n2_w1_z1_uuid PROPERTIES
    TIMEOUT 300
    SKIP_RETURN_CODE 125
    LABELS "functional;backend_all;gpu_all;variant_uuid;launcher_slr;COLLECTIVE"
    PROCESSORS 2
    WILL_FAIL FALSE
    FAIL_REGULAR_EXPRESSION "FAILED")

add_test(teamsync_n2_w16_z64_uuid "env" "ROCSHMEM_SLR_NP=2" "ROCSHMEM_MAX_NUM_CONTEXTS=16" "ROCSHMEM_HEAP_SIZE=6442450944" "ROCSHMEM_TEST_UUID=1" "../../../../share/rocshmem/test_wrapper.sh" "teamsync_n2_w16_z64_uuid" "../../../rocshmem_functional_tests" "-a" "16" "-w" "16" "-z" "64" "-localbuftype" "heap")
set_tests_properties(teamsync_n2_w16_z64_uuid PROPERTIES
    TIMEOUT 300
    SKIP_RETURN_CODE 125
    LABELS "functional;backend_all;gpu_all;variant_uuid;launcher_slr;COLLECTIVE"
    PROCESSORS 2
    WILL_FAIL FALSE
    FAIL_REGULAR_EXPRESSION "FAILED")

add_test(teamsync_n2_w32_z256_uuid "env" "ROCSHMEM_SLR_NP=2" "ROCSHMEM_MAX_NUM_CONTEXTS=32" "ROCSHMEM_HEAP_SIZE=6442450944" "ROCSHMEM_TEST_UUID=1" "../../../../share/rocshmem/test_wrapper.sh" "teamsync_n2_w32_z256_uuid" "../../../rocshmem_functional_tests" "-a" "16" "-w" "32" "-z" "256" "-localbuftype" "heap")
set_tests_properties(teamsync_n2_w32_z256_uuid PROPERTIES
    TIMEOUT 300
    SKIP_RETURN_CODE 125
    LABELS "functional;backend_all;gpu_all;variant_uuid;launcher_slr;COLLECTIVE"
    PROCESSORS 2
    WILL_FAIL FALSE
    FAIL_REGULAR_EXPRESSION "FAILED")

add_test(teamsync_n2_w39_z1024_uuid "env" "ROCSHMEM_SLR_NP=2" "ROCSHMEM_MAX_NUM_CONTEXTS=39" "ROCSHMEM_HEAP_SIZE=6442450944" "ROCSHMEM_TEST_UUID=1" "../../../../share/rocshmem/test_wrapper.sh" "teamsync_n2_w39_z1024_uuid" "../../../rocshmem_functional_tests" "-a" "16" "-w" "39" "-z" "1024" "-localbuftype" "heap")
set_tests_properties(teamsync_n2_w39_z1024_uuid PROPERTIES
    TIMEOUT 300
    SKIP_RETURN_CODE 125
    LABELS "functional;backend_all;gpu_all;variant_uuid;launcher_slr;COLLECTIVE"
    PROCESSORS 2
    WILL_FAIL FALSE
    FAIL_REGULAR_EXPRESSION "FAILED")

add_test(teamsync_n3_w16_z64_uuid "env" "ROCSHMEM_SLR_NP=3" "ROCSHMEM_MAX_NUM_CONTEXTS=16" "ROCSHMEM_HEAP_SIZE=6442450944" "ROCSHMEM_TEST_UUID=1" "../../../../share/rocshmem/test_wrapper.sh" "teamsync_n3_w16_z64_uuid" "../../../rocshmem_functional_tests" "-a" "16" "-w" "16" "-z" "64" "-localbuftype" "heap")
set_tests_properties(teamsync_n3_w16_z64_uuid PROPERTIES
    TIMEOUT 300
    SKIP_RETURN_CODE 125
    LABELS "functional;backend_all;gpu_all;variant_uuid;launcher_slr;COLLECTIVE"
    PROCESSORS 3
    WILL_FAIL FALSE
    FAIL_REGULAR_EXPRESSION "FAILED")

add_test(teamsync_n5_w16_z64_uuid "env" "ROCSHMEM_SLR_NP=5" "ROCSHMEM_MAX_NUM_CONTEXTS=16" "ROCSHMEM_HEAP_SIZE=6442450944" "ROCSHMEM_TEST_UUID=1" "../../../../share/rocshmem/test_wrapper.sh" "teamsync_n5_w16_z64_uuid" "../../../rocshmem_functional_tests" "-a" "16" "-w" "16" "-z" "64" "-localbuftype" "heap")
set_tests_properties(teamsync_n5_w16_z64_uuid PROPERTIES
    TIMEOUT 300
    SKIP_RETURN_CODE 125
    LABELS "functional;backend_all;gpu_all;variant_uuid;launcher_slr;COLLECTIVE"
    PROCESSORS 5
    WILL_FAIL FALSE
    FAIL_REGULAR_EXPRESSION "FAILED")

add_test(teamwavesync_n2_w1_z1_uuid "env" "ROCSHMEM_SLR_NP=2" "ROCSHMEM_MAX_NUM_CONTEXTS=1" "ROCSHMEM_HEAP_SIZE=6442450944" "ROCSHMEM_TEST_UUID=1" "../../../../share/rocshmem/test_wrapper.sh" "teamwavesync_n2_w1_z1_uuid" "../../../rocshmem_functional_tests" "-a" "71" "-w" "1" "-z" "1" "-localbuftype" "heap")
set_tests_properties(teamwavesync_n2_w1_z1_uuid PROPERTIES
    TIMEOUT 300
    SKIP_RETURN_CODE 125
    LABELS "functional;backend_all;gpu_all;variant_uuid;launcher_slr;COLLECTIVE"
    PROCESSORS 2
    WILL_FAIL FALSE
    FAIL_REGULAR_EXPRESSION "FAILED")

add_test(teamwavesync_n2_w16_z64_uuid "env" "ROCSHMEM_SLR_NP=2" "ROCSHMEM_MAX_NUM_CONTEXTS=16" "ROCSHMEM_HEAP_SIZE=6442450944" "ROCSHMEM_TEST_UUID=1" "../../../../share/rocshmem/test_wrapper.sh" "teamwavesync_n2_w16_z64_uuid" "../../../rocshmem_functional_tests" "-a" "71" "-w" "16" "-z" "64" "-localbuftype" "heap")
set_tests_properties(teamwavesync_n2_w16_z64_uuid PROPERTIES
    TIMEOUT 300
    SKIP_RETURN_CODE 125
    LABELS "functional;backend_all;gpu_all;variant_uuid;launcher_slr;COLLECTIVE"
    PROCESSORS 2
    WILL_FAIL FALSE
    FAIL_REGULAR_EXPRESSION "FAILED")

add_test(teamwavesync_n2_w32_z256_uuid "env" "ROCSHMEM_SLR_NP=2" "ROCSHMEM_MAX_NUM_CONTEXTS=32" "ROCSHMEM_HEAP_SIZE=6442450944" "ROCSHMEM_TEST_UUID=1" "../../../../share/rocshmem/test_wrapper.sh" "teamwavesync_n2_w32_z256_uuid" "../../../rocshmem_functional_tests" "-a" "71" "-w" "32" "-z" "256" "-localbuftype" "heap")
set_tests_properties(teamwavesync_n2_w32_z256_uuid PROPERTIES
    TIMEOUT 300
    SKIP_RETURN_CODE 125
    LABELS "functional;backend_all;gpu_all;variant_uuid;launcher_slr;COLLECTIVE"
    PROCESSORS 2
    WILL_FAIL FALSE
    FAIL_REGULAR_EXPRESSION "FAILED")

add_test(teamwavesync_n2_w39_z1024_uuid "env" "ROCSHMEM_SLR_NP=2" "ROCSHMEM_MAX_NUM_CONTEXTS=39" "ROCSHMEM_HEAP_SIZE=6442450944" "ROCSHMEM_TEST_UUID=1" "../../../../share/rocshmem/test_wrapper.sh" "teamwavesync_n2_w39_z1024_uuid" "../../../rocshmem_functional_tests" "-a" "71" "-w" "39" "-z" "1024" "-localbuftype" "heap")
set_tests_properties(teamwavesync_n2_w39_z1024_uuid PROPERTIES
    TIMEOUT 300
    SKIP_RETURN_CODE 125
    LABELS "functional;backend_all;gpu_all;variant_uuid;launcher_slr;COLLECTIVE"
    PROCESSORS 2
    WILL_FAIL FALSE
    FAIL_REGULAR_EXPRESSION "FAILED")

add_test(teamwgsync_n2_w1_z1_uuid "env" "ROCSHMEM_SLR_NP=2" "ROCSHMEM_MAX_NUM_CONTEXTS=1" "ROCSHMEM_HEAP_SIZE=6442450944" "ROCSHMEM_TEST_UUID=1" "../../../../share/rocshmem/test_wrapper.sh" "teamwgsync_n2_w1_z1_uuid" "../../../rocshmem_functional_tests" "-a" "72" "-w" "1" "-z" "1" "-localbuftype" "heap")
set_tests_properties(teamwgsync_n2_w1_z1_uuid PROPERTIES
    TIMEOUT 300
    SKIP_RETURN_CODE 125
    LABELS "functional;backend_all;gpu_all;variant_uuid;launcher_slr;COLLECTIVE"
    PROCESSORS 2
    WILL_FAIL FALSE
    FAIL_REGULAR_EXPRESSION "FAILED")

add_test(teamwgsync_n2_w16_z64_uuid "env" "ROCSHMEM_SLR_NP=2" "ROCSHMEM_MAX_NUM_CONTEXTS=16" "ROCSHMEM_HEAP_SIZE=6442450944" "ROCSHMEM_TEST_UUID=1" "../../../../share/rocshmem/test_wrapper.sh" "teamwgsync_n2_w16_z64_uuid" "../../../rocshmem_functional_tests" "-a" "72" "-w" "16" "-z" "64" "-localbuftype" "heap")
set_tests_properties(teamwgsync_n2_w16_z64_uuid PROPERTIES
    TIMEOUT 300
    SKIP_RETURN_CODE 125
    LABELS "functional;backend_all;gpu_all;variant_uuid;launcher_slr;COLLECTIVE"
    PROCESSORS 2
    WILL_FAIL FALSE
    FAIL_REGULAR_EXPRESSION "FAILED")

add_test(teamwgsync_n2_w32_z256_uuid "env" "ROCSHMEM_SLR_NP=2" "ROCSHMEM_MAX_NUM_CONTEXTS=32" "ROCSHMEM_HEAP_SIZE=6442450944" "ROCSHMEM_TEST_UUID=1" "../../../../share/rocshmem/test_wrapper.sh" "teamwgsync_n2_w32_z256_uuid" "../../../rocshmem_functional_tests" "-a" "72" "-w" "32" "-z" "256" "-localbuftype" "heap")
set_tests_properties(teamwgsync_n2_w32_z256_uuid PROPERTIES
    TIMEOUT 300
    SKIP_RETURN_CODE 125
    LABELS "functional;backend_all;gpu_all;variant_uuid;launcher_slr;COLLECTIVE"
    PROCESSORS 2
    WILL_FAIL FALSE
    FAIL_REGULAR_EXPRESSION "FAILED")

add_test(teamwgsync_n2_w39_z1024_uuid "env" "ROCSHMEM_SLR_NP=2" "ROCSHMEM_MAX_NUM_CONTEXTS=39" "ROCSHMEM_HEAP_SIZE=6442450944" "ROCSHMEM_TEST_UUID=1" "../../../../share/rocshmem/test_wrapper.sh" "teamwgsync_n2_w39_z1024_uuid" "../../../rocshmem_functional_tests" "-a" "72" "-w" "39" "-z" "1024" "-localbuftype" "heap")
set_tests_properties(teamwgsync_n2_w39_z1024_uuid PROPERTIES
    TIMEOUT 300
    SKIP_RETURN_CODE 125
    LABELS "functional;backend_all;gpu_all;variant_uuid;launcher_slr;COLLECTIVE"
    PROCESSORS 2
    WILL_FAIL FALSE
    FAIL_REGULAR_EXPRESSION "FAILED")

add_test(barrierall_n2_w1_z1_uuid "env" "ROCSHMEM_SLR_NP=2" "ROCSHMEM_MAX_NUM_CONTEXTS=1" "ROCSHMEM_HEAP_SIZE=6442450944" "ROCSHMEM_TEST_UUID=1" "../../../../share/rocshmem/test_wrapper.sh" "barrierall_n2_w1_z1_uuid" "../../../rocshmem_functional_tests" "-a" "14" "-w" "1" "-z" "1" "-localbuftype" "heap")
set_tests_properties(barrierall_n2_w1_z1_uuid PROPERTIES
    TIMEOUT 300
    SKIP_RETURN_CODE 125
    LABELS "functional;backend_all;gpu_all;variant_uuid;launcher_slr;COLLECTIVE"
    PROCESSORS 2
    WILL_FAIL FALSE
    FAIL_REGULAR_EXPRESSION "FAILED")

add_test(barrierall_n3_w1_z1_uuid "env" "ROCSHMEM_SLR_NP=3" "ROCSHMEM_MAX_NUM_CONTEXTS=1" "ROCSHMEM_HEAP_SIZE=6442450944" "ROCSHMEM_TEST_UUID=1" "../../../../share/rocshmem/test_wrapper.sh" "barrierall_n3_w1_z1_uuid" "../../../rocshmem_functional_tests" "-a" "14" "-w" "1" "-z" "1" "-localbuftype" "heap")
set_tests_properties(barrierall_n3_w1_z1_uuid PROPERTIES
    TIMEOUT 300
    SKIP_RETURN_CODE 125
    LABELS "functional;backend_all;gpu_all;variant_uuid;launcher_slr;COLLECTIVE"
    PROCESSORS 3
    WILL_FAIL FALSE
    FAIL_REGULAR_EXPRESSION "FAILED")

add_test(barrierall_n5_w1_z1_uuid "env" "ROCSHMEM_SLR_NP=5" "ROCSHMEM_MAX_NUM_CONTEXTS=1" "ROCSHMEM_HEAP_SIZE=6442450944" "ROCSHMEM_TEST_UUID=1" "../../../../share/rocshmem/test_wrapper.sh" "barrierall_n5_w1_z1_uuid" "../../../rocshmem_functional_tests" "-a" "14" "-w" "1" "-z" "1" "-localbuftype" "heap")
set_tests_properties(barrierall_n5_w1_z1_uuid PROPERTIES
    TIMEOUT 300
    SKIP_RETURN_CODE 125
    LABELS "functional;backend_all;gpu_all;variant_uuid;launcher_slr;COLLECTIVE"
    PROCESSORS 5
    WILL_FAIL FALSE
    FAIL_REGULAR_EXPRESSION "FAILED")

add_test(wavebarrierall_n2_w1_z1_uuid "env" "ROCSHMEM_SLR_NP=2" "ROCSHMEM_MAX_NUM_CONTEXTS=1" "ROCSHMEM_HEAP_SIZE=6442450944" "ROCSHMEM_TEST_UUID=1" "../../../../share/rocshmem/test_wrapper.sh" "wavebarrierall_n2_w1_z1_uuid" "../../../rocshmem_functional_tests" "-a" "65" "-w" "1" "-z" "1" "-localbuftype" "heap")
set_tests_properties(wavebarrierall_n2_w1_z1_uuid PROPERTIES
    TIMEOUT 300
    SKIP_RETURN_CODE 125
    LABELS "functional;backend_all;gpu_all;variant_uuid;launcher_slr;COLLECTIVE"
    PROCESSORS 2
    WILL_FAIL FALSE
    FAIL_REGULAR_EXPRESSION "FAILED")

add_test(wgbarrierall_n2_w1_z1_uuid "env" "ROCSHMEM_SLR_NP=2" "ROCSHMEM_MAX_NUM_CONTEXTS=1" "ROCSHMEM_HEAP_SIZE=6442450944" "ROCSHMEM_TEST_UUID=1" "../../../../share/rocshmem/test_wrapper.sh" "wgbarrierall_n2_w1_z1_uuid" "../../../rocshmem_functional_tests" "-a" "66" "-w" "1" "-z" "1" "-localbuftype" "heap")
set_tests_properties(wgbarrierall_n2_w1_z1_uuid PROPERTIES
    TIMEOUT 300
    SKIP_RETURN_CODE 125
    LABELS "functional;backend_all;gpu_all;variant_uuid;launcher_slr;COLLECTIVE"
    PROCESSORS 2
    WILL_FAIL FALSE
    FAIL_REGULAR_EXPRESSION "FAILED")

add_test(teambarrier_n2_w1_z1_uuid "env" "ROCSHMEM_SLR_NP=2" "ROCSHMEM_MAX_NUM_CONTEXTS=1" "ROCSHMEM_HEAP_SIZE=6442450944" "ROCSHMEM_TEST_UUID=1" "../../../../share/rocshmem/test_wrapper.sh" "teambarrier_n2_w1_z1_uuid" "../../../rocshmem_functional_tests" "-a" "69" "-w" "1" "-z" "1" "-localbuftype" "heap")
set_tests_properties(teambarrier_n2_w1_z1_uuid PROPERTIES
    TIMEOUT 300
    SKIP_RETURN_CODE 125
    LABELS "functional;backend_all;gpu_all;variant_uuid;launcher_slr;COLLECTIVE"
    PROCESSORS 2
    WILL_FAIL FALSE
    FAIL_REGULAR_EXPRESSION "FAILED")

add_test(teambarrier_n2_w16_z64_uuid "env" "ROCSHMEM_SLR_NP=2" "ROCSHMEM_MAX_NUM_CONTEXTS=16" "ROCSHMEM_HEAP_SIZE=6442450944" "ROCSHMEM_TEST_UUID=1" "../../../../share/rocshmem/test_wrapper.sh" "teambarrier_n2_w16_z64_uuid" "../../../rocshmem_functional_tests" "-a" "69" "-w" "16" "-z" "64" "-localbuftype" "heap")
set_tests_properties(teambarrier_n2_w16_z64_uuid PROPERTIES
    TIMEOUT 300
    SKIP_RETURN_CODE 125
    LABELS "functional;backend_all;gpu_all;variant_uuid;launcher_slr;COLLECTIVE"
    PROCESSORS 2
    WILL_FAIL FALSE
    FAIL_REGULAR_EXPRESSION "FAILED")

add_test(teambarrier_n2_w32_z256_uuid "env" "ROCSHMEM_SLR_NP=2" "ROCSHMEM_MAX_NUM_CONTEXTS=32" "ROCSHMEM_HEAP_SIZE=6442450944" "ROCSHMEM_TEST_UUID=1" "../../../../share/rocshmem/test_wrapper.sh" "teambarrier_n2_w32_z256_uuid" "../../../rocshmem_functional_tests" "-a" "69" "-w" "32" "-z" "256" "-localbuftype" "heap")
set_tests_properties(teambarrier_n2_w32_z256_uuid PROPERTIES
    TIMEOUT 300
    SKIP_RETURN_CODE 125
    LABELS "functional;backend_all;gpu_all;variant_uuid;launcher_slr;COLLECTIVE"
    PROCESSORS 2
    WILL_FAIL FALSE
    FAIL_REGULAR_EXPRESSION "FAILED")

add_test(teambarrier_n2_w39_z1024_uuid "env" "ROCSHMEM_SLR_NP=2" "ROCSHMEM_MAX_NUM_CONTEXTS=39" "ROCSHMEM_HEAP_SIZE=6442450944" "ROCSHMEM_TEST_UUID=1" "../../../../share/rocshmem/test_wrapper.sh" "teambarrier_n2_w39_z1024_uuid" "../../../rocshmem_functional_tests" "-a" "69" "-w" "39" "-z" "1024" "-localbuftype" "heap")
set_tests_properties(teambarrier_n2_w39_z1024_uuid PROPERTIES
    TIMEOUT 300
    SKIP_RETURN_CODE 125
    LABELS "functional;backend_all;gpu_all;variant_uuid;launcher_slr;COLLECTIVE"
    PROCESSORS 2
    WILL_FAIL FALSE
    FAIL_REGULAR_EXPRESSION "FAILED")

add_test(teambarrier_n3_w16_z64_uuid "env" "ROCSHMEM_SLR_NP=3" "ROCSHMEM_MAX_NUM_CONTEXTS=16" "ROCSHMEM_HEAP_SIZE=6442450944" "ROCSHMEM_TEST_UUID=1" "../../../../share/rocshmem/test_wrapper.sh" "teambarrier_n3_w16_z64_uuid" "../../../rocshmem_functional_tests" "-a" "69" "-w" "16" "-z" "64" "-localbuftype" "heap")
set_tests_properties(teambarrier_n3_w16_z64_uuid PROPERTIES
    TIMEOUT 300
    SKIP_RETURN_CODE 125
    LABELS "functional;backend_all;gpu_all;variant_uuid;launcher_slr;COLLECTIVE"
    PROCESSORS 3
    WILL_FAIL FALSE
    FAIL_REGULAR_EXPRESSION "FAILED")

add_test(teambarrier_n5_w16_z64_uuid "env" "ROCSHMEM_SLR_NP=5" "ROCSHMEM_MAX_NUM_CONTEXTS=16" "ROCSHMEM_HEAP_SIZE=6442450944" "ROCSHMEM_TEST_UUID=1" "../../../../share/rocshmem/test_wrapper.sh" "teambarrier_n5_w16_z64_uuid" "../../../rocshmem_functional_tests" "-a" "69" "-w" "16" "-z" "64" "-localbuftype" "heap")
set_tests_properties(teambarrier_n5_w16_z64_uuid PROPERTIES
    TIMEOUT 300
    SKIP_RETURN_CODE 125
    LABELS "functional;backend_all;gpu_all;variant_uuid;launcher_slr;COLLECTIVE"
    PROCESSORS 5
    WILL_FAIL FALSE
    FAIL_REGULAR_EXPRESSION "FAILED")

add_test(teamwavebarrier_n2_w1_z1_uuid "env" "ROCSHMEM_SLR_NP=2" "ROCSHMEM_MAX_NUM_CONTEXTS=1" "ROCSHMEM_HEAP_SIZE=6442450944" "ROCSHMEM_TEST_UUID=1" "../../../../share/rocshmem/test_wrapper.sh" "teamwavebarrier_n2_w1_z1_uuid" "../../../rocshmem_functional_tests" "-a" "70" "-w" "1" "-z" "1" "-localbuftype" "heap")
set_tests_properties(teamwavebarrier_n2_w1_z1_uuid PROPERTIES
    TIMEOUT 300
    SKIP_RETURN_CODE 125
    LABELS "functional;backend_all;gpu_all;variant_uuid;launcher_slr;COLLECTIVE"
    PROCESSORS 2
    WILL_FAIL FALSE
    FAIL_REGULAR_EXPRESSION "FAILED")

add_test(teamwavebarrier_n2_w16_z64_uuid "env" "ROCSHMEM_SLR_NP=2" "ROCSHMEM_MAX_NUM_CONTEXTS=16" "ROCSHMEM_HEAP_SIZE=6442450944" "ROCSHMEM_TEST_UUID=1" "../../../../share/rocshmem/test_wrapper.sh" "teamwavebarrier_n2_w16_z64_uuid" "../../../rocshmem_functional_tests" "-a" "70" "-w" "16" "-z" "64" "-localbuftype" "heap")
set_tests_properties(teamwavebarrier_n2_w16_z64_uuid PROPERTIES
    TIMEOUT 300
    SKIP_RETURN_CODE 125
    LABELS "functional;backend_all;gpu_all;variant_uuid;launcher_slr;COLLECTIVE"
    PROCESSORS 2
    WILL_FAIL FALSE
    FAIL_REGULAR_EXPRESSION "FAILED")

add_test(teamwavebarrier_n2_w32_z256_uuid "env" "ROCSHMEM_SLR_NP=2" "ROCSHMEM_MAX_NUM_CONTEXTS=32" "ROCSHMEM_HEAP_SIZE=6442450944" "ROCSHMEM_TEST_UUID=1" "../../../../share/rocshmem/test_wrapper.sh" "teamwavebarrier_n2_w32_z256_uuid" "../../../rocshmem_functional_tests" "-a" "70" "-w" "32" "-z" "256" "-localbuftype" "heap")
set_tests_properties(teamwavebarrier_n2_w32_z256_uuid PROPERTIES
    TIMEOUT 300
    SKIP_RETURN_CODE 125
    LABELS "functional;backend_all;gpu_all;variant_uuid;launcher_slr;COLLECTIVE"
    PROCESSORS 2
    WILL_FAIL FALSE
    FAIL_REGULAR_EXPRESSION "FAILED")

add_test(teamwavebarrier_n2_w39_z1024_uuid "env" "ROCSHMEM_SLR_NP=2" "ROCSHMEM_MAX_NUM_CONTEXTS=39" "ROCSHMEM_HEAP_SIZE=6442450944" "ROCSHMEM_TEST_UUID=1" "../../../../share/rocshmem/test_wrapper.sh" "teamwavebarrier_n2_w39_z1024_uuid" "../../../rocshmem_functional_tests" "-a" "70" "-w" "39" "-z" "1024" "-localbuftype" "heap")
set_tests_properties(teamwavebarrier_n2_w39_z1024_uuid PROPERTIES
    TIMEOUT 300
    SKIP_RETURN_CODE 125
    LABELS "functional;backend_all;gpu_all;variant_uuid;launcher_slr;COLLECTIVE"
    PROCESSORS 2
    WILL_FAIL FALSE
    FAIL_REGULAR_EXPRESSION "FAILED")

add_test(teamwgbarrier_n2_w1_z1_uuid "env" "ROCSHMEM_SLR_NP=2" "ROCSHMEM_MAX_NUM_CONTEXTS=1" "ROCSHMEM_HEAP_SIZE=6442450944" "ROCSHMEM_TEST_UUID=1" "../../../../share/rocshmem/test_wrapper.sh" "teamwgbarrier_n2_w1_z1_uuid" "../../../rocshmem_functional_tests" "-a" "58" "-w" "1" "-z" "1" "-localbuftype" "heap")
set_tests_properties(teamwgbarrier_n2_w1_z1_uuid PROPERTIES
    TIMEOUT 300
    SKIP_RETURN_CODE 125
    LABELS "functional;backend_all;gpu_all;variant_uuid;launcher_slr;COLLECTIVE"
    PROCESSORS 2
    WILL_FAIL FALSE
    FAIL_REGULAR_EXPRESSION "FAILED")

add_test(teamwgbarrier_n2_w16_z64_uuid "env" "ROCSHMEM_SLR_NP=2" "ROCSHMEM_MAX_NUM_CONTEXTS=16" "ROCSHMEM_HEAP_SIZE=6442450944" "ROCSHMEM_TEST_UUID=1" "../../../../share/rocshmem/test_wrapper.sh" "teamwgbarrier_n2_w16_z64_uuid" "../../../rocshmem_functional_tests" "-a" "58" "-w" "16" "-z" "64" "-localbuftype" "heap")
set_tests_properties(teamwgbarrier_n2_w16_z64_uuid PROPERTIES
    TIMEOUT 300
    SKIP_RETURN_CODE 125
    LABELS "functional;backend_all;gpu_all;variant_uuid;launcher_slr;COLLECTIVE"
    PROCESSORS 2
    WILL_FAIL FALSE
    FAIL_REGULAR_EXPRESSION "FAILED")

add_test(teamwgbarrier_n2_w32_z256_uuid "env" "ROCSHMEM_SLR_NP=2" "ROCSHMEM_MAX_NUM_CONTEXTS=32" "ROCSHMEM_HEAP_SIZE=6442450944" "ROCSHMEM_TEST_UUID=1" "../../../../share/rocshmem/test_wrapper.sh" "teamwgbarrier_n2_w32_z256_uuid" "../../../rocshmem_functional_tests" "-a" "58" "-w" "32" "-z" "256" "-localbuftype" "heap")
set_tests_properties(teamwgbarrier_n2_w32_z256_uuid PROPERTIES
    TIMEOUT 300
    SKIP_RETURN_CODE 125
    LABELS "functional;backend_all;gpu_all;variant_uuid;launcher_slr;COLLECTIVE"
    PROCESSORS 2
    WILL_FAIL FALSE
    FAIL_REGULAR_EXPRESSION "FAILED")

add_test(teamwgbarrier_n2_w39_z1024_uuid "env" "ROCSHMEM_SLR_NP=2" "ROCSHMEM_MAX_NUM_CONTEXTS=39" "ROCSHMEM_HEAP_SIZE=6442450944" "ROCSHMEM_TEST_UUID=1" "../../../../share/rocshmem/test_wrapper.sh" "teamwgbarrier_n2_w39_z1024_uuid" "../../../rocshmem_functional_tests" "-a" "58" "-w" "39" "-z" "1024" "-localbuftype" "heap")
set_tests_properties(teamwgbarrier_n2_w39_z1024_uuid PROPERTIES
    TIMEOUT 300
    SKIP_RETURN_CODE 125
    LABELS "functional;backend_all;gpu_all;variant_uuid;launcher_slr;COLLECTIVE"
    PROCESSORS 2
    WILL_FAIL FALSE
    FAIL_REGULAR_EXPRESSION "FAILED")

add_test(alltoall_n2_w1_z64_512B_uuid "env" "ROCSHMEM_SLR_NP=2" "ROCSHMEM_MAX_NUM_CONTEXTS=1" "ROCSHMEM_HEAP_SIZE=6442450944" "ROCSHMEM_TEST_UUID=1" "../../../../share/rocshmem/test_wrapper.sh" "alltoall_n2_w1_z64_512B_uuid" "../../../rocshmem_functional_tests" "-a" "19" "-w" "1" "-z" "64" "-s" "512" "-localbuftype" "heap")
set_tests_properties(alltoall_n2_w1_z64_512B_uuid PROPERTIES
    TIMEOUT 300
    SKIP_RETURN_CODE 125
    LABELS "functional;backend_all;gpu_all;variant_uuid;launcher_slr;COLLECTIVE"
    PROCESSORS 2
    WILL_FAIL FALSE
    FAIL_REGULAR_EXPRESSION "FAILED")

add_test(alltoall_n3_w1_z64_512B_uuid "env" "ROCSHMEM_SLR_NP=3" "ROCSHMEM_MAX_NUM_CONTEXTS=1" "ROCSHMEM_HEAP_SIZE=6442450944" "ROCSHMEM_TEST_UUID=1" "../../../../share/rocshmem/test_wrapper.sh" "alltoall_n3_w1_z64_512B_uuid" "../../../rocshmem_functional_tests" "-a" "19" "-w" "1" "-z" "64" "-s" "512" "-localbuftype" "heap")
set_tests_properties(alltoall_n3_w1_z64_512B_uuid PROPERTIES
    TIMEOUT 300
    SKIP_RETURN_CODE 125
    LABELS "functional;backend_all;gpu_all;variant_uuid;launcher_slr;COLLECTIVE"
    PROCESSORS 3
    WILL_FAIL FALSE
    FAIL_REGULAR_EXPRESSION "FAILED")

add_test(alltoall_n5_w1_z64_512B_uuid "env" "ROCSHMEM_SLR_NP=5" "ROCSHMEM_MAX_NUM_CONTEXTS=1" "ROCSHMEM_HEAP_SIZE=6442450944" "ROCSHMEM_TEST_UUID=1" "../../../../share/rocshmem/test_wrapper.sh" "alltoall_n5_w1_z64_512B_uuid" "../../../rocshmem_functional_tests" "-a" "19" "-w" "1" "-z" "64" "-s" "512" "-localbuftype" "heap")
set_tests_properties(alltoall_n5_w1_z64_512B_uuid PROPERTIES
    TIMEOUT 300
    SKIP_RETURN_CODE 125
    LABELS "functional;backend_all;gpu_all;variant_uuid;launcher_slr;COLLECTIVE"
    PROCESSORS 5
    WILL_FAIL FALSE
    FAIL_REGULAR_EXPRESSION "FAILED")

add_test(teambroadcast_n2_w1_z64_32768B_uuid "env" "ROCSHMEM_SLR_NP=2" "ROCSHMEM_MAX_NUM_CONTEXTS=1" "ROCSHMEM_HEAP_SIZE=6442450944" "ROCSHMEM_TEST_UUID=1" "../../../../share/rocshmem/test_wrapper.sh" "teambroadcast_n2_w1_z64_32768B_uuid" "../../../rocshmem_functional_tests" "-a" "32" "-w" "1" "-z" "64" "-s" "32768" "-localbuftype" "heap")
set_tests_properties(teambroadcast_n2_w1_z64_32768B_uuid PROPERTIES
    TIMEOUT 300
    SKIP_RETURN_CODE 125
    LABELS "functional;backend_all;gpu_all;variant_uuid;launcher_slr;COLLECTIVE"
    PROCESSORS 2
    WILL_FAIL FALSE
    FAIL_REGULAR_EXPRESSION "FAILED")

add_test(teambroadcast_n3_w1_z64_32768B_uuid "env" "ROCSHMEM_SLR_NP=3" "ROCSHMEM_MAX_NUM_CONTEXTS=1" "ROCSHMEM_HEAP_SIZE=6442450944" "ROCSHMEM_TEST_UUID=1" "../../../../share/rocshmem/test_wrapper.sh" "teambroadcast_n3_w1_z64_32768B_uuid" "../../../rocshmem_functional_tests" "-a" "32" "-w" "1" "-z" "64" "-s" "32768" "-localbuftype" "heap")
set_tests_properties(teambroadcast_n3_w1_z64_32768B_uuid PROPERTIES
    TIMEOUT 300
    SKIP_RETURN_CODE 125
    LABELS "functional;backend_all;gpu_all;variant_uuid;launcher_slr;COLLECTIVE"
    PROCESSORS 3
    WILL_FAIL FALSE
    FAIL_REGULAR_EXPRESSION "FAILED")

add_test(teambroadcast_n5_w1_z64_32768B_uuid "env" "ROCSHMEM_SLR_NP=5" "ROCSHMEM_MAX_NUM_CONTEXTS=1" "ROCSHMEM_HEAP_SIZE=6442450944" "ROCSHMEM_TEST_UUID=1" "../../../../share/rocshmem/test_wrapper.sh" "teambroadcast_n5_w1_z64_32768B_uuid" "../../../rocshmem_functional_tests" "-a" "32" "-w" "1" "-z" "64" "-s" "32768" "-localbuftype" "heap")
set_tests_properties(teambroadcast_n5_w1_z64_32768B_uuid PROPERTIES
    TIMEOUT 300
    SKIP_RETURN_CODE 125
    LABELS "functional;backend_all;gpu_all;variant_uuid;launcher_slr;COLLECTIVE"
    PROCESSORS 5
    WILL_FAIL FALSE
    FAIL_REGULAR_EXPRESSION "FAILED")

add_test(fcollect_n2_w1_z64_32768B_uuid "env" "ROCSHMEM_SLR_NP=2" "ROCSHMEM_MAX_NUM_CONTEXTS=1" "ROCSHMEM_HEAP_SIZE=6442450944" "ROCSHMEM_TEST_UUID=1" "../../../../share/rocshmem/test_wrapper.sh" "fcollect_n2_w1_z64_32768B_uuid" "../../../rocshmem_functional_tests" "-a" "18" "-w" "1" "-z" "64" "-s" "32768" "-localbuftype" "heap")
set_tests_properties(fcollect_n2_w1_z64_32768B_uuid PROPERTIES
    TIMEOUT 300
    SKIP_RETURN_CODE 125
    LABELS "functional;backend_all;gpu_all;variant_uuid;launcher_slr;COLLECTIVE"
    PROCESSORS 2
    WILL_FAIL FALSE
    FAIL_REGULAR_EXPRESSION "FAILED")

add_test(fcollect_n3_w1_z64_32768B_uuid "env" "ROCSHMEM_SLR_NP=3" "ROCSHMEM_MAX_NUM_CONTEXTS=1" "ROCSHMEM_HEAP_SIZE=6442450944" "ROCSHMEM_TEST_UUID=1" "../../../../share/rocshmem/test_wrapper.sh" "fcollect_n3_w1_z64_32768B_uuid" "../../../rocshmem_functional_tests" "-a" "18" "-w" "1" "-z" "64" "-s" "32768" "-localbuftype" "heap")
set_tests_properties(fcollect_n3_w1_z64_32768B_uuid PROPERTIES
    TIMEOUT 300
    SKIP_RETURN_CODE 125
    LABELS "functional;backend_all;gpu_all;variant_uuid;launcher_slr;COLLECTIVE"
    PROCESSORS 3
    WILL_FAIL FALSE
    FAIL_REGULAR_EXPRESSION "FAILED")

add_test(fcollect_n5_w1_z64_32768B_uuid "env" "ROCSHMEM_SLR_NP=5" "ROCSHMEM_MAX_NUM_CONTEXTS=1" "ROCSHMEM_HEAP_SIZE=6442450944" "ROCSHMEM_TEST_UUID=1" "../../../../share/rocshmem/test_wrapper.sh" "fcollect_n5_w1_z64_32768B_uuid" "../../../rocshmem_functional_tests" "-a" "18" "-w" "1" "-z" "64" "-s" "32768" "-localbuftype" "heap")
set_tests_properties(fcollect_n5_w1_z64_32768B_uuid PROPERTIES
    TIMEOUT 300
    SKIP_RETURN_CODE 125
    LABELS "functional;backend_all;gpu_all;variant_uuid;launcher_slr;COLLECTIVE"
    PROCESSORS 5
    WILL_FAIL FALSE
    FAIL_REGULAR_EXPRESSION "FAILED")

add_test(teamreduction_n2_w1_z64_32768B_uuid "env" "ROCSHMEM_SLR_NP=2" "ROCSHMEM_MAX_NUM_CONTEXTS=1" "ROCSHMEM_HEAP_SIZE=6442450944" "ROCSHMEM_TEST_UUID=1" "../../../../share/rocshmem/test_wrapper.sh" "teamreduction_n2_w1_z64_32768B_uuid" "../../../rocshmem_functional_tests" "-a" "33" "-w" "1" "-z" "64" "-s" "32768" "-localbuftype" "heap")
set_tests_properties(teamreduction_n2_w1_z64_32768B_uuid PROPERTIES
    TIMEOUT 300
    SKIP_RETURN_CODE 125
    LABELS "functional;backend_all;gpu_all;variant_uuid;launcher_slr;COLLECTIVE"
    PROCESSORS 2
    WILL_FAIL FALSE
    FAIL_REGULAR_EXPRESSION "FAILED")

add_test(teamreducescatter_n2_w1_z64_32768B_uuid "env" "ROCSHMEM_SLR_NP=2" "ROCSHMEM_MAX_NUM_CONTEXTS=1" "ROCSHMEM_HEAP_SIZE=6442450944" "ROCSHMEM_TEST_UUID=1" "../../../../share/rocshmem/test_wrapper.sh" "teamreducescatter_n2_w1_z64_32768B_uuid" "../../../rocshmem_functional_tests" "-a" "149" "-w" "1" "-z" "64" "-s" "32768" "-localbuftype" "heap")
set_tests_properties(teamreducescatter_n2_w1_z64_32768B_uuid PROPERTIES
    TIMEOUT 300
    SKIP_RETURN_CODE 125
    LABELS "functional;backend_all;gpu_all;variant_uuid;launcher_slr;COLLECTIVE"
    PROCESSORS 2
    WILL_FAIL FALSE
    FAIL_REGULAR_EXPRESSION "FAILED")

add_test(teamsplit2d_n4_w1_z1_uuid "env" "ROCSHMEM_SLR_NP=4" "ROCSHMEM_MAX_NUM_CONTEXTS=1" "ROCSHMEM_HEAP_SIZE=6442450944" "ROCSHMEM_TEST_UUID=1" "../../../../share/rocshmem/test_wrapper.sh" "teamsplit2d_n4_w1_z1_uuid" "../../../rocshmem_functional_tests" "-a" "119" "-w" "1" "-z" "1" "-localbuftype" "heap")
set_tests_properties(teamsplit2d_n4_w1_z1_uuid PROPERTIES
    TIMEOUT 300
    SKIP_RETURN_CODE 125
    LABELS "functional;backend_all;gpu_all;variant_uuid;launcher_slr;COLLECTIVE"
    PROCESSORS 4
    WILL_FAIL FALSE
    FAIL_REGULAR_EXPRESSION "FAILED")

add_test(broadcast_wave_n2_w1_wf1_32768B_uuid "env" "ROCSHMEM_SLR_NP=2" "ROCSHMEM_MAX_NUM_CONTEXTS=1" "ROCSHMEM_HEAP_SIZE=6442450944" "ROCSHMEM_TEST_UUID=1" "../../../../share/rocshmem/test_wrapper.sh" "broadcast_wave_n2_w1_wf1_32768B_uuid" "../../../rocshmem_functional_tests" "-a" "150" "-w" "1" "-s" "32768" "--num-wf" "1" "-localbuftype" "heap")
set_tests_properties(broadcast_wave_n2_w1_wf1_32768B_uuid PROPERTIES
    TIMEOUT 300
    SKIP_RETURN_CODE 125
    LABELS "functional;backend_ipc;backend_gda;gpu_all;variant_uuid;launcher_slr;COLLECTIVE"
    PROCESSORS 2
    WILL_FAIL FALSE
    FAIL_REGULAR_EXPRESSION "FAILED")

add_test(alltoall_wave_n2_w1_wf1_512B_uuid "env" "ROCSHMEM_SLR_NP=2" "ROCSHMEM_MAX_NUM_CONTEXTS=1" "ROCSHMEM_HEAP_SIZE=6442450944" "ROCSHMEM_TEST_UUID=1" "../../../../share/rocshmem/test_wrapper.sh" "alltoall_wave_n2_w1_wf1_512B_uuid" "../../../rocshmem_functional_tests" "-a" "151" "-w" "1" "-s" "512" "--num-wf" "1" "-localbuftype" "heap")
set_tests_properties(alltoall_wave_n2_w1_wf1_512B_uuid PROPERTIES
    TIMEOUT 300
    SKIP_RETURN_CODE 125
    LABELS "functional;backend_ipc;backend_gda;gpu_all;variant_uuid;launcher_slr;COLLECTIVE"
    PROCESSORS 2
    WILL_FAIL FALSE
    FAIL_REGULAR_EXPRESSION "FAILED")

add_test(fcollect_wave_n2_w1_wf1_32768B_uuid "env" "ROCSHMEM_SLR_NP=2" "ROCSHMEM_MAX_NUM_CONTEXTS=1" "ROCSHMEM_HEAP_SIZE=6442450944" "ROCSHMEM_TEST_UUID=1" "../../../../share/rocshmem/test_wrapper.sh" "fcollect_wave_n2_w1_wf1_32768B_uuid" "../../../rocshmem_functional_tests" "-a" "152" "-w" "1" "-s" "32768" "--num-wf" "1" "-localbuftype" "heap")
set_tests_properties(fcollect_wave_n2_w1_wf1_32768B_uuid PROPERTIES
    TIMEOUT 300
    SKIP_RETURN_CODE 125
    LABELS "functional;backend_ipc;backend_gda;gpu_all;variant_uuid;launcher_slr;COLLECTIVE"
    PROCESSORS 2
    WILL_FAIL FALSE
    FAIL_REGULAR_EXPRESSION "FAILED")

add_test(reduce_wave_n2_w1_wf1_32768B_uuid "env" "ROCSHMEM_SLR_NP=2" "ROCSHMEM_MAX_NUM_CONTEXTS=1" "ROCSHMEM_HEAP_SIZE=6442450944" "ROCSHMEM_TEST_UUID=1" "../../../../share/rocshmem/test_wrapper.sh" "reduce_wave_n2_w1_wf1_32768B_uuid" "../../../rocshmem_functional_tests" "-a" "153" "-w" "1" "-s" "32768" "--num-wf" "1" "-localbuftype" "heap")
set_tests_properties(reduce_wave_n2_w1_wf1_32768B_uuid PROPERTIES
    TIMEOUT 300
    SKIP_RETURN_CODE 125
    LABELS "functional;backend_ipc;backend_gda;gpu_all;variant_uuid;launcher_slr;COLLECTIVE"
    PROCESSORS 2
    WILL_FAIL FALSE
    FAIL_REGULAR_EXPRESSION "FAILED")

add_test(reducescatter_wave_n2_w1_wf1_32768B_uuid "env" "ROCSHMEM_SLR_NP=2" "ROCSHMEM_MAX_NUM_CONTEXTS=1" "ROCSHMEM_HEAP_SIZE=6442450944" "ROCSHMEM_TEST_UUID=1" "../../../../share/rocshmem/test_wrapper.sh" "reducescatter_wave_n2_w1_wf1_32768B_uuid" "../../../rocshmem_functional_tests" "-a" "154" "-w" "1" "-s" "32768" "--num-wf" "1" "-localbuftype" "heap")
set_tests_properties(reducescatter_wave_n2_w1_wf1_32768B_uuid PROPERTIES
    TIMEOUT 300
    SKIP_RETURN_CODE 125
    LABELS "functional;backend_ipc;backend_gda;gpu_all;variant_uuid;launcher_slr;COLLECTIVE"
    PROCESSORS 2
    WILL_FAIL FALSE
    FAIL_REGULAR_EXPRESSION "FAILED")

add_test(reducescatter_wave_n4_w1_wf1_32768B_uuid "env" "ROCSHMEM_SLR_NP=4" "ROCSHMEM_MAX_NUM_CONTEXTS=1" "ROCSHMEM_HEAP_SIZE=6442450944" "ROCSHMEM_TEST_UUID=1" "../../../../share/rocshmem/test_wrapper.sh" "reducescatter_wave_n4_w1_wf1_32768B_uuid" "../../../rocshmem_functional_tests" "-a" "154" "-w" "1" "-s" "32768" "--num-wf" "1" "-localbuftype" "heap")
set_tests_properties(reducescatter_wave_n4_w1_wf1_32768B_uuid PROPERTIES
    TIMEOUT 300
    SKIP_RETURN_CODE 125
    LABELS "functional;backend_ipc;backend_gda;gpu_all;variant_uuid;launcher_slr;COLLECTIVE"
    PROCESSORS 4
    WILL_FAIL FALSE
    FAIL_REGULAR_EXPRESSION "FAILED")

add_test(reducescatter_wave_n8_w1_wf1_32768B_uuid "env" "ROCSHMEM_SLR_NP=8" "ROCSHMEM_MAX_NUM_CONTEXTS=1" "ROCSHMEM_HEAP_SIZE=6442450944" "ROCSHMEM_TEST_UUID=1" "../../../../share/rocshmem/test_wrapper.sh" "reducescatter_wave_n8_w1_wf1_32768B_uuid" "../../../rocshmem_functional_tests" "-a" "154" "-w" "1" "-s" "32768" "--num-wf" "1" "-localbuftype" "heap")
set_tests_properties(reducescatter_wave_n8_w1_wf1_32768B_uuid PROPERTIES
    TIMEOUT 300
    SKIP_RETURN_CODE 125
    LABELS "functional;backend_ipc;backend_gda;gpu_all;variant_uuid;launcher_slr;COLLECTIVE"
    PROCESSORS 8
    WILL_FAIL FALSE
    FAIL_REGULAR_EXPRESSION "FAILED")

add_test(putmem_on_stream_n2_w1_z1_1048576B_uuid "env" "ROCSHMEM_SLR_NP=2" "ROCSHMEM_MAX_NUM_CONTEXTS=1" "ROCSHMEM_HEAP_SIZE=6442450944" "ROCSHMEM_TEST_UUID=1" "../../../../share/rocshmem/test_wrapper.sh" "putmem_on_stream_n2_w1_z1_1048576B_uuid" "../../../rocshmem_functional_tests" "-a" "80" "-w" "1" "-z" "1" "-s" "1048576" "-localbuftype" "heap")
set_tests_properties(putmem_on_stream_n2_w1_z1_1048576B_uuid PROPERTIES
    TIMEOUT 300
    SKIP_RETURN_CODE 125
    LABELS "functional;backend_all;gpu_all;variant_uuid;launcher_slr;RMA"
    PROCESSORS 2
    WILL_FAIL FALSE
    FAIL_REGULAR_EXPRESSION "FAILED")

add_test(putmem_on_stream_n2_w1_z1_1048576B_default_stream "env" "ROCSHMEM_SLR_NP=2" "ROCSHMEM_MAX_NUM_CONTEXTS=1" "ROCSHMEM_HEAP_SIZE=6442450944" "ROCSHMEM_TEST_USE_DEFAULT_STREAM=1" "../../../../share/rocshmem/test_wrapper.sh" "putmem_on_stream_n2_w1_z1_1048576B_default_stream" "../../../rocshmem_functional_tests" "-a" "80" "-w" "1" "-z" "1" "-s" "1048576" "-localbuftype" "heap")
set_tests_properties(putmem_on_stream_n2_w1_z1_1048576B_default_stream PROPERTIES
    TIMEOUT 300
    SKIP_RETURN_CODE 125
    LABELS "functional;backend_all;gpu_all;variant_default_stream;launcher_slr;RMA"
    PROCESSORS 2
    WILL_FAIL FALSE
    FAIL_REGULAR_EXPRESSION "FAILED")

add_test(putmem_on_stream_n2_w1_z1_1048576B_uuid_default_stream "env" "ROCSHMEM_SLR_NP=2" "ROCSHMEM_MAX_NUM_CONTEXTS=1" "ROCSHMEM_HEAP_SIZE=6442450944" "ROCSHMEM_TEST_UUID=1" "ROCSHMEM_TEST_USE_DEFAULT_STREAM=1" "../../../../share/rocshmem/test_wrapper.sh" "putmem_on_stream_n2_w1_z1_1048576B_uuid_default_stream" "../../../rocshmem_functional_tests" "-a" "80" "-w" "1" "-z" "1" "-s" "1048576" "-localbuftype" "heap")
set_tests_properties(putmem_on_stream_n2_w1_z1_1048576B_uuid_default_stream PROPERTIES
    TIMEOUT 300
    SKIP_RETURN_CODE 125
    LABELS "functional;backend_all;gpu_all;variant_uuid;variant_default_stream;launcher_slr;RMA"
    PROCESSORS 2
    WILL_FAIL FALSE
    FAIL_REGULAR_EXPRESSION "FAILED")

add_test(getmem_on_stream_n2_w1_z1_1048576B_uuid "env" "ROCSHMEM_SLR_NP=2" "ROCSHMEM_MAX_NUM_CONTEXTS=1" "ROCSHMEM_HEAP_SIZE=6442450944" "ROCSHMEM_TEST_UUID=1" "../../../../share/rocshmem/test_wrapper.sh" "getmem_on_stream_n2_w1_z1_1048576B_uuid" "../../../rocshmem_functional_tests" "-a" "79" "-w" "1" "-z" "1" "-s" "1048576" "-localbuftype" "heap")
set_tests_properties(getmem_on_stream_n2_w1_z1_1048576B_uuid PROPERTIES
    TIMEOUT 300
    SKIP_RETURN_CODE 125
    LABELS "functional;backend_ipc;backend_gda;gpu_all;variant_uuid;launcher_slr;RMA"
    PROCESSORS 2
    WILL_FAIL FALSE
    FAIL_REGULAR_EXPRESSION "FAILED")

add_test(signal_wait_until_on_stream_n2_w1_z1_uuid "env" "ROCSHMEM_SLR_NP=2" "ROCSHMEM_MAX_NUM_CONTEXTS=1" "ROCSHMEM_HEAP_SIZE=6442450944" "ROCSHMEM_TEST_UUID=1" "../../../../share/rocshmem/test_wrapper.sh" "signal_wait_until_on_stream_n2_w1_z1_uuid" "../../../rocshmem_functional_tests" "-a" "82" "-w" "1" "-z" "1" "-localbuftype" "heap")
set_tests_properties(signal_wait_until_on_stream_n2_w1_z1_uuid PROPERTIES
    TIMEOUT 300
    SKIP_RETURN_CODE 125
    LABELS "functional;backend_all;gpu_all;variant_uuid;launcher_slr;STREAM"
    PROCESSORS 2
    WILL_FAIL FALSE
    FAIL_REGULAR_EXPRESSION "FAILED")

add_test(putmem_signal_on_stream_n2_w1_z1_1048576B_uuid "env" "ROCSHMEM_SLR_NP=2" "ROCSHMEM_MAX_NUM_CONTEXTS=1" "ROCSHMEM_HEAP_SIZE=6442450944" "ROCSHMEM_TEST_UUID=1" "../../../../share/rocshmem/test_wrapper.sh" "putmem_signal_on_stream_n2_w1_z1_1048576B_uuid" "../../../rocshmem_functional_tests" "-a" "81" "-w" "1" "-z" "1" "-s" "1048576" "-localbuftype" "heap")
set_tests_properties(putmem_signal_on_stream_n2_w1_z1_1048576B_uuid PROPERTIES
    TIMEOUT 300
    SKIP_RETURN_CODE 125
    LABELS "functional;backend_ipc;backend_gda;gpu_all;variant_uuid;launcher_slr;RMA"
    PROCESSORS 2
    WILL_FAIL FALSE
    FAIL_REGULAR_EXPRESSION "FAILED")

add_test(barrier_all_on_stream_n2_w1_z1_uuid "env" "ROCSHMEM_SLR_NP=2" "ROCSHMEM_MAX_NUM_CONTEXTS=1" "ROCSHMEM_HEAP_SIZE=6442450944" "ROCSHMEM_TEST_UUID=1" "../../../../share/rocshmem/test_wrapper.sh" "barrier_all_on_stream_n2_w1_z1_uuid" "../../../rocshmem_functional_tests" "-a" "77" "-w" "1" "-z" "1" "-localbuftype" "heap")
set_tests_properties(barrier_all_on_stream_n2_w1_z1_uuid PROPERTIES
    TIMEOUT 300
    SKIP_RETURN_CODE 125
    LABELS "functional;backend_all;gpu_all;variant_uuid;launcher_slr;COLLECTIVE"
    PROCESSORS 2
    WILL_FAIL FALSE
    FAIL_REGULAR_EXPRESSION "FAILED")

add_test(quiet_on_stream_n2_w1_z1_uuid "env" "ROCSHMEM_SLR_NP=2" "ROCSHMEM_MAX_NUM_CONTEXTS=1" "ROCSHMEM_HEAP_SIZE=6442450944" "ROCSHMEM_TEST_UUID=1" "../../../../share/rocshmem/test_wrapper.sh" "quiet_on_stream_n2_w1_z1_uuid" "../../../rocshmem_functional_tests" "-a" "96" "-w" "1" "-z" "1" "-localbuftype" "heap")
set_tests_properties(quiet_on_stream_n2_w1_z1_uuid PROPERTIES
    TIMEOUT 300
    SKIP_RETURN_CODE 125
    LABELS "functional;backend_all;gpu_all;variant_uuid;launcher_slr;COLLECTIVE"
    PROCESSORS 2
    WILL_FAIL FALSE
    FAIL_REGULAR_EXPRESSION "FAILED")

add_test(sync_all_on_stream_n2_w1_z1_uuid "env" "ROCSHMEM_SLR_NP=2" "ROCSHMEM_MAX_NUM_CONTEXTS=1" "ROCSHMEM_HEAP_SIZE=6442450944" "ROCSHMEM_TEST_UUID=1" "../../../../share/rocshmem/test_wrapper.sh" "sync_all_on_stream_n2_w1_z1_uuid" "../../../rocshmem_functional_tests" "-a" "97" "-w" "1" "-z" "1" "-localbuftype" "heap")
set_tests_properties(sync_all_on_stream_n2_w1_z1_uuid PROPERTIES
    TIMEOUT 300
    SKIP_RETURN_CODE 125
    LABELS "functional;backend_all;gpu_all;variant_uuid;launcher_slr;COLLECTIVE"
    PROCESSORS 2
    WILL_FAIL FALSE
    FAIL_REGULAR_EXPRESSION "FAILED")

add_test(reduce_on_stream_n2_w1_z1_1048576B_uuid "env" "ROCSHMEM_SLR_NP=2" "ROCSHMEM_MAX_NUM_CONTEXTS=1" "ROCSHMEM_HEAP_SIZE=6442450944" "ROCSHMEM_TEST_UUID=1" "ROCSHMEM_MAX_NUM_CONTEXTS=1024" "ROCSHMEM_MAX_NUM_HOST_CONTEXTS=1024" "../../../../share/rocshmem/test_wrapper.sh" "reduce_on_stream_n2_w1_z1_1048576B_uuid" "../../../rocshmem_functional_tests" "-a" "117" "-w" "1" "-z" "1" "-s" "1048576" "-localbuftype" "heap")
set_tests_properties(reduce_on_stream_n2_w1_z1_1048576B_uuid PROPERTIES
    TIMEOUT 300
    SKIP_RETURN_CODE 125
    LABELS "functional;backend_all;gpu_all;variant_uuid;launcher_slr;COLLECTIVE"
    PROCESSORS 2
    WILL_FAIL FALSE
    FAIL_REGULAR_EXPRESSION "FAILED")

add_test(alltoallmem_on_stream_n2_w1_z64_1048576B_uuid "env" "ROCSHMEM_SLR_NP=2" "ROCSHMEM_MAX_NUM_CONTEXTS=1" "ROCSHMEM_HEAP_SIZE=6442450944" "ROCSHMEM_TEST_UUID=1" "../../../../share/rocshmem/test_wrapper.sh" "alltoallmem_on_stream_n2_w1_z64_1048576B_uuid" "../../../rocshmem_functional_tests" "-a" "76" "-w" "1" "-z" "64" "-s" "1048576" "-localbuftype" "heap")
set_tests_properties(alltoallmem_on_stream_n2_w1_z64_1048576B_uuid PROPERTIES
    TIMEOUT 300
    SKIP_RETURN_CODE 125
    LABELS "functional;backend_all;gpu_all;variant_uuid;launcher_slr;COLLECTIVE"
    PROCESSORS 2
    WILL_FAIL FALSE
    FAIL_REGULAR_EXPRESSION "FAILED")

add_test(broadcastmem_on_stream_n2_w1_z64_1048576B_uuid "env" "ROCSHMEM_SLR_NP=2" "ROCSHMEM_MAX_NUM_CONTEXTS=1" "ROCSHMEM_HEAP_SIZE=6442450944" "ROCSHMEM_TEST_UUID=1" "../../../../share/rocshmem/test_wrapper.sh" "broadcastmem_on_stream_n2_w1_z64_1048576B_uuid" "../../../rocshmem_functional_tests" "-a" "78" "-w" "1" "-z" "64" "-s" "1048576" "-localbuftype" "heap")
set_tests_properties(broadcastmem_on_stream_n2_w1_z64_1048576B_uuid PROPERTIES
    TIMEOUT 300
    SKIP_RETURN_CODE 125
    LABELS "functional;backend_all;gpu_all;variant_uuid;launcher_slr;COLLECTIVE"
    PROCESSORS 2
    WILL_FAIL FALSE
    FAIL_REGULAR_EXPRESSION "FAILED")

add_test(device_bitcode_n2_w1_z1_uuid "env" "ROCSHMEM_SLR_NP=2" "ROCSHMEM_MAX_NUM_CONTEXTS=1" "ROCSHMEM_HEAP_SIZE=6442450944" "ROCSHMEM_TEST_UUID=1" "../../../../share/rocshmem/test_wrapper.sh" "device_bitcode_n2_w1_z1_uuid" "../../../rocshmem_functional_tests" "-a" "93" "-w" "1" "-z" "1" "-localbuftype" "heap")
set_tests_properties(device_bitcode_n2_w1_z1_uuid PROPERTIES
    TIMEOUT 300
    SKIP_RETURN_CODE 125
    LABELS "functional;backend_all;gpu_all;variant_uuid;launcher_slr;OTHER"
    PROCESSORS 2
    WILL_FAIL FALSE
    FAIL_REGULAR_EXPRESSION "FAILED")

add_test(device_bitcode_n2_w32_z1024_uuid "env" "ROCSHMEM_SLR_NP=2" "ROCSHMEM_MAX_NUM_CONTEXTS=32" "ROCSHMEM_HEAP_SIZE=6442450944" "ROCSHMEM_TEST_UUID=1" "../../../../share/rocshmem/test_wrapper.sh" "device_bitcode_n2_w32_z1024_uuid" "../../../rocshmem_functional_tests" "-a" "93" "-w" "32" "-z" "1024" "-localbuftype" "heap")
set_tests_properties(device_bitcode_n2_w32_z1024_uuid PROPERTIES
    TIMEOUT 300
    SKIP_RETURN_CODE 125
    LABELS "functional;backend_all;gpu_all;variant_uuid;launcher_slr;OTHER"
    PROCESSORS 2
    WILL_FAIL FALSE
    FAIL_REGULAR_EXPRESSION "FAILED")

add_test(device_bitcode_n4_w16_z256_uuid "env" "ROCSHMEM_SLR_NP=4" "ROCSHMEM_MAX_NUM_CONTEXTS=16" "ROCSHMEM_HEAP_SIZE=6442450944" "ROCSHMEM_TEST_UUID=1" "../../../../share/rocshmem/test_wrapper.sh" "device_bitcode_n4_w16_z256_uuid" "../../../rocshmem_functional_tests" "-a" "93" "-w" "16" "-z" "256" "-localbuftype" "heap")
set_tests_properties(device_bitcode_n4_w16_z256_uuid PROPERTIES
    TIMEOUT 300
    SKIP_RETURN_CODE 125
    LABELS "functional;backend_all;gpu_all;variant_uuid;launcher_slr;OTHER"
    PROCESSORS 4
    WILL_FAIL FALSE
    FAIL_REGULAR_EXPRESSION "FAILED")

add_test(device_bitcode_n8_w16_z128_uuid "env" "ROCSHMEM_SLR_NP=8" "ROCSHMEM_MAX_NUM_CONTEXTS=16" "ROCSHMEM_HEAP_SIZE=6442450944" "ROCSHMEM_TEST_UUID=1" "../../../../share/rocshmem/test_wrapper.sh" "device_bitcode_n8_w16_z128_uuid" "../../../rocshmem_functional_tests" "-a" "93" "-w" "16" "-z" "128" "-localbuftype" "heap")
set_tests_properties(device_bitcode_n8_w16_z128_uuid PROPERTIES
    TIMEOUT 300
    SKIP_RETURN_CODE 125
    LABELS "functional;backend_all;gpu_all;variant_uuid;launcher_slr;OTHER"
    PROCESSORS 8
    WILL_FAIL FALSE
    FAIL_REGULAR_EXPRESSION "FAILED")

add_test(library_info_n2_w1_z1_uuid "env" "ROCSHMEM_SLR_NP=2" "ROCSHMEM_MAX_NUM_CONTEXTS=1" "ROCSHMEM_HEAP_SIZE=6442450944" "ROCSHMEM_TEST_UUID=1" "../../../../share/rocshmem/test_wrapper.sh" "library_info_n2_w1_z1_uuid" "../../../rocshmem_functional_tests" "-a" "94" "-w" "1" "-z" "1" "-localbuftype" "heap")
set_tests_properties(library_info_n2_w1_z1_uuid PROPERTIES
    TIMEOUT 300
    SKIP_RETURN_CODE 125
    LABELS "functional;backend_all;gpu_all;variant_uuid;launcher_slr;OTHER"
    PROCESSORS 2
    WILL_FAIL FALSE
    FAIL_REGULAR_EXPRESSION "FAILED")

add_test(hipmodule_init_n2_w1_z1_uuid "env" "ROCSHMEM_SLR_NP=2" "ROCSHMEM_MAX_NUM_CONTEXTS=1" "ROCSHMEM_HEAP_SIZE=6442450944" "ROCSHMEM_TEST_UUID=1" "../../../../share/rocshmem/test_wrapper.sh" "hipmodule_init_n2_w1_z1_uuid" "../../../rocshmem_functional_tests" "-a" "89" "-w" "1" "-z" "1" "-localbuftype" "heap")
set_tests_properties(hipmodule_init_n2_w1_z1_uuid PROPERTIES
    TIMEOUT 300
    SKIP_RETURN_CODE 125
    LABELS "functional;backend_all;gpu_all;variant_uuid;launcher_slr;OTHER"
    PROCESSORS 2
    WILL_FAIL FALSE
    FAIL_REGULAR_EXPRESSION "FAILED")

add_test(pingpong_n2_w1_z1_uuid "env" "ROCSHMEM_SLR_NP=2" "ROCSHMEM_MAX_NUM_CONTEXTS=1" "ROCSHMEM_HEAP_SIZE=6442450944" "ROCSHMEM_TEST_UUID=1" "../../../../share/rocshmem/test_wrapper.sh" "pingpong_n2_w1_z1_uuid" "../../../rocshmem_functional_tests" "-a" "12" "-w" "1" "-z" "1" "-localbuftype" "heap")
set_tests_properties(pingpong_n2_w1_z1_uuid PROPERTIES
    TIMEOUT 300
    SKIP_RETURN_CODE 125
    LABELS "functional;backend_all;gpu_all;variant_uuid;launcher_slr;OTHER"
    PROCESSORS 2
    WILL_FAIL FALSE
    FAIL_REGULAR_EXPRESSION "FAILED")

add_test(pingpong_n2_w8_z1_uuid "env" "ROCSHMEM_SLR_NP=2" "ROCSHMEM_MAX_NUM_CONTEXTS=8" "ROCSHMEM_HEAP_SIZE=6442450944" "ROCSHMEM_TEST_UUID=1" "../../../../share/rocshmem/test_wrapper.sh" "pingpong_n2_w8_z1_uuid" "../../../rocshmem_functional_tests" "-a" "12" "-w" "8" "-z" "1" "-localbuftype" "heap")
set_tests_properties(pingpong_n2_w8_z1_uuid PROPERTIES
    TIMEOUT 300
    SKIP_RETURN_CODE 125
    LABELS "functional;backend_all;gpu_all;variant_uuid;launcher_slr;OTHER"
    PROCESSORS 2
    WILL_FAIL FALSE
    FAIL_REGULAR_EXPRESSION "FAILED")

add_test(pingpong_n2_w32_z1_uuid "env" "ROCSHMEM_SLR_NP=2" "ROCSHMEM_MAX_NUM_CONTEXTS=32" "ROCSHMEM_HEAP_SIZE=6442450944" "ROCSHMEM_TEST_UUID=1" "../../../../share/rocshmem/test_wrapper.sh" "pingpong_n2_w32_z1_uuid" "../../../rocshmem_functional_tests" "-a" "12" "-w" "32" "-z" "1" "-localbuftype" "heap")
set_tests_properties(pingpong_n2_w32_z1_uuid PROPERTIES
    TIMEOUT 300
    SKIP_RETURN_CODE 125
    LABELS "functional;backend_all;gpu_all;variant_uuid;launcher_slr;OTHER"
    PROCESSORS 2
    WILL_FAIL FALSE
    FAIL_REGULAR_EXPRESSION "FAILED")

add_test(pingall_n2_w1_z1_uuid "env" "ROCSHMEM_SLR_NP=2" "ROCSHMEM_MAX_NUM_CONTEXTS=1" "ROCSHMEM_HEAP_SIZE=6442450944" "ROCSHMEM_TEST_UUID=1" "../../../../share/rocshmem/test_wrapper.sh" "pingall_n2_w1_z1_uuid" "../../../rocshmem_functional_tests" "-a" "48" "-w" "1" "-z" "1" "-localbuftype" "heap")
set_tests_properties(pingall_n2_w1_z1_uuid PROPERTIES
    TIMEOUT 300
    SKIP_RETURN_CODE 125
    LABELS "functional;backend_all;gpu_all;variant_uuid;launcher_slr;OTHER"
    PROCESSORS 2
    WILL_FAIL FALSE
    FAIL_REGULAR_EXPRESSION "FAILED")

add_test(pingall_n2_w8_z1_uuid "env" "ROCSHMEM_SLR_NP=2" "ROCSHMEM_MAX_NUM_CONTEXTS=8" "ROCSHMEM_HEAP_SIZE=6442450944" "ROCSHMEM_TEST_UUID=1" "../../../../share/rocshmem/test_wrapper.sh" "pingall_n2_w8_z1_uuid" "../../../rocshmem_functional_tests" "-a" "48" "-w" "8" "-z" "1" "-localbuftype" "heap")
set_tests_properties(pingall_n2_w8_z1_uuid PROPERTIES
    TIMEOUT 300
    SKIP_RETURN_CODE 125
    LABELS "functional;backend_all;gpu_all;variant_uuid;launcher_slr;OTHER"
    PROCESSORS 2
    WILL_FAIL FALSE
    FAIL_REGULAR_EXPRESSION "FAILED")

add_test(pingall_n2_w32_z1_uuid "env" "ROCSHMEM_SLR_NP=2" "ROCSHMEM_MAX_NUM_CONTEXTS=32" "ROCSHMEM_HEAP_SIZE=6442450944" "ROCSHMEM_TEST_UUID=1" "../../../../share/rocshmem/test_wrapper.sh" "pingall_n2_w32_z1_uuid" "../../../rocshmem_functional_tests" "-a" "48" "-w" "32" "-z" "1" "-localbuftype" "heap")
set_tests_properties(pingall_n2_w32_z1_uuid PROPERTIES
    TIMEOUT 300
    SKIP_RETURN_CODE 125
    LABELS "functional;backend_all;gpu_all;variant_uuid;launcher_slr;OTHER"
    PROCESSORS 2
    WILL_FAIL FALSE
    FAIL_REGULAR_EXPRESSION "FAILED")

add_test(shmemptr_n2_w1_z1_8B_uuid "env" "ROCSHMEM_SLR_NP=2" "ROCSHMEM_MAX_NUM_CONTEXTS=1" "ROCSHMEM_HEAP_SIZE=6442450944" "ROCSHMEM_TEST_UUID=1" "../../../../share/rocshmem/test_wrapper.sh" "shmemptr_n2_w1_z1_8B_uuid" "../../../rocshmem_functional_tests" "-a" "21" "-w" "1" "-z" "1" "-s" "8" "-localbuftype" "heap")
set_tests_properties(shmemptr_n2_w1_z1_8B_uuid PROPERTIES
    TIMEOUT 300
    SKIP_RETURN_CODE 125
    LABELS "functional;backend_all;gpu_all;variant_uuid;launcher_slr;OTHER"
    PROCESSORS 2
    WILL_FAIL FALSE
    FAIL_REGULAR_EXPRESSION "FAILED")

add_test(shmemptr_n2_w1_z1024_8B_uuid "env" "ROCSHMEM_SLR_NP=2" "ROCSHMEM_MAX_NUM_CONTEXTS=1" "ROCSHMEM_HEAP_SIZE=6442450944" "ROCSHMEM_TEST_UUID=1" "../../../../share/rocshmem/test_wrapper.sh" "shmemptr_n2_w1_z1024_8B_uuid" "../../../rocshmem_functional_tests" "-a" "21" "-w" "1" "-z" "1024" "-s" "8" "-localbuftype" "heap")
set_tests_properties(shmemptr_n2_w1_z1024_8B_uuid PROPERTIES
    TIMEOUT 300
    SKIP_RETURN_CODE 125
    LABELS "functional;backend_all;gpu_all;variant_uuid;launcher_slr;OTHER"
    PROCESSORS 2
    WILL_FAIL FALSE
    FAIL_REGULAR_EXPRESSION "FAILED")

add_test(shmemptr_n2_w8_z1_8B_uuid "env" "ROCSHMEM_SLR_NP=2" "ROCSHMEM_MAX_NUM_CONTEXTS=8" "ROCSHMEM_HEAP_SIZE=6442450944" "ROCSHMEM_TEST_UUID=1" "../../../../share/rocshmem/test_wrapper.sh" "shmemptr_n2_w8_z1_8B_uuid" "../../../rocshmem_functional_tests" "-a" "21" "-w" "8" "-z" "1" "-s" "8" "-localbuftype" "heap")
set_tests_properties(shmemptr_n2_w8_z1_8B_uuid PROPERTIES
    TIMEOUT 300
    SKIP_RETURN_CODE 125
    LABELS "functional;backend_all;gpu_all;variant_uuid;launcher_slr;OTHER"
    PROCESSORS 2
    WILL_FAIL FALSE
    FAIL_REGULAR_EXPRESSION "FAILED")

add_test(shmemptr_n2_w16_z128_8B_uuid "env" "ROCSHMEM_SLR_NP=2" "ROCSHMEM_MAX_NUM_CONTEXTS=16" "ROCSHMEM_HEAP_SIZE=6442450944" "ROCSHMEM_TEST_UUID=1" "../../../../share/rocshmem/test_wrapper.sh" "shmemptr_n2_w16_z128_8B_uuid" "../../../rocshmem_functional_tests" "-a" "21" "-w" "16" "-z" "128" "-s" "8" "-localbuftype" "heap")
set_tests_properties(shmemptr_n2_w16_z128_8B_uuid PROPERTIES
    TIMEOUT 300
    SKIP_RETURN_CODE 125
    LABELS "functional;backend_all;gpu_all;variant_uuid;launcher_slr;OTHER"
    PROCESSORS 2
    WILL_FAIL FALSE
    FAIL_REGULAR_EXPRESSION "FAILED")

add_test(flood_put_n2_w64_z1024_uuid "env" "ROCSHMEM_SLR_NP=2" "ROCSHMEM_MAX_NUM_CONTEXTS=64" "ROCSHMEM_HEAP_SIZE=6442450944" "ROCSHMEM_TEST_UUID=1" "../../../../share/rocshmem/test_wrapper.sh" "flood_put_n2_w64_z1024_uuid" "../../../rocshmem_functional_tests" "-a" "83" "-w" "64" "-z" "1024" "-localbuftype" "heap")
set_tests_properties(flood_put_n2_w64_z1024_uuid PROPERTIES
    TIMEOUT 300
    SKIP_RETURN_CODE 125
    LABELS "functional;backend_ipc;backend_gda;gpu_all;variant_uuid;launcher_slr;FLOOD"
    PROCESSORS 2
    WILL_FAIL FALSE
    FAIL_REGULAR_EXPRESSION "FAILED")

add_test(flood_put_n8_w64_z1024_uuid "env" "ROCSHMEM_SLR_NP=8" "ROCSHMEM_MAX_NUM_CONTEXTS=64" "ROCSHMEM_HEAP_SIZE=6442450944" "ROCSHMEM_TEST_UUID=1" "../../../../share/rocshmem/test_wrapper.sh" "flood_put_n8_w64_z1024_uuid" "../../../rocshmem_functional_tests" "-a" "83" "-w" "64" "-z" "1024" "-localbuftype" "heap")
set_tests_properties(flood_put_n8_w64_z1024_uuid PROPERTIES
    TIMEOUT 300
    SKIP_RETURN_CODE 125
    LABELS "functional;backend_ipc;backend_gda;gpu_all;variant_uuid;launcher_slr;FLOOD"
    PROCESSORS 8
    WILL_FAIL FALSE
    FAIL_REGULAR_EXPRESSION "FAILED")

add_test(flood_putnbi_n8_w64_z1024_uuid "env" "ROCSHMEM_SLR_NP=8" "ROCSHMEM_MAX_NUM_CONTEXTS=64" "ROCSHMEM_HEAP_SIZE=6442450944" "ROCSHMEM_TEST_UUID=1" "../../../../share/rocshmem/test_wrapper.sh" "flood_putnbi_n8_w64_z1024_uuid" "../../../rocshmem_functional_tests" "-a" "84" "-w" "64" "-z" "1024" "-localbuftype" "heap")
set_tests_properties(flood_putnbi_n8_w64_z1024_uuid PROPERTIES
    TIMEOUT 300
    SKIP_RETURN_CODE 125
    LABELS "functional;backend_ipc;backend_gda;gpu_all;variant_uuid;launcher_slr;FLOOD"
    PROCESSORS 8
    WILL_FAIL FALSE
    FAIL_REGULAR_EXPRESSION "FAILED")

add_test(flood_p_n8_w64_z1024_uuid "env" "ROCSHMEM_SLR_NP=8" "ROCSHMEM_MAX_NUM_CONTEXTS=64" "ROCSHMEM_HEAP_SIZE=6442450944" "ROCSHMEM_TEST_UUID=1" "../../../../share/rocshmem/test_wrapper.sh" "flood_p_n8_w64_z1024_uuid" "../../../rocshmem_functional_tests" "-a" "85" "-w" "64" "-z" "1024" "-localbuftype" "heap")
set_tests_properties(flood_p_n8_w64_z1024_uuid PROPERTIES
    TIMEOUT 300
    SKIP_RETURN_CODE 125
    LABELS "functional;backend_ipc;backend_gda;gpu_all;variant_uuid;launcher_slr;FLOOD"
    PROCESSORS 8
    WILL_FAIL FALSE
    FAIL_REGULAR_EXPRESSION "FAILED")

add_test(flood_get_n2_w64_z1024_uuid "env" "ROCSHMEM_SLR_NP=2" "ROCSHMEM_MAX_NUM_CONTEXTS=64" "ROCSHMEM_HEAP_SIZE=6442450944" "ROCSHMEM_TEST_UUID=1" "../../../../share/rocshmem/test_wrapper.sh" "flood_get_n2_w64_z1024_uuid" "../../../rocshmem_functional_tests" "-a" "86" "-w" "64" "-z" "1024" "-localbuftype" "heap")
set_tests_properties(flood_get_n2_w64_z1024_uuid PROPERTIES
    TIMEOUT 300
    SKIP_RETURN_CODE 125
    LABELS "functional;backend_ipc;backend_gda;gpu_all;variant_uuid;launcher_slr;FLOOD"
    PROCESSORS 2
    WILL_FAIL FALSE
    FAIL_REGULAR_EXPRESSION "FAILED")

add_test(flood_get_n8_w64_z1024_uuid "env" "ROCSHMEM_SLR_NP=8" "ROCSHMEM_MAX_NUM_CONTEXTS=64" "ROCSHMEM_HEAP_SIZE=6442450944" "ROCSHMEM_TEST_UUID=1" "../../../../share/rocshmem/test_wrapper.sh" "flood_get_n8_w64_z1024_uuid" "../../../rocshmem_functional_tests" "-a" "86" "-w" "64" "-z" "1024" "-localbuftype" "heap")
set_tests_properties(flood_get_n8_w64_z1024_uuid PROPERTIES
    TIMEOUT 300
    SKIP_RETURN_CODE 125
    LABELS "functional;backend_ipc;backend_gda;gpu_all;variant_uuid;launcher_slr;FLOOD"
    PROCESSORS 8
    WILL_FAIL FALSE
    FAIL_REGULAR_EXPRESSION "FAILED")

add_test(flood_getnbi_n8_w64_z1024_uuid "env" "ROCSHMEM_SLR_NP=8" "ROCSHMEM_MAX_NUM_CONTEXTS=64" "ROCSHMEM_HEAP_SIZE=6442450944" "ROCSHMEM_TEST_UUID=1" "../../../../share/rocshmem/test_wrapper.sh" "flood_getnbi_n8_w64_z1024_uuid" "../../../rocshmem_functional_tests" "-a" "87" "-w" "64" "-z" "1024" "-localbuftype" "heap")
set_tests_properties(flood_getnbi_n8_w64_z1024_uuid PROPERTIES
    TIMEOUT 300
    SKIP_RETURN_CODE 125
    LABELS "functional;backend_ipc;backend_gda;gpu_all;variant_uuid;launcher_slr;FLOOD"
    PROCESSORS 8
    WILL_FAIL FALSE
    FAIL_REGULAR_EXPRESSION "FAILED")

add_test(flood_g_n8_w64_z1024_uuid "env" "ROCSHMEM_SLR_NP=8" "ROCSHMEM_MAX_NUM_CONTEXTS=64" "ROCSHMEM_HEAP_SIZE=6442450944" "ROCSHMEM_TEST_UUID=1" "../../../../share/rocshmem/test_wrapper.sh" "flood_g_n8_w64_z1024_uuid" "../../../rocshmem_functional_tests" "-a" "88" "-w" "64" "-z" "1024" "-localbuftype" "heap")
set_tests_properties(flood_g_n8_w64_z1024_uuid PROPERTIES
    TIMEOUT 300
    SKIP_RETURN_CODE 125
    LABELS "functional;backend_ipc;gpu_all;variant_uuid;launcher_slr;FLOOD"
    PROCESSORS 8
    WILL_FAIL FALSE
    FAIL_REGULAR_EXPRESSION "FAILED")

add_test(flood_add_n2_w64_z1024_uuid "env" "ROCSHMEM_SLR_NP=2" "ROCSHMEM_MAX_NUM_CONTEXTS=64" "ROCSHMEM_HEAP_SIZE=6442450944" "ROCSHMEM_TEST_UUID=1" "../../../../share/rocshmem/test_wrapper.sh" "flood_add_n2_w64_z1024_uuid" "../../../rocshmem_functional_tests" "-a" "90" "-w" "64" "-z" "1024" "-localbuftype" "heap")
set_tests_properties(flood_add_n2_w64_z1024_uuid PROPERTIES
    TIMEOUT 300
    SKIP_RETURN_CODE 125
    LABELS "functional;backend_ipc;backend_gda;gpu_all;variant_uuid;launcher_slr;FLOOD"
    PROCESSORS 2
    WILL_FAIL FALSE
    FAIL_REGULAR_EXPRESSION "FAILED")

add_test(flood_add_n8_w64_z1024_uuid "env" "ROCSHMEM_SLR_NP=8" "ROCSHMEM_MAX_NUM_CONTEXTS=64" "ROCSHMEM_HEAP_SIZE=6442450944" "ROCSHMEM_TEST_UUID=1" "../../../../share/rocshmem/test_wrapper.sh" "flood_add_n8_w64_z1024_uuid" "../../../rocshmem_functional_tests" "-a" "90" "-w" "64" "-z" "1024" "-localbuftype" "heap")
set_tests_properties(flood_add_n8_w64_z1024_uuid PROPERTIES
    TIMEOUT 300
    SKIP_RETURN_CODE 125
    LABELS "functional;backend_ipc;backend_gda;gpu_all;variant_uuid;launcher_slr;FLOOD"
    PROCESSORS 8
    WILL_FAIL FALSE
    FAIL_REGULAR_EXPRESSION "FAILED")

add_test(flood_fadd_n8_w64_z1024_uuid "env" "ROCSHMEM_SLR_NP=8" "ROCSHMEM_MAX_NUM_CONTEXTS=64" "ROCSHMEM_HEAP_SIZE=6442450944" "ROCSHMEM_TEST_UUID=1" "../../../../share/rocshmem/test_wrapper.sh" "flood_fadd_n8_w64_z1024_uuid" "../../../rocshmem_functional_tests" "-a" "91" "-w" "64" "-z" "1024" "-localbuftype" "heap")
set_tests_properties(flood_fadd_n8_w64_z1024_uuid PROPERTIES
    TIMEOUT 300
    SKIP_RETURN_CODE 125
    LABELS "functional;backend_ipc;backend_gda;gpu_all;variant_uuid;launcher_slr;FLOOD"
    PROCESSORS 8
    WILL_FAIL FALSE
    FAIL_REGULAR_EXPRESSION "FAILED")

add_test(flood_waitadd_n8_w64_z1024_uuid "env" "ROCSHMEM_SLR_NP=8" "ROCSHMEM_MAX_NUM_CONTEXTS=64" "ROCSHMEM_HEAP_SIZE=6442450944" "ROCSHMEM_TEST_UUID=1" "../../../../share/rocshmem/test_wrapper.sh" "flood_waitadd_n8_w64_z1024_uuid" "../../../rocshmem_functional_tests" "-a" "92" "-w" "64" "-z" "1024" "-localbuftype" "heap")
set_tests_properties(flood_waitadd_n8_w64_z1024_uuid PROPERTIES
    TIMEOUT 300
    SKIP_RETURN_CODE 125
    LABELS "functional;backend_ipc;backend_gda;gpu_all;variant_uuid;launcher_slr;FLOOD"
    PROCESSORS 8
    WILL_FAIL FALSE
    FAIL_REGULAR_EXPRESSION "FAILED")

add_test(host_ctx_create_n2_w1_z1_uuid "env" "ROCSHMEM_SLR_NP=2" "ROCSHMEM_MAX_NUM_CONTEXTS=1" "ROCSHMEM_HEAP_SIZE=6442450944" "ROCSHMEM_TEST_UUID=1" "../../../../share/rocshmem/test_wrapper.sh" "host_ctx_create_n2_w1_z1_uuid" "../../../rocshmem_functional_tests" "-a" "118" "-w" "1" "-z" "1" "-localbuftype" "heap")
set_tests_properties(host_ctx_create_n2_w1_z1_uuid PROPERTIES
    TIMEOUT 300
    SKIP_RETURN_CODE 125
    LABELS "functional;backend_all;gpu_all;variant_uuid;launcher_slr;CTX"
    PROCESSORS 2
    WILL_FAIL FALSE
    FAIL_REGULAR_EXPRESSION "FAILED")

add_test(teamctxinfra_n2_w1_z1_uuid "env" "ROCSHMEM_SLR_NP=2" "ROCSHMEM_MAX_NUM_CONTEXTS=1" "ROCSHMEM_HEAP_SIZE=6442450944" "ROCSHMEM_TEST_UUID=1" "ROCSHMEM_MAX_NUM_CONTEXTS=1024" "../../../../share/rocshmem/test_wrapper.sh" "teamctxinfra_n2_w1_z1_uuid" "../../../rocshmem_functional_tests" "-a" "38" "-w" "1" "-z" "1" "-localbuftype" "heap")
set_tests_properties(teamctxinfra_n2_w1_z1_uuid PROPERTIES
    TIMEOUT 300
    SKIP_RETURN_CODE 125
    LABELS "functional;backend_all;gpu_all;variant_uuid;launcher_slr;TEAM"
    PROCESSORS 2
    WILL_FAIL FALSE
    FAIL_REGULAR_EXPRESSION "FAILED")

add_test(teamctxsingleinfra_n2_w1_z1_uuid "env" "ROCSHMEM_SLR_NP=2" "ROCSHMEM_MAX_NUM_CONTEXTS=1" "ROCSHMEM_HEAP_SIZE=6442450944" "ROCSHMEM_TEST_UUID=1" "ROCSHMEM_MAX_NUM_CONTEXTS=1024" "../../../../share/rocshmem/test_wrapper.sh" "teamctxsingleinfra_n2_w1_z1_uuid" "../../../rocshmem_functional_tests" "-a" "73" "-w" "1" "-z" "1" "-localbuftype" "heap")
set_tests_properties(teamctxsingleinfra_n2_w1_z1_uuid PROPERTIES
    TIMEOUT 300
    SKIP_RETURN_CODE 125
    LABELS "functional;backend_all;gpu_all;variant_uuid;launcher_slr;TEAM"
    PROCESSORS 2
    WILL_FAIL FALSE
    FAIL_REGULAR_EXPRESSION "FAILED")

add_test(teamctxblockinfra_n4_w1_z1_uuid "env" "ROCSHMEM_SLR_NP=4" "ROCSHMEM_MAX_NUM_CONTEXTS=1" "ROCSHMEM_HEAP_SIZE=6442450944" "ROCSHMEM_TEST_UUID=1" "ROCSHMEM_MAX_NUM_CONTEXTS=1024" "../../../../share/rocshmem/test_wrapper.sh" "teamctxblockinfra_n4_w1_z1_uuid" "../../../rocshmem_functional_tests" "-a" "74" "-w" "1" "-z" "1" "-localbuftype" "heap")
set_tests_properties(teamctxblockinfra_n4_w1_z1_uuid PROPERTIES
    TIMEOUT 300
    SKIP_RETURN_CODE 125
    LABELS "functional;backend_all;gpu_all;variant_uuid;launcher_slr;TEAM"
    PROCESSORS 4
    WILL_FAIL FALSE
    FAIL_REGULAR_EXPRESSION "FAILED")

add_test(teamctxblockinfra_n5_w1_z1_uuid "env" "ROCSHMEM_SLR_NP=5" "ROCSHMEM_MAX_NUM_CONTEXTS=1" "ROCSHMEM_HEAP_SIZE=6442450944" "ROCSHMEM_TEST_UUID=1" "ROCSHMEM_MAX_NUM_CONTEXTS=1024" "../../../../share/rocshmem/test_wrapper.sh" "teamctxblockinfra_n5_w1_z1_uuid" "../../../rocshmem_functional_tests" "-a" "74" "-w" "1" "-z" "1" "-localbuftype" "heap")
set_tests_properties(teamctxblockinfra_n5_w1_z1_uuid PROPERTIES
    TIMEOUT 300
    SKIP_RETURN_CODE 125
    LABELS "functional;backend_all;gpu_all;variant_uuid;launcher_slr;TEAM"
    PROCESSORS 5
    WILL_FAIL FALSE
    FAIL_REGULAR_EXPRESSION "FAILED")

add_test(teamctxoddeveninfra_n4_w1_z1_uuid "env" "ROCSHMEM_SLR_NP=4" "ROCSHMEM_MAX_NUM_CONTEXTS=1" "ROCSHMEM_HEAP_SIZE=6442450944" "ROCSHMEM_TEST_UUID=1" "ROCSHMEM_MAX_NUM_CONTEXTS=1024" "../../../../share/rocshmem/test_wrapper.sh" "teamctxoddeveninfra_n4_w1_z1_uuid" "../../../rocshmem_functional_tests" "-a" "75" "-w" "1" "-z" "1" "-localbuftype" "heap")
set_tests_properties(teamctxoddeveninfra_n4_w1_z1_uuid PROPERTIES
    TIMEOUT 300
    SKIP_RETURN_CODE 125
    LABELS "functional;backend_all;gpu_all;variant_uuid;launcher_slr;TEAM"
    PROCESSORS 4
    WILL_FAIL FALSE
    FAIL_REGULAR_EXPRESSION "FAILED")

add_test(teamctxoddeveninfra_n5_w1_z1_uuid "env" "ROCSHMEM_SLR_NP=5" "ROCSHMEM_MAX_NUM_CONTEXTS=1" "ROCSHMEM_HEAP_SIZE=6442450944" "ROCSHMEM_TEST_UUID=1" "ROCSHMEM_MAX_NUM_CONTEXTS=1024" "../../../../share/rocshmem/test_wrapper.sh" "teamctxoddeveninfra_n5_w1_z1_uuid" "../../../rocshmem_functional_tests" "-a" "75" "-w" "1" "-z" "1" "-localbuftype" "heap")
set_tests_properties(teamctxoddeveninfra_n5_w1_z1_uuid PROPERTIES
    TIMEOUT 300
    SKIP_RETURN_CODE 125
    LABELS "functional;backend_all;gpu_all;variant_uuid;launcher_slr;TEAM"
    PROCESSORS 5
    WILL_FAIL FALSE
    FAIL_REGULAR_EXPRESSION "FAILED")

add_test(teamctxsharedinfra_n2_w1_z1_uuid "env" "ROCSHMEM_SLR_NP=2" "ROCSHMEM_MAX_NUM_CONTEXTS=1" "ROCSHMEM_HEAP_SIZE=6442450944" "ROCSHMEM_TEST_UUID=1" "ROCSHMEM_MAX_NUM_CONTEXTS=1024" "../../../../share/rocshmem/test_wrapper.sh" "teamctxsharedinfra_n2_w1_z1_uuid" "../../../rocshmem_functional_tests" "-a" "95" "-w" "1" "-z" "1" "-localbuftype" "heap")
set_tests_properties(teamctxsharedinfra_n2_w1_z1_uuid PROPERTIES
    TIMEOUT 300
    SKIP_RETURN_CODE 125
    LABELS "functional;backend_all;gpu_all;variant_uuid;launcher_slr;TEAM"
    PROCESSORS 2
    WILL_FAIL FALSE
    FAIL_REGULAR_EXPRESSION "FAILED")

add_test(teamctxsharedinfra_n5_w1_z1_uuid "env" "ROCSHMEM_SLR_NP=5" "ROCSHMEM_MAX_NUM_CONTEXTS=1" "ROCSHMEM_HEAP_SIZE=6442450944" "ROCSHMEM_TEST_UUID=1" "ROCSHMEM_MAX_NUM_CONTEXTS=1024" "../../../../share/rocshmem/test_wrapper.sh" "teamctxsharedinfra_n5_w1_z1_uuid" "../../../rocshmem_functional_tests" "-a" "95" "-w" "1" "-z" "1" "-localbuftype" "heap")
set_tests_properties(teamctxsharedinfra_n5_w1_z1_uuid PROPERTIES
    TIMEOUT 300
    SKIP_RETURN_CODE 125
    LABELS "functional;backend_all;gpu_all;variant_uuid;launcher_slr;TEAM"
    PROCESSORS 5
    WILL_FAIL FALSE
    FAIL_REGULAR_EXPRESSION "FAILED")

add_test(teamctxsubsetparentinfra_n4_w1_z1_uuid "env" "ROCSHMEM_SLR_NP=4" "ROCSHMEM_MAX_NUM_CONTEXTS=1" "ROCSHMEM_HEAP_SIZE=6442450944" "ROCSHMEM_TEST_UUID=1" "ROCSHMEM_MAX_NUM_CONTEXTS=1024" "../../../../share/rocshmem/test_wrapper.sh" "teamctxsubsetparentinfra_n4_w1_z1_uuid" "../../../rocshmem_functional_tests" "-a" "98" "-w" "1" "-z" "1" "-localbuftype" "heap")
set_tests_properties(teamctxsubsetparentinfra_n4_w1_z1_uuid PROPERTIES
    TIMEOUT 300
    SKIP_RETURN_CODE 125
    LABELS "functional;backend_all;gpu_all;variant_uuid;launcher_slr;TEAM"
    PROCESSORS 4
    WILL_FAIL FALSE
    FAIL_REGULAR_EXPRESSION "FAILED")

add_test(teamctxsubsetparentinfra_n5_w1_z1_uuid "env" "ROCSHMEM_SLR_NP=5" "ROCSHMEM_MAX_NUM_CONTEXTS=1" "ROCSHMEM_HEAP_SIZE=6442450944" "ROCSHMEM_TEST_UUID=1" "ROCSHMEM_MAX_NUM_CONTEXTS=1024" "../../../../share/rocshmem/test_wrapper.sh" "teamctxsubsetparentinfra_n5_w1_z1_uuid" "../../../rocshmem_functional_tests" "-a" "98" "-w" "1" "-z" "1" "-localbuftype" "heap")
set_tests_properties(teamctxsubsetparentinfra_n5_w1_z1_uuid PROPERTIES
    TIMEOUT 300
    SKIP_RETURN_CODE 125
    LABELS "functional;backend_all;gpu_all;variant_uuid;launcher_slr;TEAM"
    PROCESSORS 5
    WILL_FAIL FALSE
    FAIL_REGULAR_EXPRESSION "FAILED")

add_test(fence_putwavesignal_n2_w1_z64_1048576B_uuid "env" "ROCSHMEM_SLR_NP=2" "ROCSHMEM_MAX_NUM_CONTEXTS=1" "ROCSHMEM_HEAP_SIZE=6442450944" "ROCSHMEM_TEST_UUID=1" "../../../../share/rocshmem/test_wrapper.sh" "fence_putwavesignal_n2_w1_z64_1048576B_uuid" "../../../rocshmem_functional_tests" "-a" "99" "-w" "1" "-z" "64" "-s" "1048576" "-localbuftype" "heap")
set_tests_properties(fence_putwavesignal_n2_w1_z64_1048576B_uuid PROPERTIES
    TIMEOUT 300
    SKIP_RETURN_CODE 125
    LABELS "functional;backend_ipc;backend_gda;gpu_all;variant_uuid;launcher_slr;FENCE"
    PROCESSORS 2
    WILL_FAIL FALSE
    FAIL_REGULAR_EXPRESSION "FAILED")

add_test(fence_putwavesignal_n2_w8_z256_1048576B_uuid "env" "ROCSHMEM_SLR_NP=2" "ROCSHMEM_MAX_NUM_CONTEXTS=8" "ROCSHMEM_HEAP_SIZE=6442450944" "ROCSHMEM_TEST_UUID=1" "../../../../share/rocshmem/test_wrapper.sh" "fence_putwavesignal_n2_w8_z256_1048576B_uuid" "../../../rocshmem_functional_tests" "-a" "99" "-w" "8" "-z" "256" "-s" "1048576" "-localbuftype" "heap")
set_tests_properties(fence_putwavesignal_n2_w8_z256_1048576B_uuid PROPERTIES
    TIMEOUT 300
    SKIP_RETURN_CODE 125
    LABELS "functional;backend_ipc;backend_gda;gpu_all;variant_uuid;launcher_slr;FENCE"
    PROCESSORS 2
    WILL_FAIL FALSE
    FAIL_REGULAR_EXPRESSION "FAILED")

add_test(fence_putwavesignal_n2_w32_z1024_65536B_uuid "env" "ROCSHMEM_SLR_NP=2" "ROCSHMEM_MAX_NUM_CONTEXTS=32" "ROCSHMEM_HEAP_SIZE=6442450944" "ROCSHMEM_TEST_UUID=1" "../../../../share/rocshmem/test_wrapper.sh" "fence_putwavesignal_n2_w32_z1024_65536B_uuid" "../../../rocshmem_functional_tests" "-a" "99" "-w" "32" "-z" "1024" "-s" "65536" "-localbuftype" "heap")
set_tests_properties(fence_putwavesignal_n2_w32_z1024_65536B_uuid PROPERTIES
    TIMEOUT 300
    SKIP_RETURN_CODE 125
    LABELS "functional;backend_ipc;backend_gda;gpu_all;variant_uuid;launcher_slr;FENCE"
    PROCESSORS 2
    WILL_FAIL FALSE
    FAIL_REGULAR_EXPRESSION "FAILED")

add_test(fence_putlargesmall_n2_w1_z64_4096B_uuid "env" "ROCSHMEM_SLR_NP=2" "ROCSHMEM_MAX_NUM_CONTEXTS=1" "ROCSHMEM_HEAP_SIZE=6442450944" "ROCSHMEM_TEST_UUID=1" "../../../../share/rocshmem/test_wrapper.sh" "fence_putlargesmall_n2_w1_z64_4096B_uuid" "../../../rocshmem_functional_tests" "-a" "100" "-w" "1" "-z" "64" "-s" "4096" "-localbuftype" "heap")
set_tests_properties(fence_putlargesmall_n2_w1_z64_4096B_uuid PROPERTIES
    TIMEOUT 300
    SKIP_RETURN_CODE 125
    LABELS "functional;backend_ipc;backend_gda;gpu_all;variant_uuid;launcher_slr;FENCE"
    PROCESSORS 2
    WILL_FAIL FALSE
    FAIL_REGULAR_EXPRESSION "FAILED")

add_test(fence_putlargesmall_n2_w8_z256_65536B_uuid "env" "ROCSHMEM_SLR_NP=2" "ROCSHMEM_MAX_NUM_CONTEXTS=8" "ROCSHMEM_HEAP_SIZE=6442450944" "ROCSHMEM_TEST_UUID=1" "../../../../share/rocshmem/test_wrapper.sh" "fence_putlargesmall_n2_w8_z256_65536B_uuid" "../../../rocshmem_functional_tests" "-a" "100" "-w" "8" "-z" "256" "-s" "65536" "-localbuftype" "heap")
set_tests_properties(fence_putlargesmall_n2_w8_z256_65536B_uuid PROPERTIES
    TIMEOUT 300
    SKIP_RETURN_CODE 125
    LABELS "functional;backend_ipc;backend_gda;gpu_all;variant_uuid;launcher_slr;FENCE"
    PROCESSORS 2
    WILL_FAIL FALSE
    FAIL_REGULAR_EXPRESSION "FAILED")

add_test(fence_fanout_n2_w1_z64_1048576B_uuid "env" "ROCSHMEM_SLR_NP=2" "ROCSHMEM_MAX_NUM_CONTEXTS=1" "ROCSHMEM_HEAP_SIZE=6442450944" "ROCSHMEM_TEST_UUID=1" "../../../../share/rocshmem/test_wrapper.sh" "fence_fanout_n2_w1_z64_1048576B_uuid" "../../../rocshmem_functional_tests" "-a" "101" "-w" "1" "-z" "64" "-s" "1048576" "-localbuftype" "heap")
set_tests_properties(fence_fanout_n2_w1_z64_1048576B_uuid PROPERTIES
    TIMEOUT 300
    SKIP_RETURN_CODE 125
    LABELS "functional;backend_ipc;backend_gda;gpu_all;variant_uuid;launcher_slr;FENCE"
    PROCESSORS 2
    WILL_FAIL FALSE
    FAIL_REGULAR_EXPRESSION "FAILED")

add_test(fence_fanout_n4_w4_z256_65536B_uuid "env" "ROCSHMEM_SLR_NP=4" "ROCSHMEM_MAX_NUM_CONTEXTS=4" "ROCSHMEM_HEAP_SIZE=6442450944" "ROCSHMEM_TEST_UUID=1" "../../../../share/rocshmem/test_wrapper.sh" "fence_fanout_n4_w4_z256_65536B_uuid" "../../../rocshmem_functional_tests" "-a" "101" "-w" "4" "-z" "256" "-s" "65536" "-localbuftype" "heap")
set_tests_properties(fence_fanout_n4_w4_z256_65536B_uuid PROPERTIES
    TIMEOUT 300
    SKIP_RETURN_CODE 125
    LABELS "functional;backend_ipc;backend_gda;gpu_all;variant_uuid;launcher_slr;FENCE"
    PROCESSORS 4
    WILL_FAIL FALSE
    FAIL_REGULAR_EXPRESSION "FAILED")

add_test(fence_fanout_n8_w8_z256_65536B_uuid "env" "ROCSHMEM_SLR_NP=8" "ROCSHMEM_MAX_NUM_CONTEXTS=8" "ROCSHMEM_HEAP_SIZE=6442450944" "ROCSHMEM_TEST_UUID=1" "../../../../share/rocshmem/test_wrapper.sh" "fence_fanout_n8_w8_z256_65536B_uuid" "../../../rocshmem_functional_tests" "-a" "101" "-w" "8" "-z" "256" "-s" "65536" "-localbuftype" "heap")
set_tests_properties(fence_fanout_n8_w8_z256_65536B_uuid PROPERTIES
    TIMEOUT 300
    SKIP_RETURN_CODE 125
    LABELS "functional;backend_ipc;backend_gda;gpu_all;variant_uuid;launcher_slr;FENCE"
    PROCESSORS 8
    WILL_FAIL FALSE
    FAIL_REGULAR_EXPRESSION "FAILED")

add_test(fence_putwavenbichunks_n2_w1_z64_1048576B_uuid "env" "ROCSHMEM_SLR_NP=2" "ROCSHMEM_MAX_NUM_CONTEXTS=1" "ROCSHMEM_HEAP_SIZE=6442450944" "ROCSHMEM_TEST_UUID=1" "../../../../share/rocshmem/test_wrapper.sh" "fence_putwavenbichunks_n2_w1_z64_1048576B_uuid" "../../../rocshmem_functional_tests" "-a" "102" "-w" "1" "-z" "64" "-s" "1048576" "-localbuftype" "heap")
set_tests_properties(fence_putwavenbichunks_n2_w1_z64_1048576B_uuid PROPERTIES
    TIMEOUT 300
    SKIP_RETURN_CODE 125
    LABELS "functional;backend_ipc;backend_gda;gpu_all;variant_uuid;launcher_slr;FENCE"
    PROCESSORS 2
    WILL_FAIL FALSE
    FAIL_REGULAR_EXPRESSION "FAILED")

add_test(fence_putwavenbichunks_n2_w8_z256_65536B_uuid "env" "ROCSHMEM_SLR_NP=2" "ROCSHMEM_MAX_NUM_CONTEXTS=8" "ROCSHMEM_HEAP_SIZE=6442450944" "ROCSHMEM_TEST_UUID=1" "../../../../share/rocshmem/test_wrapper.sh" "fence_putwavenbichunks_n2_w8_z256_65536B_uuid" "../../../rocshmem_functional_tests" "-a" "102" "-w" "8" "-z" "256" "-s" "65536" "-localbuftype" "heap")
set_tests_properties(fence_putwavenbichunks_n2_w8_z256_65536B_uuid PROPERTIES
    TIMEOUT 300
    SKIP_RETURN_CODE 125
    LABELS "functional;backend_ipc;backend_gda;gpu_all;variant_uuid;launcher_slr;FENCE"
    PROCESSORS 2
    WILL_FAIL FALSE
    FAIL_REGULAR_EXPRESSION "FAILED")

add_test(buffer_register_symmetric_n2_w1_z1_64B_uuid "env" "ROCSHMEM_SLR_NP=2" "ROCSHMEM_MAX_NUM_CONTEXTS=1" "ROCSHMEM_HEAP_SIZE=6442450944" "ROCSHMEM_TEST_UUID=1" "ROCSHMEM_HEAP_ALLOCATOR_TYPE=vmm_posix" "../../../../share/rocshmem/test_wrapper.sh" "buffer_register_symmetric_n2_w1_z1_64B_uuid" "../../../rocshmem_functional_tests" "-a" "162" "-w" "1" "-z" "1" "-s" "64" "-localbuftype" "heap")
set_tests_properties(buffer_register_symmetric_n2_w1_z1_64B_uuid PROPERTIES
    TIMEOUT 300
    SKIP_RETURN_CODE 125
    LABELS "functional;backend_ipc;backend_gda;gpu_all;variant_uuid;launcher_slr;MEMORY;RMA;SYMMETRIC"
    PROCESSORS 2
    WILL_FAIL FALSE
    FAIL_REGULAR_EXPRESSION "FAILED")

add_test(buffer_register_symmetric_n2_w2_z64_64B_uuid "env" "ROCSHMEM_SLR_NP=2" "ROCSHMEM_MAX_NUM_CONTEXTS=2" "ROCSHMEM_HEAP_SIZE=6442450944" "ROCSHMEM_TEST_UUID=1" "ROCSHMEM_HEAP_ALLOCATOR_TYPE=vmm_posix" "../../../../share/rocshmem/test_wrapper.sh" "buffer_register_symmetric_n2_w2_z64_64B_uuid" "../../../rocshmem_functional_tests" "-a" "162" "-w" "2" "-z" "64" "-s" "64" "-localbuftype" "heap")
set_tests_properties(buffer_register_symmetric_n2_w2_z64_64B_uuid PROPERTIES
    TIMEOUT 300
    SKIP_RETURN_CODE 125
    LABELS "functional;backend_ipc;backend_gda;gpu_all;variant_uuid;launcher_slr;MEMORY;RMA;SYMMETRIC"
    PROCESSORS 2
    WILL_FAIL FALSE
    FAIL_REGULAR_EXPRESSION "FAILED")

add_test(buffer_register_symmetric_n4_w1_z1_64B_uuid "env" "ROCSHMEM_SLR_NP=4" "ROCSHMEM_MAX_NUM_CONTEXTS=1" "ROCSHMEM_HEAP_SIZE=6442450944" "ROCSHMEM_TEST_UUID=1" "ROCSHMEM_HEAP_ALLOCATOR_TYPE=vmm_posix" "../../../../share/rocshmem/test_wrapper.sh" "buffer_register_symmetric_n4_w1_z1_64B_uuid" "../../../rocshmem_functional_tests" "-a" "162" "-w" "1" "-z" "1" "-s" "64" "-localbuftype" "heap")
set_tests_properties(buffer_register_symmetric_n4_w1_z1_64B_uuid PROPERTIES
    TIMEOUT 300
    SKIP_RETURN_CODE 125
    LABELS "functional;backend_ipc;backend_gda;gpu_all;variant_uuid;launcher_slr;MEMORY;RMA;SYMMETRIC"
    PROCESSORS 4
    WILL_FAIL FALSE
    FAIL_REGULAR_EXPRESSION "FAILED")

add_test(buffer_register_symmetric_n4_w2_z64_64B_uuid "env" "ROCSHMEM_SLR_NP=4" "ROCSHMEM_MAX_NUM_CONTEXTS=2" "ROCSHMEM_HEAP_SIZE=6442450944" "ROCSHMEM_TEST_UUID=1" "ROCSHMEM_HEAP_ALLOCATOR_TYPE=vmm_posix" "../../../../share/rocshmem/test_wrapper.sh" "buffer_register_symmetric_n4_w2_z64_64B_uuid" "../../../rocshmem_functional_tests" "-a" "162" "-w" "2" "-z" "64" "-s" "64" "-localbuftype" "heap")
set_tests_properties(buffer_register_symmetric_n4_w2_z64_64B_uuid PROPERTIES
    TIMEOUT 300
    SKIP_RETURN_CODE 125
    LABELS "functional;backend_ipc;backend_gda;gpu_all;variant_uuid;launcher_slr;MEMORY;RMA;SYMMETRIC"
    PROCESSORS 4
    WILL_FAIL FALSE
    FAIL_REGULAR_EXPRESSION "FAILED")

add_test(get_n2_w1_z1_v1048576B_uuid "env" "ROCSHMEM_SLR_NP=2" "ROCSHMEM_MAX_NUM_CONTEXTS=1" "ROCSHMEM_HEAP_SIZE=6442450944" "ROCSHMEM_TEST_UUID=1" "../../../../share/rocshmem/test_wrapper.sh" "get_n2_w1_z1_v1048576B_uuid" "../../../rocshmem_functional_tests" "-a" "0" "-w" "1" "-z" "1" "-v" "1048576" "-noverif" "-localbuftype" "heap")
set_tests_properties(get_n2_w1_z1_v1048576B_uuid PROPERTIES
    SKIP_RETURN_CODE 125
    LABELS "functional;backend_ipc;backend_gda;gpu_all;variant_uuid;launcher_slr;HEATMAP"
    PROCESSORS 2
    WILL_FAIL FALSE
    FAIL_REGULAR_EXPRESSION "FAILED")

add_test(get_n2_w32_z1024_v1073741824B_uuid "env" "ROCSHMEM_SLR_NP=2" "ROCSHMEM_MAX_NUM_CONTEXTS=32" "ROCSHMEM_HEAP_SIZE=6442450944" "ROCSHMEM_TEST_UUID=1" "../../../../share/rocshmem/test_wrapper.sh" "get_n2_w32_z1024_v1073741824B_uuid" "../../../rocshmem_functional_tests" "-a" "0" "-w" "32" "-z" "1024" "-v" "1073741824" "-noverif" "-localbuftype" "heap")
set_tests_properties(get_n2_w32_z1024_v1073741824B_uuid PROPERTIES
    SKIP_RETURN_CODE 125
    LABELS "functional;backend_ipc;backend_gda;gpu_all;variant_uuid;launcher_slr;HEATMAP"
    PROCESSORS 2
    WILL_FAIL FALSE
    FAIL_REGULAR_EXPRESSION "FAILED")

add_test(waveget_n2_w1_z64_v1073741824B_uuid "env" "ROCSHMEM_SLR_NP=2" "ROCSHMEM_MAX_NUM_CONTEXTS=1" "ROCSHMEM_HEAP_SIZE=6442450944" "ROCSHMEM_TEST_UUID=1" "../../../../share/rocshmem/test_wrapper.sh" "waveget_n2_w1_z64_v1073741824B_uuid" "../../../rocshmem_functional_tests" "-a" "28" "-w" "1" "-z" "64" "-v" "1073741824" "-noverif" "-localbuftype" "heap")
set_tests_properties(waveget_n2_w1_z64_v1073741824B_uuid PROPERTIES
    SKIP_RETURN_CODE 125
    LABELS "functional;backend_ipc;backend_gda;gpu_all;variant_uuid;launcher_slr;HEATMAP"
    PROCESSORS 2
    WILL_FAIL FALSE
    FAIL_REGULAR_EXPRESSION "FAILED")

add_test(waveget_n2_w2_z64_v1073741824B_uuid "env" "ROCSHMEM_SLR_NP=2" "ROCSHMEM_MAX_NUM_CONTEXTS=2" "ROCSHMEM_HEAP_SIZE=6442450944" "ROCSHMEM_TEST_UUID=1" "../../../../share/rocshmem/test_wrapper.sh" "waveget_n2_w2_z64_v1073741824B_uuid" "../../../rocshmem_functional_tests" "-a" "28" "-w" "2" "-z" "64" "-v" "1073741824" "-noverif" "-localbuftype" "heap")
set_tests_properties(waveget_n2_w2_z64_v1073741824B_uuid PROPERTIES
    SKIP_RETURN_CODE 125
    LABELS "functional;backend_ipc;backend_gda;gpu_all;variant_uuid;launcher_slr;HEATMAP"
    PROCESSORS 2
    WILL_FAIL FALSE
    FAIL_REGULAR_EXPRESSION "FAILED")

add_test(waveget_n2_w16_z1024_v1073741824B_uuid "env" "ROCSHMEM_SLR_NP=2" "ROCSHMEM_MAX_NUM_CONTEXTS=16" "ROCSHMEM_HEAP_SIZE=6442450944" "ROCSHMEM_TEST_UUID=1" "../../../../share/rocshmem/test_wrapper.sh" "waveget_n2_w16_z1024_v1073741824B_uuid" "../../../rocshmem_functional_tests" "-a" "28" "-w" "16" "-z" "1024" "-v" "1073741824" "-noverif" "-localbuftype" "heap")
set_tests_properties(waveget_n2_w16_z1024_v1073741824B_uuid PROPERTIES
    SKIP_RETURN_CODE 125
    LABELS "functional;backend_ipc;backend_gda;gpu_all;variant_uuid;launcher_slr;HEATMAP"
    PROCESSORS 2
    WILL_FAIL FALSE
    FAIL_REGULAR_EXPRESSION "FAILED")

add_test(wgget_n2_w1_z1024_v1073741824B_uuid "env" "ROCSHMEM_SLR_NP=2" "ROCSHMEM_MAX_NUM_CONTEXTS=1" "ROCSHMEM_HEAP_SIZE=6442450944" "ROCSHMEM_TEST_UUID=1" "../../../../share/rocshmem/test_wrapper.sh" "wgget_n2_w1_z1024_v1073741824B_uuid" "../../../rocshmem_functional_tests" "-a" "24" "-w" "1" "-z" "1024" "-v" "1073741824" "-noverif" "-localbuftype" "heap")
set_tests_properties(wgget_n2_w1_z1024_v1073741824B_uuid PROPERTIES
    SKIP_RETURN_CODE 125
    LABELS "functional;backend_ipc;backend_gda;gpu_all;variant_uuid;launcher_slr;HEATMAP"
    PROCESSORS 2
    WILL_FAIL FALSE
    FAIL_REGULAR_EXPRESSION "FAILED")

add_test(wgget_n2_w16_z1024_v1073741824B_uuid "env" "ROCSHMEM_SLR_NP=2" "ROCSHMEM_MAX_NUM_CONTEXTS=16" "ROCSHMEM_HEAP_SIZE=6442450944" "ROCSHMEM_TEST_UUID=1" "../../../../share/rocshmem/test_wrapper.sh" "wgget_n2_w16_z1024_v1073741824B_uuid" "../../../rocshmem_functional_tests" "-a" "24" "-w" "16" "-z" "1024" "-v" "1073741824" "-noverif" "-localbuftype" "heap")
set_tests_properties(wgget_n2_w16_z1024_v1073741824B_uuid PROPERTIES
    SKIP_RETURN_CODE 125
    LABELS "functional;backend_ipc;backend_gda;gpu_all;variant_uuid;launcher_slr;HEATMAP"
    PROCESSORS 2
    WILL_FAIL FALSE
    FAIL_REGULAR_EXPRESSION "FAILED")

add_test(put_n2_w1_z1_v1048576B_uuid "env" "ROCSHMEM_SLR_NP=2" "ROCSHMEM_MAX_NUM_CONTEXTS=1" "ROCSHMEM_HEAP_SIZE=6442450944" "ROCSHMEM_TEST_UUID=1" "../../../../share/rocshmem/test_wrapper.sh" "put_n2_w1_z1_v1048576B_uuid" "../../../rocshmem_functional_tests" "-a" "2" "-w" "1" "-z" "1" "-v" "1048576" "-noverif" "-localbuftype" "heap")
set_tests_properties(put_n2_w1_z1_v1048576B_uuid PROPERTIES
    SKIP_RETURN_CODE 125
    LABELS "functional;backend_all;gpu_all;variant_uuid;launcher_slr;HEATMAP"
    PROCESSORS 2
    WILL_FAIL FALSE
    FAIL_REGULAR_EXPRESSION "FAILED")

add_test(put_n2_w32_z1024_v1073741824B_uuid "env" "ROCSHMEM_SLR_NP=2" "ROCSHMEM_MAX_NUM_CONTEXTS=32" "ROCSHMEM_HEAP_SIZE=6442450944" "ROCSHMEM_TEST_UUID=1" "../../../../share/rocshmem/test_wrapper.sh" "put_n2_w32_z1024_v1073741824B_uuid" "../../../rocshmem_functional_tests" "-a" "2" "-w" "32" "-z" "1024" "-v" "1073741824" "-noverif" "-localbuftype" "heap")
set_tests_properties(put_n2_w32_z1024_v1073741824B_uuid PROPERTIES
    SKIP_RETURN_CODE 125
    LABELS "functional;backend_all;gpu_all;variant_uuid;launcher_slr;HEATMAP"
    PROCESSORS 2
    WILL_FAIL FALSE
    FAIL_REGULAR_EXPRESSION "FAILED")

add_test(waveput_n2_w1_z64_v1073741824B_uuid "env" "ROCSHMEM_SLR_NP=2" "ROCSHMEM_MAX_NUM_CONTEXTS=1" "ROCSHMEM_HEAP_SIZE=6442450944" "ROCSHMEM_TEST_UUID=1" "../../../../share/rocshmem/test_wrapper.sh" "waveput_n2_w1_z64_v1073741824B_uuid" "../../../rocshmem_functional_tests" "-a" "30" "-w" "1" "-z" "64" "-v" "1073741824" "-noverif" "-localbuftype" "heap")
set_tests_properties(waveput_n2_w1_z64_v1073741824B_uuid PROPERTIES
    SKIP_RETURN_CODE 125
    LABELS "functional;backend_all;gpu_all;variant_uuid;launcher_slr;HEATMAP"
    PROCESSORS 2
    WILL_FAIL FALSE
    FAIL_REGULAR_EXPRESSION "FAILED")

add_test(waveput_n2_w2_z64_v1073741824B_uuid "env" "ROCSHMEM_SLR_NP=2" "ROCSHMEM_MAX_NUM_CONTEXTS=2" "ROCSHMEM_HEAP_SIZE=6442450944" "ROCSHMEM_TEST_UUID=1" "../../../../share/rocshmem/test_wrapper.sh" "waveput_n2_w2_z64_v1073741824B_uuid" "../../../rocshmem_functional_tests" "-a" "30" "-w" "2" "-z" "64" "-v" "1073741824" "-noverif" "-localbuftype" "heap")
set_tests_properties(waveput_n2_w2_z64_v1073741824B_uuid PROPERTIES
    SKIP_RETURN_CODE 125
    LABELS "functional;backend_all;gpu_all;variant_uuid;launcher_slr;HEATMAP"
    PROCESSORS 2
    WILL_FAIL FALSE
    FAIL_REGULAR_EXPRESSION "FAILED")

add_test(waveput_n2_w16_z1024_v1073741824B_uuid "env" "ROCSHMEM_SLR_NP=2" "ROCSHMEM_MAX_NUM_CONTEXTS=16" "ROCSHMEM_HEAP_SIZE=6442450944" "ROCSHMEM_TEST_UUID=1" "../../../../share/rocshmem/test_wrapper.sh" "waveput_n2_w16_z1024_v1073741824B_uuid" "../../../rocshmem_functional_tests" "-a" "30" "-w" "16" "-z" "1024" "-v" "1073741824" "-noverif" "-localbuftype" "heap")
set_tests_properties(waveput_n2_w16_z1024_v1073741824B_uuid PROPERTIES
    SKIP_RETURN_CODE 125
    LABELS "functional;backend_all;gpu_all;variant_uuid;launcher_slr;HEATMAP"
    PROCESSORS 2
    WILL_FAIL FALSE
    FAIL_REGULAR_EXPRESSION "FAILED")

add_test(wgput_n2_w1_z1024_v1073741824B_uuid "env" "ROCSHMEM_SLR_NP=2" "ROCSHMEM_MAX_NUM_CONTEXTS=1" "ROCSHMEM_HEAP_SIZE=6442450944" "ROCSHMEM_TEST_UUID=1" "../../../../share/rocshmem/test_wrapper.sh" "wgput_n2_w1_z1024_v1073741824B_uuid" "../../../rocshmem_functional_tests" "-a" "26" "-w" "1" "-z" "1024" "-v" "1073741824" "-noverif" "-localbuftype" "heap")
set_tests_properties(wgput_n2_w1_z1024_v1073741824B_uuid PROPERTIES
    SKIP_RETURN_CODE 125
    LABELS "functional;backend_all;gpu_all;variant_uuid;launcher_slr;HEATMAP"
    PROCESSORS 2
    WILL_FAIL FALSE
    FAIL_REGULAR_EXPRESSION "FAILED")

add_test(wgput_n2_w16_z1024_v1073741824B_uuid "env" "ROCSHMEM_SLR_NP=2" "ROCSHMEM_MAX_NUM_CONTEXTS=16" "ROCSHMEM_HEAP_SIZE=6442450944" "ROCSHMEM_TEST_UUID=1" "../../../../share/rocshmem/test_wrapper.sh" "wgput_n2_w16_z1024_v1073741824B_uuid" "../../../rocshmem_functional_tests" "-a" "26" "-w" "16" "-z" "1024" "-v" "1073741824" "-noverif" "-localbuftype" "heap")
set_tests_properties(wgput_n2_w16_z1024_v1073741824B_uuid PROPERTIES
    SKIP_RETURN_CODE 125
    LABELS "functional;backend_all;gpu_all;variant_uuid;launcher_slr;HEATMAP"
    PROCESSORS 2
    WILL_FAIL FALSE
    FAIL_REGULAR_EXPRESSION "FAILED")

add_test(alltoall_n2_w1_z256_v1073741824B_uuid "env" "ROCSHMEM_SLR_NP=2" "ROCSHMEM_MAX_NUM_CONTEXTS=1" "ROCSHMEM_HEAP_SIZE=6442450944" "ROCSHMEM_TEST_UUID=1" "../../../../share/rocshmem/test_wrapper.sh" "alltoall_n2_w1_z256_v1073741824B_uuid" "../../../rocshmem_functional_tests" "-a" "19" "-w" "1" "-z" "256" "-v" "1073741824" "-noverif" "-localbuftype" "heap")
set_tests_properties(alltoall_n2_w1_z256_v1073741824B_uuid PROPERTIES
    SKIP_RETURN_CODE 125
    LABELS "functional;backend_all;gpu_all;variant_uuid;launcher_slr;HEATMAP"
    PROCESSORS 2
    WILL_FAIL FALSE
    FAIL_REGULAR_EXPRESSION "FAILED")

add_test(alltoall_n4_w1_z256_v1073741824B_uuid "env" "ROCSHMEM_SLR_NP=4" "ROCSHMEM_MAX_NUM_CONTEXTS=1" "ROCSHMEM_HEAP_SIZE=6442450944" "ROCSHMEM_TEST_UUID=1" "../../../../share/rocshmem/test_wrapper.sh" "alltoall_n4_w1_z256_v1073741824B_uuid" "../../../rocshmem_functional_tests" "-a" "19" "-w" "1" "-z" "256" "-v" "1073741824" "-noverif" "-localbuftype" "heap")
set_tests_properties(alltoall_n4_w1_z256_v1073741824B_uuid PROPERTIES
    SKIP_RETURN_CODE 125
    LABELS "functional;backend_all;gpu_all;variant_uuid;launcher_slr;HEATMAP"
    PROCESSORS 4
    WILL_FAIL FALSE
    FAIL_REGULAR_EXPRESSION "FAILED")

add_test(alltoall_n8_w1_z256_v1073741824B_uuid "env" "ROCSHMEM_SLR_NP=8" "ROCSHMEM_MAX_NUM_CONTEXTS=1" "ROCSHMEM_HEAP_SIZE=6442450944" "ROCSHMEM_TEST_UUID=1" "../../../../share/rocshmem/test_wrapper.sh" "alltoall_n8_w1_z256_v1073741824B_uuid" "../../../rocshmem_functional_tests" "-a" "19" "-w" "1" "-z" "256" "-v" "1073741824" "-noverif" "-localbuftype" "heap")
set_tests_properties(alltoall_n8_w1_z256_v1073741824B_uuid PROPERTIES
    SKIP_RETURN_CODE 125
    LABELS "functional;backend_all;gpu_all;variant_uuid;launcher_slr;HEATMAP"
    PROCESSORS 8
    WILL_FAIL FALSE
    FAIL_REGULAR_EXPRESSION "FAILED")

add_test(alltoall_n16_w1_z256_v1073741824B_uuid "env" "ROCSHMEM_SLR_NP=16" "ROCSHMEM_MAX_NUM_CONTEXTS=1" "ROCSHMEM_HEAP_SIZE=6442450944" "ROCSHMEM_TEST_UUID=1" "../../../../share/rocshmem/test_wrapper.sh" "alltoall_n16_w1_z256_v1073741824B_uuid" "../../../rocshmem_functional_tests" "-a" "19" "-w" "1" "-z" "256" "-v" "1073741824" "-noverif" "-localbuftype" "heap")
set_tests_properties(alltoall_n16_w1_z256_v1073741824B_uuid PROPERTIES
    SKIP_RETURN_CODE 125
    LABELS "functional;backend_all;gpu_all;variant_uuid;launcher_slr;HEATMAP"
    PROCESSORS 16
    WILL_FAIL FALSE
    FAIL_REGULAR_EXPRESSION "FAILED")

add_test(alltoall_n32_w1_z256_v1073741824B_uuid "env" "ROCSHMEM_SLR_NP=32" "ROCSHMEM_MAX_NUM_CONTEXTS=1" "ROCSHMEM_HEAP_SIZE=6442450944" "ROCSHMEM_TEST_UUID=1" "../../../../share/rocshmem/test_wrapper.sh" "alltoall_n32_w1_z256_v1073741824B_uuid" "../../../rocshmem_functional_tests" "-a" "19" "-w" "1" "-z" "256" "-v" "1073741824" "-noverif" "-localbuftype" "heap")
set_tests_properties(alltoall_n32_w1_z256_v1073741824B_uuid PROPERTIES
    SKIP_RETURN_CODE 125
    LABELS "functional;backend_all;gpu_all;variant_uuid;launcher_slr;HEATMAP"
    PROCESSORS 32
    WILL_FAIL FALSE
    FAIL_REGULAR_EXPRESSION "FAILED")

add_test(alltoall_n64_w1_z256_v1073741824B_uuid "env" "ROCSHMEM_SLR_NP=64" "ROCSHMEM_MAX_NUM_CONTEXTS=1" "ROCSHMEM_HEAP_SIZE=6442450944" "ROCSHMEM_TEST_UUID=1" "../../../../share/rocshmem/test_wrapper.sh" "alltoall_n64_w1_z256_v1073741824B_uuid" "../../../rocshmem_functional_tests" "-a" "19" "-w" "1" "-z" "256" "-v" "1073741824" "-noverif" "-localbuftype" "heap")
set_tests_properties(alltoall_n64_w1_z256_v1073741824B_uuid PROPERTIES
    SKIP_RETURN_CODE 125
    LABELS "functional;backend_all;gpu_all;variant_uuid;launcher_slr;HEATMAP"
    PROCESSORS 64
    WILL_FAIL FALSE
    FAIL_REGULAR_EXPRESSION "FAILED")

add_test(tile_put_contiguous_n2_w1_z1_1048576B_uuid "env" "ROCSHMEM_SLR_NP=2" "ROCSHMEM_MAX_NUM_CONTEXTS=1" "ROCSHMEM_HEAP_SIZE=6442450944" "ROCSHMEM_TEST_UUID=1" "../../../../share/rocshmem/test_wrapper.sh" "tile_put_contiguous_n2_w1_z1_1048576B_uuid" "../../../rocshmem_functional_tests" "-a" "103" "-w" "1" "-z" "1" "-s" "1048576" "-localbuftype" "heap")
set_tests_properties(tile_put_contiguous_n2_w1_z1_1048576B_uuid PROPERTIES
    TIMEOUT 300
    SKIP_RETURN_CODE 125
    LABELS "functional;backend_ipc;backend_gda;gpu_all;variant_uuid;launcher_slr;TILE"
    PROCESSORS 2
    WILL_FAIL FALSE
    FAIL_REGULAR_EXPRESSION "FAILED")

add_test(tile_put_rowmajor_n2_w1_z1_1048576B_uuid "env" "ROCSHMEM_SLR_NP=2" "ROCSHMEM_MAX_NUM_CONTEXTS=1" "ROCSHMEM_HEAP_SIZE=6442450944" "ROCSHMEM_TEST_UUID=1" "../../../../share/rocshmem/test_wrapper.sh" "tile_put_rowmajor_n2_w1_z1_1048576B_uuid" "../../../rocshmem_functional_tests" "-a" "104" "-w" "1" "-z" "1" "-s" "1048576" "-localbuftype" "heap")
set_tests_properties(tile_put_rowmajor_n2_w1_z1_1048576B_uuid PROPERTIES
    TIMEOUT 300
    SKIP_RETURN_CODE 125
    LABELS "functional;backend_ipc;backend_gda;gpu_all;variant_uuid;launcher_slr;TILE"
    PROCESSORS 2
    WILL_FAIL FALSE
    FAIL_REGULAR_EXPRESSION "FAILED")

add_test(tile_put_colmajor_n2_w1_z1_1048576B_uuid "env" "ROCSHMEM_SLR_NP=2" "ROCSHMEM_MAX_NUM_CONTEXTS=1" "ROCSHMEM_HEAP_SIZE=6442450944" "ROCSHMEM_TEST_UUID=1" "../../../../share/rocshmem/test_wrapper.sh" "tile_put_colmajor_n2_w1_z1_1048576B_uuid" "../../../rocshmem_functional_tests" "-a" "105" "-w" "1" "-z" "1" "-s" "1048576" "-localbuftype" "heap")
set_tests_properties(tile_put_colmajor_n2_w1_z1_1048576B_uuid PROPERTIES
    TIMEOUT 300
    SKIP_RETURN_CODE 125
    LABELS "functional;backend_ipc;backend_gda;gpu_all;variant_uuid;launcher_slr;TILE"
    PROCESSORS 2
    WILL_FAIL FALSE
    FAIL_REGULAR_EXPRESSION "FAILED")

add_test(tile_put_arbitrary_n2_w1_z1_16384B_uuid "env" "ROCSHMEM_SLR_NP=2" "ROCSHMEM_MAX_NUM_CONTEXTS=1" "ROCSHMEM_HEAP_SIZE=6442450944" "ROCSHMEM_TEST_UUID=1" "../../../../share/rocshmem/test_wrapper.sh" "tile_put_arbitrary_n2_w1_z1_16384B_uuid" "../../../rocshmem_functional_tests" "-a" "106" "-w" "1" "-z" "1" "-s" "16384" "-localbuftype" "heap")
set_tests_properties(tile_put_arbitrary_n2_w1_z1_16384B_uuid PROPERTIES
    TIMEOUT 300
    SKIP_RETURN_CODE 125
    LABELS "functional;backend_ipc;backend_gda;gpu_all;variant_uuid;launcher_slr;TILE"
    PROCESSORS 2
    WILL_FAIL FALSE
    FAIL_REGULAR_EXPRESSION "FAILED")

add_test(tile_put_1d_n2_w1_z1_1048576B_uuid "env" "ROCSHMEM_SLR_NP=2" "ROCSHMEM_MAX_NUM_CONTEXTS=1" "ROCSHMEM_HEAP_SIZE=6442450944" "ROCSHMEM_TEST_UUID=1" "../../../../share/rocshmem/test_wrapper.sh" "tile_put_1d_n2_w1_z1_1048576B_uuid" "../../../rocshmem_functional_tests" "-a" "111" "-w" "1" "-z" "1" "-s" "1048576" "-localbuftype" "heap")
set_tests_properties(tile_put_1d_n2_w1_z1_1048576B_uuid PROPERTIES
    TIMEOUT 300
    SKIP_RETURN_CODE 125
    LABELS "functional;backend_ipc;backend_gda;gpu_all;variant_uuid;launcher_slr;TILE"
    PROCESSORS 2
    WILL_FAIL FALSE
    FAIL_REGULAR_EXPRESSION "FAILED")

add_test(tile_put_wave_contiguous_n2_w1_wf1_1048576B_uuid "env" "ROCSHMEM_SLR_NP=2" "ROCSHMEM_MAX_NUM_CONTEXTS=1" "ROCSHMEM_HEAP_SIZE=6442450944" "ROCSHMEM_TEST_UUID=1" "../../../../share/rocshmem/test_wrapper.sh" "tile_put_wave_contiguous_n2_w1_wf1_1048576B_uuid" "../../../rocshmem_functional_tests" "-a" "107" "-w" "1" "-s" "1048576" "--num-wf" "1" "-localbuftype" "heap")
set_tests_properties(tile_put_wave_contiguous_n2_w1_wf1_1048576B_uuid PROPERTIES
    TIMEOUT 300
    SKIP_RETURN_CODE 125
    LABELS "functional;backend_ipc;backend_gda;gpu_all;variant_uuid;launcher_slr;TILE"
    PROCESSORS 2
    WILL_FAIL FALSE
    FAIL_REGULAR_EXPRESSION "FAILED")

add_test(tile_put_wave_rowmajor_n2_w1_wf1_1048576B_uuid "env" "ROCSHMEM_SLR_NP=2" "ROCSHMEM_MAX_NUM_CONTEXTS=1" "ROCSHMEM_HEAP_SIZE=6442450944" "ROCSHMEM_TEST_UUID=1" "../../../../share/rocshmem/test_wrapper.sh" "tile_put_wave_rowmajor_n2_w1_wf1_1048576B_uuid" "../../../rocshmem_functional_tests" "-a" "163" "-w" "1" "-s" "1048576" "--num-wf" "1" "-localbuftype" "heap")
set_tests_properties(tile_put_wave_rowmajor_n2_w1_wf1_1048576B_uuid PROPERTIES
    TIMEOUT 300
    SKIP_RETURN_CODE 125
    LABELS "functional;backend_ipc;backend_gda;gpu_all;variant_uuid;launcher_slr;TILE"
    PROCESSORS 2
    WILL_FAIL FALSE
    FAIL_REGULAR_EXPRESSION "FAILED")

add_test(tile_put_wave_colmajor_n2_w1_wf1_1048576B_uuid "env" "ROCSHMEM_SLR_NP=2" "ROCSHMEM_MAX_NUM_CONTEXTS=1" "ROCSHMEM_HEAP_SIZE=6442450944" "ROCSHMEM_TEST_UUID=1" "../../../../share/rocshmem/test_wrapper.sh" "tile_put_wave_colmajor_n2_w1_wf1_1048576B_uuid" "../../../rocshmem_functional_tests" "-a" "164" "-w" "1" "-s" "1048576" "--num-wf" "1" "-localbuftype" "heap")
set_tests_properties(tile_put_wave_colmajor_n2_w1_wf1_1048576B_uuid PROPERTIES
    TIMEOUT 300
    SKIP_RETURN_CODE 125
    LABELS "functional;backend_ipc;backend_gda;gpu_all;variant_uuid;launcher_slr;TILE"
    PROCESSORS 2
    WILL_FAIL FALSE
    FAIL_REGULAR_EXPRESSION "FAILED")

add_test(tile_put_wg_contiguous_n2_w1_wf16_1048576B_uuid "env" "ROCSHMEM_SLR_NP=2" "ROCSHMEM_MAX_NUM_CONTEXTS=16" "ROCSHMEM_HEAP_SIZE=6442450944" "ROCSHMEM_TEST_UUID=1" "../../../../share/rocshmem/test_wrapper.sh" "tile_put_wg_contiguous_n2_w1_wf16_1048576B_uuid" "../../../rocshmem_functional_tests" "-a" "108" "-w" "1" "-s" "1048576" "--num-wf" "16" "-localbuftype" "heap")
set_tests_properties(tile_put_wg_contiguous_n2_w1_wf16_1048576B_uuid PROPERTIES
    TIMEOUT 300
    SKIP_RETURN_CODE 125
    LABELS "functional;backend_ipc;backend_gda;gpu_all;variant_uuid;launcher_slr;TILE"
    PROCESSORS 2
    WILL_FAIL FALSE
    FAIL_REGULAR_EXPRESSION "FAILED")

add_test(tile_put_wg_rowmajor_n2_w1_wf16_1048576B_uuid "env" "ROCSHMEM_SLR_NP=2" "ROCSHMEM_MAX_NUM_CONTEXTS=16" "ROCSHMEM_HEAP_SIZE=6442450944" "ROCSHMEM_TEST_UUID=1" "../../../../share/rocshmem/test_wrapper.sh" "tile_put_wg_rowmajor_n2_w1_wf16_1048576B_uuid" "../../../rocshmem_functional_tests" "-a" "167" "-w" "1" "-s" "1048576" "--num-wf" "16" "-localbuftype" "heap")
set_tests_properties(tile_put_wg_rowmajor_n2_w1_wf16_1048576B_uuid PROPERTIES
    TIMEOUT 300
    SKIP_RETURN_CODE 125
    LABELS "functional;backend_ipc;backend_gda;gpu_all;variant_uuid;launcher_slr;TILE"
    PROCESSORS 2
    WILL_FAIL FALSE
    FAIL_REGULAR_EXPRESSION "FAILED")

add_test(tile_put_wg_colmajor_n2_w1_wf16_1048576B_uuid "env" "ROCSHMEM_SLR_NP=2" "ROCSHMEM_MAX_NUM_CONTEXTS=16" "ROCSHMEM_HEAP_SIZE=6442450944" "ROCSHMEM_TEST_UUID=1" "../../../../share/rocshmem/test_wrapper.sh" "tile_put_wg_colmajor_n2_w1_wf16_1048576B_uuid" "../../../rocshmem_functional_tests" "-a" "168" "-w" "1" "-s" "1048576" "--num-wf" "16" "-localbuftype" "heap")
set_tests_properties(tile_put_wg_colmajor_n2_w1_wf16_1048576B_uuid PROPERTIES
    TIMEOUT 300
    SKIP_RETURN_CODE 125
    LABELS "functional;backend_ipc;backend_gda;gpu_all;variant_uuid;launcher_slr;TILE"
    PROCESSORS 2
    WILL_FAIL FALSE
    FAIL_REGULAR_EXPRESSION "FAILED")

add_test(tile_get_contiguous_n2_w1_z1_1048576B_uuid "env" "ROCSHMEM_SLR_NP=2" "ROCSHMEM_MAX_NUM_CONTEXTS=1" "ROCSHMEM_HEAP_SIZE=6442450944" "ROCSHMEM_TEST_UUID=1" "../../../../share/rocshmem/test_wrapper.sh" "tile_get_contiguous_n2_w1_z1_1048576B_uuid" "../../../rocshmem_functional_tests" "-a" "109" "-w" "1" "-z" "1" "-s" "1048576" "-localbuftype" "heap")
set_tests_properties(tile_get_contiguous_n2_w1_z1_1048576B_uuid PROPERTIES
    TIMEOUT 300
    SKIP_RETURN_CODE 125
    LABELS "functional;backend_ipc;backend_gda;gpu_all;variant_uuid;launcher_slr;TILE"
    PROCESSORS 2
    WILL_FAIL FALSE
    FAIL_REGULAR_EXPRESSION "FAILED")

add_test(tile_get_rowmajor_n2_w1_z1_1048576B_uuid "env" "ROCSHMEM_SLR_NP=2" "ROCSHMEM_MAX_NUM_CONTEXTS=1" "ROCSHMEM_HEAP_SIZE=6442450944" "ROCSHMEM_TEST_UUID=1" "../../../../share/rocshmem/test_wrapper.sh" "tile_get_rowmajor_n2_w1_z1_1048576B_uuid" "../../../rocshmem_functional_tests" "-a" "114" "-w" "1" "-z" "1" "-s" "1048576" "-localbuftype" "heap")
set_tests_properties(tile_get_rowmajor_n2_w1_z1_1048576B_uuid PROPERTIES
    TIMEOUT 300
    SKIP_RETURN_CODE 125
    LABELS "functional;backend_ipc;backend_gda;gpu_all;variant_uuid;launcher_slr;TILE"
    PROCESSORS 2
    WILL_FAIL FALSE
    FAIL_REGULAR_EXPRESSION "FAILED")

add_test(tile_get_colmajor_n2_w1_z1_1048576B_uuid "env" "ROCSHMEM_SLR_NP=2" "ROCSHMEM_MAX_NUM_CONTEXTS=1" "ROCSHMEM_HEAP_SIZE=6442450944" "ROCSHMEM_TEST_UUID=1" "../../../../share/rocshmem/test_wrapper.sh" "tile_get_colmajor_n2_w1_z1_1048576B_uuid" "../../../rocshmem_functional_tests" "-a" "115" "-w" "1" "-z" "1" "-s" "1048576" "-localbuftype" "heap")
set_tests_properties(tile_get_colmajor_n2_w1_z1_1048576B_uuid PROPERTIES
    TIMEOUT 300
    SKIP_RETURN_CODE 125
    LABELS "functional;backend_ipc;backend_gda;gpu_all;variant_uuid;launcher_slr;TILE"
    PROCESSORS 2
    WILL_FAIL FALSE
    FAIL_REGULAR_EXPRESSION "FAILED")

add_test(tile_get_arbitrary_n2_w1_z1_16384B_uuid "env" "ROCSHMEM_SLR_NP=2" "ROCSHMEM_MAX_NUM_CONTEXTS=1" "ROCSHMEM_HEAP_SIZE=6442450944" "ROCSHMEM_TEST_UUID=1" "../../../../share/rocshmem/test_wrapper.sh" "tile_get_arbitrary_n2_w1_z1_16384B_uuid" "../../../rocshmem_functional_tests" "-a" "116" "-w" "1" "-z" "1" "-s" "16384" "-localbuftype" "heap")
set_tests_properties(tile_get_arbitrary_n2_w1_z1_16384B_uuid PROPERTIES
    TIMEOUT 300
    SKIP_RETURN_CODE 125
    LABELS "functional;backend_ipc;backend_gda;gpu_all;variant_uuid;launcher_slr;TILE"
    PROCESSORS 2
    WILL_FAIL FALSE
    FAIL_REGULAR_EXPRESSION "FAILED")

add_test(tile_get_1d_n2_w1_z1_1048576B_uuid "env" "ROCSHMEM_SLR_NP=2" "ROCSHMEM_MAX_NUM_CONTEXTS=1" "ROCSHMEM_HEAP_SIZE=6442450944" "ROCSHMEM_TEST_UUID=1" "../../../../share/rocshmem/test_wrapper.sh" "tile_get_1d_n2_w1_z1_1048576B_uuid" "../../../rocshmem_functional_tests" "-a" "112" "-w" "1" "-z" "1" "-s" "1048576" "-localbuftype" "heap")
set_tests_properties(tile_get_1d_n2_w1_z1_1048576B_uuid PROPERTIES
    TIMEOUT 300
    SKIP_RETURN_CODE 125
    LABELS "functional;backend_ipc;backend_gda;gpu_all;variant_uuid;launcher_slr;TILE"
    PROCESSORS 2
    WILL_FAIL FALSE
    FAIL_REGULAR_EXPRESSION "FAILED")

add_test(tile_get_wave_contiguous_n2_w1_wf1_1048576B_uuid "env" "ROCSHMEM_SLR_NP=2" "ROCSHMEM_MAX_NUM_CONTEXTS=1" "ROCSHMEM_HEAP_SIZE=6442450944" "ROCSHMEM_TEST_UUID=1" "../../../../share/rocshmem/test_wrapper.sh" "tile_get_wave_contiguous_n2_w1_wf1_1048576B_uuid" "../../../rocshmem_functional_tests" "-a" "113" "-w" "1" "-s" "1048576" "--num-wf" "1" "-localbuftype" "heap")
set_tests_properties(tile_get_wave_contiguous_n2_w1_wf1_1048576B_uuid PROPERTIES
    TIMEOUT 300
    SKIP_RETURN_CODE 125
    LABELS "functional;backend_ipc;backend_gda;gpu_all;variant_uuid;launcher_slr;TILE"
    PROCESSORS 2
    WILL_FAIL FALSE
    FAIL_REGULAR_EXPRESSION "FAILED")

add_test(tile_get_wave_rowmajor_n2_w1_wf1_1048576B_uuid "env" "ROCSHMEM_SLR_NP=2" "ROCSHMEM_MAX_NUM_CONTEXTS=1" "ROCSHMEM_HEAP_SIZE=6442450944" "ROCSHMEM_TEST_UUID=1" "../../../../share/rocshmem/test_wrapper.sh" "tile_get_wave_rowmajor_n2_w1_wf1_1048576B_uuid" "../../../rocshmem_functional_tests" "-a" "165" "-w" "1" "-s" "1048576" "--num-wf" "1" "-localbuftype" "heap")
set_tests_properties(tile_get_wave_rowmajor_n2_w1_wf1_1048576B_uuid PROPERTIES
    TIMEOUT 300
    SKIP_RETURN_CODE 125
    LABELS "functional;backend_ipc;backend_gda;gpu_all;variant_uuid;launcher_slr;TILE"
    PROCESSORS 2
    WILL_FAIL FALSE
    FAIL_REGULAR_EXPRESSION "FAILED")

add_test(tile_get_wave_colmajor_n2_w1_wf1_1048576B_uuid "env" "ROCSHMEM_SLR_NP=2" "ROCSHMEM_MAX_NUM_CONTEXTS=1" "ROCSHMEM_HEAP_SIZE=6442450944" "ROCSHMEM_TEST_UUID=1" "../../../../share/rocshmem/test_wrapper.sh" "tile_get_wave_colmajor_n2_w1_wf1_1048576B_uuid" "../../../rocshmem_functional_tests" "-a" "166" "-w" "1" "-s" "1048576" "--num-wf" "1" "-localbuftype" "heap")
set_tests_properties(tile_get_wave_colmajor_n2_w1_wf1_1048576B_uuid PROPERTIES
    TIMEOUT 300
    SKIP_RETURN_CODE 125
    LABELS "functional;backend_ipc;backend_gda;gpu_all;variant_uuid;launcher_slr;TILE"
    PROCESSORS 2
    WILL_FAIL FALSE
    FAIL_REGULAR_EXPRESSION "FAILED")

add_test(tile_get_wg_contiguous_n2_w1_wf16_1048576B_uuid "env" "ROCSHMEM_SLR_NP=2" "ROCSHMEM_MAX_NUM_CONTEXTS=16" "ROCSHMEM_HEAP_SIZE=6442450944" "ROCSHMEM_TEST_UUID=1" "../../../../share/rocshmem/test_wrapper.sh" "tile_get_wg_contiguous_n2_w1_wf16_1048576B_uuid" "../../../rocshmem_functional_tests" "-a" "110" "-w" "1" "-s" "1048576" "--num-wf" "16" "-localbuftype" "heap")
set_tests_properties(tile_get_wg_contiguous_n2_w1_wf16_1048576B_uuid PROPERTIES
    TIMEOUT 300
    SKIP_RETURN_CODE 125
    LABELS "functional;backend_ipc;backend_gda;gpu_all;variant_uuid;launcher_slr;TILE"
    PROCESSORS 2
    WILL_FAIL FALSE
    FAIL_REGULAR_EXPRESSION "FAILED")

add_test(tile_get_wg_rowmajor_n2_w1_wf16_1048576B_uuid "env" "ROCSHMEM_SLR_NP=2" "ROCSHMEM_MAX_NUM_CONTEXTS=16" "ROCSHMEM_HEAP_SIZE=6442450944" "ROCSHMEM_TEST_UUID=1" "../../../../share/rocshmem/test_wrapper.sh" "tile_get_wg_rowmajor_n2_w1_wf16_1048576B_uuid" "../../../rocshmem_functional_tests" "-a" "169" "-w" "1" "-s" "1048576" "--num-wf" "16" "-localbuftype" "heap")
set_tests_properties(tile_get_wg_rowmajor_n2_w1_wf16_1048576B_uuid PROPERTIES
    TIMEOUT 300
    SKIP_RETURN_CODE 125
    LABELS "functional;backend_ipc;backend_gda;gpu_all;variant_uuid;launcher_slr;TILE"
    PROCESSORS 2
    WILL_FAIL FALSE
    FAIL_REGULAR_EXPRESSION "FAILED")

add_test(tile_get_wg_colmajor_n2_w1_wf16_1048576B_uuid "env" "ROCSHMEM_SLR_NP=2" "ROCSHMEM_MAX_NUM_CONTEXTS=16" "ROCSHMEM_HEAP_SIZE=6442450944" "ROCSHMEM_TEST_UUID=1" "../../../../share/rocshmem/test_wrapper.sh" "tile_get_wg_colmajor_n2_w1_wf16_1048576B_uuid" "../../../rocshmem_functional_tests" "-a" "170" "-w" "1" "-s" "1048576" "--num-wf" "16" "-localbuftype" "heap")
set_tests_properties(tile_get_wg_colmajor_n2_w1_wf16_1048576B_uuid PROPERTIES
    TIMEOUT 300
    SKIP_RETURN_CODE 125
    LABELS "functional;backend_ipc;backend_gda;gpu_all;variant_uuid;launcher_slr;TILE"
    PROCESSORS 2
    WILL_FAIL FALSE
    FAIL_REGULAR_EXPRESSION "FAILED")

add_test(tile_get_wg_contiguous_n2_w4_wf16_1048576B_uuid "env" "ROCSHMEM_SLR_NP=2" "ROCSHMEM_MAX_NUM_CONTEXTS=64" "ROCSHMEM_HEAP_SIZE=6442450944" "ROCSHMEM_TEST_UUID=1" "../../../../share/rocshmem/test_wrapper.sh" "tile_get_wg_contiguous_n2_w4_wf16_1048576B_uuid" "../../../rocshmem_functional_tests" "-a" "110" "-w" "4" "-s" "1048576" "--num-wf" "16" "-localbuftype" "heap")
set_tests_properties(tile_get_wg_contiguous_n2_w4_wf16_1048576B_uuid PROPERTIES
    TIMEOUT 300
    SKIP_RETURN_CODE 125
    LABELS "functional;backend_ipc;backend_gda;gpu_all;variant_uuid;launcher_slr;TILE"
    PROCESSORS 2
    WILL_FAIL FALSE
    FAIL_REGULAR_EXPRESSION "FAILED")

add_test(tile_put_rowmajor_start_coord_n2_w1_z1_uuid "env" "ROCSHMEM_SLR_NP=2" "ROCSHMEM_MAX_NUM_CONTEXTS=1" "ROCSHMEM_HEAP_SIZE=6442450944" "ROCSHMEM_TEST_UUID=1" "../../../../share/rocshmem/test_wrapper.sh" "tile_put_rowmajor_start_coord_n2_w1_z1_uuid" "../../../rocshmem_functional_tests" "-a" "177" "-w" "1" "-z" "1" "-localbuftype" "heap")
set_tests_properties(tile_put_rowmajor_start_coord_n2_w1_z1_uuid PROPERTIES
    TIMEOUT 300
    SKIP_RETURN_CODE 125
    LABELS "functional;backend_ipc;backend_gda;gpu_all;variant_uuid;launcher_slr;TILE"
    PROCESSORS 2
    WILL_FAIL FALSE
    FAIL_REGULAR_EXPRESSION "FAILED")

add_test(tile_put_wave_rowmajor_start_coord_n2_w1_wf1_uuid "env" "ROCSHMEM_SLR_NP=2" "ROCSHMEM_MAX_NUM_CONTEXTS=1" "ROCSHMEM_HEAP_SIZE=6442450944" "ROCSHMEM_TEST_UUID=1" "../../../../share/rocshmem/test_wrapper.sh" "tile_put_wave_rowmajor_start_coord_n2_w1_wf1_uuid" "../../../rocshmem_functional_tests" "-a" "179" "-w" "1" "--num-wf" "1" "-localbuftype" "heap")
set_tests_properties(tile_put_wave_rowmajor_start_coord_n2_w1_wf1_uuid PROPERTIES
    TIMEOUT 300
    SKIP_RETURN_CODE 125
    LABELS "functional;backend_ipc;backend_gda;gpu_all;variant_uuid;launcher_slr;TILE"
    PROCESSORS 2
    WILL_FAIL FALSE
    FAIL_REGULAR_EXPRESSION "FAILED")

add_test(tile_put_wg_rowmajor_start_coord_n2_w1_wf16_uuid "env" "ROCSHMEM_SLR_NP=2" "ROCSHMEM_MAX_NUM_CONTEXTS=16" "ROCSHMEM_HEAP_SIZE=6442450944" "ROCSHMEM_TEST_UUID=1" "../../../../share/rocshmem/test_wrapper.sh" "tile_put_wg_rowmajor_start_coord_n2_w1_wf16_uuid" "../../../rocshmem_functional_tests" "-a" "181" "-w" "1" "--num-wf" "16" "-localbuftype" "heap")
set_tests_properties(tile_put_wg_rowmajor_start_coord_n2_w1_wf16_uuid PROPERTIES
    TIMEOUT 300
    SKIP_RETURN_CODE 125
    LABELS "functional;backend_ipc;backend_gda;gpu_all;variant_uuid;launcher_slr;TILE"
    PROCESSORS 2
    WILL_FAIL FALSE
    FAIL_REGULAR_EXPRESSION "FAILED")

add_test(tile_get_rowmajor_start_coord_n2_w1_z1_uuid "env" "ROCSHMEM_SLR_NP=2" "ROCSHMEM_MAX_NUM_CONTEXTS=1" "ROCSHMEM_HEAP_SIZE=6442450944" "ROCSHMEM_TEST_UUID=1" "../../../../share/rocshmem/test_wrapper.sh" "tile_get_rowmajor_start_coord_n2_w1_z1_uuid" "../../../rocshmem_functional_tests" "-a" "178" "-w" "1" "-z" "1" "-localbuftype" "heap")
set_tests_properties(tile_get_rowmajor_start_coord_n2_w1_z1_uuid PROPERTIES
    TIMEOUT 300
    SKIP_RETURN_CODE 125
    LABELS "functional;backend_ipc;backend_gda;gpu_all;variant_uuid;launcher_slr;TILE"
    PROCESSORS 2
    WILL_FAIL FALSE
    FAIL_REGULAR_EXPRESSION "FAILED")

add_test(tile_get_wave_rowmajor_start_coord_n2_w1_wf1_uuid "env" "ROCSHMEM_SLR_NP=2" "ROCSHMEM_MAX_NUM_CONTEXTS=1" "ROCSHMEM_HEAP_SIZE=6442450944" "ROCSHMEM_TEST_UUID=1" "../../../../share/rocshmem/test_wrapper.sh" "tile_get_wave_rowmajor_start_coord_n2_w1_wf1_uuid" "../../../rocshmem_functional_tests" "-a" "180" "-w" "1" "--num-wf" "1" "-localbuftype" "heap")
set_tests_properties(tile_get_wave_rowmajor_start_coord_n2_w1_wf1_uuid PROPERTIES
    TIMEOUT 300
    SKIP_RETURN_CODE 125
    LABELS "functional;backend_ipc;backend_gda;gpu_all;variant_uuid;launcher_slr;TILE"
    PROCESSORS 2
    WILL_FAIL FALSE
    FAIL_REGULAR_EXPRESSION "FAILED")

add_test(tile_get_wg_rowmajor_start_coord_n2_w1_wf16_uuid "env" "ROCSHMEM_SLR_NP=2" "ROCSHMEM_MAX_NUM_CONTEXTS=16" "ROCSHMEM_HEAP_SIZE=6442450944" "ROCSHMEM_TEST_UUID=1" "../../../../share/rocshmem/test_wrapper.sh" "tile_get_wg_rowmajor_start_coord_n2_w1_wf16_uuid" "../../../rocshmem_functional_tests" "-a" "182" "-w" "1" "--num-wf" "16" "-localbuftype" "heap")
set_tests_properties(tile_get_wg_rowmajor_start_coord_n2_w1_wf16_uuid PROPERTIES
    TIMEOUT 300
    SKIP_RETURN_CODE 125
    LABELS "functional;backend_ipc;backend_gda;gpu_all;variant_uuid;launcher_slr;TILE"
    PROCESSORS 2
    WILL_FAIL FALSE
    FAIL_REGULAR_EXPRESSION "FAILED")

add_test(tile_put_3d_n2_w1_z1_uuid "env" "ROCSHMEM_SLR_NP=2" "ROCSHMEM_MAX_NUM_CONTEXTS=1" "ROCSHMEM_HEAP_SIZE=6442450944" "ROCSHMEM_TEST_UUID=1" "../../../../share/rocshmem/test_wrapper.sh" "tile_put_3d_n2_w1_z1_uuid" "../../../rocshmem_functional_tests" "-a" "183" "-w" "1" "-z" "1" "-localbuftype" "heap")
set_tests_properties(tile_put_3d_n2_w1_z1_uuid PROPERTIES
    TIMEOUT 300
    SKIP_RETURN_CODE 125
    LABELS "functional;backend_ipc;backend_gda;gpu_all;variant_uuid;launcher_slr;TILE"
    PROCESSORS 2
    WILL_FAIL FALSE
    FAIL_REGULAR_EXPRESSION "FAILED")

add_test(tile_get_3d_n2_w1_z1_uuid "env" "ROCSHMEM_SLR_NP=2" "ROCSHMEM_MAX_NUM_CONTEXTS=1" "ROCSHMEM_HEAP_SIZE=6442450944" "ROCSHMEM_TEST_UUID=1" "../../../../share/rocshmem/test_wrapper.sh" "tile_get_3d_n2_w1_z1_uuid" "../../../rocshmem_functional_tests" "-a" "184" "-w" "1" "-z" "1" "-localbuftype" "heap")
set_tests_properties(tile_get_3d_n2_w1_z1_uuid PROPERTIES
    TIMEOUT 300
    SKIP_RETURN_CODE 125
    LABELS "functional;backend_ipc;backend_gda;gpu_all;variant_uuid;launcher_slr;TILE"
    PROCESSORS 2
    WILL_FAIL FALSE
    FAIL_REGULAR_EXPRESSION "FAILED")

add_test(tile_put_wg_3d_n2_w4_wf16_uuid "env" "ROCSHMEM_SLR_NP=2" "ROCSHMEM_MAX_NUM_CONTEXTS=64" "ROCSHMEM_HEAP_SIZE=6442450944" "ROCSHMEM_TEST_UUID=1" "../../../../share/rocshmem/test_wrapper.sh" "tile_put_wg_3d_n2_w4_wf16_uuid" "../../../rocshmem_functional_tests" "-a" "185" "-w" "4" "--num-wf" "16" "-localbuftype" "heap")
set_tests_properties(tile_put_wg_3d_n2_w4_wf16_uuid PROPERTIES
    TIMEOUT 300
    SKIP_RETURN_CODE 125
    LABELS "functional;backend_ipc;backend_gda;gpu_all;variant_uuid;launcher_slr;TILE"
    PROCESSORS 2
    WILL_FAIL FALSE
    FAIL_REGULAR_EXPRESSION "FAILED")

add_test(tile_get_wg_3d_n2_w4_wf16_uuid "env" "ROCSHMEM_SLR_NP=2" "ROCSHMEM_MAX_NUM_CONTEXTS=64" "ROCSHMEM_HEAP_SIZE=6442450944" "ROCSHMEM_TEST_UUID=1" "../../../../share/rocshmem/test_wrapper.sh" "tile_get_wg_3d_n2_w4_wf16_uuid" "../../../rocshmem_functional_tests" "-a" "186" "-w" "4" "--num-wf" "16" "-localbuftype" "heap")
set_tests_properties(tile_get_wg_3d_n2_w4_wf16_uuid PROPERTIES
    TIMEOUT 300
    SKIP_RETURN_CODE 125
    LABELS "functional;backend_ipc;backend_gda;gpu_all;variant_uuid;launcher_slr;TILE"
    PROCESSORS 2
    WILL_FAIL FALSE
    FAIL_REGULAR_EXPRESSION "FAILED")

add_test(tile_put_4d_n2_w1_z1_uuid "env" "ROCSHMEM_SLR_NP=2" "ROCSHMEM_MAX_NUM_CONTEXTS=1" "ROCSHMEM_HEAP_SIZE=6442450944" "ROCSHMEM_TEST_UUID=1" "../../../../share/rocshmem/test_wrapper.sh" "tile_put_4d_n2_w1_z1_uuid" "../../../rocshmem_functional_tests" "-a" "187" "-w" "1" "-z" "1" "-localbuftype" "heap")
set_tests_properties(tile_put_4d_n2_w1_z1_uuid PROPERTIES
    TIMEOUT 300
    SKIP_RETURN_CODE 125
    LABELS "functional;backend_ipc;backend_gda;gpu_all;variant_uuid;launcher_slr;TILE"
    PROCESSORS 2
    WILL_FAIL FALSE
    FAIL_REGULAR_EXPRESSION "FAILED")

add_test(tile_get_4d_n2_w1_z1_uuid "env" "ROCSHMEM_SLR_NP=2" "ROCSHMEM_MAX_NUM_CONTEXTS=1" "ROCSHMEM_HEAP_SIZE=6442450944" "ROCSHMEM_TEST_UUID=1" "../../../../share/rocshmem/test_wrapper.sh" "tile_get_4d_n2_w1_z1_uuid" "../../../rocshmem_functional_tests" "-a" "188" "-w" "1" "-z" "1" "-localbuftype" "heap")
set_tests_properties(tile_get_4d_n2_w1_z1_uuid PROPERTIES
    TIMEOUT 300
    SKIP_RETURN_CODE 125
    LABELS "functional;backend_ipc;backend_gda;gpu_all;variant_uuid;launcher_slr;TILE"
    PROCESSORS 2
    WILL_FAIL FALSE
    FAIL_REGULAR_EXPRESSION "FAILED")

add_test(tile_put_wg_4d_n2_w4_wf16_uuid "env" "ROCSHMEM_SLR_NP=2" "ROCSHMEM_MAX_NUM_CONTEXTS=64" "ROCSHMEM_HEAP_SIZE=6442450944" "ROCSHMEM_TEST_UUID=1" "../../../../share/rocshmem/test_wrapper.sh" "tile_put_wg_4d_n2_w4_wf16_uuid" "../../../rocshmem_functional_tests" "-a" "189" "-w" "4" "--num-wf" "16" "-localbuftype" "heap")
set_tests_properties(tile_put_wg_4d_n2_w4_wf16_uuid PROPERTIES
    TIMEOUT 300
    SKIP_RETURN_CODE 125
    LABELS "functional;backend_ipc;backend_gda;gpu_all;variant_uuid;launcher_slr;TILE"
    PROCESSORS 2
    WILL_FAIL FALSE
    FAIL_REGULAR_EXPRESSION "FAILED")

add_test(tile_get_wg_4d_n2_w4_wf16_uuid "env" "ROCSHMEM_SLR_NP=2" "ROCSHMEM_MAX_NUM_CONTEXTS=64" "ROCSHMEM_HEAP_SIZE=6442450944" "ROCSHMEM_TEST_UUID=1" "../../../../share/rocshmem/test_wrapper.sh" "tile_get_wg_4d_n2_w4_wf16_uuid" "../../../rocshmem_functional_tests" "-a" "190" "-w" "4" "--num-wf" "16" "-localbuftype" "heap")
set_tests_properties(tile_get_wg_4d_n2_w4_wf16_uuid PROPERTIES
    TIMEOUT 300
    SKIP_RETURN_CODE 125
    LABELS "functional;backend_ipc;backend_gda;gpu_all;variant_uuid;launcher_slr;TILE"
    PROCESSORS 2
    WILL_FAIL FALSE
    FAIL_REGULAR_EXPRESSION "FAILED")

add_test(tile_put_wave_3d_n2_w1_wf4_uuid "env" "ROCSHMEM_SLR_NP=2" "ROCSHMEM_MAX_NUM_CONTEXTS=4" "ROCSHMEM_HEAP_SIZE=6442450944" "ROCSHMEM_TEST_UUID=1" "../../../../share/rocshmem/test_wrapper.sh" "tile_put_wave_3d_n2_w1_wf4_uuid" "../../../rocshmem_functional_tests" "-a" "191" "-w" "1" "--num-wf" "4" "-localbuftype" "heap")
set_tests_properties(tile_put_wave_3d_n2_w1_wf4_uuid PROPERTIES
    TIMEOUT 300
    SKIP_RETURN_CODE 125
    LABELS "functional;backend_ipc;backend_gda;gpu_all;variant_uuid;launcher_slr;TILE"
    PROCESSORS 2
    WILL_FAIL FALSE
    FAIL_REGULAR_EXPRESSION "FAILED")

add_test(tile_get_wave_3d_n2_w1_wf4_uuid "env" "ROCSHMEM_SLR_NP=2" "ROCSHMEM_MAX_NUM_CONTEXTS=4" "ROCSHMEM_HEAP_SIZE=6442450944" "ROCSHMEM_TEST_UUID=1" "../../../../share/rocshmem/test_wrapper.sh" "tile_get_wave_3d_n2_w1_wf4_uuid" "../../../rocshmem_functional_tests" "-a" "192" "-w" "1" "--num-wf" "4" "-localbuftype" "heap")
set_tests_properties(tile_get_wave_3d_n2_w1_wf4_uuid PROPERTIES
    TIMEOUT 300
    SKIP_RETURN_CODE 125
    LABELS "functional;backend_ipc;backend_gda;gpu_all;variant_uuid;launcher_slr;TILE"
    PROCESSORS 2
    WILL_FAIL FALSE
    FAIL_REGULAR_EXPRESSION "FAILED")

add_test(tile_put_wave_4d_n2_w1_wf4_uuid "env" "ROCSHMEM_SLR_NP=2" "ROCSHMEM_MAX_NUM_CONTEXTS=4" "ROCSHMEM_HEAP_SIZE=6442450944" "ROCSHMEM_TEST_UUID=1" "../../../../share/rocshmem/test_wrapper.sh" "tile_put_wave_4d_n2_w1_wf4_uuid" "../../../rocshmem_functional_tests" "-a" "193" "-w" "1" "--num-wf" "4" "-localbuftype" "heap")
set_tests_properties(tile_put_wave_4d_n2_w1_wf4_uuid PROPERTIES
    TIMEOUT 300
    SKIP_RETURN_CODE 125
    LABELS "functional;backend_ipc;backend_gda;gpu_all;variant_uuid;launcher_slr;TILE"
    PROCESSORS 2
    WILL_FAIL FALSE
    FAIL_REGULAR_EXPRESSION "FAILED")

add_test(tile_get_wave_4d_n2_w1_wf4_uuid "env" "ROCSHMEM_SLR_NP=2" "ROCSHMEM_MAX_NUM_CONTEXTS=4" "ROCSHMEM_HEAP_SIZE=6442450944" "ROCSHMEM_TEST_UUID=1" "../../../../share/rocshmem/test_wrapper.sh" "tile_get_wave_4d_n2_w1_wf4_uuid" "../../../rocshmem_functional_tests" "-a" "194" "-w" "1" "--num-wf" "4" "-localbuftype" "heap")
set_tests_properties(tile_get_wave_4d_n2_w1_wf4_uuid PROPERTIES
    TIMEOUT 300
    SKIP_RETURN_CODE 125
    LABELS "functional;backend_ipc;backend_gda;gpu_all;variant_uuid;launcher_slr;TILE"
    PROCESSORS 2
    WILL_FAIL FALSE
    FAIL_REGULAR_EXPRESSION "FAILED")

add_test(tile_broadcast_n2_w1_z1_1048576B_uuid "env" "ROCSHMEM_SLR_NP=2" "ROCSHMEM_MAX_NUM_CONTEXTS=1" "ROCSHMEM_HEAP_SIZE=6442450944" "ROCSHMEM_TEST_UUID=1" "../../../../share/rocshmem/test_wrapper.sh" "tile_broadcast_n2_w1_z1_1048576B_uuid" "../../../rocshmem_functional_tests" "-a" "132" "-w" "1" "-z" "1" "-s" "1048576" "-localbuftype" "heap")
set_tests_properties(tile_broadcast_n2_w1_z1_1048576B_uuid PROPERTIES
    TIMEOUT 300
    SKIP_RETURN_CODE 125
    LABELS "functional;backend_ipc;backend_gda;gpu_all;variant_uuid;launcher_slr;TILE"
    PROCESSORS 2
    WILL_FAIL FALSE
    FAIL_REGULAR_EXPRESSION "FAILED")

add_test(tile_broadcast_n4_w1_z1_1048576B_uuid "env" "ROCSHMEM_SLR_NP=4" "ROCSHMEM_MAX_NUM_CONTEXTS=1" "ROCSHMEM_HEAP_SIZE=6442450944" "ROCSHMEM_TEST_UUID=1" "../../../../share/rocshmem/test_wrapper.sh" "tile_broadcast_n4_w1_z1_1048576B_uuid" "../../../rocshmem_functional_tests" "-a" "132" "-w" "1" "-z" "1" "-s" "1048576" "-localbuftype" "heap")
set_tests_properties(tile_broadcast_n4_w1_z1_1048576B_uuid PROPERTIES
    TIMEOUT 300
    SKIP_RETURN_CODE 125
    LABELS "functional;backend_ipc;backend_gda;gpu_all;variant_uuid;launcher_slr;TILE"
    PROCESSORS 4
    WILL_FAIL FALSE
    FAIL_REGULAR_EXPRESSION "FAILED")

add_test(tile_broadcast_wave_n2_w1_wf4_1048576B_uuid "env" "ROCSHMEM_SLR_NP=2" "ROCSHMEM_MAX_NUM_CONTEXTS=4" "ROCSHMEM_HEAP_SIZE=6442450944" "ROCSHMEM_TEST_UUID=1" "../../../../share/rocshmem/test_wrapper.sh" "tile_broadcast_wave_n2_w1_wf4_1048576B_uuid" "../../../rocshmem_functional_tests" "-a" "133" "-w" "1" "-s" "1048576" "--num-wf" "4" "-localbuftype" "heap")
set_tests_properties(tile_broadcast_wave_n2_w1_wf4_1048576B_uuid PROPERTIES
    TIMEOUT 300
    SKIP_RETURN_CODE 125
    LABELS "functional;backend_ipc;backend_gda;gpu_all;variant_uuid;launcher_slr;TILE"
    PROCESSORS 2
    WILL_FAIL FALSE
    FAIL_REGULAR_EXPRESSION "FAILED")

add_test(tile_broadcast_wave_n4_w1_wf4_1048576B_uuid "env" "ROCSHMEM_SLR_NP=4" "ROCSHMEM_MAX_NUM_CONTEXTS=4" "ROCSHMEM_HEAP_SIZE=6442450944" "ROCSHMEM_TEST_UUID=1" "../../../../share/rocshmem/test_wrapper.sh" "tile_broadcast_wave_n4_w1_wf4_1048576B_uuid" "../../../rocshmem_functional_tests" "-a" "133" "-w" "1" "-s" "1048576" "--num-wf" "4" "-localbuftype" "heap")
set_tests_properties(tile_broadcast_wave_n4_w1_wf4_1048576B_uuid PROPERTIES
    TIMEOUT 300
    SKIP_RETURN_CODE 125
    LABELS "functional;backend_ipc;backend_gda;gpu_all;variant_uuid;launcher_slr;TILE"
    PROCESSORS 4
    WILL_FAIL FALSE
    FAIL_REGULAR_EXPRESSION "FAILED")

add_test(tile_broadcast_wg_n2_w4_wf1_1048576B_uuid "env" "ROCSHMEM_SLR_NP=2" "ROCSHMEM_MAX_NUM_CONTEXTS=4" "ROCSHMEM_HEAP_SIZE=6442450944" "ROCSHMEM_TEST_UUID=1" "../../../../share/rocshmem/test_wrapper.sh" "tile_broadcast_wg_n2_w4_wf1_1048576B_uuid" "../../../rocshmem_functional_tests" "-a" "134" "-w" "4" "-s" "1048576" "--num-wf" "1" "-localbuftype" "heap")
set_tests_properties(tile_broadcast_wg_n2_w4_wf1_1048576B_uuid PROPERTIES
    TIMEOUT 300
    SKIP_RETURN_CODE 125
    LABELS "functional;backend_ipc;backend_gda;gpu_all;variant_uuid;launcher_slr;TILE"
    PROCESSORS 2
    WILL_FAIL FALSE
    FAIL_REGULAR_EXPRESSION "FAILED")

add_test(tile_broadcast_wg_n4_w4_wf1_1048576B_uuid "env" "ROCSHMEM_SLR_NP=4" "ROCSHMEM_MAX_NUM_CONTEXTS=4" "ROCSHMEM_HEAP_SIZE=6442450944" "ROCSHMEM_TEST_UUID=1" "../../../../share/rocshmem/test_wrapper.sh" "tile_broadcast_wg_n4_w4_wf1_1048576B_uuid" "../../../rocshmem_functional_tests" "-a" "134" "-w" "4" "-s" "1048576" "--num-wf" "1" "-localbuftype" "heap")
set_tests_properties(tile_broadcast_wg_n4_w4_wf1_1048576B_uuid PROPERTIES
    TIMEOUT 300
    SKIP_RETURN_CODE 125
    LABELS "functional;backend_ipc;backend_gda;gpu_all;variant_uuid;launcher_slr;TILE"
    PROCESSORS 4
    WILL_FAIL FALSE
    FAIL_REGULAR_EXPRESSION "FAILED")

add_test(tile_allgather_n2_w1_z1_1048576B_uuid "env" "ROCSHMEM_SLR_NP=2" "ROCSHMEM_MAX_NUM_CONTEXTS=1" "ROCSHMEM_HEAP_SIZE=6442450944" "ROCSHMEM_TEST_UUID=1" "../../../../share/rocshmem/test_wrapper.sh" "tile_allgather_n2_w1_z1_1048576B_uuid" "../../../rocshmem_functional_tests" "-a" "135" "-w" "1" "-z" "1" "-s" "1048576" "-localbuftype" "heap")
set_tests_properties(tile_allgather_n2_w1_z1_1048576B_uuid PROPERTIES
    TIMEOUT 300
    SKIP_RETURN_CODE 125
    LABELS "functional;backend_ipc;backend_gda;gpu_all;variant_uuid;launcher_slr;TILE"
    PROCESSORS 2
    WILL_FAIL FALSE
    FAIL_REGULAR_EXPRESSION "FAILED")

add_test(tile_allgather_n4_w1_z1_1048576B_uuid "env" "ROCSHMEM_SLR_NP=4" "ROCSHMEM_MAX_NUM_CONTEXTS=1" "ROCSHMEM_HEAP_SIZE=6442450944" "ROCSHMEM_TEST_UUID=1" "../../../../share/rocshmem/test_wrapper.sh" "tile_allgather_n4_w1_z1_1048576B_uuid" "../../../rocshmem_functional_tests" "-a" "135" "-w" "1" "-z" "1" "-s" "1048576" "-localbuftype" "heap")
set_tests_properties(tile_allgather_n4_w1_z1_1048576B_uuid PROPERTIES
    TIMEOUT 300
    SKIP_RETURN_CODE 125
    LABELS "functional;backend_ipc;backend_gda;gpu_all;variant_uuid;launcher_slr;TILE"
    PROCESSORS 4
    WILL_FAIL FALSE
    FAIL_REGULAR_EXPRESSION "FAILED")

add_test(tile_allgather_wave_n2_w1_wf4_1048576B_uuid "env" "ROCSHMEM_SLR_NP=2" "ROCSHMEM_MAX_NUM_CONTEXTS=4" "ROCSHMEM_HEAP_SIZE=6442450944" "ROCSHMEM_TEST_UUID=1" "../../../../share/rocshmem/test_wrapper.sh" "tile_allgather_wave_n2_w1_wf4_1048576B_uuid" "../../../rocshmem_functional_tests" "-a" "136" "-w" "1" "-s" "1048576" "--num-wf" "4" "-localbuftype" "heap")
set_tests_properties(tile_allgather_wave_n2_w1_wf4_1048576B_uuid PROPERTIES
    TIMEOUT 300
    SKIP_RETURN_CODE 125
    LABELS "functional;backend_ipc;backend_gda;gpu_all;variant_uuid;launcher_slr;TILE"
    PROCESSORS 2
    WILL_FAIL FALSE
    FAIL_REGULAR_EXPRESSION "FAILED")

add_test(tile_allgather_wave_n4_w1_wf4_1048576B_uuid "env" "ROCSHMEM_SLR_NP=4" "ROCSHMEM_MAX_NUM_CONTEXTS=4" "ROCSHMEM_HEAP_SIZE=6442450944" "ROCSHMEM_TEST_UUID=1" "../../../../share/rocshmem/test_wrapper.sh" "tile_allgather_wave_n4_w1_wf4_1048576B_uuid" "../../../rocshmem_functional_tests" "-a" "136" "-w" "1" "-s" "1048576" "--num-wf" "4" "-localbuftype" "heap")
set_tests_properties(tile_allgather_wave_n4_w1_wf4_1048576B_uuid PROPERTIES
    TIMEOUT 300
    SKIP_RETURN_CODE 125
    LABELS "functional;backend_ipc;backend_gda;gpu_all;variant_uuid;launcher_slr;TILE"
    PROCESSORS 4
    WILL_FAIL FALSE
    FAIL_REGULAR_EXPRESSION "FAILED")

add_test(tile_allgather_wg_n2_w4_wf1_1048576B_uuid "env" "ROCSHMEM_SLR_NP=2" "ROCSHMEM_MAX_NUM_CONTEXTS=4" "ROCSHMEM_HEAP_SIZE=6442450944" "ROCSHMEM_TEST_UUID=1" "../../../../share/rocshmem/test_wrapper.sh" "tile_allgather_wg_n2_w4_wf1_1048576B_uuid" "../../../rocshmem_functional_tests" "-a" "137" "-w" "4" "-s" "1048576" "--num-wf" "1" "-localbuftype" "heap")
set_tests_properties(tile_allgather_wg_n2_w4_wf1_1048576B_uuid PROPERTIES
    TIMEOUT 300
    SKIP_RETURN_CODE 125
    LABELS "functional;backend_ipc;backend_gda;gpu_all;variant_uuid;launcher_slr;TILE"
    PROCESSORS 2
    WILL_FAIL FALSE
    FAIL_REGULAR_EXPRESSION "FAILED")

add_test(tile_allgather_wg_n4_w4_wf1_1048576B_uuid "env" "ROCSHMEM_SLR_NP=4" "ROCSHMEM_MAX_NUM_CONTEXTS=4" "ROCSHMEM_HEAP_SIZE=6442450944" "ROCSHMEM_TEST_UUID=1" "../../../../share/rocshmem/test_wrapper.sh" "tile_allgather_wg_n4_w4_wf1_1048576B_uuid" "../../../rocshmem_functional_tests" "-a" "137" "-w" "4" "-s" "1048576" "--num-wf" "1" "-localbuftype" "heap")
set_tests_properties(tile_allgather_wg_n4_w4_wf1_1048576B_uuid PROPERTIES
    TIMEOUT 300
    SKIP_RETURN_CODE 125
    LABELS "functional;backend_ipc;backend_gda;gpu_all;variant_uuid;launcher_slr;TILE"
    PROCESSORS 4
    WILL_FAIL FALSE
    FAIL_REGULAR_EXPRESSION "FAILED")

add_test(tile_reduce_n2_w1_z1_1048576B_uuid "env" "ROCSHMEM_SLR_NP=2" "ROCSHMEM_MAX_NUM_CONTEXTS=1" "ROCSHMEM_HEAP_SIZE=6442450944" "ROCSHMEM_TEST_UUID=1" "../../../../share/rocshmem/test_wrapper.sh" "tile_reduce_n2_w1_z1_1048576B_uuid" "../../../rocshmem_functional_tests" "-a" "155" "-w" "1" "-z" "1" "-s" "1048576" "-localbuftype" "heap")
set_tests_properties(tile_reduce_n2_w1_z1_1048576B_uuid PROPERTIES
    TIMEOUT 300
    SKIP_RETURN_CODE 125
    LABELS "functional;backend_ipc;backend_gda;gpu_all;variant_uuid;launcher_slr;TILE"
    PROCESSORS 2
    WILL_FAIL FALSE
    FAIL_REGULAR_EXPRESSION "FAILED")

add_test(tile_reduce_n4_w1_z1_65536B_uuid "env" "ROCSHMEM_SLR_NP=4" "ROCSHMEM_MAX_NUM_CONTEXTS=1" "ROCSHMEM_HEAP_SIZE=6442450944" "ROCSHMEM_TEST_UUID=1" "../../../../share/rocshmem/test_wrapper.sh" "tile_reduce_n4_w1_z1_65536B_uuid" "../../../rocshmem_functional_tests" "-a" "155" "-w" "1" "-z" "1" "-s" "65536" "-localbuftype" "heap")
set_tests_properties(tile_reduce_n4_w1_z1_65536B_uuid PROPERTIES
    TIMEOUT 300
    SKIP_RETURN_CODE 125
    LABELS "functional;backend_ipc;backend_gda;gpu_all;variant_uuid;launcher_slr;TILE"
    PROCESSORS 4
    WILL_FAIL FALSE
    FAIL_REGULAR_EXPRESSION "FAILED")

add_test(tile_reduce_wave_n2_w1_wf4_1048576B_uuid "env" "ROCSHMEM_SLR_NP=2" "ROCSHMEM_MAX_NUM_CONTEXTS=4" "ROCSHMEM_HEAP_SIZE=6442450944" "ROCSHMEM_TEST_UUID=1" "../../../../share/rocshmem/test_wrapper.sh" "tile_reduce_wave_n2_w1_wf4_1048576B_uuid" "../../../rocshmem_functional_tests" "-a" "156" "-w" "1" "-s" "1048576" "--num-wf" "4" "-localbuftype" "heap")
set_tests_properties(tile_reduce_wave_n2_w1_wf4_1048576B_uuid PROPERTIES
    TIMEOUT 300
    SKIP_RETURN_CODE 125
    LABELS "functional;backend_ipc;backend_gda;gpu_all;variant_uuid;launcher_slr;TILE"
    PROCESSORS 2
    WILL_FAIL FALSE
    FAIL_REGULAR_EXPRESSION "FAILED")

add_test(tile_reduce_wave_n4_w1_wf4_65536B_uuid "env" "ROCSHMEM_SLR_NP=4" "ROCSHMEM_MAX_NUM_CONTEXTS=4" "ROCSHMEM_HEAP_SIZE=6442450944" "ROCSHMEM_TEST_UUID=1" "../../../../share/rocshmem/test_wrapper.sh" "tile_reduce_wave_n4_w1_wf4_65536B_uuid" "../../../rocshmem_functional_tests" "-a" "156" "-w" "1" "-s" "65536" "--num-wf" "4" "-localbuftype" "heap")
set_tests_properties(tile_reduce_wave_n4_w1_wf4_65536B_uuid PROPERTIES
    TIMEOUT 300
    SKIP_RETURN_CODE 125
    LABELS "functional;backend_ipc;backend_gda;gpu_all;variant_uuid;launcher_slr;TILE"
    PROCESSORS 4
    WILL_FAIL FALSE
    FAIL_REGULAR_EXPRESSION "FAILED")

add_test(tile_reduce_wg_n2_w4_wf1_1048576B_uuid "env" "ROCSHMEM_SLR_NP=2" "ROCSHMEM_MAX_NUM_CONTEXTS=4" "ROCSHMEM_HEAP_SIZE=6442450944" "ROCSHMEM_TEST_UUID=1" "../../../../share/rocshmem/test_wrapper.sh" "tile_reduce_wg_n2_w4_wf1_1048576B_uuid" "../../../rocshmem_functional_tests" "-a" "157" "-w" "4" "-s" "1048576" "--num-wf" "1" "-localbuftype" "heap")
set_tests_properties(tile_reduce_wg_n2_w4_wf1_1048576B_uuid PROPERTIES
    TIMEOUT 300
    SKIP_RETURN_CODE 125
    LABELS "functional;backend_ipc;backend_gda;gpu_all;variant_uuid;launcher_slr;TILE"
    PROCESSORS 2
    WILL_FAIL FALSE
    FAIL_REGULAR_EXPRESSION "FAILED")

add_test(tile_reduce_wg_n4_w4_wf1_65536B_uuid "env" "ROCSHMEM_SLR_NP=4" "ROCSHMEM_MAX_NUM_CONTEXTS=4" "ROCSHMEM_HEAP_SIZE=6442450944" "ROCSHMEM_TEST_UUID=1" "../../../../share/rocshmem/test_wrapper.sh" "tile_reduce_wg_n4_w4_wf1_65536B_uuid" "../../../rocshmem_functional_tests" "-a" "157" "-w" "4" "-s" "65536" "--num-wf" "1" "-localbuftype" "heap")
set_tests_properties(tile_reduce_wg_n4_w4_wf1_65536B_uuid PROPERTIES
    TIMEOUT 300
    SKIP_RETURN_CODE 125
    LABELS "functional;backend_ipc;backend_gda;gpu_all;variant_uuid;launcher_slr;TILE"
    PROCESSORS 4
    WILL_FAIL FALSE
    FAIL_REGULAR_EXPRESSION "FAILED")

add_test(host_putmem_n2_w1_z1_65536B_uuid "env" "ROCSHMEM_SLR_NP=2" "ROCSHMEM_MAX_NUM_CONTEXTS=1" "ROCSHMEM_HEAP_SIZE=6442450944" "ROCSHMEM_TEST_UUID=1" "ROCSHMEM_TEST_UUID=1" "../../../../share/rocshmem/test_wrapper.sh" "host_putmem_n2_w1_z1_65536B_uuid" "../../../rocshmem_functional_tests" "-a" "121" "-w" "1" "-z" "1" "-s" "65536" "-localbuftype" "heap")
set_tests_properties(host_putmem_n2_w1_z1_65536B_uuid PROPERTIES
    TIMEOUT 300
    SKIP_RETURN_CODE 125
    LABELS "functional;backend_ipc;gpu_all;variant_uuid;launcher_slr;HOST"
    PROCESSORS 2
    WILL_FAIL FALSE
    FAIL_REGULAR_EXPRESSION "FAILED")

add_test(host_getmem_n2_w1_z1_65536B_uuid "env" "ROCSHMEM_SLR_NP=2" "ROCSHMEM_MAX_NUM_CONTEXTS=1" "ROCSHMEM_HEAP_SIZE=6442450944" "ROCSHMEM_TEST_UUID=1" "ROCSHMEM_TEST_UUID=1" "../../../../share/rocshmem/test_wrapper.sh" "host_getmem_n2_w1_z1_65536B_uuid" "../../../rocshmem_functional_tests" "-a" "122" "-w" "1" "-z" "1" "-s" "65536" "-localbuftype" "heap")
set_tests_properties(host_getmem_n2_w1_z1_65536B_uuid PROPERTIES
    TIMEOUT 300
    SKIP_RETURN_CODE 125
    LABELS "functional;backend_ipc;gpu_all;variant_uuid;launcher_slr;HOST"
    PROCESSORS 2
    WILL_FAIL FALSE
    FAIL_REGULAR_EXPRESSION "FAILED")

add_test(host_amo_fadd_n2_w1_z1_uuid "env" "ROCSHMEM_SLR_NP=2" "ROCSHMEM_MAX_NUM_CONTEXTS=1" "ROCSHMEM_HEAP_SIZE=6442450944" "ROCSHMEM_TEST_UUID=1" "ROCSHMEM_TEST_UUID=1" "../../../../share/rocshmem/test_wrapper.sh" "host_amo_fadd_n2_w1_z1_uuid" "../../../rocshmem_functional_tests" "-a" "123" "-w" "1" "-z" "1" "-localbuftype" "heap")
set_tests_properties(host_amo_fadd_n2_w1_z1_uuid PROPERTIES
    TIMEOUT 300
    SKIP_RETURN_CODE 125
    LABELS "functional;backend_ipc;gpu_all;variant_uuid;launcher_slr;HOST"
    PROCESSORS 2
    WILL_FAIL FALSE
    FAIL_REGULAR_EXPRESSION "FAILED")

add_test(host_amo_fcswap_n2_w1_z1_uuid "env" "ROCSHMEM_SLR_NP=2" "ROCSHMEM_MAX_NUM_CONTEXTS=1" "ROCSHMEM_HEAP_SIZE=6442450944" "ROCSHMEM_TEST_UUID=1" "ROCSHMEM_TEST_UUID=1" "../../../../share/rocshmem/test_wrapper.sh" "host_amo_fcswap_n2_w1_z1_uuid" "../../../rocshmem_functional_tests" "-a" "124" "-w" "1" "-z" "1" "-localbuftype" "heap")
set_tests_properties(host_amo_fcswap_n2_w1_z1_uuid PROPERTIES
    TIMEOUT 300
    SKIP_RETURN_CODE 125
    LABELS "functional;backend_ipc;gpu_all;variant_uuid;launcher_slr;HOST"
    PROCESSORS 2
    WILL_FAIL FALSE
    FAIL_REGULAR_EXPRESSION "FAILED")

add_test(host_int_amo_fadd_n2_w1_z1_uuid "env" "ROCSHMEM_SLR_NP=2" "ROCSHMEM_MAX_NUM_CONTEXTS=1" "ROCSHMEM_HEAP_SIZE=6442450944" "ROCSHMEM_TEST_UUID=1" "ROCSHMEM_TEST_UUID=1" "../../../../share/rocshmem/test_wrapper.sh" "host_int_amo_fadd_n2_w1_z1_uuid" "../../../rocshmem_functional_tests" "-a" "127" "-w" "1" "-z" "1" "-localbuftype" "heap")
set_tests_properties(host_int_amo_fadd_n2_w1_z1_uuid PROPERTIES
    TIMEOUT 300
    SKIP_RETURN_CODE 125
    LABELS "functional;backend_ipc;gpu_all;variant_uuid;launcher_slr;HOST"
    PROCESSORS 2
    WILL_FAIL FALSE
    FAIL_REGULAR_EXPRESSION "FAILED")

add_test(host_int_amo_fcswap_n2_w1_z1_uuid "env" "ROCSHMEM_SLR_NP=2" "ROCSHMEM_MAX_NUM_CONTEXTS=1" "ROCSHMEM_HEAP_SIZE=6442450944" "ROCSHMEM_TEST_UUID=1" "ROCSHMEM_TEST_UUID=1" "../../../../share/rocshmem/test_wrapper.sh" "host_int_amo_fcswap_n2_w1_z1_uuid" "../../../rocshmem_functional_tests" "-a" "128" "-w" "1" "-z" "1" "-localbuftype" "heap")
set_tests_properties(host_int_amo_fcswap_n2_w1_z1_uuid PROPERTIES
    TIMEOUT 300
    SKIP_RETURN_CODE 125
    LABELS "functional;backend_ipc;gpu_all;variant_uuid;launcher_slr;HOST"
    PROCESSORS 2
    WILL_FAIL FALSE
    FAIL_REGULAR_EXPRESSION "FAILED")

add_test(host_amo_add_n2_w1_z1_uuid "env" "ROCSHMEM_SLR_NP=2" "ROCSHMEM_MAX_NUM_CONTEXTS=1" "ROCSHMEM_HEAP_SIZE=6442450944" "ROCSHMEM_TEST_UUID=1" "ROCSHMEM_TEST_UUID=1" "ROCSHMEM_MAX_NUM_HOST_CONTEXTS=2" "../../../../share/rocshmem/test_wrapper.sh" "host_amo_add_n2_w1_z1_uuid" "../../../rocshmem_functional_tests" "-a" "131" "-w" "1" "-z" "1" "-localbuftype" "heap")
set_tests_properties(host_amo_add_n2_w1_z1_uuid PROPERTIES
    TIMEOUT 300
    SKIP_RETURN_CODE 125
    LABELS "functional;backend_ipc;gpu_all;variant_uuid;launcher_slr;HOST"
    PROCESSORS 2
    WILL_FAIL FALSE
    FAIL_REGULAR_EXPRESSION "FAILED")

add_test(host_ctx_putmem_n2_w1_z1_65536B_uuid "env" "ROCSHMEM_SLR_NP=2" "ROCSHMEM_MAX_NUM_CONTEXTS=1" "ROCSHMEM_HEAP_SIZE=6442450944" "ROCSHMEM_TEST_UUID=1" "ROCSHMEM_TEST_UUID=1" "ROCSHMEM_MAX_NUM_HOST_CONTEXTS=2" "../../../../share/rocshmem/test_wrapper.sh" "host_ctx_putmem_n2_w1_z1_65536B_uuid" "../../../rocshmem_functional_tests" "-a" "125" "-w" "1" "-z" "1" "-s" "65536" "-localbuftype" "heap")
set_tests_properties(host_ctx_putmem_n2_w1_z1_65536B_uuid PROPERTIES
    TIMEOUT 300
    SKIP_RETURN_CODE 125
    LABELS "functional;backend_ipc;gpu_all;variant_uuid;launcher_slr;HOST"
    PROCESSORS 2
    WILL_FAIL FALSE
    FAIL_REGULAR_EXPRESSION "FAILED")

add_test(host_ctx_getmem_n2_w1_z1_65536B_uuid "env" "ROCSHMEM_SLR_NP=2" "ROCSHMEM_MAX_NUM_CONTEXTS=1" "ROCSHMEM_HEAP_SIZE=6442450944" "ROCSHMEM_TEST_UUID=1" "ROCSHMEM_TEST_UUID=1" "ROCSHMEM_MAX_NUM_HOST_CONTEXTS=2" "../../../../share/rocshmem/test_wrapper.sh" "host_ctx_getmem_n2_w1_z1_65536B_uuid" "../../../rocshmem_functional_tests" "-a" "126" "-w" "1" "-z" "1" "-s" "65536" "-localbuftype" "heap")
set_tests_properties(host_ctx_getmem_n2_w1_z1_65536B_uuid PROPERTIES
    TIMEOUT 300
    SKIP_RETURN_CODE 125
    LABELS "functional;backend_ipc;gpu_all;variant_uuid;launcher_slr;HOST"
    PROCESSORS 2
    WILL_FAIL FALSE
    FAIL_REGULAR_EXPRESSION "FAILED")

add_test(host_amo_all_pes_n4_w1_z1_uuid "env" "ROCSHMEM_SLR_NP=4" "ROCSHMEM_MAX_NUM_CONTEXTS=1" "ROCSHMEM_HEAP_SIZE=6442450944" "ROCSHMEM_TEST_UUID=1" "ROCSHMEM_TEST_UUID=1" "../../../../share/rocshmem/test_wrapper.sh" "host_amo_all_pes_n4_w1_z1_uuid" "../../../rocshmem_functional_tests" "-a" "129" "-w" "1" "-z" "1" "-localbuftype" "heap")
set_tests_properties(host_amo_all_pes_n4_w1_z1_uuid PROPERTIES
    TIMEOUT 300
    SKIP_RETURN_CODE 125
    LABELS "functional;backend_ipc;gpu_all;variant_uuid;launcher_slr;HOST"
    PROCESSORS 4
    WILL_FAIL FALSE
    FAIL_REGULAR_EXPRESSION "FAILED")

add_test(host_amo_self_n4_w1_z1_uuid "env" "ROCSHMEM_SLR_NP=4" "ROCSHMEM_MAX_NUM_CONTEXTS=1" "ROCSHMEM_HEAP_SIZE=6442450944" "ROCSHMEM_TEST_UUID=1" "ROCSHMEM_TEST_UUID=1" "../../../../share/rocshmem/test_wrapper.sh" "host_amo_self_n4_w1_z1_uuid" "../../../rocshmem_functional_tests" "-a" "130" "-w" "1" "-z" "1" "-localbuftype" "heap")
set_tests_properties(host_amo_self_n4_w1_z1_uuid PROPERTIES
    TIMEOUT 300
    SKIP_RETURN_CODE 125
    LABELS "functional;backend_ipc;gpu_all;variant_uuid;launcher_slr;HOST"
    PROCESSORS 4
    WILL_FAIL FALSE
    FAIL_REGULAR_EXPRESSION "FAILED")

add_test(host_wait_until_n2_w1_z1_uuid "env" "ROCSHMEM_SLR_NP=2" "ROCSHMEM_MAX_NUM_CONTEXTS=1" "ROCSHMEM_HEAP_SIZE=6442450944" "ROCSHMEM_TEST_UUID=1" "ROCSHMEM_TEST_UUID=1" "../../../../share/rocshmem/test_wrapper.sh" "host_wait_until_n2_w1_z1_uuid" "../../../rocshmem_functional_tests" "-a" "138" "-w" "1" "-z" "1" "-localbuftype" "heap")
set_tests_properties(host_wait_until_n2_w1_z1_uuid PROPERTIES
    TIMEOUT 300
    SKIP_RETURN_CODE 125
    LABELS "functional;backend_ipc;gpu_all;variant_uuid;launcher_slr;HOST"
    PROCESSORS 2
    WILL_FAIL FALSE
    FAIL_REGULAR_EXPRESSION "FAILED")

add_test(host_test_n2_w1_z1_uuid "env" "ROCSHMEM_SLR_NP=2" "ROCSHMEM_MAX_NUM_CONTEXTS=1" "ROCSHMEM_HEAP_SIZE=6442450944" "ROCSHMEM_TEST_UUID=1" "ROCSHMEM_TEST_UUID=1" "../../../../share/rocshmem/test_wrapper.sh" "host_test_n2_w1_z1_uuid" "../../../rocshmem_functional_tests" "-a" "139" "-w" "1" "-z" "1" "-localbuftype" "heap")
set_tests_properties(host_test_n2_w1_z1_uuid PROPERTIES
    TIMEOUT 300
    SKIP_RETURN_CODE 125
    LABELS "functional;backend_ipc;gpu_all;variant_uuid;launcher_slr;HOST"
    PROCESSORS 2
    WILL_FAIL FALSE
    FAIL_REGULAR_EXPRESSION "FAILED")

add_test(host_wait_until_all_n2_w1_z1_uuid "env" "ROCSHMEM_SLR_NP=2" "ROCSHMEM_MAX_NUM_CONTEXTS=1" "ROCSHMEM_HEAP_SIZE=6442450944" "ROCSHMEM_TEST_UUID=1" "ROCSHMEM_TEST_UUID=1" "../../../../share/rocshmem/test_wrapper.sh" "host_wait_until_all_n2_w1_z1_uuid" "../../../rocshmem_functional_tests" "-a" "140" "-w" "1" "-z" "1" "-localbuftype" "heap")
set_tests_properties(host_wait_until_all_n2_w1_z1_uuid PROPERTIES
    TIMEOUT 300
    SKIP_RETURN_CODE 125
    LABELS "functional;backend_ipc;gpu_all;variant_uuid;launcher_slr;HOST"
    PROCESSORS 2
    WILL_FAIL FALSE
    FAIL_REGULAR_EXPRESSION "FAILED")

add_test(host_wait_until_any_n2_w1_z1_uuid "env" "ROCSHMEM_SLR_NP=2" "ROCSHMEM_MAX_NUM_CONTEXTS=1" "ROCSHMEM_HEAP_SIZE=6442450944" "ROCSHMEM_TEST_UUID=1" "ROCSHMEM_TEST_UUID=1" "../../../../share/rocshmem/test_wrapper.sh" "host_wait_until_any_n2_w1_z1_uuid" "../../../rocshmem_functional_tests" "-a" "141" "-w" "1" "-z" "1" "-localbuftype" "heap")
set_tests_properties(host_wait_until_any_n2_w1_z1_uuid PROPERTIES
    TIMEOUT 300
    SKIP_RETURN_CODE 125
    LABELS "functional;backend_ipc;gpu_all;variant_uuid;launcher_slr;HOST"
    PROCESSORS 2
    WILL_FAIL FALSE
    FAIL_REGULAR_EXPRESSION "FAILED")

add_test(host_wait_until_some_n2_w1_z1_uuid "env" "ROCSHMEM_SLR_NP=2" "ROCSHMEM_MAX_NUM_CONTEXTS=1" "ROCSHMEM_HEAP_SIZE=6442450944" "ROCSHMEM_TEST_UUID=1" "ROCSHMEM_TEST_UUID=1" "../../../../share/rocshmem/test_wrapper.sh" "host_wait_until_some_n2_w1_z1_uuid" "../../../rocshmem_functional_tests" "-a" "142" "-w" "1" "-z" "1" "-localbuftype" "heap")
set_tests_properties(host_wait_until_some_n2_w1_z1_uuid PROPERTIES
    TIMEOUT 300
    SKIP_RETURN_CODE 125
    LABELS "functional;backend_ipc;gpu_all;variant_uuid;launcher_slr;HOST"
    PROCESSORS 2
    WILL_FAIL FALSE
    FAIL_REGULAR_EXPRESSION "FAILED")

add_test(host_wait_until_all_vector_n2_w1_z1_uuid "env" "ROCSHMEM_SLR_NP=2" "ROCSHMEM_MAX_NUM_CONTEXTS=1" "ROCSHMEM_HEAP_SIZE=6442450944" "ROCSHMEM_TEST_UUID=1" "ROCSHMEM_TEST_UUID=1" "../../../../share/rocshmem/test_wrapper.sh" "host_wait_until_all_vector_n2_w1_z1_uuid" "../../../rocshmem_functional_tests" "-a" "143" "-w" "1" "-z" "1" "-localbuftype" "heap")
set_tests_properties(host_wait_until_all_vector_n2_w1_z1_uuid PROPERTIES
    TIMEOUT 300
    SKIP_RETURN_CODE 125
    LABELS "functional;backend_ipc;gpu_all;variant_uuid;launcher_slr;HOST"
    PROCESSORS 2
    WILL_FAIL FALSE
    FAIL_REGULAR_EXPRESSION "FAILED")

add_test(host_wait_until_any_vector_n2_w1_z1_uuid "env" "ROCSHMEM_SLR_NP=2" "ROCSHMEM_MAX_NUM_CONTEXTS=1" "ROCSHMEM_HEAP_SIZE=6442450944" "ROCSHMEM_TEST_UUID=1" "ROCSHMEM_TEST_UUID=1" "../../../../share/rocshmem/test_wrapper.sh" "host_wait_until_any_vector_n2_w1_z1_uuid" "../../../rocshmem_functional_tests" "-a" "144" "-w" "1" "-z" "1" "-localbuftype" "heap")
set_tests_properties(host_wait_until_any_vector_n2_w1_z1_uuid PROPERTIES
    TIMEOUT 300
    SKIP_RETURN_CODE 125
    LABELS "functional;backend_ipc;gpu_all;variant_uuid;launcher_slr;HOST"
    PROCESSORS 2
    WILL_FAIL FALSE
    FAIL_REGULAR_EXPRESSION "FAILED")

add_test(host_wait_until_some_vector_n2_w1_z1_uuid "env" "ROCSHMEM_SLR_NP=2" "ROCSHMEM_MAX_NUM_CONTEXTS=1" "ROCSHMEM_HEAP_SIZE=6442450944" "ROCSHMEM_TEST_UUID=1" "ROCSHMEM_TEST_UUID=1" "../../../../share/rocshmem/test_wrapper.sh" "host_wait_until_some_vector_n2_w1_z1_uuid" "../../../rocshmem_functional_tests" "-a" "145" "-w" "1" "-z" "1" "-localbuftype" "heap")
set_tests_properties(host_wait_until_some_vector_n2_w1_z1_uuid PROPERTIES
    TIMEOUT 300
    SKIP_RETURN_CODE 125
    LABELS "functional;backend_ipc;gpu_all;variant_uuid;launcher_slr;HOST"
    PROCESSORS 2
    WILL_FAIL FALSE
    FAIL_REGULAR_EXPRESSION "FAILED")

add_test(host_wait_until_all_status_n2_w1_z1_uuid "env" "ROCSHMEM_SLR_NP=2" "ROCSHMEM_MAX_NUM_CONTEXTS=1" "ROCSHMEM_HEAP_SIZE=6442450944" "ROCSHMEM_TEST_UUID=1" "ROCSHMEM_TEST_UUID=1" "../../../../share/rocshmem/test_wrapper.sh" "host_wait_until_all_status_n2_w1_z1_uuid" "../../../rocshmem_functional_tests" "-a" "146" "-w" "1" "-z" "1" "-localbuftype" "heap")
set_tests_properties(host_wait_until_all_status_n2_w1_z1_uuid PROPERTIES
    TIMEOUT 300
    SKIP_RETURN_CODE 125
    LABELS "functional;backend_ipc;gpu_all;variant_uuid;launcher_slr;HOST"
    PROCESSORS 2
    WILL_FAIL FALSE
    FAIL_REGULAR_EXPRESSION "FAILED")

add_test(host_wait_until_any_status_n2_w1_z1_uuid "env" "ROCSHMEM_SLR_NP=2" "ROCSHMEM_MAX_NUM_CONTEXTS=1" "ROCSHMEM_HEAP_SIZE=6442450944" "ROCSHMEM_TEST_UUID=1" "ROCSHMEM_TEST_UUID=1" "../../../../share/rocshmem/test_wrapper.sh" "host_wait_until_any_status_n2_w1_z1_uuid" "../../../rocshmem_functional_tests" "-a" "147" "-w" "1" "-z" "1" "-localbuftype" "heap")
set_tests_properties(host_wait_until_any_status_n2_w1_z1_uuid PROPERTIES
    TIMEOUT 300
    SKIP_RETURN_CODE 125
    LABELS "functional;backend_ipc;gpu_all;variant_uuid;launcher_slr;HOST"
    PROCESSORS 2
    WILL_FAIL FALSE
    FAIL_REGULAR_EXPRESSION "FAILED")

add_test(host_wait_until_some_status_n2_w1_z1_uuid "env" "ROCSHMEM_SLR_NP=2" "ROCSHMEM_MAX_NUM_CONTEXTS=1" "ROCSHMEM_HEAP_SIZE=6442450944" "ROCSHMEM_TEST_UUID=1" "ROCSHMEM_TEST_UUID=1" "../../../../share/rocshmem/test_wrapper.sh" "host_wait_until_some_status_n2_w1_z1_uuid" "../../../rocshmem_functional_tests" "-a" "148" "-w" "1" "-z" "1" "-localbuftype" "heap")
set_tests_properties(host_wait_until_some_status_n2_w1_z1_uuid PROPERTIES
    TIMEOUT 300
    SKIP_RETURN_CODE 125
    LABELS "functional;backend_ipc;gpu_all;variant_uuid;launcher_slr;HOST"
    PROCESSORS 2
    WILL_FAIL FALSE
    FAIL_REGULAR_EXPRESSION "FAILED")

# Tier labels for monorepo CI (generated by ApplyInstallTierLabels.cmake)
# Categories: quick, standard, comprehensive, full

set_tests_properties(put_n2_w1_z1_1048576B_uuid PROPERTIES LABELS "standard;comprehensive;full;pr")
set_tests_properties(put_n2_w8_z1_1048576B_uuid PROPERTIES LABELS "quick;standard;comprehensive;full;smoke;pre-commit;pr")
set_tests_properties(defaultctxput_n2_w4_z128_1024B_uuid PROPERTIES LABELS "quick;standard;comprehensive;full;smoke;pre-commit;nightly")
set_tests_properties(teamctxput_n2_w4_z128_1024B_uuid PROPERTIES LABELS "comprehensive;full;nightly")
set_tests_properties(teamctxput_n2_w16_z256_1024B_uuid PROPERTIES LABELS "comprehensive;full;nightly")
set_tests_properties(wgput_n2_w1_z64_1048576B_uuid PROPERTIES LABELS "quick;standard;comprehensive;full;smoke;pre-commit;nightly")
set_tests_properties(wgput_n2_w2_z64_1048576B_uuid PROPERTIES LABELS "comprehensive;full;nightly")
set_tests_properties(wgput_n2_w16_z64_8B_uuid PROPERTIES LABELS "comprehensive;full;nightly")
set_tests_properties(waveput_n2_w1_z64_1048576B_uuid PROPERTIES LABELS "comprehensive;full;nightly")
set_tests_properties(waveput_n2_w2_z64_1048576B_uuid PROPERTIES LABELS "comprehensive;full;nightly")
set_tests_properties(waveput_n2_w2_z128_1048576B_uuid PROPERTIES LABELS "comprehensive;full;nightly")
set_tests_properties(waveput_n2_w16_z128_8B_uuid PROPERTIES LABELS "comprehensive;full;nightly")
set_tests_properties(p_n2_w1_z1_128B_uuid PROPERTIES LABELS "quick;standard;comprehensive;full;smoke;pre-commit;nightly")
set_tests_properties(p_n2_w1_z1024_2B_uuid PROPERTIES LABELS "quick;standard;comprehensive;full;smoke;pre-commit;nightly")
set_tests_properties(p_n2_w8_z1_32B_uuid PROPERTIES LABELS "quick;standard;comprehensive;full;smoke;pre-commit;nightly")
set_tests_properties(p_n2_w16_z128_4B_uuid PROPERTIES LABELS "quick;standard;comprehensive;full;smoke;pre-commit;nightly")
set_tests_properties(putnbi_n2_w1_z1_1048576B_uuid PROPERTIES LABELS "standard;comprehensive;full;pr")
set_tests_properties(putnbi_n2_w8_z1_1048576B_uuid PROPERTIES LABELS "standard;comprehensive;full;pr")
set_tests_properties(defaultctxputnbi_n2_w4_z128_1024B_uuid PROPERTIES LABELS "comprehensive;full;nightly")
set_tests_properties(teamctxputnbi_n2_w4_z128_1024B_uuid PROPERTIES LABELS "comprehensive;full;nightly")
set_tests_properties(teamctxputnbi_n2_w16_z256_1024B_uuid PROPERTIES LABELS "comprehensive;full;nightly")
set_tests_properties(wgputnbi_n2_w1_z64_1048576B_uuid PROPERTIES LABELS "comprehensive;full;nightly")
set_tests_properties(wgputnbi_n2_w2_z64_1048576B_uuid PROPERTIES LABELS "comprehensive;full;nightly")
set_tests_properties(wgputnbi_n2_w16_z64_8B_uuid PROPERTIES LABELS "comprehensive;full;nightly")
set_tests_properties(waveputnbi_n2_w1_z64_1048576B_uuid PROPERTIES LABELS "comprehensive;full;nightly")
set_tests_properties(waveputnbi_n2_w2_z64_1048576B_uuid PROPERTIES LABELS "comprehensive;full;nightly")
set_tests_properties(waveputnbi_n2_w2_z128_1048576B_uuid PROPERTIES LABELS "comprehensive;full;nightly")
set_tests_properties(waveputnbi_n2_w16_z128_8B_uuid PROPERTIES LABELS "comprehensive;full;nightly")
set_tests_properties(get_n2_w1_z1_1048576B_uuid PROPERTIES LABELS "standard;comprehensive;full;pr")
set_tests_properties(get_n2_w8_z1_1048576B_uuid PROPERTIES LABELS "quick;standard;comprehensive;full;smoke;pre-commit;pr")
set_tests_properties(defaultctxget_n2_w4_z128_1024B_uuid PROPERTIES LABELS "quick;standard;comprehensive;full;smoke;pre-commit;nightly")
set_tests_properties(teamctxget_n2_w4_z128_1024B_uuid PROPERTIES LABELS "comprehensive;full;nightly")
set_tests_properties(teamctxget_n2_w16_z256_1024B_uuid PROPERTIES LABELS "comprehensive;full;nightly")
set_tests_properties(wgget_n2_w1_z64_1048576B_uuid PROPERTIES LABELS "quick;standard;comprehensive;full;smoke;pre-commit;nightly")
set_tests_properties(wgget_n2_w2_z64_1048576B_uuid PROPERTIES LABELS "comprehensive;full;nightly")
set_tests_properties(wgget_n2_w16_z64_8B_uuid PROPERTIES LABELS "comprehensive;full;nightly")
set_tests_properties(waveget_n2_w1_z64_1048576B_uuid PROPERTIES LABELS "comprehensive;full;nightly")
set_tests_properties(waveget_n2_w2_z64_1048576B_uuid PROPERTIES LABELS "comprehensive;full;nightly")
set_tests_properties(waveget_n2_w2_z128_1048576B_uuid PROPERTIES LABELS "comprehensive;full;nightly")
set_tests_properties(waveget_n2_w16_z128_8B_uuid PROPERTIES LABELS "comprehensive;full;nightly")
set_tests_properties(g_n2_w1_z1_128B_uuid PROPERTIES LABELS "quick;standard;comprehensive;full;smoke;pre-commit;nightly")
set_tests_properties(g_n2_w1_z1024_1B_uuid PROPERTIES LABELS "quick;standard;comprehensive;full;smoke;pre-commit;nightly")
set_tests_properties(g_n2_w8_z1_32B_uuid PROPERTIES LABELS "quick;standard;comprehensive;full;smoke;pre-commit;nightly")
set_tests_properties(g_n2_w16_z128_4B_uuid PROPERTIES LABELS "quick;standard;comprehensive;full;smoke;pre-commit;nightly")
set_tests_properties(getnbi_n2_w1_z1_1048576B_uuid PROPERTIES LABELS "standard;comprehensive;full;pr")
set_tests_properties(getnbi_n2_w8_z1_1048576B_uuid PROPERTIES LABELS "standard;comprehensive;full;pr")
set_tests_properties(defaultctxgetnbi_n2_w4_z128_1024B_uuid PROPERTIES LABELS "comprehensive;full;nightly")
set_tests_properties(teamctxgetnbi_n2_w4_z128_1024B_uuid PROPERTIES LABELS "comprehensive;full;nightly")
set_tests_properties(teamctxgetnbi_n2_w16_z256_1024B_uuid PROPERTIES LABELS "comprehensive;full;nightly")
set_tests_properties(wggetnbi_n2_w1_z64_1048576B_uuid PROPERTIES LABELS "comprehensive;full;nightly")
set_tests_properties(wggetnbi_n2_w2_z64_1048576B_uuid PROPERTIES LABELS "comprehensive;full;nightly")
set_tests_properties(wggetnbi_n2_w16_z64_8B_uuid PROPERTIES LABELS "comprehensive;full;nightly")
set_tests_properties(wavegetnbi_n2_w1_z64_1048576B_uuid PROPERTIES LABELS "comprehensive;full;nightly")
set_tests_properties(wavegetnbi_n2_w2_z64_1048576B_uuid PROPERTIES LABELS "comprehensive;full;nightly")
set_tests_properties(wavegetnbi_n2_w2_z128_1048576B_uuid PROPERTIES LABELS "comprehensive;full;nightly")
set_tests_properties(wavegetnbi_n2_w16_z128_8B_uuid PROPERTIES LABELS "comprehensive;full;nightly")
set_tests_properties(amo_add_n2_w1_z1_uuid PROPERTIES LABELS "standard;comprehensive;full;pr")
set_tests_properties(amo_add_n2_w1_z1024_uuid PROPERTIES LABELS "standard;comprehensive;full;pr")
set_tests_properties(amo_add_n2_w8_z1_uuid PROPERTIES LABELS "standard;comprehensive;full;pr")
set_tests_properties(amo_add_n2_w32_z128_uuid PROPERTIES LABELS "standard;comprehensive;full;pr")
set_tests_properties(amo_fadd_n2_w1_z1_uuid PROPERTIES LABELS "quick;standard;comprehensive;full;smoke;pre-commit")
set_tests_properties(amo_fadd_n2_w1_z1024_uuid PROPERTIES LABELS "quick;standard;comprehensive;full;smoke;pre-commit")
set_tests_properties(amo_fadd_n2_w8_z1_uuid PROPERTIES LABELS "quick;standard;comprehensive;full;smoke;pre-commit")
set_tests_properties(amo_fadd_n2_w32_z128_uuid PROPERTIES LABELS "quick;standard;comprehensive;full;smoke;pre-commit")
set_tests_properties(amo_inc_n2_w1_z1_uuid PROPERTIES LABELS "standard;comprehensive;full;pr")
set_tests_properties(amo_inc_n2_w1_z1024_uuid PROPERTIES LABELS "standard;comprehensive;full;pr")
set_tests_properties(amo_inc_n2_w8_z1_uuid PROPERTIES LABELS "standard;comprehensive;full;pr")
set_tests_properties(amo_inc_n2_w32_z128_uuid PROPERTIES LABELS "standard;comprehensive;full;pr")
set_tests_properties(amo_finc_n2_w1_z1_uuid PROPERTIES LABELS "standard;comprehensive;full;pr")
set_tests_properties(amo_finc_n2_w1_z1024_uuid PROPERTIES LABELS "standard;comprehensive;full;pr")
set_tests_properties(amo_finc_n2_w8_z1_uuid PROPERTIES LABELS "standard;comprehensive;full;pr")
set_tests_properties(amo_finc_n2_w32_z128_uuid PROPERTIES LABELS "standard;comprehensive;full;pr")
set_tests_properties(amo_set_n2_w1_z1_uuid PROPERTIES LABELS "comprehensive;full;nightly")
set_tests_properties(amo_set_n2_w8_z1_uuid PROPERTIES LABELS "comprehensive;full;nightly")
set_tests_properties(amo_set_n2_w32_z1_uuid PROPERTIES LABELS "comprehensive;full;nightly")
set_tests_properties(amo_fetch_n2_w1_z1_uuid PROPERTIES LABELS "comprehensive;full;nightly")
set_tests_properties(amo_fetch_n2_w1_z1024_uuid PROPERTIES LABELS "comprehensive;full;nightly")
set_tests_properties(amo_fetch_n2_w8_z1_uuid PROPERTIES LABELS "comprehensive;full;nightly")
set_tests_properties(amo_fetch_n2_w32_z128_uuid PROPERTIES LABELS "comprehensive;full;nightly")
set_tests_properties(amo_fcswap_n2_w1_z1_uuid PROPERTIES LABELS "comprehensive;full;nightly")
set_tests_properties(amo_fcswap_n2_w32_z1_uuid PROPERTIES LABELS "comprehensive;full;nightly")
set_tests_properties(amo_fcswap_n2_w8_z1_uuid PROPERTIES LABELS "comprehensive;full;nightly")
set_tests_properties(amo_and_n2_w1_z1_uuid PROPERTIES LABELS "comprehensive;full;nightly")
set_tests_properties(amo_fetchand_n2_w1_z1_uuid PROPERTIES LABELS "comprehensive;full;nightly")
set_tests_properties(amo_xor_n2_w1_z1_uuid PROPERTIES LABELS "comprehensive;full;nightly")
set_tests_properties(putsignal_n2_w1_z1_1048576B_uuid PROPERTIES LABELS "standard;comprehensive;full;pr")
set_tests_properties(putsignal_n2_w2_z32_1048576B_uuid PROPERTIES LABELS "standard;comprehensive;full;pr")
set_tests_properties(wgputsignal_n2_w2_z32_1048576B_uuid PROPERTIES LABELS "comprehensive;full;nightly")
set_tests_properties(waveputsignal_n2_w1_z32_1048576B_uuid PROPERTIES LABELS "comprehensive;full;nightly")
set_tests_properties(waveputsignal_n2_w2_z64_1048576B_uuid PROPERTIES LABELS "comprehensive;full;nightly")
set_tests_properties(putsignalnbi_n2_w1_z1_1048576B_uuid PROPERTIES LABELS "comprehensive;full;nightly")
set_tests_properties(putsignalnbi_n2_w2_z32_1048576B_uuid PROPERTIES LABELS "comprehensive;full;nightly")
set_tests_properties(wgputsignalnbi_n2_w2_z32_1048576B_uuid PROPERTIES LABELS "comprehensive;full;nightly")
set_tests_properties(waveputsignalnbi_n2_w1_z32_1048576B_uuid PROPERTIES LABELS "comprehensive;full;nightly")
set_tests_properties(waveputsignalnbi_n2_w2_z64_1048576B_uuid PROPERTIES LABELS "comprehensive;full;nightly")
set_tests_properties(signalfetch_n2_w1_z1_uuid PROPERTIES LABELS "standard;comprehensive;full;pr")
set_tests_properties(wgsignalfetch_n2_w2_z32_uuid PROPERTIES LABELS "comprehensive;full;nightly")
set_tests_properties(wavesignalfetch_n2_w1_z32_uuid PROPERTIES LABELS "comprehensive;full;nightly")
set_tests_properties(wavesignalfetch_n2_w1_z64_uuid PROPERTIES LABELS "comprehensive;full;nightly")
set_tests_properties(signaladd_n2_w2_z32_uuid PROPERTIES LABELS "standard;comprehensive;full;pr")
set_tests_properties(signalset_n2_w2_z32_uuid PROPERTIES LABELS "standard;comprehensive;full;pr")
set_tests_properties(signalwaituntil_n2_w1_z1_uuid PROPERTIES LABELS "standard;comprehensive;full;pr")
set_tests_properties(init_n2_w1_z1_uuid PROPERTIES LABELS "comprehensive;full;nightly")
set_tests_properties(syncall_n2_w1_z1_uuid PROPERTIES LABELS "standard;comprehensive;full;pr")
set_tests_properties(syncall_n3_w1_z1_uuid PROPERTIES LABELS "standard;comprehensive;full;pr")
set_tests_properties(syncall_n5_w1_z1_uuid PROPERTIES LABELS "standard;comprehensive;full;pr")
set_tests_properties(wavesyncall_n2_w1_z1_uuid PROPERTIES LABELS "standard;comprehensive;full;pr")
set_tests_properties(wgsyncall_n2_w1_z1_uuid PROPERTIES LABELS "standard;comprehensive;full;pr")
set_tests_properties(teamsync_n2_w1_z1_uuid PROPERTIES LABELS "standard;comprehensive;full;pr")
set_tests_properties(teamsync_n2_w16_z64_uuid PROPERTIES LABELS "standard;comprehensive;full;pr")
set_tests_properties(teamsync_n2_w32_z256_uuid PROPERTIES LABELS "standard;comprehensive;full;pr")
set_tests_properties(teamsync_n2_w39_z1024_uuid PROPERTIES LABELS "standard;comprehensive;full;pr")
set_tests_properties(teamsync_n3_w16_z64_uuid PROPERTIES LABELS "standard;comprehensive;full;pr")
set_tests_properties(teamsync_n5_w16_z64_uuid PROPERTIES LABELS "standard;comprehensive;full;pr")
set_tests_properties(teamwavesync_n2_w1_z1_uuid PROPERTIES LABELS "comprehensive;full;nightly")
set_tests_properties(teamwavesync_n2_w16_z64_uuid PROPERTIES LABELS "comprehensive;full;nightly")
set_tests_properties(teamwavesync_n2_w32_z256_uuid PROPERTIES LABELS "comprehensive;full;nightly")
set_tests_properties(teamwavesync_n2_w39_z1024_uuid PROPERTIES LABELS "comprehensive;full;nightly")
set_tests_properties(teamwgsync_n2_w1_z1_uuid PROPERTIES LABELS "comprehensive;full;nightly")
set_tests_properties(teamwgsync_n2_w16_z64_uuid PROPERTIES LABELS "comprehensive;full;nightly")
set_tests_properties(teamwgsync_n2_w32_z256_uuid PROPERTIES LABELS "comprehensive;full;nightly")
set_tests_properties(teamwgsync_n2_w39_z1024_uuid PROPERTIES LABELS "comprehensive;full;nightly")
set_tests_properties(barrierall_n2_w1_z1_uuid PROPERTIES LABELS "standard;comprehensive;full;pr")
set_tests_properties(barrierall_n3_w1_z1_uuid PROPERTIES LABELS "standard;comprehensive;full;pr")
set_tests_properties(barrierall_n5_w1_z1_uuid PROPERTIES LABELS "standard;comprehensive;full;pr")
set_tests_properties(wavebarrierall_n2_w1_z1_uuid PROPERTIES LABELS "standard;comprehensive;full;pr")
set_tests_properties(wgbarrierall_n2_w1_z1_uuid PROPERTIES LABELS "standard;comprehensive;full;pr")
set_tests_properties(teambarrier_n2_w1_z1_uuid PROPERTIES LABELS "comprehensive;full;nightly")
set_tests_properties(teambarrier_n2_w16_z64_uuid PROPERTIES LABELS "comprehensive;full;nightly")
set_tests_properties(teambarrier_n2_w32_z256_uuid PROPERTIES LABELS "comprehensive;full;nightly")
set_tests_properties(teambarrier_n2_w39_z1024_uuid PROPERTIES LABELS "comprehensive;full;nightly")
set_tests_properties(teambarrier_n3_w16_z64_uuid PROPERTIES LABELS "comprehensive;full;nightly")
set_tests_properties(teambarrier_n5_w16_z64_uuid PROPERTIES LABELS "comprehensive;full;nightly")
set_tests_properties(teamwavebarrier_n2_w1_z1_uuid PROPERTIES LABELS "comprehensive;full;nightly")
set_tests_properties(teamwavebarrier_n2_w16_z64_uuid PROPERTIES LABELS "comprehensive;full;nightly")
set_tests_properties(teamwavebarrier_n2_w32_z256_uuid PROPERTIES LABELS "comprehensive;full;nightly")
set_tests_properties(teamwavebarrier_n2_w39_z1024_uuid PROPERTIES LABELS "comprehensive;full;nightly")
set_tests_properties(teamwgbarrier_n2_w1_z1_uuid PROPERTIES LABELS "comprehensive;full;nightly")
set_tests_properties(teamwgbarrier_n2_w16_z64_uuid PROPERTIES LABELS "comprehensive;full;nightly")
set_tests_properties(teamwgbarrier_n2_w32_z256_uuid PROPERTIES LABELS "comprehensive;full;nightly")
set_tests_properties(teamwgbarrier_n2_w39_z1024_uuid PROPERTIES LABELS "comprehensive;full;nightly")
set_tests_properties(alltoall_n2_w1_z64_512B_uuid PROPERTIES LABELS "standard;comprehensive;full;pr")
set_tests_properties(alltoall_n3_w1_z64_512B_uuid PROPERTIES LABELS "standard;comprehensive;full;pr")
set_tests_properties(alltoall_n5_w1_z64_512B_uuid PROPERTIES LABELS "standard;comprehensive;full;pr")
set_tests_properties(teambroadcast_n2_w1_z64_32768B_uuid PROPERTIES LABELS "comprehensive;full;nightly")
set_tests_properties(teambroadcast_n3_w1_z64_32768B_uuid PROPERTIES LABELS "comprehensive;full;nightly")
set_tests_properties(teambroadcast_n5_w1_z64_32768B_uuid PROPERTIES LABELS "comprehensive;full;nightly")
set_tests_properties(fcollect_n2_w1_z64_32768B_uuid PROPERTIES LABELS "comprehensive;full;nightly")
set_tests_properties(fcollect_n3_w1_z64_32768B_uuid PROPERTIES LABELS "comprehensive;full;nightly")
set_tests_properties(fcollect_n5_w1_z64_32768B_uuid PROPERTIES LABELS "comprehensive;full;nightly")
set_tests_properties(teamreduction_n2_w1_z64_32768B_uuid PROPERTIES LABELS "comprehensive;full;nightly")
set_tests_properties(teamreducescatter_n2_w1_z64_32768B_uuid PROPERTIES LABELS "comprehensive;full;nightly")
set_tests_properties(teamsplit2d_n4_w1_z1_uuid PROPERTIES LABELS "comprehensive;full;nightly")
set_tests_properties(broadcast_wave_n2_w1_wf1_32768B_uuid PROPERTIES LABELS "comprehensive;full;nightly")
set_tests_properties(alltoall_wave_n2_w1_wf1_512B_uuid PROPERTIES LABELS "comprehensive;full;nightly")
set_tests_properties(fcollect_wave_n2_w1_wf1_32768B_uuid PROPERTIES LABELS "comprehensive;full;nightly")
set_tests_properties(reduce_wave_n2_w1_wf1_32768B_uuid PROPERTIES LABELS "comprehensive;full;nightly")
set_tests_properties(putmem_on_stream_n2_w1_z1_1048576B_uuid PROPERTIES LABELS "quick;standard;comprehensive;full;smoke;pre-commit")
set_tests_properties(putmem_on_stream_n2_w1_z1_1048576B_default_stream PROPERTIES LABELS "quick;standard;comprehensive;full;smoke;pre-commit")
set_tests_properties(putmem_on_stream_n2_w1_z1_1048576B_uuid_default_stream PROPERTIES LABELS "quick;standard;comprehensive;full;smoke;pre-commit")
set_tests_properties(getmem_on_stream_n2_w1_z1_1048576B_uuid PROPERTIES LABELS "comprehensive;full;nightly")
set_tests_properties(signal_wait_until_on_stream_n2_w1_z1_uuid PROPERTIES LABELS "standard;comprehensive;full;pr")
set_tests_properties(putmem_signal_on_stream_n2_w1_z1_1048576B_uuid PROPERTIES LABELS "standard;comprehensive;full;pr")
set_tests_properties(barrier_all_on_stream_n2_w1_z1_uuid PROPERTIES LABELS "standard;comprehensive;full;pr")
set_tests_properties(quiet_on_stream_n2_w1_z1_uuid PROPERTIES LABELS "comprehensive;full;nightly")
set_tests_properties(sync_all_on_stream_n2_w1_z1_uuid PROPERTIES LABELS "comprehensive;full;nightly")
set_tests_properties(alltoallmem_on_stream_n2_w1_z64_1048576B_uuid PROPERTIES LABELS "comprehensive;full;nightly")
set_tests_properties(broadcastmem_on_stream_n2_w1_z64_1048576B_uuid PROPERTIES LABELS "comprehensive;full;nightly")
set_tests_properties(device_bitcode_n2_w1_z1_uuid PROPERTIES LABELS "quick;standard;comprehensive;full;smoke;pre-commit")
set_tests_properties(device_bitcode_n2_w32_z1024_uuid PROPERTIES LABELS "quick;standard;comprehensive;full;smoke;pre-commit")
set_tests_properties(device_bitcode_n4_w16_z256_uuid PROPERTIES LABELS "quick;standard;comprehensive;full;smoke;pre-commit")
set_tests_properties(device_bitcode_n8_w16_z128_uuid PROPERTIES LABELS "quick;standard;comprehensive;full;smoke;pre-commit")
set_tests_properties(library_info_n2_w1_z1_uuid PROPERTIES LABELS "standard;comprehensive;full;pr")
set_tests_properties(hipmodule_init_n2_w1_z1_uuid PROPERTIES LABELS "standard;comprehensive;full;pr")
set_tests_properties(pingpong_n2_w1_z1_uuid PROPERTIES LABELS "quick;standard;comprehensive;full;smoke;pre-commit")
set_tests_properties(pingpong_n2_w8_z1_uuid PROPERTIES LABELS "quick;standard;comprehensive;full;smoke;pre-commit")
set_tests_properties(pingpong_n2_w32_z1_uuid PROPERTIES LABELS "quick;standard;comprehensive;full;smoke;pre-commit")
set_tests_properties(pingall_n2_w1_z1_uuid PROPERTIES LABELS "comprehensive;full;nightly")
set_tests_properties(pingall_n2_w8_z1_uuid PROPERTIES LABELS "comprehensive;full;nightly")
set_tests_properties(pingall_n2_w32_z1_uuid PROPERTIES LABELS "comprehensive;full;nightly")
set_tests_properties(shmemptr_n2_w1_z1_8B_uuid PROPERTIES LABELS "standard;comprehensive;full;pr")
set_tests_properties(shmemptr_n2_w1_z1024_8B_uuid PROPERTIES LABELS "standard;comprehensive;full;pr")
set_tests_properties(shmemptr_n2_w8_z1_8B_uuid PROPERTIES LABELS "standard;comprehensive;full;pr")
set_tests_properties(shmemptr_n2_w16_z128_8B_uuid PROPERTIES LABELS "standard;comprehensive;full;pr")
set_tests_properties(flood_put_n2_w64_z1024_uuid PROPERTIES LABELS "full;weekly")
set_tests_properties(flood_put_n8_w64_z1024_uuid PROPERTIES LABELS "full;weekly")
set_tests_properties(flood_putnbi_n8_w64_z1024_uuid PROPERTIES LABELS "full;weekly")
set_tests_properties(flood_p_n8_w64_z1024_uuid PROPERTIES LABELS "full;weekly")
set_tests_properties(flood_get_n2_w64_z1024_uuid PROPERTIES LABELS "full;weekly")
set_tests_properties(flood_get_n8_w64_z1024_uuid PROPERTIES LABELS "full;weekly")
set_tests_properties(flood_getnbi_n8_w64_z1024_uuid PROPERTIES LABELS "full;weekly")
set_tests_properties(flood_g_n8_w64_z1024_uuid PROPERTIES LABELS "full;weekly")
set_tests_properties(flood_add_n2_w64_z1024_uuid PROPERTIES LABELS "full;weekly")
set_tests_properties(flood_add_n8_w64_z1024_uuid PROPERTIES LABELS "full;weekly")
set_tests_properties(flood_fadd_n8_w64_z1024_uuid PROPERTIES LABELS "full;weekly")
set_tests_properties(flood_waitadd_n8_w64_z1024_uuid PROPERTIES LABELS "full;weekly")
set_tests_properties(teamctxinfra_n2_w1_z1_uuid PROPERTIES LABELS "comprehensive;full;nightly")
set_tests_properties(teamctxsingleinfra_n2_w1_z1_uuid PROPERTIES LABELS "comprehensive;full;nightly")
set_tests_properties(teamctxblockinfra_n4_w1_z1_uuid PROPERTIES LABELS "comprehensive;full;nightly")
set_tests_properties(teamctxblockinfra_n5_w1_z1_uuid PROPERTIES LABELS "comprehensive;full;nightly")
set_tests_properties(teamctxoddeveninfra_n4_w1_z1_uuid PROPERTIES LABELS "comprehensive;full;nightly")
set_tests_properties(teamctxoddeveninfra_n5_w1_z1_uuid PROPERTIES LABELS "comprehensive;full;nightly")
set_tests_properties(teamctxsharedinfra_n2_w1_z1_uuid PROPERTIES LABELS "comprehensive;full;nightly")
set_tests_properties(teamctxsharedinfra_n5_w1_z1_uuid PROPERTIES LABELS "comprehensive;full;nightly")
set_tests_properties(teamctxsubsetparentinfra_n4_w1_z1_uuid PROPERTIES LABELS "comprehensive;full;nightly")
set_tests_properties(teamctxsubsetparentinfra_n5_w1_z1_uuid PROPERTIES LABELS "comprehensive;full;nightly")
set_tests_properties(fence_putwavesignal_n2_w1_z64_1048576B_uuid PROPERTIES LABELS "full;weekly")
set_tests_properties(fence_putwavesignal_n2_w8_z256_1048576B_uuid PROPERTIES LABELS "full;weekly")
set_tests_properties(fence_putwavesignal_n2_w32_z1024_65536B_uuid PROPERTIES LABELS "full;weekly")
set_tests_properties(fence_putlargesmall_n2_w1_z64_4096B_uuid PROPERTIES LABELS "full;weekly")
set_tests_properties(fence_putlargesmall_n2_w8_z256_65536B_uuid PROPERTIES LABELS "full;weekly")
set_tests_properties(fence_fanout_n2_w1_z64_1048576B_uuid PROPERTIES LABELS "full;weekly")
set_tests_properties(fence_fanout_n4_w4_z256_65536B_uuid PROPERTIES LABELS "full;weekly")
set_tests_properties(fence_fanout_n8_w8_z256_65536B_uuid PROPERTIES LABELS "full;weekly")
set_tests_properties(fence_putwavenbichunks_n2_w1_z64_1048576B_uuid PROPERTIES LABELS "full;weekly")
set_tests_properties(fence_putwavenbichunks_n2_w8_z256_65536B_uuid PROPERTIES LABELS "full;weekly")
set_tests_properties(buffer_register_symmetric_n2_w1_z1_64B_uuid PROPERTIES LABELS "standard;comprehensive;full;pr")
set_tests_properties(buffer_register_symmetric_n2_w2_z64_64B_uuid PROPERTIES LABELS "standard;comprehensive;full;pr")
set_tests_properties(buffer_register_symmetric_n4_w1_z1_64B_uuid PROPERTIES LABELS "standard;comprehensive;full;pr")
set_tests_properties(buffer_register_symmetric_n4_w2_z64_64B_uuid PROPERTIES LABELS "standard;comprehensive;full;pr")
set_tests_properties(get_n2_w1_z1_v1048576B_uuid PROPERTIES LABELS "standard;comprehensive;full;pr")
set_tests_properties(waveget_n2_w1_z64_v1073741824B_uuid PROPERTIES LABELS "comprehensive;full;nightly")
set_tests_properties(waveget_n2_w2_z64_v1073741824B_uuid PROPERTIES LABELS "comprehensive;full;nightly")
set_tests_properties(waveget_n2_w16_z1024_v1073741824B_uuid PROPERTIES LABELS "comprehensive;full;nightly")
set_tests_properties(wgget_n2_w1_z1024_v1073741824B_uuid PROPERTIES LABELS "comprehensive;full;nightly")
set_tests_properties(wgget_n2_w16_z1024_v1073741824B_uuid PROPERTIES LABELS "comprehensive;full;nightly")
set_tests_properties(put_n2_w1_z1_v1048576B_uuid PROPERTIES LABELS "standard;comprehensive;full;pr")
set_tests_properties(waveput_n2_w1_z64_v1073741824B_uuid PROPERTIES LABELS "comprehensive;full;nightly")
set_tests_properties(waveput_n2_w2_z64_v1073741824B_uuid PROPERTIES LABELS "comprehensive;full;nightly")
set_tests_properties(waveput_n2_w16_z1024_v1073741824B_uuid PROPERTIES LABELS "comprehensive;full;nightly")
set_tests_properties(wgput_n2_w1_z1024_v1073741824B_uuid PROPERTIES LABELS "comprehensive;full;nightly")
set_tests_properties(wgput_n2_w16_z1024_v1073741824B_uuid PROPERTIES LABELS "comprehensive;full;nightly")
set_tests_properties(alltoall_n2_w1_z256_v1073741824B_uuid PROPERTIES LABELS "standard;comprehensive;full;pr")
set_tests_properties(alltoall_n4_w1_z256_v1073741824B_uuid PROPERTIES LABELS "standard;comprehensive;full;pr")
set_tests_properties(alltoall_n8_w1_z256_v1073741824B_uuid PROPERTIES LABELS "standard;comprehensive;full;pr")
set_tests_properties(alltoall_n16_w1_z256_v1073741824B_uuid PROPERTIES LABELS "standard;comprehensive;full;pr")
set_tests_properties(alltoall_n32_w1_z256_v1073741824B_uuid PROPERTIES LABELS "standard;comprehensive;full;pr")
set_tests_properties(alltoall_n64_w1_z256_v1073741824B_uuid PROPERTIES LABELS "standard;comprehensive;full;pr")
set_tests_properties(tile_put_contiguous_n2_w1_z1_1048576B_uuid PROPERTIES LABELS "comprehensive;full;nightly")
set_tests_properties(tile_put_rowmajor_n2_w1_z1_1048576B_uuid PROPERTIES LABELS "comprehensive;full;nightly")
set_tests_properties(tile_put_colmajor_n2_w1_z1_1048576B_uuid PROPERTIES LABELS "comprehensive;full;nightly")
set_tests_properties(tile_put_arbitrary_n2_w1_z1_16384B_uuid PROPERTIES LABELS "comprehensive;full;nightly")
set_tests_properties(tile_put_1d_n2_w1_z1_1048576B_uuid PROPERTIES LABELS "comprehensive;full;nightly")
set_tests_properties(tile_put_wave_contiguous_n2_w1_wf1_1048576B_uuid PROPERTIES LABELS "comprehensive;full;nightly")
set_tests_properties(tile_put_wave_rowmajor_n2_w1_wf1_1048576B_uuid PROPERTIES LABELS "comprehensive;full;nightly")
set_tests_properties(tile_put_wave_colmajor_n2_w1_wf1_1048576B_uuid PROPERTIES LABELS "comprehensive;full;nightly")
set_tests_properties(tile_put_wg_contiguous_n2_w1_wf16_1048576B_uuid PROPERTIES LABELS "comprehensive;full;nightly")
set_tests_properties(tile_put_wg_rowmajor_n2_w1_wf16_1048576B_uuid PROPERTIES LABELS "comprehensive;full;nightly")
set_tests_properties(tile_put_wg_colmajor_n2_w1_wf16_1048576B_uuid PROPERTIES LABELS "comprehensive;full;nightly")
set_tests_properties(tile_get_contiguous_n2_w1_z1_1048576B_uuid PROPERTIES LABELS "comprehensive;full;nightly")
set_tests_properties(tile_get_rowmajor_n2_w1_z1_1048576B_uuid PROPERTIES LABELS "comprehensive;full;nightly")
set_tests_properties(tile_get_colmajor_n2_w1_z1_1048576B_uuid PROPERTIES LABELS "comprehensive;full;nightly")
set_tests_properties(tile_get_arbitrary_n2_w1_z1_16384B_uuid PROPERTIES LABELS "comprehensive;full;nightly")
set_tests_properties(tile_get_1d_n2_w1_z1_1048576B_uuid PROPERTIES LABELS "comprehensive;full;nightly")
set_tests_properties(tile_get_wave_contiguous_n2_w1_wf1_1048576B_uuid PROPERTIES LABELS "comprehensive;full;nightly")
set_tests_properties(tile_get_wave_rowmajor_n2_w1_wf1_1048576B_uuid PROPERTIES LABELS "comprehensive;full;nightly")
set_tests_properties(tile_get_wave_colmajor_n2_w1_wf1_1048576B_uuid PROPERTIES LABELS "comprehensive;full;nightly")
set_tests_properties(tile_get_wg_contiguous_n2_w1_wf16_1048576B_uuid PROPERTIES LABELS "comprehensive;full;nightly")
set_tests_properties(tile_get_wg_rowmajor_n2_w1_wf16_1048576B_uuid PROPERTIES LABELS "comprehensive;full;nightly")
set_tests_properties(tile_get_wg_colmajor_n2_w1_wf16_1048576B_uuid PROPERTIES LABELS "comprehensive;full;nightly")
set_tests_properties(tile_get_wg_contiguous_n2_w4_wf16_1048576B_uuid PROPERTIES LABELS "comprehensive;full;nightly")
set_tests_properties(tile_put_rowmajor_start_coord_n2_w1_z1_uuid PROPERTIES LABELS "comprehensive;full;nightly")
set_tests_properties(tile_put_wave_rowmajor_start_coord_n2_w1_wf1_uuid PROPERTIES LABELS "comprehensive;full;nightly")
set_tests_properties(tile_put_wg_rowmajor_start_coord_n2_w1_wf16_uuid PROPERTIES LABELS "comprehensive;full;nightly")
set_tests_properties(tile_get_rowmajor_start_coord_n2_w1_z1_uuid PROPERTIES LABELS "comprehensive;full;nightly")
set_tests_properties(tile_get_wave_rowmajor_start_coord_n2_w1_wf1_uuid PROPERTIES LABELS "comprehensive;full;nightly")
set_tests_properties(tile_get_wg_rowmajor_start_coord_n2_w1_wf16_uuid PROPERTIES LABELS "comprehensive;full;nightly")
set_tests_properties(tile_put_3d_n2_w1_z1_uuid PROPERTIES LABELS "comprehensive;full;nightly")
set_tests_properties(tile_get_3d_n2_w1_z1_uuid PROPERTIES LABELS "comprehensive;full;nightly")
set_tests_properties(tile_put_wg_3d_n2_w4_wf16_uuid PROPERTIES LABELS "comprehensive;full;nightly")
set_tests_properties(tile_get_wg_3d_n2_w4_wf16_uuid PROPERTIES LABELS "comprehensive;full;nightly")
set_tests_properties(tile_put_4d_n2_w1_z1_uuid PROPERTIES LABELS "comprehensive;full;nightly")
set_tests_properties(tile_get_4d_n2_w1_z1_uuid PROPERTIES LABELS "comprehensive;full;nightly")
set_tests_properties(tile_put_wg_4d_n2_w4_wf16_uuid PROPERTIES LABELS "comprehensive;full;nightly")
set_tests_properties(tile_get_wg_4d_n2_w4_wf16_uuid PROPERTIES LABELS "comprehensive;full;nightly")
set_tests_properties(tile_put_wave_3d_n2_w1_wf4_uuid PROPERTIES LABELS "comprehensive;full;nightly")
set_tests_properties(tile_get_wave_3d_n2_w1_wf4_uuid PROPERTIES LABELS "comprehensive;full;nightly")
set_tests_properties(tile_put_wave_4d_n2_w1_wf4_uuid PROPERTIES LABELS "comprehensive;full;nightly")
set_tests_properties(tile_get_wave_4d_n2_w1_wf4_uuid PROPERTIES LABELS "comprehensive;full;nightly")
set_tests_properties(tile_broadcast_n2_w1_z1_1048576B_uuid PROPERTIES LABELS "comprehensive;full;nightly")
set_tests_properties(tile_broadcast_n4_w1_z1_1048576B_uuid PROPERTIES LABELS "comprehensive;full;nightly")
set_tests_properties(tile_broadcast_wave_n2_w1_wf4_1048576B_uuid PROPERTIES LABELS "comprehensive;full;nightly")
set_tests_properties(tile_broadcast_wave_n4_w1_wf4_1048576B_uuid PROPERTIES LABELS "comprehensive;full;nightly")
set_tests_properties(tile_broadcast_wg_n2_w4_wf1_1048576B_uuid PROPERTIES LABELS "comprehensive;full;nightly")
set_tests_properties(tile_broadcast_wg_n4_w4_wf1_1048576B_uuid PROPERTIES LABELS "comprehensive;full;nightly")
set_tests_properties(tile_allgather_n2_w1_z1_1048576B_uuid PROPERTIES LABELS "comprehensive;full;nightly")
set_tests_properties(tile_allgather_n4_w1_z1_1048576B_uuid PROPERTIES LABELS "comprehensive;full;nightly")
set_tests_properties(tile_allgather_wave_n2_w1_wf4_1048576B_uuid PROPERTIES LABELS "comprehensive;full;nightly")
set_tests_properties(tile_allgather_wave_n4_w1_wf4_1048576B_uuid PROPERTIES LABELS "comprehensive;full;nightly")
set_tests_properties(tile_allgather_wg_n2_w4_wf1_1048576B_uuid PROPERTIES LABELS "comprehensive;full;nightly")
set_tests_properties(tile_allgather_wg_n4_w4_wf1_1048576B_uuid PROPERTIES LABELS "comprehensive;full;nightly")
set_tests_properties(tile_reduce_n2_w1_z1_1048576B_uuid PROPERTIES LABELS "comprehensive;full;nightly")
set_tests_properties(tile_reduce_n4_w1_z1_65536B_uuid PROPERTIES LABELS "comprehensive;full;nightly")
set_tests_properties(tile_reduce_wave_n2_w1_wf4_1048576B_uuid PROPERTIES LABELS "comprehensive;full;nightly")
set_tests_properties(tile_reduce_wave_n4_w1_wf4_65536B_uuid PROPERTIES LABELS "comprehensive;full;nightly")
set_tests_properties(tile_reduce_wg_n2_w4_wf1_1048576B_uuid PROPERTIES LABELS "comprehensive;full;nightly")
set_tests_properties(tile_reduce_wg_n4_w4_wf1_65536B_uuid PROPERTIES LABELS "comprehensive;full;nightly")
set_tests_properties(host_putmem_n2_w1_z1_65536B_uuid PROPERTIES LABELS "comprehensive;full;nightly")
set_tests_properties(host_getmem_n2_w1_z1_65536B_uuid PROPERTIES LABELS "comprehensive;full;nightly")
set_tests_properties(host_amo_fadd_n2_w1_z1_uuid PROPERTIES LABELS "comprehensive;full;nightly")
set_tests_properties(host_amo_fcswap_n2_w1_z1_uuid PROPERTIES LABELS "comprehensive;full;nightly")
set_tests_properties(host_int_amo_fadd_n2_w1_z1_uuid PROPERTIES LABELS "comprehensive;full;nightly")
set_tests_properties(host_int_amo_fcswap_n2_w1_z1_uuid PROPERTIES LABELS "comprehensive;full;nightly")
set_tests_properties(host_ctx_putmem_n2_w1_z1_65536B_uuid PROPERTIES LABELS "comprehensive;full;nightly")
set_tests_properties(host_ctx_getmem_n2_w1_z1_65536B_uuid PROPERTIES LABELS "comprehensive;full;nightly")
set_tests_properties(host_amo_all_pes_n4_w1_z1_uuid PROPERTIES LABELS "comprehensive;full;nightly")
set_tests_properties(host_amo_self_n4_w1_z1_uuid PROPERTIES LABELS "comprehensive;full;nightly")
set_tests_properties(host_wait_until_n2_w1_z1_uuid PROPERTIES LABELS "comprehensive;full;nightly")
set_tests_properties(host_test_n2_w1_z1_uuid PROPERTIES LABELS "comprehensive;full;nightly")
set_tests_properties(host_wait_until_all_n2_w1_z1_uuid PROPERTIES LABELS "comprehensive;full;nightly")
set_tests_properties(host_wait_until_any_n2_w1_z1_uuid PROPERTIES LABELS "comprehensive;full;nightly")
set_tests_properties(host_wait_until_some_n2_w1_z1_uuid PROPERTIES LABELS "comprehensive;full;nightly")
set_tests_properties(host_wait_until_all_vector_n2_w1_z1_uuid PROPERTIES LABELS "comprehensive;full;nightly")
set_tests_properties(host_wait_until_any_vector_n2_w1_z1_uuid PROPERTIES LABELS "comprehensive;full;nightly")
set_tests_properties(host_wait_until_some_vector_n2_w1_z1_uuid PROPERTIES LABELS "comprehensive;full;nightly")
set_tests_properties(host_wait_until_all_status_n2_w1_z1_uuid PROPERTIES LABELS "comprehensive;full;nightly")
set_tests_properties(host_wait_until_any_status_n2_w1_z1_uuid PROPERTIES LABELS "comprehensive;full;nightly")
set_tests_properties(host_wait_until_some_status_n2_w1_z1_uuid PROPERTIES LABELS "comprehensive;full;nightly")
