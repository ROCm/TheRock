# Gtest

hipblaslt-test is the main regression gtest for hipBLASLt. All test items should pass.

```shell
# Go to hipBLASLt build directory
cd hipBLASLt; cd build/release

# Run full gtest tests
./clients/hipblaslt-test

# Run gtest tests with filter
./clients/hipblaslt-test --gtest_filter=<test pattern>

# Demo: gtest tests with filter
./clients/hipblaslt-test --gtest_filter=*quick*
```
