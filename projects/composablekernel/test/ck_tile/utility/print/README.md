# Print Function Tests

This directory contains unit tests for testing the print functionality of various data structures and coordinate transformations in the composable_kernel library.

## Tests Included

### test_print_sequence.cpp
Tests the print functionality for `sequence<...>` containers:
- Simple sequences with multiple elements
- Single element sequences
- Empty sequences
- Longer sequences

### test_print_array.cpp
Tests the print functionality for `array<T, N>` containers:
- Arrays with integer values
- Single element arrays
- Empty arrays (size 0)
- Arrays with floating point values

### test_print_tuple.cpp
Tests the print functionality for `tuple<...>` containers:
- Simple tuples with numbers
- Single element tuples
- Empty tuples
- Mixed type tuples

### test_print_coordinate_transform.cpp
Tests the print functionality for coordinate transformation structures:
- `pass_through` transform
- `embed` transform
- `merge` transform
- `unmerge` transform
- `freeze` transform

## Testing Approach

All tests use Google Test's `CaptureStdout()` functionality to capture the output from print functions and verify the formatting:

```cpp
testing::internal::CaptureStdout();
print(object);
std::string output = testing::internal::GetCapturedStdout();
EXPECT_EQ(output, "expected_format");
```

This approach enables testing of print function output without affecting the console during test execution.

## Building and Running

The tests are integrated into the CMake build system. To build and run the print tests:

```bash
# Build the specific test
make test_print_sequence

# Run the test
./test_print_sequence

# Or run all print tests using CTest
ctest -R "test_print"
```

## Adding New Tests

To add tests for new data structures:

1. Create a new test file: `test_print_<structure_name>.cpp`
2. Follow the existing pattern using `CaptureStdout()`
3. Add the test executable to `CMakeLists.txt`
