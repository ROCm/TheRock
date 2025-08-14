import torch
import pytest


class TestROCmAvailability:
    def test_rocm_available(self):
        assert (
            torch.cuda.is_available()
        ), "ROCm is not available or not detected by PyTorch"


class TestMatrixOperations:
    def test_matrix_multiplication(self):
        matrix1 = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], device="cuda")
        matrix2 = torch.tensor(
            [[7.0, 8.0, 9.0, 10.0], [11.0, 12.0, 13.0, 14.0], [15.0, 16.0, 17.0, 18.0]],
            device="cuda",
        )
        expected = torch.tensor(
            [[74.0, 80.0, 86.0, 92.0], [173.0, 188.0, 203.0, 218.0]], device="cuda"
        )
        result = torch.mm(matrix1, matrix2)
        assert torch.allclose(result, expected)
        assert result.device.type == "cuda"

    def test_batch_matrix_multiplication(self):
        batch_matrix1 = torch.ones(10, 2, 3, device="cuda")
        batch_matrix2 = torch.ones(10, 3, 4, device="cuda")
        expected = torch.full((10, 2, 4), 3.0, device="cuda")
        result = torch.bmm(batch_matrix1, batch_matrix2)
        assert torch.allclose(result, expected)
        assert result.device.type == "cuda"

    def test_matrix_multiplication_at_operator(self):
        matrix1 = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], device="cuda")
        matrix2 = torch.tensor(
            [[7.0, 8.0, 9.0, 10.0], [11.0, 12.0, 13.0, 14.0], [15.0, 16.0, 17.0, 18.0]],
            device="cuda",
        )
        expected = torch.tensor(
            [[74.0, 80.0, 86.0, 92.0], [173.0, 188.0, 203.0, 218.0]], device="cuda"
        )
        result = matrix1 @ matrix2
        assert torch.allclose(result, expected)
        assert result.device.type == "cuda"

    def test_elementwise_multiplication(self):
        matrix1 = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], device="cuda")
        matrix2 = torch.tensor([[7.0, 8.0, 9.0], [10.0, 11.0, 12.0]], device="cuda")
        expected = torch.tensor([[7.0, 16.0, 27.0], [40.0, 55.0, 72.0]], device="cuda")
        result = matrix1 * matrix2
        assert torch.allclose(result, expected)
        assert result.device.type == "cuda"

    def test_transpose(self):
        matrix = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], device="cuda")
        expected = torch.tensor([[1.0, 4.0], [2.0, 5.0], [3.0, 6.0]], device="cuda")
        transposed = torch.t(matrix)
        assert torch.allclose(transposed, expected)
        assert transposed.device.type == "cuda"

    def test_dot_product(self):
        vector1 = torch.tensor([1.0, 2.0, 3.0], device="cuda")
        vector2 = torch.tensor([4.0, 5.0, 6.0], device="cuda")
        expected = torch.tensor(32.0, device="cuda")
        result = torch.dot(vector1, vector2)
        assert torch.allclose(result, expected)
        assert result.device.type == "cuda"

    def test_matrix_vector_multiplication(self):
        matrix = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], device="cuda")
        vector = torch.tensor([7.0, 8.0, 9.0], device="cuda")
        expected = torch.tensor([50.0, 122.0], device="cuda")
        result = torch.mv(matrix, vector)
        assert torch.allclose(result, expected)
        assert result.device.type == "cuda"

    def test_matrix_multiplication_matmul(self):
        matrix1 = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], device="cuda")
        matrix2 = torch.tensor(
            [[7.0, 8.0, 9.0, 10.0], [11.0, 12.0, 13.0, 14.0], [15.0, 16.0, 17.0, 18.0]],
            device="cuda",
        )
        expected = torch.tensor(
            [[74.0, 80.0, 86.0, 92.0], [173.0, 188.0, 203.0, 218.0]], device="cuda"
        )
        result = torch.matmul(matrix1, matrix2)
        assert torch.allclose(result, expected)
        assert result.device.type == "cuda"


class TestConvolutions:
    def teardown_method(self):
        # TODO(#999): fix tests stalling on exit without this
        torch.cuda.synchronize()

    def test_conv_transpose2d(self):
        inputs = torch.randn(1, 4, 5, 5, device="cuda")
        weights = torch.randn(4, 8, 3, 3, device="cuda")
        # Simply running any conv op exercises MIOpen and library loading.
        # On Windows, this may fail if `amd_comgr_3.dll` (from build output) is
        # used instead of `amd_comgr0605.dll` that is expected at runtime.
        result = torch.nn.functional.conv_transpose2d(inputs, weights, padding=1)

        # TODO: check conv output values (and don't use randn)
        assert result.device.type == "cuda"

    # Lifted from
    # https://github.com/pytorch/pytorch/blob/main/test/nn/test_convolution.py
    def test_conv_cudnn_nhwc_support(self):
        input = torch.randn(
            (1, 16, 1, 1), dtype=torch.float, device="cuda", requires_grad=True
        )
        weight = torch.randn(
            (8, 16, 3, 3), dtype=torch.float, device="cuda", requires_grad=True
        )
        weight = weight.to(memory_format=torch.channels_last)
        o = torch.conv2d(input, weight, None, (2, 1), (1, 1), (1, 1), 1)
        assert o.is_contiguous(memory_format=torch.channels_last)
