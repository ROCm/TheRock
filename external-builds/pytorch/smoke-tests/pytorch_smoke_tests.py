import torch
import pytest


@pytest.mark.skipif(not torch.cuda.is_available(), reason="ROCm is not available")
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
