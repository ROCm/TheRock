import torch

def test_matrix_multiplication():
    matrix1 = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    matrix2 = torch.tensor([[7.0, 8.0, 9.0, 10.0], [11.0, 12.0, 13.0, 14.0], [15.0, 16.0, 17.0, 18.0]])
    expected = torch.tensor([[74.0, 80.0, 86.0, 92.0], [173.0, 188.0, 203.0, 218.0]])
    result = torch.mm(matrix1, matrix2)
    assert torch.allclose(result, expected)

def test_batch_matrix_multiplication():
    batch_matrix1 = torch.ones(10, 2, 3)
    batch_matrix2 = torch.ones(10, 3, 4)
    expected = torch.full((10, 2, 4), 3.0)
    result = torch.bmm(batch_matrix1, batch_matrix2)
    assert torch.allclose(result, expected)

def test_matrix_multiplication_at_operator():
    matrix1 = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    matrix2 = torch.tensor([[7.0, 8.0, 9.0, 10.0], [11.0, 12.0, 13.0, 14.0], [15.0, 16.0, 17.0, 18.0]])
    expected = torch.tensor([[74.0, 80.0, 86.0, 92.0], [173.0, 188.0, 203.0, 218.0]])
    result = matrix1 @ matrix2
    assert torch.allclose(result, expected)

def test_elementwise_multiplication():
    matrix1 = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    matrix2 = torch.tensor([[7.0, 8.0, 9.0], [10.0, 11.0, 12.0]])
    expected = torch.tensor([[7.0, 16.0, 27.0], [40.0, 55.0, 72.0]])
    result = matrix1 * matrix2
    assert torch.allclose(result, expected)

def test_transpose():
    matrix = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    expected = torch.tensor([[1.0, 4.0], [2.0, 5.0], [3.0, 6.0]])
    transposed = torch.t(matrix)
    assert torch.allclose(transposed, expected)

def test_dot_product():
    vector1 = torch.tensor([1.0, 2.0, 3.0])
    vector2 = torch.tensor([4.0, 5.0, 6.0])
    expected = torch.tensor(32.0)
    result = torch.dot(vector1, vector2)
    assert torch.allclose(result, expected)

def test_matrix_vector_multiplication():
    matrix = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    vector = torch.tensor([7.0, 8.0, 9.0])
    expected = torch.tensor([50.0, 122.0])
    result = torch.mv(matrix, vector)
    assert torch.allclose(result, expected)

def test_matrix_multiplication_matmul():
    matrix1 = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    matrix2 = torch.tensor([[7.0, 8.0, 9.0, 10.0], [11.0, 12.0, 13.0, 14.0], [15.0, 16.0, 17.0, 18.0]])
    expected = torch.tensor([[74.0, 80.0, 86.0, 92.0], [173.0, 188.0, 203.0, 218.0]])
    result = torch.matmul(matrix1, matrix2)
    assert torch.allclose(result, expected)
