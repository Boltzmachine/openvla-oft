import torch
import argparse

def matrix_multiply(a, b):
    return torch.matmul(a, b)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Matrix Multiplication Script")
    parser.add_argument("--size", type=int, default=3, help="Size of the square matrices")
    args = parser.parse_args()

    size = args.size
    
    while True:
        a = torch.randn(size, size, device='cuda')
        b = torch.randn(size, size, device='cuda')

        result = matrix_multiply(a, b)