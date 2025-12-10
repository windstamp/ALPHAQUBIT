
import torch
from ai_models.model import AlphaQubitDecoder

def test_crash():
    # Grid size 3 -> 9 stabilizers. S=9.
    # F=1 (features)
    # hidden_dim=16 (must be divisible by num_heads=2)
    model = AlphaQubitDecoder(num_features=1, hidden_dim=16, num_stabilizers=9, grid_size=3, num_heads=2, num_layers=1)
    
    B = 1
    R = 2
    S = 9
    F = 1
    inputs = torch.randn(B, R, S, F)
    basis = torch.zeros(B, dtype=torch.long)
    final_mask = torch.zeros(B, S, dtype=torch.long)
    
    try:
        output = model(inputs, basis, final_mask)
        print("Model ran successfully")
    except NameError as e:
        print(f"Caught expected error: {e}")
    except Exception as e:
        print(f"Caught unexpected error: {e}")

if __name__ == "__main__":
    test_crash()
