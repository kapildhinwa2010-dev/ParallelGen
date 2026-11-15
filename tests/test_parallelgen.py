import pytest
import torch
from parallelgen import ParallelGen, create_parallelgen

class TestParallelGen:
    def test_init(self):
        m = ParallelGen(dim=256, hidden_dim=512)
        assert m.dim == 256
    
    def test_forward(self):
        m = ParallelGen(dim=256)
        x = torch.randn(32, 256)
        assert m(x).shape == (32, 256)
    
    def test_factory(self):
        m = create_parallelgen()
        assert isinstance(m, ParallelGen)

if __name__ == '__main__':
    pytest.main([__file__, '-v'])
