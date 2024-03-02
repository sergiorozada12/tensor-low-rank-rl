import numpy as np
import torch


class PARAFAC(torch.nn.Module):
    """This class is only introduced to experiment 9, as it was requested  as a follow up"""
    def __init__(self, dims: np.ndarray, k: int, scale: float = 1.0, nA: int = 1, bias=0.0) -> None:
        super().__init__()

        self.nA = nA
        self.k = k
        self.n_factors = len(dims)

        factors = []
        for dim in dims:
            factor = scale * (torch.randn(dim, k, dtype=torch.double, requires_grad=True) - bias) 
            factors.append(torch.nn.Parameter(factor))
        self.factors = torch.nn.ParameterList(factors)

    def forward(self, indices: np.ndarray) -> torch.Tensor:
        prod = torch.ones(self.k, dtype=torch.double)
        for i in range(len(indices)):
            idx = indices[i]
            factor = self.factors[i]
            prod *= factor[idx, :]
        if len(indices) < len(self.factors):
            res = []
            for cols in zip(
                *[self.factors[-(a + 1)].t() for a in reversed(range(self.nA))]
            ):
                kr = cols[0]
                for j in range(1, self.nA):
                    kr = torch.kron(kr, cols[j])
                res.append(kr)
            factors_action = torch.stack(res, dim=1)
            return torch.matmul(prod, factors_action.T)
        return torch.sum(prod, dim=-1)
