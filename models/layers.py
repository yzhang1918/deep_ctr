from torch import nn
from torch_sparse import spmm


class SPLinear(nn.Linear):

    def forward(self, x):
        y = spmm(x._indices(), x._values(), x.size(0), self.weight.t())
        if self.bias is not None:
            y += self.bias
        return y

