from .functional import revgrad
from torch.nn import Module
from torch import tensor

class RevGrad(Module):
    def __init__(self, alpha = 1, *args, **kwargs):
        '''
        A gradient reversal layer

        This layer has no parameters, and simply reverses the gradient in the backward through
        '''

        super().__init__(*args, **kwargs)

        self.alpha = tensor(alpha, requires_grad = False)
    def forward(self, input_):
        return revgrad(input_, self.alpha)