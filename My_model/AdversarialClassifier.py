"""
Common classifier and Adversarial classifier
"""
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_revgrad import RevGrad


class LinearNorm(nn.Module):
    def __init__(self, in_dim, out_dim, bias=True, w_init_gain='linear'):
        super(LinearNorm, self).__init__()
        self.linear_layer = nn.Linear(in_dim, out_dim, bias=bias)
        torch.nn.init.xavier_uniform_(self.linear_layer.weight, gain=torch.nn.init.calculate_gain(w_init_gain))

    def forward(self, x):
        return self.linear_layer(x)


class AdversarialClassifier(nn.Module):
    def __init__(self, input_emb, dim_content=82, num_classes=100):
        super(AdversarialClassifier, self).__init__()
        self.fullfeed = nn.Sequential(LinearNorm(input_emb, 256), LinearNorm(256, num_classes))
        if input_emb == dim_content:
            print("adversarial classifier for content embedding")
            self.revnetwork = torch.nn.Sequential(RevGrad(), self.fullfeed)  # 逆转梯度？
        else:
            print("common classifier for speaker embedding")
            self.revnetwork = torch.nn.Sequential(self.fullfeed)

    def forward(self, x):  # (batch, 1, dim_emb)
        domain_class = self.revnetwork(x)  # to (batch, 1, num_classes)
        domain_preds = torch.mean(domain_class, dim=1)  # to (batch, num_classes)
        # domain_preds = F.softmax(domain_class, dim=1)
        return domain_preds


if __name__ == '__main__':
    adv_classifier = AdversarialClassifier(input_emb=32, dim_content=32, num_classes=174)
    dataX = torch.rand(16, 271, 32)
    domain_predict = adv_classifier(dataX)
    print(domain_predict.shape)
