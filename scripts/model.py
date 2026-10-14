import torch
from torch import nn
from torch.nn import BatchNorm2d, Conv2d, Module, PReLU, Sequential
import math
import torch.nn.functional as F
from torch.nn import Module, Parameter

def conv_bn(inp, oup, stride = 1):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU6(inplace=True)
    )
    
def conv_dw(inp, oup, stride = 1):
    return nn.Sequential(
        nn.Conv2d(inp, inp, 3, stride, 1, groups=inp, bias=False),
        nn.BatchNorm2d(inp),
        nn.ReLU6(inplace=True),

        nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU6(inplace=True),
    )
    
class MobileNetV1(nn.Module):
    fc_scale = 7 * 7
    def __init__(self, dropout_keep_prob, embedding_size):
        super(MobileNetV1, self).__init__()
        self.stage1 = nn.Sequential(
            conv_bn(3, 32, 1),    # 3
            conv_dw(32, 64, 1),   # 7

            conv_dw(64, 128, 2),  # 11
            conv_dw(128, 128, 1),  # 19

            conv_dw(128, 256, 2),  # 27
            conv_dw(256, 256, 1),  # 43
        )
        self.stage2 = nn.Sequential(
            conv_dw(256, 512, 2),  # 43 + 16 = 59
            conv_dw(512, 512, 1), # 59 + 32 = 91
            conv_dw(512, 512, 1), # 91 + 32 = 123
            conv_dw(512, 512, 1), # 123 + 32 = 155
            conv_dw(512, 512, 1), # 155 + 32 = 187
            conv_dw(512, 512, 1), # 187 + 32 = 219
        )
        self.stage3 = nn.Sequential(
            conv_dw(512, 1024, 2), # 219 +3 2 = 241
            conv_dw(1024, 1024, 1), # 241 + 64 = 301
        )

        self.sep        = nn.Conv2d(1024, 512, kernel_size=1, bias=False)
        self.sep_bn     = nn.BatchNorm2d(512)
        self.prelu      = nn.PReLU(512)

        self.bn2        = nn.BatchNorm2d(512, eps=1e-05)
        self.dropout    = nn.Dropout(p=dropout_keep_prob, inplace=True)
        self.linear     = nn.Linear(512 * self.fc_scale, embedding_size)
        self.features   = nn.BatchNorm1d(embedding_size, eps=1e-05)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, 0, 0.1)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
                
    def forward(self, x):
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)

        x = self.sep(x)
        x = self.sep_bn(x)
        x = self.prelu(x)

        x = self.bn2(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.linear(x)
        x = self.features(x)
        return x
    
class Arcface_Head(Module):
    def __init__(self, num_classes, embedding_size=512, s=64., m=0.5):
        super(Arcface_Head, self).__init__()
        self.s = s
        self.m = m
        self.weight = Parameter(torch.FloatTensor(num_classes, embedding_size))
        nn.init.xavier_uniform_(self.weight)

        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m

    def forward(self, input, label):
        cosine  = F.linear(input, F.normalize(self.weight))
        sine    = torch.sqrt((1.0 - torch.pow(cosine, 2)).clamp(0, 1))
        phi     = cosine * self.cos_m - sine * self.sin_m
        phi     = torch.where(cosine.float() > self.th, phi.float(), cosine.float() - self.mm)

        one_hot = torch.zeros(cosine.size()).type_as(phi).long()
        one_hot.scatter_(1, label.view(-1, 1).long(), 1)
        output  = (one_hot * phi) + ((1.0 - one_hot) * cosine) 
        output  *= self.s
        return output
    
class Arcface(nn.Module):
    def __init__(self, num_classes=10572, embedding_size=512, s = 64, m = 0.5):
        super(Arcface, self).__init__()
        self.arcface = MobileNetV1(dropout_keep_prob=0.5, embedding_size=embedding_size)
        self.head = Arcface_Head(num_classes=num_classes, embedding_size=embedding_size, s=s, m=m)

    def forward(self, x):
        x = self.arcface(x)
        x = x.view(x.size()[0], -1)
        x = F.normalize(x)
        return x

    def forward_train(self, x, y):
        x = self.arcface(x)
        x = x.view(x.size()[0], -1)
        x = F.normalize(x)
        x = self.head(x, y)
        return x