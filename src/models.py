import torch
from torch import nn

class Block(nn.Module):
    def __init__(self,a,b):
        super().__init__(); self.net=nn.Sequential(nn.Conv2d(a,b,3,padding=1),nn.GELU(),nn.Conv2d(b,b,3,padding=1),nn.GELU())
    def forward(self,x): return self.net(x)

class ForecastNet(nn.Module):
    def __init__(self,w=32):
        super().__init__()
        self.enc1=Block(2,w); self.enc2=Block(w,w*2); self.enc3=Block(w*2,w*4); self.mid=Block(w*4,w*4)
        self.pool=nn.AvgPool2d(2); self.up=nn.Upsample(scale_factor=2,mode='bilinear',align_corners=False)
        self.dec2=Block(w*6,w*2); self.dec1=Block(w*3,w); self.out=nn.Conv2d(w,1,1)
    def forward(self,x):
        a=self.enc1(x); b=self.enc2(self.pool(a)); c=self.mid(self.enc3(self.pool(b)))
        return self.out(self.dec1(torch.cat((self.up(self.dec2(torch.cat((self.up(c),b),1))),a),1)))
