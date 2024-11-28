import torch
from torch import nn, fft
from torch.nn import functional as F
from einops import rearrange

class depthwise_conv(nn.Module):
    def __init__(self, nin, nout,kernal_size,pad=True):
        super(depthwise_conv, self).__init__()
        self.depthwise = nn.Conv2d(nin, nin, kernel_size=kernal_size, padding=(kernal_size-1)//2 if pad else 0, groups=nin)
        self.pointwise = nn.Conv2d(nin, nout, kernel_size=1)

    def forward(self, x):
        out = self.depthwise(x)
        out = self.pointwise(out)
        return out

class MultiHeadImageSelfAttention(nn.Module):
    def __init__(self, in_channels, hidden_channels,patch_size=32):
        super(MultiHeadImageSelfAttention, self).__init__()

        self.conv1_1x1 = nn.Conv2d(in_channels, hidden_channels, kernel_size=1, padding=0)
        self.conv1_3x3 = nn.Conv2d(in_channels, hidden_channels, kernel_size=3, padding=1)
        self.conv1_5x5 = nn.Conv2d(in_channels, hidden_channels, kernel_size=5, padding=2)
        self.query = nn.Conv2d(hidden_channels, hidden_channels, 1)
        self.key = nn.Conv2d(hidden_channels, hidden_channels, 1)
        self.patch_size = patch_size

    def forward(self, input_image):
        # input_image has shape (N, C, H, W)

        x1 = self.conv1_1x1(input_image)
        x2 = self.conv1_3x3(input_image)
        x3 = self.conv1_5x5(input_image)
        # x1, x2, x3 have shape (N, C', H, W)

        x1 = rearrange(x1, 'b c (h patch1) (w patch2) -> (b h w) c patch1 patch2', patch1=self.patch_size,
                            patch2=self.patch_size)
        x2 = rearrange(x2, 'b c (h patch1) (w patch2) -> (b h w) c patch1 patch2', patch1=self.patch_size,
                       patch2=self.patch_size)
        x3 = rearrange(x3, 'b c (h patch1) (w patch2) -> (b h w) c patch1 patch2', patch1=self.patch_size,
                       patch2=self.patch_size)

        Q1 = self.query(x1).view(*x1.shape[:-2], -1)
        K1 = self.key(x1).view(*x1.shape[:2], -1)
        Q2 = self.query(x2).view(*x2.shape[:-2], -1)
        K2 = self.key(x2).view(*x2.shape[:-2], -1)
        Q3 = self.query(x3).view(*x3.shape[:-2], -1)
        K3 = self.key(x3).view(*x3.shape[:-2], -1)
        # Q1, K1, Q2, K2, Q3, K3 have shape (N*S*S, C, patch_size*patch_size)

        attention_weights1 = F.softmax(Q1.transpose(-2,-1) @ K1 / (x1.shape[1] ** 0.5), dim=-1)
        attention_weights2 = F.softmax(Q2.transpose(-2,-1) @ K2 / (x2.shape[1] ** 0.5), dim=-1)
        attention_weights3 = F.softmax(Q3.transpose(-2,-1) @ K3 / (x3.shape[1] ** 0.5), dim=-1)
        # attention_weights1, attention_weights2, attention_weights3 have shape (N*S*S, patch_size*patch_size)

        return attention_weights1, attention_weights2, attention_weights3
        # outputs have shape (N, H*W, H*W)


class MLP(nn.Module):
    def __init__(self,size,ratio1=0.01):
        super(MLP, self).__init__()
        size = int(size)
        self.fc1 = nn.Linear(size, int(size*ratio1))
        self.fc2 = nn.Linear(int(size*ratio1),size)

    def forward(self, y):
        # Flatten the input
        x = y.view(y.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return x.view(*y.shape)

class ResBlock(nn.Module):
    def __init__(self,nin,nout):
        super(ResBlock, self).__init__()
        self.layer = nn.Sequential(nn.Conv2d(nin,nin,1),nn.ReLU(),nn.Conv2d(nin,nout,3,padding=1))

    def forward(self,x):
        return self.layer(x)+x

class MSAttBlock(nn.Module):

    def __init__(self,nin,patch_size=8):
        super(MSAttBlock, self).__init__()
        self.ms_to_hidden = nn.Sequential(nn.Conv2d(nin,nin*3,3,padding=1),nn.ReLU(),nn.Conv2d(nin*3,nin*6,3,padding=1))
        self.pan_to_hidden = nn.Conv2d(nin,nin*3,3,padding=1)
        self.patch_size = patch_size
        self.after_attention = nn.Conv2d(nin*3,nin,3,padding=1)

    def forward(self,ms,pan):
        fpan = self.pan_to_hidden(pan)
        hidden_ms = self.ms_to_hidden(ms)
        q_ms, k_ms = hidden_ms.chunk(2, dim=1)
        q_patch = rearrange(q_ms, 'b c (h patch1) (w patch2) -> b c h w patch1 patch2', patch1=self.patch_size,
                            patch2=self.patch_size)
        k_patch = rearrange(k_ms, 'b c (h patch1) (w patch2) -> b c h w patch1 patch2', patch1=self.patch_size,
                            patch2=self.patch_size)
        q_fft = torch.fft.fft2(q_patch.float())
        k_fft = torch.fft.fft2(k_patch.float())
        out = q_fft * k_fft
        out = torch.fft.irfft2(out, s=(self.patch_size, self.patch_size))
        out = rearrange(out, 'b c h w patch1 patch2 -> b c (h patch1) (w patch2)', patch1=self.patch_size,
                        patch2=self.patch_size)
        attention_map_ms = F.layer_norm(out,out.shape[-2:])
        attention1 = attention_map_ms*fpan
        attention1 = self.after_attention(attention1)

        return attention1+ms

class PanAttBlock(nn.Module):

    def __init__(self,nin,patch_size=32):
        super(PanAttBlock, self).__init__()
        self.pan_attention = MultiHeadImageSelfAttention(nin,nin,patch_size)
        self.after_att = nn.Sequential(nn.Conv2d(nin*3,nin*3,3,padding=1),nn.ReLU(),depthwise_conv(nin*3,nin,3))
        self.patch_size = patch_size

    def forward(self,ms,pan):
        att1,att2,att3 = self.pan_attention(pan)
        # output has shape (N*S*S, patch_size*patch_size, patch_size*patch_size)
        hidden_ms_reshape = rearrange(ms.contiguous(),'b c (h patch1) (w patch2) -> (b h w) c (patch1 patch2)', patch1=self.patch_size,
                            patch2=self.patch_size)
        # shape (N*S*S, C, patch_size*patch_size)
        h = int(ms.size()[2]/self.patch_size)
        att1 = rearrange(hidden_ms_reshape @ att1,'(b h w) c (patch1 patch2) -> b c (h patch1) (w patch2)', patch1=self.patch_size,
                            patch2=self.patch_size,h=h,w=h)
        att2 = rearrange(hidden_ms_reshape @ att2,'(b h w) c (patch1 patch2) -> b c (h patch1) (w patch2)', patch1=self.patch_size,
                            patch2=self.patch_size,h=h,w=h)
        att3 = rearrange(hidden_ms_reshape @ att3,'(b h w) c (patch1 patch2) -> b c (h patch1) (w patch2)', patch1=self.patch_size,
                            patch2=self.patch_size,h=h,w=h)
        attention2 = self.after_att(torch.concat([att1,att2,att3],dim=1))

        return attention2+pan

class Merger(nn.Module):
    def __init__(self,nin):
        super(Merger, self).__init__()
        self.deal_diff1,self.deal_diff2 = [nn.Sequential(nn.Conv2d(nin,nin*3,3,padding=1),nn.ReLU(),depthwise_conv(nin*3,nin*3,3),nn.ReLU(),nn.Conv2d(nin*3,nin,3,padding=1)) for _ in range(2)]

    def forward(self,ms,pan):
        minus = pan - ms
        return self.deal_diff1(-minus) + ms, pan-self.deal_diff2(minus)

class CrossNet(nn.Module):

    def __init__(self,nin=8):
        super(CrossNet, self).__init__()
        self.pan_mult = nn.Conv2d(1,nin,3,padding=1)
        self.mt1, self.mt2, self.mt3, self.mt4, self.mt5 = [MSAttBlock(nin) for _ in range(5)]
        self.pt1, self.pt2, self.pt3, self.pt4, self.pt5 = [PanAttBlock(nin) for _ in range(5)]
        self.res12, self.res23, self.res34, self.res45 = [ResBlock(nin,nin) for _ in range(4)]
        self.res121, self.res231, self.res341, self.res451 = [ResBlock(nin, nin) for _ in range(4)]
        self.fusion12, self.fusion23, self.fusion34, self.fusion45,self.fusion5 = [Merger(nin) for _ in range(5)]
        self.merge = nn.Sequential(nn.Conv2d(nin*2,nin*4,3,padding=1),nn.ReLU(),nn.Conv2d(nin*4,nin,3,padding=1))

    def forward(self,ms,pan):
        fpan = self.pan_mult(pan)

        pan1 = self.res12(self.mt1(ms,fpan))
        ms1 = self.res121(self.pt1(ms,fpan))
        ms1,pan1 = self.fusion12(ms1,pan1)

        pan2 = self.res23(self.mt2(ms1,pan1))
        ms2 = self.res231(self.pt2(ms1,pan1))
        ms2,pan2 = self.fusion23(ms2,pan2)

        pan3 = self.res34(self.mt3(ms2,pan2))
        ms3 = self.res341(self.pt3(ms2,pan2))
        ms3,pan3 = self.fusion34(ms3,pan3)

        pan4 = self.res45(self.mt4(ms3,pan3))
        ms4 = self.res451(self.pt4(ms3,pan3))
        ms4,pan4 = self.fusion45(ms4,pan4)

        pan5 = self.mt5(ms4,pan4)
        ms5 = self.pt5(ms4,pan4)
        ms5,pan5 = self.fusion5(ms5,pan5)

        out = self.merge(torch.concatenate([pan5,ms5],dim=1))

        return out+ms

if __name__ == '__main__':
    from torchsummary import summary
    device = torch.device('cuda:0')
    net = CrossNet(4)
    summary(net.to(device),[(4,64,64),(1,64,64)])