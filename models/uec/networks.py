import torch
import torch.nn as nn
import torch.nn.functional as F


class ExNonlinearOp(nn.Module):
    def __init__(self, in_nc=3, out_nc=3,base_nf=64):
        super(ExNonlinearOp,self).__init__()
        self.base_nf = base_nf
        self.out_nc = out_nc
        self.encoder = nn.Conv2d(in_nc, base_nf, 1, 1)
        self.mid_conv = nn.Conv2d(base_nf, base_nf, 1, 1)
        self.decoder = nn.Conv2d(base_nf, out_nc, 1, 1)
        self.act = nn.LeakyReLU(inplace=True)

    def forward(self,x,val):
        x_code = self.encoder(x)
        x_code = self.act(x_code)
        x_code = self.act(self.mid_conv(x_code))
        y = self.decoder(x_code)
        return val*y + (1-val)*x

class ExCorrector(nn.Module):
    def __init__(self, in_nc=3, out_nc=3,base_nf=64):
        super(ExCorrector,self).__init__()
        self.in_nc = in_nc
        self.base_nf = base_nf
        self.out_nc = out_nc
        self.ex_block = ExNonlinearOp(in_nc,out_nc,base_nf)

    def forward(self,img,val):
        return self.ex_block(img,val)

class Encoder(nn.Module):
    def __init__(self, in_nc=3, encode_nf=32):
        super(Encoder, self).__init__()
        stride = 2
        pad = 0
        self.pad = nn.ZeroPad2d(1)
        self.conv1 = nn.Conv2d(in_nc, encode_nf, 7, stride, pad, bias=True)
        self.conv2 = nn.Conv2d(encode_nf, encode_nf, 3, stride, pad, bias=True)
        self.act = nn.ReLU(inplace=True)
        self.max = nn.AdaptiveMaxPool2d((1,1))

    def forward(self, x):
        b, _,_,_ = x.size()
        conv1_out = self.act(self.conv1(self.pad(x)))
        conv2_out = self.act(self.conv2(self.pad(conv1_out)))
        std, mean = torch.std_mean(conv2_out, dim=[2, 3], keepdim=False)
        maxs = self.max(conv2_out).squeeze(2).squeeze(2)
        out = torch.cat([std, mean, maxs], dim=1)
        return out

class DiffPredictor(nn.Module):
    def __init__(self,fea_dim1=96,fea_dim2=8):
        super(DiffPredictor,self).__init__()
        self.fc3 = nn.Linear(fea_dim1,fea_dim2)
        self.tanh = nn.Tanh()
        self.fc4 = nn.Linear(fea_dim2*2,1)
    def forward(self,img_fea1, img_fea2):
        val1 = self.tanh(self.fc3(img_fea1))
        val2 = self.tanh(self.fc3(img_fea2))
        val = torch.cat([val1,val2],dim=1)
        val = self.fc4(val)
        return val

class UECNetwork(nn.Module):
    def __init__(self, in_nc=3, out_nc = 3, base_nf = 64, encode_nf =32):
        super(UECNetwork,self).__init__()
        self.fea_dim = encode_nf * 3
        self.image_encoder = Encoder(in_nc,encode_nf)
        self.mExCorrector = ExCorrector()
        self.mDiffPredictor =  DiffPredictor(self.fea_dim)
        self.renderers = [self.mExCorrector]
        self.predict_heads = [self.mDiffPredictor]

    def render(self,img,vals):
        b,_,h,w = img.shape
        imgs = []
        for render, scalar in zip(self.renderers,vals):
            img = render(img,scalar)
            output_img = torch.clamp(img, 0, 1.0)
            imgs.append(output_img)
        return imgs

    def forward(self,img1, img2, return_vals = True):
        b,_,h,w = img1.shape
        vals = []
        for render, predict_head in zip(self.renderers,self.predict_heads):
            img1_resized = F.interpolate(input=img1, size=(256, int(256*w/h)), mode='bilinear', align_corners=False)
            img2_resized = F.interpolate(input=img2, size=(256, int(256*w/h)), mode='bilinear', align_corners=False)
            feat1 = self.image_encoder(img1_resized)
            feat2 = self.image_encoder(img2_resized)
            scalar = predict_head(feat1,feat2)
            vals.append(scalar)
            img = render(img1,scalar)
        img = torch.clamp(img, 0, 1.0)
        if return_vals:
            return img,vals
        else:
            return img