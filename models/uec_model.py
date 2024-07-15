from .uec.networks import *
from .uec.loss import *
from .base_model import BaseModel
from . import networks
# import os
# import logging
# from collections import OrderedDict, defaultdict
import torch
import torch.nn as nn
# import random
import torchvision.models as models
import torch.nn.functional as F

class UECModel(BaseModel):
    def __init__(self, opt):
        super(UECModel, self).__init__(opt)
        self.opt = opt
        self.loss_names = ['pix','mon','psnr']
        self.model_names = ['G']
        self.visual_names = ['img1','fake_img','img2']
        # self.netG = NeurOP(net_opt['in_nc'],net_opt['out_nc'],net_opt['base_nf'],net_opt['cond_nf'],net_opt['init_model'])
        netG = UECNetwork()
        # self.set_requires_grad(netG.mExCorrector, False)
        self.netG = networks.init_net(netG, opt.init_type, opt.init_gain, self.gpu_ids)
        self.optimizers = []
        self.softmax = nn.Softmax()
        # self.netG.load_state_dict(torch.load(weights))

        # self.print_network()
        if self.isTrain:
            self.netG.train()
            # train_opt = opt['train']
            self.cri_pix = nn.MSELoss()
            # self.cri_pix = nn.L1Loss()
            # self.cri_cos = CosineLoss()
            self.cri_ratio = 1.0 / 10.0
            self.optimizer_G = torch.optim.Adam(self.netG.parameters(), lr=opt.lr*opt.e_lr_ratio,
                                                betas=(opt.beta1, 0.999))
            self.optimizers.append(self.optimizer_G)

    def set_input(self, inputimg):
        self.img1 = inputimg['image_pair'][0].to(self.device)
        self.img2 = inputimg['image_pair'][1].to(self.device)
        self.image_paths = inputimg['image_path']
        if(self.opt.phase == "train"):
            self.ref1 = inputimg['image_pair2'][0].to(self.device)
            self.ref2 = inputimg['image_pair2'][1].to(self.device)


    def forward(self):
        self.fake_img = self.netG(self.img1, self.img2)[0]
        self.fake_img = self.netG(self.fake_img, self.img2)[0]
        self.fake_img = self.netG(self.fake_img, self.img2)[0]
        if(self.opt.phase == "train"):
            self.fake_ref1 = self.netG(self.img1, self.ref1)[0]
            self.fake_ref2 = self.netG(self.img1, self.ref2)[0]

    def backward(self):
        self.loss_pix = self.cri_pix(self.fake_img, self.img2)
        self.loss_psnr = 10 * torch.log10(1 / self.loss_pix)
        self.loss_mon = torch.mean(F.relu(self.fake_ref1 - self.fake_ref2))
        self.loss = self.loss_pix + self.loss_mon
        self.loss.backward()


    def optimize_parameters(self):
        self.optimizer_G.zero_grad()
        self.forward()
        self.backward()
        self.optimizer_G.step()

    def compute_visuals(self):
        pass
