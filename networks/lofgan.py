import random

import numpy as np
import torch
from torch import autograd

from networks.blocks import *
from networks.loss import *
import networks.mixtotorch as mixformer
from einops import rearrange, repeat

import torch.nn as nn
import torchvision
from sklearn.decomposition import FastICA


class LoFGAN(nn.Module):
    def __init__(self, config):
        super(LoFGAN, self).__init__()
        #nf: 32  n_downs: 2  norm: bn
        self.gen = Generator(config['gen'])
        #nf: 64  n_res_blks: 4  num_classes: 1802
        self.dis = Discriminator(config['dis'])

        self.w_adv_g = config['w_adv_g']
        self.w_adv_d = config['w_adv_d']
        self.w_recon = config['w_recon']
        self.w_cls = config['w_cls']
        self.w_gp = config['w_gp']
        self.n_sample = config['n_sample_train']

    def forward(self, xs, y, mode):
        if mode == 'gen_update':
            fake_x, base_index = self.gen(xs)


            feat_real, _, _ = self.dis(xs)
            feat_fake, logit_adv_fake, logit_c_fake = self.dis(fake_x)
            loss_adv_gen = torch.mean(-logit_adv_fake)
            loss_cls_gen = F.cross_entropy(logit_c_fake, y.squeeze())

            loss_adv_gen = loss_adv_gen * self.w_adv_g
            loss_cls_gen = loss_cls_gen * self.w_cls

            loss_total = loss_adv_gen + loss_cls_gen
            loss_total.backward()

            return {'loss_total': loss_total,
                    'loss_adv_gen': loss_adv_gen,
                    'loss_cls_gen': loss_cls_gen}

        elif mode == 'dis_update':
            xs.requires_grad_()
            _, logit_adv_real, logit_c_real = self.dis(xs)
            loss_adv_dis_real = torch.nn.ReLU()(1.0 - logit_adv_real).mean()
            loss_adv_dis_real = loss_adv_dis_real * self.w_adv_d
            loss_adv_dis_real.backward(retain_graph=True)

            y_extend = y.repeat(1, self.n_sample).view(-1)
            index = torch.LongTensor(range(y_extend.size(0))).cuda()
            logit_c_real_forgp = logit_c_real[index, y_extend].unsqueeze(1)
            loss_reg_dis = self.calc_grad2(logit_c_real_forgp, xs)

            loss_reg_dis = loss_reg_dis * self.w_gp
            loss_reg_dis.backward(retain_graph=True)

            loss_cls_dis = F.cross_entropy(logit_c_real, y_extend)
            loss_cls_dis = loss_cls_dis * self.w_cls
            loss_cls_dis.backward()

            with torch.no_grad():
                fake_x = self.gen(xs)[0]

            _, logit_adv_fake, _ = self.dis(fake_x.detach())
            loss_adv_dis_fake = torch.nn.ReLU()(1.0 + logit_adv_fake).mean()
            loss_adv_dis_fake = loss_adv_dis_fake * self.w_adv_d
            loss_adv_dis_fake.backward()

            loss_total = loss_adv_dis_real + loss_adv_dis_fake + loss_cls_dis
            return {'loss_total': loss_total,
                    'loss_adv_dis': loss_adv_dis_fake + loss_adv_dis_real,
                    'loss_adv_dis_real': loss_adv_dis_real,
                    'loss_adv_dis_fake': loss_adv_dis_fake,
                    'loss_cls_dis': loss_cls_dis,
                    'loss_reg': loss_reg_dis}

        else:
            assert 0, 'Not support operation'

    def generate(self, xs):
        fake_x = self.gen(xs)[0]
        return fake_x
    
    def calc_grad2(self, d_out, x_in):
        batch_size = x_in.size(0)
        grad_dout = autograd.grad(outputs=d_out.mean(),
                                  inputs=x_in,
                                  create_graph=True,
                                  retain_graph=True,
                                  only_inputs=True)[0]
        grad_dout2 = grad_dout.pow(2)
        assert (grad_dout2.size() == x_in.size())
        reg = grad_dout2.sum()
        reg /= batch_size
        return reg


class Discriminator(nn.Module):
    def __init__(self, config):
        super(Discriminator, self).__init__()
        self.soft_label = False
        #64
        nf = config['nf']
        #119
        n_class = config['num_classes']
        #4
        n_res_blks = config['n_res_blks']
        
        cnn_f = [Conv2dBlock(3, nf, 5, 1, 2,
                             pad_type='reflect',
                             norm='sn',
                             activation='none')]
        for i in range(n_res_blks):
            nf_out = np.min([nf * 2, 1024])
            cnn_f += [ActFirstResBlock(nf, nf_out, fhid=None, activation='lrelu', norm='sn')]
            cnn_f += [nn.ReflectionPad2d(1)]
            cnn_f += [nn.AvgPool2d(kernel_size=3, stride=2)]
            nf = np.min([nf * 2, 1024])

        nf_out = np.min([nf * 2, 1024])
        cnn_f += [ActFirstResBlock(nf, nf_out, fhid=None, activation='lrelu', norm='sn')]
        cnn_adv = [nn.AdaptiveAvgPool2d(1),
                   Conv2dBlock(nf_out, 1, 1, 1,
                               norm='none',
                               activation='none',
                               activation_first=False)]
        cnn_c = [nn.AdaptiveAvgPool2d(1),
                 Conv2dBlock(nf_out, n_class, 1, 1,
                             norm='none',
                             activation='none',
                             activation_first=False)]
        self.cnn_f = nn.Sequential(*cnn_f)
        self.cnn_adv = nn.Sequential(*cnn_adv)
        self.cnn_c = nn.Sequential(*cnn_c)

    def forward(self, x):
        if len(x.size()) == 5:
            B, K, C, H, W = x.size()
            x = x.view(B * K, C, H, W)
        else:
            B, C, H, W = x.size()
            K = 1
        feat = self.cnn_f(x)
        logit_adv = self.cnn_adv(feat).view(B * K, -1)
        logit_c = self.cnn_c(feat).view(B * K, -1)
        return feat, logit_adv, logit_c


class Generator(nn.Module):
    def __init__(self,config):
        super(Generator, self).__init__()
        #编码层
        self.encoder = mixformer.MixFormer()
        #解码层
        self.decoder = Decoder()

        self.convencoder = Encoder()
        self.nla = NonLocalBlock(128)
        self.alpha = torch.nn.Parameter(torch.empty(1, 1), requires_grad=True)
        self.beta = torch.nn.Parameter(torch.empty(1, 1), requires_grad=True)
        self.gama = torch.nn.Parameter(torch.empty(1, 1), requires_grad=True)
        nn.init.constant_(self.alpha, 44)
        nn.init.constant_(self.beta, 42)
        nn.init.constant_(self.gama, 42)
        
    def forward(self, xs):
        b, k, C, H, W = xs.shape
        xs_1 = xs.reshape([-1, C, H, W])
        querys = self.encoder(xs_1)
        querys = querys.permute(0,2,1)
        querys = querys.view(-1,128,8,8)
        c, h, w = querys.size()[-3:]
        querys = querys.view(b, k, c, h, w)

        base_index = random.choice(range(k))
        base_feat = querys[:, base_index, :, :, :]
        refs = torch.cat([querys[:, :base_index, :, :, :], querys[:, (base_index + 1):, :, :, :]], dim=1)
        ref_index = random.choice(range(2))
        ref_feat_1 = refs[:, ref_index, :, :, :]
        ref_feat_2 = refs[:, 1-ref_index, :, :, :]

        xs_2 = xs[:, base_index, :, :, :]
        ec1, ec2, ec3, ec4, ec5 = self.convencoder(xs_2)

        feat_gen = self.nla(ref_feat_1, ref_feat_2, base_feat)
        if self.alpha < 0:
            self.alpha.data = self.alpha + 10

        if self.beta < 0:
            self.beta.data = self.beta + 10
        #print(f'alpha={self.alpha.item()},beta={self.beta.item()}')

        feat_gen = feat_gen.view(b, 128, 64)
        ref_feat_1 = ref_feat_1.view(b, 128, 64)
        ref_feat_2 = ref_feat_2.view(b, 128, 64)
        feat_gen_1 = feat_gen.detach().cpu().numpy()
        ref_feat_1_1 = ref_feat_1.detach().cpu().numpy()
        ref_feat_2_1 = ref_feat_2.detach().cpu().numpy()
        r1= []
        for i in range(b):
            feat_gen_2 = feat_gen_1[i,:,:]
            ref_feat_1_2 = ref_feat_1_1[i,:,:]
            ref_feat_2_2 = ref_feat_2_1[i,:,:]
            feat_gen_2_centered = feat_gen_2 - feat_gen_2.mean(axis=0)
            feat_gen_2_centered -= feat_gen_2_centered.mean(axis=1).reshape(128, -1)
            ref_feat_1_2_centered = ref_feat_1_2 - ref_feat_1_2.mean(axis=0)
            ref_feat_1_2_centered -= ref_feat_1_2_centered.mean(axis=1).reshape(128, -1)
            ref_feat_2_2_centered = ref_feat_2_2 - ref_feat_2_2.mean(axis=0)
            ref_feat_2_2_centered -= ref_feat_2_2_centered.mean(axis=1).reshape(128, -1)
            mix_1 = feat_gen_2_centered.T
            mix_2 = ref_feat_1_2_centered.T
            mix_3 = ref_feat_2_2_centered.T
            u_1 = FastICA(n_components=int(self.alpha)).fit_transform(mix_1)
            u_1 = u_1.T
            u_2 = FastICA(n_components=int(self.beta)).fit_transform(mix_2)
            u_2 = u_2.T
            u_3 = FastICA(n_components=(128-int(self.alpha)-int(self.beta))).fit_transform(mix_3)
            u_3 = u_3.T
            s = np.concatenate((u_1,u_2),axis=0)
            s = np.concatenate((s,u_3),axis=0)
            result_tensor = torch.from_numpy(s)
            r3 = result_tensor.unsqueeze(dim=0)
            r1.append(r3)
        feat_fuse = r1[0]
        for k in range(1,b):
            feat_fuse = torch.cat((feat_fuse,r1[k]), 0)
        feat_fuse = feat_fuse.view(b, 128, 8, 8).float().cuda()
        feat_fuse_1 = feat_fuse.clone()
        fake_x = self.decoder(feat_fuse, ec1, ec2, ec3, ec4, ec5, feat_fuse_1)
        
        return fake_x, base_index

class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()

        self.Ec1 = Conv2dBlock(3, 32, 5, 1, 2,
                             norm='bn',
                             activation='lrelu',
                             pad_type='reflect')
        self.Ec2 = Conv2dBlock(32, 64, 3, 2, 1,
                             norm='bn',
                             activation='lrelu',
                             pad_type='reflect')
        self.Ec3 = Conv2dBlock(64, 128, 3, 2, 1,
                             norm='bn',
                             activation='lrelu',
                             pad_type='reflect')
        self.Ec4 = Conv2dBlock(128, 128, 3, 2, 1,
                             norm='bn',
                             activation='lrelu',
                             pad_type='reflect')
        self.Ec5 = Conv2dBlock(128, 128, 3, 2, 1,
                             norm='bn',
                             activation='lrelu',
                             pad_type='reflect')

    def forward(self, x):
        ec1 = self.Ec1(x)
        ec2 = self.Ec2(ec1)
        ec3 = self.Ec3(ec2)
        ec4 = self.Ec4(ec3)
        ec5 = self.Ec5(ec4)
        return ec1, ec2, ec3, ec4, ec5


class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.Upsample = nn.Upsample(scale_factor=2)
        self.Dc1 = Conv2dBlock(128, 128, 3, 1, 1,
                             norm='bn',
                             activation='lrelu',
                             pad_type='reflect')
        self.Dc2 = Conv2dBlock(128, 128, 3, 1, 1,
                             norm='bn',
                             activation='lrelu',
                             pad_type='reflect')
        self.Dc3 = Conv2dBlock(128, 64, 3, 1, 1,
                             norm='bn',
                             activation='lrelu',
                             pad_type='reflect')
        self.Dc4 =  Conv2dBlock(64, 32, 3, 1, 1,
                             norm='bn',
                             activation='lrelu',
                             pad_type='reflect')
        self.Dc5 = Conv2dBlock(32, 3, 5, 1, 2,
                             norm='none',
                             activation='tanh',
                             pad_type='reflect')
        self.concat1 = Conv2dBlock(384,128,1,1)
        self.concat2 = Conv2dBlock(384,128,1,1)
        self.concat3 = Conv2dBlock(384,128,1,1)
        self.concat4 = Conv2dBlock(192,64,1,1)
        self.concat5 = Conv2dBlock(96,32,1,1)

    def forward(self, x, ec1, ec2, ec3, ec4, ec5, feat_fuse_1):
        x = torch.cat([x, ec5, feat_fuse_1], dim=1)
        x = self.concat1(x)
        x = self.Upsample(x)
        feat_fuse_1 = self.Upsample(feat_fuse_1)
        x = self.Dc1(x)
        feat_fuse_1 = self.Dc1(feat_fuse_1)
        x = torch.cat([x, ec4, feat_fuse_1],dim = 1)
        x = self.concat2(x)
        x = self.Upsample(x)
        feat_fuse_1 = self.Upsample(feat_fuse_1)
        x = self.Dc2(x)
        feat_fuse_1 = self.Dc2(feat_fuse_1)
        x = torch.cat([x, ec3, feat_fuse_1],dim=1)
        x = self.concat3(x)
        x = self.Upsample(x)
        feat_fuse_1 = self.Upsample(feat_fuse_1)
        x = self.Dc3(x)
        feat_fuse_1 = self.Dc3(feat_fuse_1)
        x = torch.cat([x, ec2, feat_fuse_1],dim=1)
        x = self.concat4(x)
        x = self.Upsample(x)
        feat_fuse_1 = self.Upsample(feat_fuse_1)
        x = self.Dc4(x)
        feat_fuse_1 = self.Dc4(feat_fuse_1)
        x = torch.cat([x, ec1, feat_fuse_1],dim=1)
        x = self.concat5(x)
        x = self.Dc5(x)
        return x

class LN(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
    def forward(self, x):
        return self.norm(x)


class MLP(nn.Module):
    def __init__(self, in_feat, hid_feat = None, out_feat = None, dropout = 0.):
        super().__init__()
        if not hid_feat:
            hid_feat = in_feat
        if not out_feat:
            out_feat = in_feat
        self.linear1 = nn.Linear(in_feat, hid_feat)
        self.activation = nn.GELU()
        self.linear2 = nn.Linear(hid_feat, out_feat)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.linear1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.linear2(x)
        return self.dropout(x)

class NonLocalBlock(nn.Module):
    def __init__(self, channel):
        super(NonLocalBlock, self).__init__()
        self.inter_channel = channel // 2
        self.conv_phi = nn.Conv2d(in_channels=channel, out_channels=self.inter_channel, kernel_size=1, stride=1,padding=0, bias=False)
        self.conv_theta = nn.Conv2d(in_channels=channel, out_channels=self.inter_channel, kernel_size=1, stride=1, padding=0, bias=False)
        self.conv_g = nn.Conv2d(in_channels=channel, out_channels=self.inter_channel, kernel_size=1, stride=1, padding=0, bias=False)
        self.softmax = nn.Softmax(dim=1)
        self.conv_mask = nn.Conv2d(in_channels=self.inter_channel, out_channels=channel, kernel_size=1, stride=1, padding=0, bias=False)

    def forward(self, x1, x2, x3):
        # [N, C, H , W]
        b, c, h, w = x1.size()
        # [N, C/2, H * W]
        x_phi = self.conv_phi(x1).view(b, c/2, -1)
        # [N, H * W, C/2]
        x_theta = self.conv_theta(x2).contiguous().view(b, c/2, -1).permute(0, 2, 1)
        x_g = self.conv_g(x3).contiguous().view(b, c/2, -1).permute(0, 2, 1)
        # [N, H * W, H * W]
        mul_theta_phi = torch.matmul(x_theta, x_phi)
        mul_theta_phi = self.softmax(mul_theta_phi)
        # [N, H * W, C/2]
        mul_theta_phi_g = torch.matmul(mul_theta_phi, x_g)
        # [N, C/2, H, W]
        mul_theta_phi_g = mul_theta_phi_g.permute(0,2,1).contiguous().view(b,self.inter_channel, h, w)
        # [N, C, H , W]
        mask = self.conv_mask(mul_theta_phi_g)
        out = mask + x3
        return out


if __name__ == '__main__':
    config = {}
    model = Generator(config).cuda()
    x = torch.randn(8, 3, 3, 128, 128).cuda()
    y,_ = model(x)
    print("--------",y.size())
