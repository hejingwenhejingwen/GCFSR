import cv2
import math
import numpy as np
import random
import torch
from collections import OrderedDict
from os import path as osp
import os

from tqdm import tqdm

from torch.nn import functional as F

from basicsr.archs import build_network
from basicsr.losses import build_loss
from basicsr.losses.losses import g_path_regularize, r1_penalty
from basicsr.utils import imwrite, tensor2img
from basicsr.utils.registry import MODEL_REGISTRY
from .base_model import BaseModel


@MODEL_REGISTRY.register()
class GCFSR_Model(BaseModel):
    """StyleGAN2 model."""

    def __init__(self, opt):
        super(GCFSR_Model, self).__init__(opt)

        # define network net_g
        self.net_g = build_network(opt['network_g'])
        self.net_g = self.model_to_device(self.net_g)
        self.print_network(self.net_g)
        # load pretrained model
        load_path = self.opt['path'].get('pretrain_network_g', None)
        if load_path is not None:
            param_key = self.opt['path'].get('param_key_g', 'params')
            self.load_network(self.net_g, load_path, self.opt['path'].get('strict_load_g', True), param_key)

        # latent dimension: self.num_style_feat
        self.num_style_feat = opt['network_g']['num_style_feat']

        if self.is_train:
            self.init_training_settings()

    def init_training_settings(self):
        train_opt = self.opt['train']

        # define network net_d
        self.net_d = build_network(self.opt['network_d'])
        self.net_d = self.model_to_device(self.net_d)
        self.print_network(self.net_d)

        # load pretrained model
        load_path = self.opt['path'].get('pretrain_network_d', None)
        if load_path is not None:
            self.load_network(self.net_d, load_path, self.opt['path'].get('strict_load_d', True))

        # define network net_g with Exponential Moving Average (EMA)
        # net_g_ema only used for testing on one GPU and saving, do not need to
        # wrap with DistributedDataParallel
        self.net_g_ema = build_network(self.opt['network_g']).to(self.device)
        # load pretrained model
        load_path = self.opt['path'].get('pretrain_network_g', None)
        if load_path is not None:
            self.load_network(self.net_g_ema, load_path, self.opt['path'].get('strict_load_g', True), 'params_ema')
        else:
            self.model_ema(0)  # copy net_g weight

        self.net_g.train()
        self.net_d.train()
        self.net_g_ema.eval()

        # define losses
        # gan loss (wgan)
        self.cri_gan = build_loss(train_opt['gan_opt']).to(self.device)

        if train_opt.get('pixel_opt'):
            self.cri_pix = build_loss(train_opt['pixel_opt']).to(self.device)
        else:
            self.cri_pix = None

        if train_opt.get('perceptual_opt'):
            self.cri_perceptual = build_loss(train_opt['perceptual_opt']).to(self.device)
        else:
            self.cri_perceptual = None

        # regularization weights 
        self.r1_reg_weight = train_opt['r1_reg_weight']  # for discriminator
        self.path_reg_weight = train_opt['path_reg_weight']  # for generator

        self.net_g_reg_every = train_opt['net_g_reg_every']
        self.net_d_reg_every = train_opt['net_d_reg_every']
        self.mixing_prob = train_opt['mixing_prob']

        self.net_g_reg_l1_every = train_opt['net_g_reg_l1_every'] if 'net_g_reg_l1_every' in train_opt else None

        self.mean_path_length = 0

        self.pixel_LR = train_opt.get('pixel_LR')

        # set up optimizers and schedulers
        self.setup_optimizers()
        self.setup_schedulers()

    def setup_optimizers(self):
        train_opt = self.opt['train']
        # optimizer g
        net_g_reg_ratio = self.net_g_reg_every / (self.net_g_reg_every + 1)
        if self.opt['network_g']['type'] == 'StyleGAN2GeneratorC':
            normal_params = []
            style_mlp_params = []
            modulation_conv_params = []
            for name, param in self.net_g.named_parameters():
                if 'modulation' in name:
                    normal_params.append(param)
                elif 'style_mlp' in name:
                    style_mlp_params.append(param)
                elif 'modulated_conv' in name:
                    modulation_conv_params.append(param)
                else:
                    normal_params.append(param)
            optim_params_g = [
                {  # add normal params first
                    'params': normal_params,
                    'lr': train_opt['optim_g']['lr']
                },
                {
                    'params': style_mlp_params,
                    'lr': train_opt['optim_g']['lr'] * 0.01
                },
                {
                    'params': modulation_conv_params,
                    'lr': train_opt['optim_g']['lr'] / 3
                }
            ]
        else:
            normal_params = []
            for name, param in self.net_g.named_parameters():
                normal_params.append(param)
            optim_params_g = [{  # add normal params first
                'params': normal_params,
                'lr': train_opt['optim_g']['lr']
            }]

        optim_type = train_opt['optim_g'].pop('type')
        lr = train_opt['optim_g']['lr'] * net_g_reg_ratio
        betas = (0**net_g_reg_ratio, 0.99**net_g_reg_ratio)
        self.optimizer_g = self.get_optimizer(optim_type, optim_params_g, lr, betas=betas)
        self.optimizers.append(self.optimizer_g)

        # optimizer d
        net_d_reg_ratio = self.net_d_reg_every / (self.net_d_reg_every + 1)
        if self.opt['network_d']['type'] == 'StyleGAN2DiscriminatorC':
            normal_params = []
            linear_params = []
            for name, param in self.net_d.named_parameters():
                if 'final_linear' in name:
                    linear_params.append(param)
                else:
                    normal_params.append(param)
            optim_params_d = [
                {  # add normal params first
                    'params': normal_params,
                    'lr': train_opt['optim_d']['lr']
                },
                {
                    'params': linear_params,
                    'lr': train_opt['optim_d']['lr'] * (1 / math.sqrt(512))
                }
            ]
        else:
            normal_params = []
            for name, param in self.net_d.named_parameters():
                normal_params.append(param)
            optim_params_d = [{  # add normal params first
                'params': normal_params,
                'lr': train_opt['optim_d']['lr']
            }]

        optim_type = train_opt['optim_d'].pop('type')
        lr = train_opt['optim_d']['lr'] * net_d_reg_ratio
        betas = (0**net_d_reg_ratio, 0.99**net_d_reg_ratio)
        self.optimizer_d = self.get_optimizer(optim_type, optim_params_d, lr, betas=betas)
        self.optimizers.append(self.optimizer_d)

    def feed_data(self, data):
        if 'gt' in data:
            self.real_img = data['gt'].to(self.device)
        if 'in_size' in data:
            self.in_size = data['in_size'].to(self.device)
        self.lq = data['lq'].to(self.device)

    def make_noise(self, batch, num_noise):
        if num_noise == 1:
            noises = torch.randn(batch, self.num_style_feat, device=self.device)
        else:
            noises = torch.randn(num_noise, batch, self.num_style_feat, device=self.device).unbind(0)
        return noises

    def mixing_noise(self, batch, prob):
        if random.random() < prob:
            return self.make_noise(batch, 2)
        else:
            return [self.make_noise(batch, 1)]

    def optimize_parameters(self, current_iter):
        loss_dict = OrderedDict()

        # optimize net_d
        for p in self.net_d.parameters():
            p.requires_grad = True
        self.optimizer_d.zero_grad()

        fake_img, _ = self.net_g(self.lq, self.in_size)
        fake_pred = self.net_d(fake_img.detach())

        real_pred = self.net_d(self.real_img)
        # wgan loss with softplus (logistic loss) for discriminator
        l_d = self.cri_gan(real_pred, True, is_disc=True) + self.cri_gan(fake_pred, False, is_disc=True)
        loss_dict['l_d'] = l_d
        # In wgan, real_score should be positive and fake_score should be
        # negative
        loss_dict['real_score'] = real_pred.detach().mean()
        loss_dict['fake_score'] = fake_pred.detach().mean()
        l_d.backward()

        if current_iter % self.net_d_reg_every == 0:
            self.real_img.requires_grad = True
            real_pred = self.net_d(self.real_img)
            l_d_r1 = r1_penalty(real_pred, self.real_img)
            l_d_r1 = (self.r1_reg_weight / 2 * l_d_r1 * self.net_d_reg_every + 0 * real_pred[0])
            # TODO: why do we need to add 0 * real_pred, otherwise, a runtime
            # error will arise: RuntimeError: Expected to have finished
            # reduction in the prior iteration before starting a new one.
            # This error indicates that your module has parameters that were
            # not used in producing loss.
            loss_dict['l_d_r1'] = l_d_r1.detach().mean()
            l_d_r1.backward()

        self.optimizer_d.step()

        # optimize net_g
        for p in self.net_d.parameters():
            p.requires_grad = False
        self.optimizer_g.zero_grad()

        fake_img, _ = self.net_g(self.lq, self.in_size)
        fake_pred = self.net_d(fake_img)
        
        # wgan loss with softplus (non-saturating loss) for generator
        l_g = self.cri_gan(fake_pred, True, is_disc=False)
        loss_dict['l_g_wgan'] = l_g

        if self.cri_pix: 
            l_g_pix = self.cri_pix(fake_img, self.real_img)
            l_g += l_g_pix
            loss_dict['l_g_pix'] = l_g_pix

        if self.cri_perceptual:
            l_g_percep, l_g_style = self.cri_perceptual(fake_img, self.real_img)
            if l_g_percep is not None:
                l_g += l_g_percep
                loss_dict['l_g_percep'] = l_g_percep
            if l_g_style is not None:
                l_g += l_g_style
                loss_dict['l_g_style'] = l_g_style

        l_g.backward()

        self.optimizer_g.step()

        self.log_dict = self.reduce_loss_dict(loss_dict)

        # EMA
        self.model_ema(decay=0.5**(32 / (10 * 1000)))

    def test(self):
        with torch.no_grad():
            self.net_g_ema.eval()
            self.output, _ = self.net_g_ema(self.lq, self.in_size)
            # self.output, _ = self.net_g_ema(self.lq)

    def dist_validation(self, dataloader, current_iter, tb_logger, save_img):
        if self.opt['rank'] == 0:
            self.nondist_validation(dataloader, current_iter, tb_logger, save_img)

    def nondist_validation(self, dataloader, current_iter, tb_logger, save_img):
        dataset_name = dataloader.dataset.opt['name']
        pbar = tqdm(total=len(dataloader), unit='image')

        for idx, val_data in enumerate(dataloader):
            img_name = osp.splitext(osp.basename(val_data['lq_path'][0]))[0]
            save_dir = osp.join(self.opt['path']['visualization'], dataset_name, img_name)
            os.makedirs(save_dir, exist_ok=True)
            self.feed_data(val_data)

            self.test()

            visuals = self.get_current_visuals()
            sr_img = tensor2img([visuals['sr']], min_max=(-1, 1))

            # tentative for out of GPU memory
            del self.output
            torch.cuda.empty_cache()

            if save_img:
                if self.opt['is_train']:
                    save_img_path = osp.join(save_dir, f'{img_name}_{current_iter}.png')
                else:
                    if self.opt['val']['suffix']:
                        save_img_path = osp.join(self.opt['path']['visualization'], dataset_name,
                                                f'{img_name}_{self.opt["val"]["suffix"]}.png')
                    else:
                        save_img_path = osp.join(self.opt['path']['visualization'], dataset_name,
                                                f'{img_name}_{self.opt["name"]}.png')
                imwrite(sr_img, save_img_path)

            # tentative for out of GPU memory
            del self.lq
            torch.cuda.empty_cache()

            pbar.update(1)
            pbar.set_description(f'Test {img_name}')
        pbar.close()


    def get_current_visuals(self):
        out_dict = OrderedDict()
        if hasattr(self, 'gt'):
            out_dict['gt'] = self.gt.detach().cpu()
        out_dict['sr'] = self.output.detach().cpu()
        return out_dict

    def save(self, epoch, current_iter):
        self.save_network([self.net_g, self.net_g_ema], 'net_g', current_iter, param_key=['params', 'params_ema'])
        self.save_network(self.net_d, 'net_d', current_iter)
        self.save_training_state(epoch, current_iter)
