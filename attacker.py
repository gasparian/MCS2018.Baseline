'''
FGSM attack on student model
'''
import os
import time
import argparse

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.autograd import Variable
import torch.nn.functional as F
import torch.utils.data as data
import torchvision.transforms as transforms
from tqdm import tqdm
from PIL import Image
from skimage.measure import compare_ssim

import matplotlib.pyplot as plt
from student_net_learning.models import *

def reverse_normalize(tensor, mean, std):
    '''reverese normalize to convert tensor -> PIL Image'''
    tensor_copy = tensor.clone()
    for t, m, s in zip(tensor_copy, mean, std):
        t.div_(s).sub_(m)
    return tensor_copy

def get_model(model_name, checkpoint_path):
    '''
    Model architecture choosing
    '''
    if model_name == 'ResNet18':
        net = ResNet18()
    elif model_name == 'ResNet34':
        net = ResNet34()
    elif model_name == 'ResNet50':
        net = ResNet50()
    elif model_name == 'DenseNet':
        net = DenseNet121()
    elif model_name == 'VGG11':
        net = VGG('VGG11')
    checkpoint = torch.load(checkpoint_path)
    net.load_state_dict(checkpoint['net'])
    return net

class FGSM_Attacker():
    '''
    FGSM attacker: https://arxiv.org/pdf/1412.6572.pdf
    model -- white-box model for attack
    eps -- const * Clipped Noise
    ssim_thr -- min value for ssim compare
    transform -- img to tensor transform without CenterCrop and Scale
    '''
    def __init__(self, model, eps, ssim_thr, transform, img2tensor, 
                 reverse_std, reverse_mean, args, max_iter=50):
        self.model = model
        self.model.eval()
        self.eps = eps
        self.ssim_thr = ssim_thr
        self.max_iter = max_iter
        self.reverse_std = reverse_std
        self.reverse_mean = reverse_mean
        self.transform = transform
        self.cropping = transforms.Compose([
                                      transforms.CenterCrop(224),
                                      transforms.Scale(112)
                                      ])
        self.img2tensor = img2tensor
        self.args = args
        self.loss = nn.MSELoss()

    def tensor2img(self, tensor, on_cuda=False):
        tensor = reverse_normalize(tensor, self.reverse_mean, self.reverse_std)
        # clipping
        tensor[tensor > 1] = 1
        tensor[tensor < 0] = 0
        tensor = tensor.squeeze(0)
        if on_cuda:
            tensor = tensor.cpu()
        return transforms.ToPILImage()(tensor)

    def attack(self, attack_pairs):
        '''
        Args:
            attack_pairs (dict) - id pair, 'source': 5 imgs,
                                           'target': 5 imgs
        '''
        target_img_names = attack_pairs['target']
        target_descriptors = np.ones((len(attack_pairs['target']), 512), 
                                     dtype=np.float32)

        for idx, img_name in enumerate(target_img_names):
            img_name = os.path.join(self.args["root"], img_name)
            img = Image.open(img_name)
            tensor = self.transform(img).unsqueeze(0)
            if self.args["cuda"]:
                tensor = tensor.cuda(async=True)

            res = self.model(Variable(tensor, requires_grad=False))\
                      .data.cpu().numpy().squeeze()
            target_descriptors[idx] = res
        target_img = img

        for img_name in attack_pairs['source']:
            print ('TEST: attack on image {0}'.format(img_name))

            #img is attacked
            if os.path.isfile(os.path.join(self.args["save_root"], img_name)):
                continue

            img = Image.open(os.path.join(self.args["root"], img_name))
            original_img = self.cropping(img)
            attacked_img = original_img
            tensor = self.transform(img)
            input_var = Variable(tensor.unsqueeze(0), requires_grad=True)
    
            print ('TEST: start iterations')
            tick = time.time()
            
            for iter_number in tqdm(range(self.max_iter)):
                adv_noise = torch.zeros((3,112,112))
                
                if self.args["cuda"]:
                    adv_noise = adv_noise.cuda(async=True)

                for target_descriptor in target_descriptors:
                    target_out = Variable(torch.from_numpy(target_descriptor)\
                                          .unsqueeze(0), requires_grad=False)

                    input_var.grad = None
                    out = self.model(input_var)
                    calc_loss = self.loss(out, target_out)
                    calc_loss.backward()
                    noise = self.eps * torch.sign(input_var.grad.data).squeeze()
                    adv_noise = adv_noise + noise

                input_var.data = input_var.data - adv_noise
                changed_img = self.tensor2img(input_var.data.cpu().squeeze())

                #SSIM checking
                ssim = compare_ssim(np.array(original_img, dtype=np.float32), 
                                    np.array(changed_img, dtype=np.float32), 
                                    multichannel=True)
                if ssim < self.ssim_thr:
                    break
                else:
                    attacked_img = changed_img
            tock = time.time()
            print ('TEST: end iterations. SSIM: {0}; Time: {1:.2f}sec\n'.format(ssim, tock - tick))

            if not os.path.isdir(self.args["save_root"]):
                os.makedirs(self.args["save_root"])
            attacked_img.save(os.path.join(self.args["save_root"], img_name.replace('.jpg', '.png')))
            
            if self.args["imshow"]:
                # display images difference
                plt.figure(figsize=(20, 5))
                
                plt.subplot(1, 4, 1)
                plt.axis('off')
                plt.imshow(original_img)
                
                plt.subplot(1, 4, 2)
                plt.axis('off')
                plt.imshow(np.abs(np.array(original_img) - np.array(attacked_img)))
            
                plt.subplot(1, 4, 3)
                plt.axis('off')
                plt.imshow(attacked_img)
                
                plt.subplot(1, 4, 4)
                plt.axis('off')
                plt.imshow(np.array(target_img))
                
                plt.show()

if __name__ == '__main__':
    SSIM_THR = 0.95

    MEAN = [0.485, 0.456, 0.406]
    STD = [0.229, 0.224, 0.225] 

    REVERSE_MEAN = [-0.485, -0.456, -0.406]
    REVERSE_STD = [1/0.229, 1/0.224, 1/0.225]

    parser = argparse.ArgumentParser(description='PyTorch student network training')

    parser.add_argument('--root', 
                        required=True,
                        type=str, 
                        help='data root path')
    parser.add_argument('--save_root',
                        required=True,
                        type=str,
                        help='path to store results',
                        default='./changed_imgs')
    parser.add_argument('--datalist', 
                        required=True,
                        type=str, 
                        help='datalist path')
    parser.add_argument('--model_name',
                        type=str, 
                        help='model name', 
                        default='ResNet18')
    parser.add_argument('--checkpoint_path',
                        required=True,
                        type=str,
                        help='path to learned student model checkpoints')
    parser.add_argument('--cuda',
                        action='store_true', 
                        help='use CUDA')

    args = parser.parse_args()

    main()
