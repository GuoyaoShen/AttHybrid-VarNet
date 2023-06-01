import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F

import fastmri
from fastmri.data import transforms

from utils.evaluation_utils import *



def recon_hybridvarnet(
        dataloader,
        net,
        device,
        idx_case,
        show_info=True,
):
    '''
    Reconstruct image from the dataloader
    '''
    net = net.to(device)
    net.eval()

    with torch.no_grad():
        for idx, data in enumerate(dataloader):
            if idx != idx_case:
                continue

            Xk, yk, mask, y = data

            Xk = Xk.to(device).float()
            mask = mask.to(device).float()
            y = y.to(device).float()

            # network forward
            y_pred, y_pred_mid = net(Xk, mask)

            # evaluation metrics
            tg = y.detach()  # [B,H,W]
            pred = y_pred.detach()
            if show_info:
                print('tg.shape:', tg.shape)

            i_nmse = calc_nmse_tensor(tg, pred)
            i_psnr = calc_psnr_tensor(tg, pred)
            i_ssim = calc_ssim_tensor(tg, pred)
            if show_info:
                print('NMSE: ' + str(i_nmse) + '|| PSNR: ' + str(i_psnr) + '|| SSIM: ' + str(i_ssim))

            # ZF-MRI
            zf = fastmri.ifft2c(Xk)  # [B,Nc,H,W,2]
            zf = fastmri.complex_abs(zf)  # [B,Nc,H,W]
            zf = fastmri.rss(zf, dim=1)  # [B,H,W]
            break
    return pred.cpu().numpy(), tg.cpu().numpy(), zf.cpu().numpy()


def recon_varnet(
        dataloader,
        net,
        device,
        idx_case,
        show_info=True,
):
    '''
    Reconstruct image from the dataloader
    '''
    net = net.to(device)
    net.eval()

    with torch.no_grad():
        for idx, data in enumerate(dataloader):
            if idx != idx_case:
                continue

            Xk, yk, mask, y = data

            Xk = Xk.to(device).float()
            mask = mask.to(device).float()
            y = y.to(device).float()

            # network forward
            y_pred = net(Xk, mask)

            # evaluation metrics
            tg = y.detach()  # [B,H,W]
            pred = y_pred.detach()
            if show_info:
                print('tg.shape:', tg.shape)

            i_nmse = calc_nmse_tensor(tg, pred)
            i_psnr = calc_psnr_tensor(tg, pred)
            i_ssim = calc_ssim_tensor(tg, pred)
            if show_info:
                print('NMSE: ' + str(i_nmse) + '|| PSNR: ' + str(i_psnr) + '|| SSIM: ' + str(i_ssim))

            # ZF-MRI
            zf = fastmri.ifft2c(Xk)  # [B,Nc,H,W,2]
            zf = fastmri.complex_abs(zf)  # [B,Nc,H,W]
            zf = fastmri.rss(zf, dim=1)  # [B,H,W]
            break
    return pred.cpu().numpy(), tg.cpu().numpy(), zf.cpu().numpy()


def recon_wnet(
        dataloader,
        net,
        device,
        idx_case,
        show_info=True,
):
    '''
    Reconstruct image from the dataloader
    '''
    net = net.to(device)
    net.eval()

    with torch.no_grad():
        for idx, data in enumerate(dataloader):
            if idx != idx_case:
                continue

            Xk, yk, mask, y = data

            Xk = Xk.to(device).float()
            mask = mask.to(device).float()
            y = y.to(device).float()

            # network forward
            y_pred, k_pred_mid = net(Xk, mask)

            # evaluation metrics
            tg = y.detach()  # [B,H,W]
            pred = y_pred.detach()
            if show_info:
                print('tg.shape:', tg.shape)

            i_nmse = calc_nmse_tensor(tg, pred)
            i_psnr = calc_psnr_tensor(tg, pred)
            i_ssim = calc_ssim_tensor(tg, pred)
            if show_info:
                print('NMSE: ' + str(i_nmse) + '|| PSNR: ' + str(i_psnr) + '|| SSIM: ' + str(i_ssim))

            # ZF-MRI
            zf = fastmri.ifft2c(Xk)  # [B,Nc,H,W,2]
            zf = fastmri.complex_abs(zf)  # [B,Nc,H,W]
            zf = fastmri.rss(zf, dim=1)  # [B,H,W]
            break
    return pred.cpu().numpy(), tg.cpu().numpy(), zf.cpu().numpy()


def recon_unet(
        dataloader,
        net,
        device,
        idx_case,
        show_info=True,
):
    '''
    Reconstruct image from the dataloader
    '''
    net = net.to(device)
    net.eval()

    with torch.no_grad():
        for idx, data in enumerate(dataloader):
            if idx != idx_case:
                continue

            X, y, mask = data

            X = X.to(device).float()
            y = y.to(device).float()

            # network forward
            y_pred = net(X)

            # evaluation metrics
            tg = y.detach().squeeze(1)  # [B,H,W]
            pred = y_pred.detach().squeeze(1)
            if show_info:
                print('tg.shape:', tg.shape)

            i_nmse = calc_nmse_tensor(tg, pred)
            i_psnr = calc_psnr_tensor(tg, pred)
            i_ssim = calc_ssim_tensor(tg, pred)
            if show_info:
                print('NMSE: ' + str(i_nmse) + '|| PSNR: ' + str(i_psnr) + '|| SSIM: ' + str(i_ssim))

            # ZF-MRI
            zf = X.detach().squeeze(1)
            break
    return pred.cpu().numpy(), tg.cpu().numpy(), zf.cpu().numpy()