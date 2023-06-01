import numpy as np
import matplotlib.pyplot as plt
from tqdm.autonotebook import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F

import fastmri
from fastmri.data import transforms

from utils.evaluation_utils import *



def train_hybridvarnet(
        train_dataloader,
        test_dataloader,
        optimizer,
        loss_mid,
        loss_img,
        alpha,
        net,
        device,
        PATH_MODEL,
        NUM_EPOCH=5,
        show_step=-1,
        show_test=False):
    '''
    Train the HybridVarNet.
    '''

    net = net.to(device)
    net.train()

    pbar = tqdm(range(NUM_EPOCH), desc='LOSS')
    for i in pbar:
        running_loss = 0.0
        for idx, data in enumerate(train_dataloader):
            Xk, yk, mask, y = data

            Xk = Xk.to(device).float()
            mask = mask.to(device).float()
            y = y.to(device).float()

            # network forward
            y_pred, y_pred_mid = net(Xk, mask)
            if i == 0 and idx == 0:
                print('Xk.shape:', Xk.shape)
                print('mask.shape:', mask.shape)
                print('y.shape:', y.shape)
                print('y_pred.shape:', y_pred.shape)
            optimizer.zero_grad()
            loss_train = alpha * loss_mid(y_pred_mid, y.detach().clone()) + loss_img(y_pred, y)

            loss_train.backward()
            optimizer.step()
            running_loss += loss_train.item()

        running_loss /= len(train_dataloader)

        pbar.set_description("Loss=%f" % (running_loss))
        if show_step > 0:
            if (i + 1) % show_step == 0:
                print('*** EPOCH ' + str(i + 1) + ' || AVG LOSS: ' + str(running_loss))

    # test model
    if show_test:
        loss_test, nmse, psnr, ssim = test_hybridvarnet(
            test_dataloader,
            loss_mid,
            loss_img,
            alpha,
            net,
            device)

    # save model
    torch.save({
        'model_state_dict': net.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, PATH_MODEL)
    print('MODEL SAVED.')

    return net


def test_hybridvarnet(
        test_dataloader,
        loss_mid,
        loss_img,
        alpha,
        net,
        device):
    '''
    Test the reconstruction performance. HybridVarNet.
    '''
    net = net.to(device)
    net.eval()

    running_loss = 0.0
    nmse = 0.0
    psnr = 0.0
    ssim = 0.0
    with torch.no_grad():
        for idx, data in enumerate(test_dataloader):
            Xk, yk, mask, y = data

            Xk = Xk.to(device).float()
            mask = mask.to(device).float()
            y = y.to(device).float()

            # network forward
            y_pred, y_pred_mid = net(Xk, mask)
            loss_test = alpha * loss_mid(y_pred_mid, y.detach().clone()) + loss_img(y_pred, y)
            running_loss += loss_test.item()

            # evaluation metrics
            tg = y.detach()
            pred = y_pred.detach()

            # print('tg.shape:', tg.shape)
            i_nmse = calc_nmse_tensor(tg, pred)
            i_psnr = calc_psnr_tensor(tg, pred)
            i_ssim = calc_ssim_tensor(tg, pred)

            nmse += i_nmse
            psnr += i_psnr
            ssim += i_ssim

    nmse /= len(test_dataloader)
    psnr /= len(test_dataloader)
    ssim /= len(test_dataloader)
    running_loss /= len(test_dataloader)
    print('### TEST LOSS: ', str(running_loss) + '|| NMSE: ' + str(nmse) + '|| PSNR: ' + str(psnr) + '|| SSIM: ' + str(ssim))
    print('----------------------------------------------------------------------')

    return running_loss, nmse, psnr, ssim


def train_wnet(
        train_dataloader,
        test_dataloader,
        optimizer,
        loss_mid,
        loss_img,
        alpha,
        net,
        device,
        PATH_MODEL,
        NUM_EPOCH=5,
        show_step=-1,
        show_test=False):
    '''
    Train the WNet.
    '''

    net = net.to(device)
    net.train()

    pbar = tqdm(range(NUM_EPOCH), desc='LOSS')
    for i in pbar:
        running_loss = 0.0
        for idx, data in enumerate(train_dataloader):
            Xk, yk, mask, y = data

            Xk = Xk.to(device).float()
            yk = yk.to(device).float()
            mask = mask.to(device).float()
            y = y.to(device).float()

            # network forward
            y_pred, k_pred_mid = net(Xk, mask)
            if i == 0 and idx == 0:
                print('Xk.shape:', Xk.shape)
                print('mask.shape:', mask.shape)
                print('y.shape:', y.shape)
                print('y_pred.shape:', y_pred.shape)
            optimizer.zero_grad()
            loss_train = alpha * loss_mid(k_pred_mid, yk) + loss_img(y_pred, y)

            loss_train.backward()
            optimizer.step()
            running_loss += loss_train.item()

        running_loss /= len(train_dataloader)

        pbar.set_description("Loss=%f" % (running_loss))
        if show_step > 0:
            if (i + 1) % show_step == 0:
                print('*** EPOCH ' + str(i + 1) + ' || AVG LOSS: ' + str(running_loss))

    # test model
    if show_test:
        loss_test, nmse, psnr, ssim = test_wnet(
            test_dataloader,
            loss_mid,
            loss_img,
            alpha,
            net,
            device)

    # save model
    torch.save({
        'model_state_dict': net.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, PATH_MODEL)
    print('MODEL SAVED.')

    return net


def test_wnet(
        test_dataloader,
        loss_mid,
        loss_img,
        alpha,
        net,
        device):
    '''
    Test the reconstruction performance. WNet.
    '''
    net = net.to(device)
    net.eval()

    running_loss = 0.0
    nmse = 0.0
    psnr = 0.0
    ssim = 0.0
    with torch.no_grad():
        for idx, data in enumerate(test_dataloader):
            Xk, yk, mask, y = data

            Xk = Xk.to(device).float()
            yk = yk.to(device).float()
            mask = mask.to(device).float()
            y = y.to(device).float()

            # network forward
            y_pred, k_pred_mid = net(Xk, mask)
            loss_test = alpha * loss_mid(k_pred_mid, yk) + loss_img(y_pred, y)
            running_loss += loss_test.item()

            # evaluation metrics
            tg = y.detach()
            pred = y_pred.detach()

            # print('tg.shape:', tg.shape)
            i_nmse = calc_nmse_tensor(tg, pred)
            i_psnr = calc_psnr_tensor(tg, pred)
            i_ssim = calc_ssim_tensor(tg, pred)

            nmse += i_nmse
            psnr += i_psnr
            ssim += i_ssim

    nmse /= len(test_dataloader)
    psnr /= len(test_dataloader)
    ssim /= len(test_dataloader)
    running_loss /= len(test_dataloader)
    print('### TEST LOSS: ', str(running_loss) + '|| NMSE: ' + str(nmse) + '|| PSNR: ' + str(psnr) + '|| SSIM: ' + str(ssim))
    print('----------------------------------------------------------------------')

    return running_loss, nmse, psnr, ssim

def train_varnet(
        train_dataloader,
        test_dataloader,
        optimizer,
        loss,
        net,
        device,
        PATH_MODEL,
        NUM_EPOCH=5,
        show_step=-1,
        show_test=False):
    '''
    Train the VarNet.
    '''

    net = net.to(device)
    net.train()

    pbar = tqdm(range(NUM_EPOCH), desc='LOSS')
    for i in pbar:
        running_loss = 0.0
        for idx, data in enumerate(train_dataloader):
            Xk, yk, mask, y = data

            Xk = Xk.to(device).float()
            mask = mask.to(device).float()
            y = y.to(device).float()

            # network forward
            y_pred = net(Xk, mask)
            if i == 0 and idx == 0:
                print('Xk.shape:', Xk.shape)
                print('mask.shape:', mask.shape)
                print('y.shape:', y.shape)
                print('y_pred.shape:', y_pred.shape)
            optimizer.zero_grad()
            loss_train = loss(y_pred, y)

            loss_train.backward()
            optimizer.step()
            running_loss += loss_train.item()

        running_loss /= len(train_dataloader)

        pbar.set_description("Loss=%f" % (running_loss))
        if show_step > 0:
            if (i + 1) % show_step == 0:
                print('*** EPOCH ' + str(i + 1) + ' || AVG LOSS: ' + str(running_loss))

    # test model
    if show_test:
        loss_test, nmse, psnr, ssim = test_varnet(
            test_dataloader,
            loss,
            net,
            device)

    # save model
    torch.save({
        'model_state_dict': net.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, PATH_MODEL)
    print('MODEL SAVED.')

    return net


def test_varnet(
        test_dataloader,
        loss,
        net,
        device):
    '''
    Test the reconstruction performance. VarNet.
    '''
    net = net.to(device)
    net.eval()

    running_loss = 0.0
    nmse = 0.0
    psnr = 0.0
    ssim = 0.0
    with torch.no_grad():
        for idx, data in enumerate(test_dataloader):
            Xk, yk, mask, y = data

            Xk = Xk.to(device).float()
            mask = mask.to(device).float()
            y = y.to(device).float()

            # network forward
            y_pred = net(Xk, mask)
            running_loss += loss(y_pred, y).item()

            # evaluation metrics
            tg = y.detach()
            pred = y_pred.detach()

            # print('tg.shape:', tg.shape)
            i_nmse = calc_nmse_tensor(tg, pred)
            i_psnr = calc_psnr_tensor(tg, pred)
            i_ssim = calc_ssim_tensor(tg, pred)

            nmse += i_nmse
            psnr += i_psnr
            ssim += i_ssim

    nmse /= len(test_dataloader)
    psnr /= len(test_dataloader)
    ssim /= len(test_dataloader)
    running_loss /= len(test_dataloader)
    print('### TEST LOSS: ', str(running_loss) + '|| NMSE: ' + str(nmse) + '|| PSNR: ' + str(psnr) + '|| SSIM: ' + str(ssim))
    print('----------------------------------------------------------------------')

    return running_loss, nmse, psnr, ssim


def train_unet(
        train_dataloader,
        test_dataloader,
        optimizer,
        loss,
        net,
        device,
        PATH_MODEL,
        NUM_EPOCH=5,
        show_step=-1,
        show_test=False):
    '''
    Train the U-Net.
    :param train_dataloader: training dataloader.
    :param test_dataloader: test dataloader.
    :param optimizer: optimizer.
    :param loss: loss function object.
    :param net: network object.
    :param device: device, gpu or cpu.
    :param NUM_EPOCH: number of epoch, default=5.
    :param show_step: int, default=-1. Steps to show intermediate loss during training. -1 for not showing.
    :param show_test: flag. Whether to show test after training.
    '''

    net = net.to(device)
    net.train()

    pbar = tqdm(range(NUM_EPOCH), desc='LOSS')
    for i in pbar:
        running_loss = 0.0
        for idx, data in enumerate(train_dataloader):
            X, y, mask = data

            X = X.to(device).float()
            y = y.to(device).float()

            # network forward
            y_pred = net(X)
            if i==0 and idx==0:
                print('X.shape:', X.shape)
                print('mask.shape:', mask.shape)
                print('y_pred.shape:', y_pred.shape)

            optimizer.zero_grad()
            loss_train = loss(y_pred, y)

            loss_train.backward()
            optimizer.step()
            running_loss += loss_train.item()

        running_loss /= len(train_dataloader)

        pbar.set_description("Loss=%f" % (running_loss))
        if show_step > 0:
            if (i + 1) % show_step == 0:
                print('*** EPOCH ' + str(i + 1) + ' || AVG LOSS: ' + str(running_loss))

    # test model
    if show_test:
        loss_test, nmse, psnr, ssim = test_unet(
            test_dataloader,
            loss,
            net,
            device)

    # save model
    torch.save({
        'model_state_dict': net.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, PATH_MODEL)
    print('MODEL SAVED.')

    return net


def test_unet(
        test_dataloader,
        loss,
        net,
        device):
    '''
    Test the reconstruction performance. U-Net.
    '''
    net = net.to(device)
    net.eval()

    running_loss = 0.0
    nmse = 0.0
    psnr = 0.0
    ssim = 0.0
    with torch.no_grad():
        for idx, data in enumerate(test_dataloader):
            X, y, mask = data

            X = X.to(device).float()  #[B,1,H,W]
            y = y.to(device).float()

            # network forward
            y_pred = net(X)
            running_loss += loss(y_pred, y).item()

            # evaluation metrics
            tg = y.detach()  # [B,1,H,W]
            pred = y_pred.detach()

            if idx==0:
                print('tg.shape:', tg.shape)
                print('pred.shape:', pred.shape)
            i_nmse = calc_nmse_tensor(tg.squeeze(1), pred.squeeze(1))
            i_psnr = calc_psnr_tensor(tg.squeeze(1), pred.squeeze(1))
            i_ssim = calc_ssim_tensor(tg.squeeze(1), pred.squeeze(1))

            nmse += i_nmse
            psnr += i_psnr
            ssim += i_ssim

    nmse /= len(test_dataloader)
    psnr /= len(test_dataloader)
    ssim /= len(test_dataloader)
    running_loss /= len(test_dataloader)
    print('### TEST LOSS: ', str(running_loss) + '|| NMSE: ' + str(nmse) + '|| PSNR: ' + str(psnr) + '|| SSIM: ' + str(ssim))
    print('----------------------------------------------------------------------')

    return running_loss, nmse, psnr, ssim
