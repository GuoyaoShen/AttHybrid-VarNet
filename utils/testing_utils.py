import numpy as np
import scipy.stats

from utils.evaluation_utils import *



def voltest_hybridvarnet(
        test_dataloader,
        net,
        device,
):
    '''
    Volume test the reconstruction performance. HybridVarNet.
    '''
    net = net.to(device)
    net.eval()

    nmse = 0.0
    psnr = 0.0
    ssim = 0.0
    with torch.no_grad():
        for idx, data in enumerate(test_dataloader):
            Xk, yk, mask, y = data

            if Xk.shape[0] != 1:
                raise ValueError('Volume test dataloader batch size must be 1.')

            Xk = Xk.squeeze(0).to(device).float()  # [D,Nc,H,W,2]
            mask = mask.squeeze(0).to(device).float()
            y = y.squeeze(0).to(device).float()  # [D,H,W]

            # network forward
            y_pred = torch.zeros_like(y)
            for idxs in range(y.shape[0]):
                y_predi, y_pred_midi = net(Xk[idxs].unsqueeze(0), mask[idxs].unsqueeze(0))
                y_pred[idxs] = y_predi

            # evaluation metrics
            tg = y.detach()
            pred = y_pred.detach()

            # print('tg.shape:', tg.shape)
            i_nmse = volume_nmse_tensor(tg, pred)
            i_psnr = volume_psnr_tensor(tg, pred)
            i_ssim = volume_ssim_tensor(tg, pred)

            nmse += i_nmse
            psnr += i_psnr
            ssim += i_ssim

    nmse /= len(test_dataloader)
    psnr /= len(test_dataloader)
    ssim /= len(test_dataloader)
    print('### NMSE: ' + str(nmse) + '|| PSNR: ' + str(psnr) + '|| SSIM: ' + str(ssim))
    print('----------------------------------------------------------------------')

    return nmse, psnr, ssim


def voltest_varnet(
        test_dataloader,
        net,
        device,
):
    '''
    Volume test the reconstruction performance. VarNet.
    '''
    net = net.to(device)
    net.eval()

    nmse = 0.0
    psnr = 0.0
    ssim = 0.0
    with torch.no_grad():
        for idx, data in enumerate(test_dataloader):
            Xk, yk, mask, y = data

            if Xk.shape[0] != 1:
                raise ValueError('Volume test dataloader batch size must be 1.')

            Xk = Xk.squeeze(0).to(device).float()  # [D,Nc,H,W,2]
            mask = mask.squeeze(0).to(device).float()
            y = y.squeeze(0).to(device).float()  # [D,H,W]

            # network forward
            y_pred = torch.zeros_like(y)
            for idxs in range(y.shape[0]):
                y_predi = net(Xk[idxs].unsqueeze(0), mask[idxs].unsqueeze(0))
                y_pred[idxs] = y_predi

            # evaluation metrics
            tg = y.detach()
            pred = y_pred.detach()

            # print('tg.shape:', tg.shape)
            i_nmse = volume_nmse_tensor(tg, pred)
            i_psnr = volume_psnr_tensor(tg, pred)
            i_ssim = volume_ssim_tensor(tg, pred)

            nmse += i_nmse
            psnr += i_psnr
            ssim += i_ssim

    nmse /= len(test_dataloader)
    psnr /= len(test_dataloader)
    ssim /= len(test_dataloader)
    print('### NMSE: ' + str(nmse) + '|| PSNR: ' + str(psnr) + '|| SSIM: ' + str(ssim))
    print('----------------------------------------------------------------------')

    return nmse, psnr, ssim


def voltest_wnet(
        test_dataloader,
        net,
        device,
):
    '''
    Volume test the reconstruction performance. WNet.
    '''
    net = net.to(device)
    net.eval()

    nmse = 0.0
    psnr = 0.0
    ssim = 0.0
    with torch.no_grad():
        for idx, data in enumerate(test_dataloader):
            Xk, yk, mask, y = data

            if Xk.shape[0] != 1:
                raise ValueError('Volume test dataloader batch size must be 1.')

            Xk = Xk.squeeze(0).to(device).float()  # [D,Nc,H,W,2]
            yk = yk.squeeze(0).to(device).float()
            mask = mask.squeeze(0).to(device).float()
            y = y.squeeze(0).to(device).float()  # [D,H,W]

            # network forward
            y_pred = torch.zeros_like(y)
            for idxs in range(y.shape[0]):
                y_predi, k_pred_midi = net(Xk[idxs].unsqueeze(0), mask[idxs].unsqueeze(0))
                y_pred[idxs] = y_predi

            # evaluation metrics
            tg = y.detach()
            pred = y_pred.detach()

            # print('tg.shape:', tg.shape)
            i_nmse = volume_nmse_tensor(tg, pred)
            i_psnr = volume_psnr_tensor(tg, pred)
            i_ssim = volume_ssim_tensor(tg, pred)

            nmse += i_nmse
            psnr += i_psnr
            ssim += i_ssim

    nmse /= len(test_dataloader)
    psnr /= len(test_dataloader)
    ssim /= len(test_dataloader)
    print('### NMSE: ' + str(nmse) + '|| PSNR: ' + str(psnr) + '|| SSIM: ' + str(ssim))
    print('----------------------------------------------------------------------')

    return nmse, psnr, ssim


def voltest_unet(
        test_dataloader,
        net,
        device,
):
    '''
    Volume test the reconstruction performance. UNet.
    '''
    net = net.to(device)
    net.eval()

    nmse = 0.0
    psnr = 0.0
    ssim = 0.0
    with torch.no_grad():
        for idx, data in enumerate(test_dataloader):
            X, y, mask = data

            if X.shape[0] != 1:
                raise ValueError('Volume test dataloader batch size must be 1.')

            X = X.squeeze(0).to(device).float()  # [D,Nc,H,W]
            y = y.squeeze(0).to(device).float()

            # network forward
            y_pred = torch.zeros_like(y)
            for idxs in range(y.shape[0]):
                y_predi = net(X[idxs].unsqueeze(0))
                y_pred[idxs] = y_predi

            # evaluation metrics
            tg = y.detach()
            pred = y_pred.detach()

            # print('tg.shape:', tg.shape)
            i_nmse = volume_nmse_tensor(tg.squeeze(1), pred.squeeze(1))
            i_psnr = volume_psnr_tensor(tg.squeeze(1), pred.squeeze(1))
            i_ssim = volume_ssim_tensor(tg.squeeze(1), pred.squeeze(1))

            nmse += i_nmse
            psnr += i_psnr
            ssim += i_ssim

    nmse /= len(test_dataloader)
    psnr /= len(test_dataloader)
    ssim /= len(test_dataloader)
    print('### NMSE: ' + str(nmse) + '|| PSNR: ' + str(psnr) + '|| SSIM: ' + str(ssim))
    print('----------------------------------------------------------------------')

    return nmse, psnr, ssim


# testing utils with confidence intervals

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
    nmse_tensor = torch.zeros(len(test_dataloader))
    psnr_tensor = torch.zeros(len(test_dataloader))
    ssim_tensor = torch.zeros(len(test_dataloader))
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

            nmse_tensor[idx] = i_nmse
            psnr_tensor[idx] = i_psnr
            ssim_tensor[idx] = i_ssim

    nmse /= len(test_dataloader)
    psnr /= len(test_dataloader)
    ssim /= len(test_dataloader)
    running_loss /= len(test_dataloader)
    print('### TEST LOSS: ', str(running_loss) + '|| NMSE: ' + str(nmse) + '|| PSNR: ' + str(psnr) + '|| SSIM: ' + str(ssim))
    print('----------------------------------------------------------------------')

    return running_loss, nmse_tensor, psnr_tensor, ssim_tensor


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
    nmse_tensor = torch.zeros(len(test_dataloader))
    psnr_tensor = torch.zeros(len(test_dataloader))
    ssim_tensor = torch.zeros(len(test_dataloader))
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

            nmse_tensor[idx] = i_nmse
            psnr_tensor[idx] = i_psnr
            ssim_tensor[idx] = i_ssim

    nmse /= len(test_dataloader)
    psnr /= len(test_dataloader)
    ssim /= len(test_dataloader)
    running_loss /= len(test_dataloader)
    print('### TEST LOSS: ', str(running_loss) + '|| NMSE: ' + str(nmse) + '|| PSNR: ' + str(psnr) + '|| SSIM: ' + str(ssim))
    print('----------------------------------------------------------------------')

    return running_loss, nmse_tensor, psnr_tensor, ssim_tensor


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
    nmse_tensor = torch.zeros(len(test_dataloader))
    psnr_tensor = torch.zeros(len(test_dataloader))
    ssim_tensor = torch.zeros(len(test_dataloader))
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

            nmse_tensor[idx] = i_nmse
            psnr_tensor[idx] = i_psnr
            ssim_tensor[idx] = i_ssim

    nmse /= len(test_dataloader)
    psnr /= len(test_dataloader)
    ssim /= len(test_dataloader)
    running_loss /= len(test_dataloader)
    print('### TEST LOSS: ', str(running_loss) + '|| NMSE: ' + str(nmse) + '|| PSNR: ' + str(psnr) + '|| SSIM: ' + str(ssim))
    print('----------------------------------------------------------------------')

    return running_loss, nmse_tensor, psnr_tensor, ssim_tensor


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
    nmse_tensor = torch.zeros(len(test_dataloader))
    psnr_tensor = torch.zeros(len(test_dataloader))
    ssim_tensor = torch.zeros(len(test_dataloader))
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

            nmse_tensor[idx] = i_nmse
            psnr_tensor[idx] = i_psnr
            ssim_tensor[idx] = i_ssim

    nmse /= len(test_dataloader)
    psnr /= len(test_dataloader)
    ssim /= len(test_dataloader)
    running_loss /= len(test_dataloader)
    print('### TEST LOSS: ', str(running_loss) + '|| NMSE: ' + str(nmse) + '|| PSNR: ' + str(psnr) + '|| SSIM: ' + str(ssim))
    print('----------------------------------------------------------------------')

    return running_loss, nmse_tensor, psnr_tensor, ssim_tensor


def mean_confidence_interval(data, confidence=0.95):
    '''
    data: np array.
    '''
    a = 1.0 * data
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * scipy.stats.t.ppf((1 + confidence) / 2., n-1)
    return m, h