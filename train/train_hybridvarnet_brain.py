import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from fastmri.data import subsample, mri_data

from utils.training_utils import *
from net.att_hybrid_varnet.hybrid_varnet import HybridVarNet
from net.losses import *
from utils.data_transform import DataTransform_VarNet
from utils.misc import calc_model_size

print(torch.__version__)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print('device:', device)



# ****** TRAINING SETTINGS ******
# dataset settings
acc = 4  # acceleration factor
frac_c = 0.08  # center fraction
path_dir_train = 'C:/TortoiseGitRepos/data/fastmri/brain_multicoil_train/'
path_dir_test = 'C:/TortoiseGitRepos/data/fastmri/brain_multicoil_test/'
bhsz = 1
# training settings
NUM_EPOCH = 50
learning_rate = 1e-3
# save settings
PATH_MODEL = './saved_models/fastmri/hybridvarnet_brain_'+str(acc)+'x_E'+str(NUM_EPOCH)+'.pt'
PATH_CKPOINT = './saved_models/fastmri/hybridvarnet_brain_'+str(acc)+'x_E'+str(NUM_EPOCH)+'_ck.pt'


# ====== Construct dataset ======
# initialize mask
mask_func = subsample.EquispacedMaskFractionFunc(
    center_fractions=[frac_c],
    accelerations=[acc]
)

# initialize dataset
data_transform = DataTransform_VarNet(
    mask_func,
    img_size=320,
    flag_singlecoil=False,
)

# training set
dataset_train = mri_data.SliceDataset(
    root=pathlib.Path(path_dir_train),
    transform=data_transform,
    challenge='multicoil'
)

# test set
dataset_test = mri_data.SliceDataset(
    root=pathlib.Path(path_dir_test),
    transform=data_transform,
    challenge='multicoil'
)

dataloader_train = DataLoader(dataset_train, batch_size=bhsz, shuffle=True)
dataloader_test = DataLoader(dataset_test, batch_size=bhsz, shuffle=True)
print('len dataloader train:', len(dataloader_train))
print('len dataloader test:', len(dataloader_test))


# ====== Construct model ======
net = HybridVarNet(
    num_cascades=12,
    sens_chans=24,
    sens_pools=4,
    chans=36,
    pools=4,
    use_attention=True,
    use_res=True,
)
print('model size: %.3f MB' % (calc_model_size(net)))


# ====== Train ======
optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate, amsgrad=False)
criteon_mid = NRMSELoss(flag_l1=False)
criteon_img = NRMSELoss(flag_l1=False)
alpha = 1

net = train_hybridvarnet(
    dataloader_train,
    dataloader_test,
    optimizer,
    criteon_mid,
    criteon_img,
    alpha,
    net,
    device,
    PATH_MODEL=PATH_CKPOINT,
    NUM_EPOCH=NUM_EPOCH,
    show_step=5,
    show_test=True)


# ====== Save model ======
torch.save(net.state_dict(), PATH_MODEL)
print('MODEL SAVED')
