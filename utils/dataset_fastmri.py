import os
from pathlib import Path
from typing import (
    Any,
    Callable,
    Dict,
    List,
    NamedTuple,
    Optional,
    Sequence,
    Tuple,
    Union,
)

import h5py
import numpy as np
import pandas as pd
import torch


class VolumeDataset(torch.utils.data.Dataset):
    """
    A PyTorch Dataset that provides access to MR image volumes.
    """

    def __init__(
        self,
        root: Union[str, Path, os.PathLike],
        transform: Optional[Callable] = None,
    ):
        """
        Args:
            root: Path to the dataset.
            challenge: "singlecoil" or "multicoil" depending on which challenge
                to use.
            transform: Optional; A callable object that pre-processes the raw
                data into appropriate form. The transform function should take
                'kspace', 'target', 'attributes', 'filename', and 'slice' as
                inputs. 'target' may be null for test data.
        """
        self.root = root
        self.list_file = os.listdir(root)  # list of all files in root_dir
        self.transform = transform

    def __len__(self):
        return len(self.list_file)

    def __getitem__(self, i: int):
        fname = self.root + self.list_file[i]

        with h5py.File(fname, "r") as hf:
            num_slice = hf['kspace'][()].shape[0]  # [D,N_coil,H,W]

            for dataslice in range(num_slice):
                kspace = hf["kspace"][dataslice]
                if self.transform is None:
                    sample = (kspace, None, None, None, None, None)
                else:
                    sample = self.transform(kspace, None, None, None, None, None)

                if len(sample) == 3:  # unet, [image_masked, image_full, mask], img [Nc,H,W], mask [1,H,W]
                    flag_unet = True
                    image_masked_s, image_full_s, mask_s = sample
                    if dataslice == 0:
                        image_masked = image_masked_s[None, ...]
                        image_full = image_full_s[None, ...]
                        mask = mask_s[None, ...]
                    else:
                        image_masked = torch.cat((image_masked, image_masked_s[None, ...]), 0)
                        image_full = torch.cat((image_full, image_full_s[None, ...]), 0)
                        mask = torch.cat((mask, mask_s[None, ...]), 0)
                elif len(sample) == 4:  # wnet & varnet, [masked_kspace, kspace, mask, image_full], kspace [Nc,H,W,2], mask [Nc,H,W,2], image [H,W]
                    flag_unet = False
                    masked_kspace_s, kspace_s, mask_s, image_full_s = sample
                    if dataslice == 0:
                        masked_kspace = masked_kspace_s[None, ...]
                        full_kspace = kspace_s[None, ...]
                        mask = mask_s[None, ...]
                        image_full = image_full_s[None, ...]
                    else:
                        masked_kspace = torch.cat((masked_kspace, masked_kspace_s[None, ...]), 0)
                        full_kspace = torch.cat((full_kspace, kspace_s[None, ...]), 0)
                        mask = torch.cat((mask, mask_s[None, ...]), 0)
                        image_full = torch.cat((image_full, image_full_s[None, ...]), 0)
                else:
                    raise ValueError('Unrecognized sampling method!')

        if flag_unet:
            # img [D,Nc,H,W], mask [D,1,H,W]
            return image_masked, image_full, mask
        else:
            # kspace [D,Nc,H,W,2], mask [D,Nc,H,W,2], image [D,H,W]
            return masked_kspace, full_kspace, mask, image_full