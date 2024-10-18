# AttHybrid-VarNet
This is the implementation of attention hybrid variational net for accelerated MRI reconstruction from:

[Attention hybrid variational net for accelerated MRI reconstruction](https://doi.org/10.1063/5.0165485)


## Overview

We propose a deep learning-based attention hybrid variational network that performs learning in both the k-space and image domains.

More specifically, we build an attention hybrid variational network (AttHybrid-VarNet) that benefits from the superior k-space reconstruction ability and an image-domain refinement network to further improve the image quality. Furthermore, spatial- and channel-wise attention also enables the convolutional module to further fine tune the weights for different channels and regions in the feature maps according to the attention scores.

Overall model structure
![Overall structure](/imgs/fig1_model_overall.png)

Reconstruction samples
![recon_fastmri](/imgs/fig3_recon_fastmri.png)
![recon_b1000](/imgs/fig4_recon_b1000.png)