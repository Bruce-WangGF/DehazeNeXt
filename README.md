# DehazeNeXt

Implementation of **DehazeNeXt: Rethinking Convolutional Network Design for Efficient Image Dehazing**.

The network is in [`basicsr/archs/DehazeNeXt_arch.py`](basicsr/archs/DehazeNeXt_arch.py). Its two named modules are the Dual-input Feature Seesaw Module (DFSM) and Dynamic Local Refinement Module (DLRM).

## Environment

Use a CUDA GPU. Install a CUDA-compatible PyTorch and torchvision pair, plus the CuPy wheel matching your CUDA version. For example, for CUDA 11.x:

```bash
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
python -m pip install cupy-cuda11x
python -m pip install -e .
```

The dynamic convolution in DLRM uses a CuPy CUDA kernel and requires a CUDA GPU. The code has been exercised with Python 3.8, PyTorch 2.1.2+cu118, torchvision 0.16.2+cu118, and CuPy 12.3.0.

## Checkpoints

Pretrained checkpoints: [Google Drive folder](https://drive.google.com/drive/folders/1H-JtuX5UcpYQSNmOewT09hFyW1u-Au6v).

## Training

Place paired training images in `datasets/ITS/train/{hazy,clear}` or `datasets/OTS/train/{hazy,clear}`, then run one of the four configurations:

```bash
python -m basicsr.train -opt options/train/DehazeNeXt_B_ITS.yml
```

The default configurations train for 600,000 iterations and validate on the corresponding `datasets/<ITS-or-OTS>/test` set. Adjust paths and training settings for your data and hardware.

## Acknowledgment

The training framework is adapted from [BasicSR](https://github.com/XPixelGroup/BasicSR). The repository retains the Apache 2.0 license.
