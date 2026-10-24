from torch.utils import data as data
from torchvision.transforms.functional import normalize
import cv2

from basicsr.data.data_util import paired_paths_from_folder, paired_paths_from_lmdb, paired_paths_from_meta_info_file
from basicsr.data.transforms import augment, paired_random_crop
from basicsr.utils import FileClient, imfrombytes, img2tensor
from basicsr.utils.matlab_functions import rgb2ycbcr
from basicsr.utils.registry import DATASET_REGISTRY
from basicsr.metrics import calculate_niqe
import numpy as np
@DATASET_REGISTRY.register()
class PairedImageDataset(data.Dataset):
    """Paired image dataset for image restoration.

    Read LQ (Low Quality, e.g. LR (Low Resolution), blurry, noisy, etc) and GT image pairs.

    There are three modes:
    1. 'lmdb': Use lmdb files.
        If opt['io_backend'] == lmdb.
    2. 'meta_info_file': Use meta information file to generate paths.
        If opt['io_backend'] != lmdb and opt['meta_info_file'] is not None.
    3. 'folder': Scan folders to generate paths.
        The rest.

    Args:
        opt (dict): Config for train datasets. It contains the following keys:
            dataroot_gt (str): Data root path for gt.
            dataroot_lq (str): Data root path for lq.
            meta_info_file (str): Path for meta information file.
            io_backend (dict): IO backend type and other kwarg.
            filename_tmpl (str): Template for each filename. Note that the template excludes the file extension.
                Default: '{}'.
            gt_size (int): Cropped patched size for gt patches.
            use_hflip (bool): Use horizontal flips.
            use_rot (bool): Use rotation (use vertical flip and transposing h and w for implementation).

            scale (bool): Scale, which will be added automatically.
            phase (str): 'train' or 'val'.
    """

    def __init__(self, opt):
        super(PairedImageDataset, self).__init__()
        self.opt = opt
        # file client (io backend)
        self.file_client = None
        self.io_backend_opt = opt['io_backend']
        self.mean = opt['mean'] if 'mean' in opt else None
        self.std = opt['std'] if 'std' in opt else None

        self.gt_folder, self.lq_folder = opt['dataroot_gt'], opt['dataroot_lq']
        if 'filename_tmpl' in opt:
            self.filename_tmpl = opt['filename_tmpl']
        else:
            self.filename_tmpl = '{}'

        if self.io_backend_opt['type'] == 'lmdb':
            self.io_backend_opt['db_paths'] = [self.lq_folder, self.gt_folder]
            self.io_backend_opt['client_keys'] = ['lq', 'gt']
            self.paths = paired_paths_from_lmdb([self.lq_folder, self.gt_folder], ['lq', 'gt'])
        elif 'meta_info_file' in self.opt and self.opt['meta_info_file'] is not None:
            self.paths = paired_paths_from_meta_info_file([self.lq_folder, self.gt_folder], ['lq', 'gt'],
                                                          self.opt['meta_info_file'], self.filename_tmpl)
        else:
            self.paths = paired_paths_from_folder([self.lq_folder, self.gt_folder], ['lq', 'gt'], self.filename_tmpl)

    def __getitem__(self, index):
        if self.file_client is None:
            self.file_client = FileClient(self.io_backend_opt.pop('type'), **self.io_backend_opt)

        scale = self.opt['scale']

        # 读取图像（HWC, BGR, [0,1], float32）
        gt_path = self.paths[index]['gt_path']
        img_bytes = self.file_client.get(gt_path, 'gt')
        img_gt, _ = imfrombytes(img_bytes, float32=True)

        lq_path = self.paths[index]['lq_path']
        img_bytes = self.file_client.get(lq_path, 'lq')
        img_lq, _ = imfrombytes(img_bytes, float32=True)

        # ===================== 训练阶段：先做“对称补齐”，再裁剪 =====================
        if self.opt['phase'] == 'train':
            gt_size = self.opt['gt_size']
            lq_patch = gt_size // scale

            # 目标 LQ 至少能裁出 lq_patch；GT 与 LQ*scale 严格对齐，且 GT 至少 gt_size
            H_lq, W_lq = img_lq.shape[:2]
            target_lq_h = max(H_lq, lq_patch)
            target_lq_w = max(W_lq, lq_patch)

            target_gt_h = target_lq_h * scale
            target_gt_w = target_lq_w * scale

            # 兜底：确保 GT 端至少 gt_size；若增大了 GT，需回推同步增大 LQ 以保持对齐
            target_gt_h = max(target_gt_h, gt_size)
            target_gt_w = max(target_gt_w, gt_size)
            target_lq_h = max(target_lq_h, (target_gt_h + scale - 1) // scale)
            target_lq_w = max(target_lq_w, (target_gt_w + scale - 1) // scale)
            target_gt_h = target_lq_h * scale
            target_gt_w = target_lq_w * scale

            # 对称补齐（默认使用 reflect，更适合视觉任务；如需 0 填充改 mode='constant', constant=0）
            img_lq = self._pad_to_size_symmetric(img_lq, target_lq_h, target_lq_w, mode='reflect')
            img_gt = self._pad_to_size_symmetric(img_gt, target_gt_h, target_gt_w, mode='reflect')

            # 成对随机裁剪 + 增广
            img_gt, img_lq = paired_random_crop(img_gt, img_lq, gt_size, scale, gt_path)
            img_gt, img_lq = augment([img_gt, img_lq], self.opt['use_hflip'], self.opt['use_rot'])

        # ===================== 非训练阶段：保持原逻辑的对齐裁剪 =====================
        # 可选颜色空间变换
        if 'color' in self.opt and self.opt['color'] == 'y':
            img_gt = rgb2ycbcr(img_gt, y_only=True)[..., None]
            img_lq = rgb2ycbcr(img_lq, y_only=True)[..., None]

        # 验证/测试阶段：将 GT 裁到 LQ*scale，避免尺寸不匹配
        if self.opt['phase'] != 'train':
            img_gt = img_gt[0:img_lq.shape[0] * scale, 0:img_lq.shape[1] * scale, :]

        # BGR->RGB, HWC->CHW, numpy->tensor
        img_gt, img_lq = img2tensor([img_gt, img_lq], bgr2rgb=True, float32=True)

        # 归一化
        if self.mean is not None or self.std is not None:
            normalize(img_lq, self.mean, self.std, inplace=True)
            normalize(img_gt, self.mean, self.std, inplace=True)

        return {'lq': img_lq, 'gt': img_gt, 'lq_path': lq_path, 'gt_path': gt_path}

    # def __getitem__(self, index):
    #     if self.file_client is None:
    #         self.file_client = FileClient(self.io_backend_opt.pop('type'), **self.io_backend_opt)
    #
    #     scale = self.opt['scale']
    #
    #     # Load gt and lq images. Dimension order: HWC; channel order: BGR;
    #     # image range: [0, 1], float32.
    #     gt_path = self.paths[index]['gt_path']
    #     img_bytes = self.file_client.get(gt_path, 'gt')
    #     img_gt,_ = imfrombytes(img_bytes, float32=True)
    #     # print(img_gt.shape)
    #     lq_path = self.paths[index]['lq_path']
    #     img_bytes = self.file_client.get(lq_path, 'lq')
    #     img_lq,img_lq_src = imfrombytes(img_bytes, float32=True)
    #
    #
    #
    #
    #
    #     # augmentation for training
    #     if self.opt['phase'] == 'train':
    #         gt_size = self.opt['gt_size']
    #         # random crop
    #         img_gt, img_lq = paired_random_crop(img_gt, img_lq, gt_size, scale, gt_path)
    #         # flip, rotation
    #         img_gt, img_lq = augment([img_gt, img_lq], self.opt['use_hflip'], self.opt['use_rot'])
    #
    #
    #
    #
    #     # color space transform
    #     if 'color' in self.opt and self.opt['color'] == 'y':
    #         img_gt = rgb2ycbcr(img_gt, y_only=True)[..., None]
    #         img_lq = rgb2ycbcr(img_lq, y_only=True)[..., None]
    #
    #     # crop the unmatched GT images during validation or testing, especially for SR benchmark datasets
    #     # TODO: It is better to update the datasets, rather than force to crop
    #     if self.opt['phase'] != 'train':
    #         # print('Ori',img_gt.shape)
    #         img_gt = img_gt[0:img_lq.shape[0] * scale, 0:img_lq.shape[1] * scale, :]
    #         # print('After',img_gt.shape)
    #
    #     # BGR to RGB, HWC to CHW, numpy to tensor
    #     img_gt, img_lq = img2tensor([img_gt, img_lq], bgr2rgb=True, float32=True)
    #     # normalize
    #     if self.mean is not None or self.std is not None:
    #         normalize(img_lq, self.mean, self.std, inplace=True)
    #         normalize(img_gt, self.mean, self.std, inplace=True)
    #
    #     return {'lq': img_lq, 'gt': img_gt, 'lq_path': lq_path, 'gt_path': gt_path}

    def __len__(self):
        return len(self.paths)

    def _pad_to_size_symmetric(self,img, th, tw, mode='reflect', constant=0):
        """
        将 HWC 图像对称补到目标尺寸 (th, tw)。
        - 不足才补；足够则原样返回。
        - 对称补：上下、左右尽量平均分配，若需要奇数像素则把多出来的像素补在“下/右”侧。
        - mode: 'reflect'（默认）、'edge'、'symmetric'、'constant' 等 numpy.pad 支持的模式。
        """
        h, w = img.shape[:2]
        if h >= th and w >= tw:
            return img

        pad_h = max(th - h, 0)
        pad_w = max(tw - w, 0)

        top = pad_h // 2
        bottom = pad_h - top
        left = pad_w // 2
        right = pad_w - left

        pad = ((top, bottom), (left, right), (0, 0))
        if mode == 'constant':
            return np.pad(img, pad, mode='constant', constant_values=constant)
        else:
            return np.pad(img, pad, mode=mode)
