"""DehazeNeXt architecture with DFSM and DLRM modules."""

from collections import namedtuple
from string import Template

import cupy
import torch
from torch import nn
from torch.autograd import Function
from torch.nn.modules.utils import _pair

from basicsr.utils.registry import ARCH_REGISTRY


Stream = namedtuple('Stream', ['ptr'])


def Dtype(t):
    if isinstance(t, torch.cuda.FloatTensor):
        return 'float'
    elif isinstance(t, torch.cuda.DoubleTensor):
        return 'double'


@cupy._util.memoize(for_each_device=True)
def load_kernel(kernel_name, code, **kwargs):
    code = Template(code).substitute(**kwargs)
    kernel_code = cupy.cuda.compile_with_cache(code)
    return kernel_code.get_function(kernel_name)


CUDA_NUM_THREADS = 1024


kernel_loop = '''
#define CUDA_KERNEL_LOOP(i, n)                        \
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; \
      i < (n);                                       \
      i += blockDim.x * gridDim.x)
'''


def GET_BLOCKS(N):
    return (N + CUDA_NUM_THREADS - 1) // CUDA_NUM_THREADS


_idynamic_kernel = kernel_loop + '''
extern "C"
__global__ void idynamic_forward_kernel(
const ${Dtype}* bottom_data, const ${Dtype}* weight_data, ${Dtype}* top_data) {
  CUDA_KERNEL_LOOP(index, ${nthreads}) { //nthreads = H*W*C egg:28*38*18
    const int n = index / ${channels} / ${top_height} / ${top_width}; //n is [0-batchsize-1]
    const int c = (index / ${top_height} / ${top_width}) % ${channels}; // c  is 0-channels-1
    const int h = (index / ${top_width}) % ${top_height};
    const int w = index % ${top_width};
    const int g = c / (${channels} / ${groups});
    //printf("%d ",g);
    ${Dtype} value = 0;
    #pragma unroll
    //printf("%d ", ${bottom_height});
    //dilation_h = dilation_w = 1;
    //stride_h = stride_w = 1;
    //pad_h = pad_w = 3
    for (int kh = 0; kh < ${kernel_h}; ++kh) {
      #pragma unroll
      for (int kw = 0; kw < ${kernel_w}; ++kw) {
        const int h_in = -${pad_h} + h * ${stride_h} + kh * ${dilation_h};
        const int w_in = -${pad_w} + w * ${stride_w} + kw * ${dilation_w};
        if ((h_in >= 0) && (h_in < ${bottom_height})
          && (w_in >= 0) && (w_in < ${bottom_width})) {
          const int offset = ((n * ${channels} + c) * ${bottom_height} + h_in)
            * ${bottom_width} + w_in;
          const int offset_weight = ((((n * ${groups} + g) * ${kernel_h} + kh) * ${kernel_w} + kw) * ${top_height} + h)
            * ${top_width} + w;
          //printf("%f ", weight_data[offset_weight]);
          value += weight_data[offset_weight] * bottom_data[offset];
        }
      }
    }
    top_data[index] = value;
  }
}
'''


_idynamic_kernel_backward_grad_input = kernel_loop + '''
extern "C"
__global__ void idynamic_backward_grad_input_kernel(
    const ${Dtype}* const top_diff, const ${Dtype}* const weight_data, ${Dtype}* const bottom_diff) {
  CUDA_KERNEL_LOOP(index, ${nthreads}) {
    const int n = index / ${channels} / ${bottom_height} / ${bottom_width};
    const int c = (index / ${bottom_height} / ${bottom_width}) % ${channels};
    const int h = (index / ${bottom_width}) % ${bottom_height};
    const int w = index % ${bottom_width};
    const int g = c / (${channels} / ${groups});
    ${Dtype} value = 0;
    #pragma unroll
    for (int kh = 0; kh < ${kernel_h}; ++kh) {
      #pragma unroll
      for (int kw = 0; kw < ${kernel_w}; ++kw) {
        const int h_out_s = h + ${pad_h} - kh * ${dilation_h};
        const int w_out_s = w + ${pad_w} - kw * ${dilation_w};
        if (((h_out_s % ${stride_h}) == 0) && ((w_out_s % ${stride_w}) == 0)) {
          const int h_out = h_out_s / ${stride_h};
          const int w_out = w_out_s / ${stride_w};
          if ((h_out >= 0) && (h_out < ${top_height})
                && (w_out >= 0) && (w_out < ${top_width})) {
            const int offset = ((n * ${channels} + c) * ${top_height} + h_out)
                  * ${top_width} + w_out;
            const int offset_weight = ((((n * ${groups} + g) * ${kernel_h} + kh) * ${kernel_w} + kw) * ${top_height} + h_out)
                  * ${top_width} + w_out;
            value += weight_data[offset_weight] * top_diff[offset];
          }
        }
      }
    }
    bottom_diff[index] = value;
  }
}
'''


_idynamic_kernel_backward_grad_weight = kernel_loop + '''
extern "C"
__global__ void idynamic_backward_grad_weight_kernel(
    const ${Dtype}* const top_diff, const ${Dtype}* const bottom_data, ${Dtype}* const buffer_data) {
  CUDA_KERNEL_LOOP(index, ${nthreads}) {
    const int h = (index / ${top_width}) % ${top_height};
    const int w = index % ${top_width};
    const int kh = (index / ${kernel_w} / ${top_height} / ${top_width})
          % ${kernel_h};
    const int kw = (index / ${top_height} / ${top_width}) % ${kernel_w};
    const int h_in = -${pad_h} + h * ${stride_h} + kh * ${dilation_h};
    const int w_in = -${pad_w} + w * ${stride_w} + kw * ${dilation_w};
    if ((h_in >= 0) && (h_in < ${bottom_height})
          && (w_in >= 0) && (w_in < ${bottom_width})) {
      const int g = (index / ${kernel_h} / ${kernel_w} / ${top_height} / ${top_width}) % ${groups};
      const int n = (index / ${groups} / ${kernel_h} / ${kernel_w} / ${top_height} / ${top_width}) % ${num};
      ${Dtype} value = 0;
      #pragma unroll
      for (int c = g * (${channels} / ${groups}); c < (g + 1) * (${channels} / ${groups}); ++c) {
        const int top_offset = ((n * ${channels} + c) * ${top_height} + h)
              * ${top_width} + w;
        const int bottom_offset = ((n * ${channels} + c) * ${bottom_height} + h_in)
              * ${bottom_width} + w_in;
        value += top_diff[top_offset] * bottom_data[bottom_offset];
      }
      buffer_data[index] = value;
    } else {
      buffer_data[index] = 0;
    }
  }
}
'''


class _idynamic(Function):
    @staticmethod
    def forward(ctx, input, weight, stride, padding, dilation):
        assert input.dim() == 4 and input.is_cuda
        assert weight.dim() == 6 and weight.is_cuda
        batch_size, channels, height, width = input.size()
        kernel_h, kernel_w = weight.size()[2:4]
        output_h = int((height + 2 * padding[0] - (dilation[0] * (kernel_h - 1) + 1)) / stride[0] + 1)
        output_w = int((width + 2 * padding[1] - (dilation[1] * (kernel_w - 1) + 1)) / stride[1] + 1)
        output = input.new(batch_size, channels, output_h, output_w)
        n = output.numel()
        with torch.cuda.device_of(input):
            f = load_kernel('idynamic_forward_kernel', _idynamic_kernel, Dtype=Dtype(input), nthreads=n,
                            num=batch_size, channels=channels, groups=weight.size()[1],
                            bottom_height=height, bottom_width=width,
                            top_height=output_h, top_width=output_w,
                            kernel_h=kernel_h, kernel_w=kernel_w,
                            stride_h=stride[0], stride_w=stride[1],
                            dilation_h=dilation[0], dilation_w=dilation[1],
                            pad_h=padding[0], pad_w=padding[1])
            f(block=(CUDA_NUM_THREADS, 1, 1),
              grid=(GET_BLOCKS(n), 1, 1),
              args=[input.data_ptr(), weight.data_ptr(), output.data_ptr()],
              stream=Stream(ptr=torch.cuda.current_stream().cuda_stream))
        ctx.save_for_backward(input, weight)
        ctx.stride, ctx.padding, ctx.dilation = stride, padding, dilation
        return output
    @staticmethod
    def backward(ctx, grad_output):
        assert grad_output.is_cuda
        if not grad_output.is_contiguous():
            grad_output = grad_output.contiguous()
        input, weight = ctx.saved_tensors
        stride, padding, dilation = ctx.stride, ctx.padding, ctx.dilation
        batch_size, channels, height, width = input.size()
        kernel_h, kernel_w = weight.size()[2:4]
        output_h, output_w = grad_output.size()[2:]
        grad_input, grad_weight = None, None
        opt = dict(Dtype=Dtype(grad_output),
                   num=batch_size, channels=channels, groups=weight.size()[1],
                   bottom_height=height, bottom_width=width,
                   top_height=output_h, top_width=output_w,
                   kernel_h=kernel_h, kernel_w=kernel_w,
                   stride_h=stride[0], stride_w=stride[1],
                   dilation_h=dilation[0], dilation_w=dilation[1],
                   pad_h=padding[0], pad_w=padding[1])
        with torch.cuda.device_of(input):
            if ctx.needs_input_grad[0]:
                grad_input = input.new(input.size())
                n = grad_input.numel()
                opt['nthreads'] = n
                f = load_kernel('idynamic_backward_grad_input_kernel',
                                _idynamic_kernel_backward_grad_input, **opt)
                f(block=(CUDA_NUM_THREADS, 1, 1),
                  grid=(GET_BLOCKS(n), 1, 1),
                  args=[grad_output.data_ptr(), weight.data_ptr(), grad_input.data_ptr()],
                  stream=Stream(ptr=torch.cuda.current_stream().cuda_stream))
            if ctx.needs_input_grad[1]:
                grad_weight = weight.new(weight.size())
                n = grad_weight.numel()
                opt['nthreads'] = n
                f = load_kernel('idynamic_backward_grad_weight_kernel',
                                _idynamic_kernel_backward_grad_weight, **opt)
                f(block=(CUDA_NUM_THREADS, 1, 1),
                  grid=(GET_BLOCKS(n), 1, 1),
                  args=[grad_output.data_ptr(), input.data_ptr(), grad_weight.data_ptr()],
                  stream=Stream(ptr=torch.cuda.current_stream().cuda_stream))
        return grad_input, grad_weight, None, None, None


def _idynamic_cuda(input, weight, bias=None, stride=1, padding=0, dilation=1):
    """ idynamic kernel
    """
    assert input.size(0) == weight.size(0)
    assert input.size(-2) // stride == weight.size(-2)
    assert input.size(-1) // stride == weight.size(-1)
    if input.is_cuda:
        out = _idynamic.apply(input, weight, _pair(stride), _pair(padding), _pair(dilation))
        if bias is not None:
            out += bias.view(1, -1, 1, 1)
    else:
        raise NotImplementedError
    return out


class IDynamicDWConv(nn.Module):
    """Generate a grouped dynamic depthwise kernel at each spatial position."""

    def __init__(self, channels, kernel_size, bias=True):
        super().__init__()
        self.kernel_size = kernel_size
        self.channels = channels
        reduction_ratio = 4
        self.groups = max(1, self.channels // 4)
        self.conv1 = nn.Sequential(
            nn.Conv2d(channels, channels // reduction_ratio, 1, bias=bias),
            nn.Conv2d(channels // reduction_ratio, channels // reduction_ratio,
                      kernel_size=kernel_size, padding=kernel_size // 2,
                      groups=channels // reduction_ratio, bias=bias),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(channels // reduction_ratio, kernel_size ** 2 * self.groups, 1, bias=bias)
        )
        self.feature_reduction = nn.Conv2d(channels, self.groups, 1, bias=bias)
        self.feature_expansion = nn.Conv2d(self.groups, channels, 1, bias=bias)

    def forward(self, x):
        b, _, h, w = x.shape
        weight = self.conv2(self.conv1(x))
        weight = weight.view(b, self.groups, self.kernel_size, self.kernel_size, h, w)
        x_reduced = self.feature_reduction(x)
        out = _idynamic_cuda(x_reduced, weight, stride=1,
                             padding=(self.kernel_size - 1) // 2)
        out = self.feature_expansion(out)
        return out


class ResidualBlock(torch.nn.Module):
    def __init__(self, channels, kernelsize=3):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=kernelsize, stride=1, padding=kernelsize // 2)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=kernelsize, stride=1, padding=kernelsize // 2)
        self.relu = nn.ReLU(inplace=True)
    def forward(self, x):
        out = self.relu(self.conv1(x))
        out = self.conv2(out)
        return out + x


class ResidualGroup(torch.nn.Module):
    def __init__(self, channels, nResblock):
        super().__init__()
        self.Engroup = nn.Sequential(*[ResidualBlock(channels) for _ in range(nResblock)])
    def forward(self, x):
        x1 = self.Engroup(x)
        return x1 + x


class DFSM(nn.Module):
    def __init__(self, channel, nResblock):
        super().__init__()
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(channel, channel, 3, 1, 1)
        self.conv2 = nn.Conv2d(channel, channel, 5, 1, 2)
        self.conv3 = nn.Conv2d(channel * 2, channel // 2, 1, 1, 0)
        self.conv4 = nn.Conv2d(channel, channel, 3, 1, 1)
        self.conv5 = nn.Conv2d(channel, channel, 3, 1, 1)
        self.ResGroup = ResidualGroup(channel // 2, nResblock)
        self.conv6 = nn.Conv2d(channel // 2, channel, 3, 1, 1)
        self.sig = nn.Sigmoid()
    def forward(self, x_dehaze, x_hazy):
        x1_dehaze = self.relu(self.conv1(x_dehaze))
        x1_hazy = self.relu(self.conv2(x_hazy))
        x2 = self.conv6(self.ResGroup(self.conv3(torch.cat((x1_hazy, x1_dehaze), 1))))
        x3_dehaze = self.conv4(x2)
        x3_hazy = self.conv5(x2)
        x4 = self.sig(x3_dehaze + x3_hazy)
        x5 = x_dehaze * x4
        x6 = x_hazy * (1 - x4)
        return x5 + x6


class DLRM(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, 1, 1)
        self.relu = nn.ReLU(inplace=True)
        self.conv3 = nn.Conv2d(channels, channels // 2, 1, 1, 0)
        self.DynamicConv = IDynamicDWConv(channels // 2, kernel_size=7)
        self.conv6 = nn.Conv2d(channels // 2, channels * 2, 1, 1, 0)
        self.Gelu = nn.GELU()
    def forward(self, x):
        x2 = self.relu(self.conv1(x))
        x2 = self.conv3(x2)
        x3 = self.DynamicConv(x2)
        x4 = self.conv6(x3)
        x4_1, x4_2 = x4.chunk(2, dim=1)
        x5 = self.Gelu(x4_1) * x4_2
        return x5 + x


class TopNet(nn.Module):
    def __init__(self, channels, kernelsize, nResblock):
        super(TopNet, self).__init__()
        self.conv1 = nn.Conv2d(3, channels, kernel_size=kernelsize, stride=1, padding=kernelsize // 2)
        self.ResGroup = ResidualGroup(channels, nResblock)
    def forward(self, x):
        x1 = self.conv1(x)
        out1 = self.ResGroup(x1)
        return out1


class BottomNet(nn.Module):
    def __init__(self, channels, kernelsize, nResblock=15):
        super(BottomNet, self).__init__()
        self.conv1 = nn.Conv2d(3, channels, kernel_size=kernelsize, stride=1, padding=kernelsize // 2)
        self.dfsm = DFSM(channels, nResblock)
    def forward(self, x_up, x):
        x1 = self.conv1(x)
        out1 = self.dfsm(x_up, x1)
        return out1


class DeBottomNet(nn.Module):
    def __init__(self, channels):
        super(DeBottomNet, self).__init__()
        self.dlrm = DLRM(channels)
        self.ResBlock = ResidualBlock(channels)
        self.Up22 = nn.Upsample(scale_factor=2)
    def forward(self, x):
        x = self.dlrm(x)
        x1 = self.ResBlock(x)
        x3 = self.Up22(x1)
        return x3


class DeTopNet(nn.Module):
    def __init__(self, channels, IsUp=True, nResblock=15):
        super().__init__()
        self.IsUp = IsUp
        self.dlrm = DLRM(channels)
        self.ResBlock = ResidualBlock(channels)
        self.dfsm = DFSM(channels, nResblock)
        self.Up22 = nn.Upsample(scale_factor=2)
    def forward(self, x, x_dowm):
        x1 = self.dfsm(x, x_dowm)
        x1 = self.dlrm(x1)
        x2 = self.ResBlock(x1)
        if not self.IsUp:
            return x2
        x4 = self.Up22(x2)
        return x4


@ARCH_REGISTRY.register()
class DehazeNeXt(nn.Module):
    def __init__(self, dim=64, nResblock=15):
        super().__init__()
        channels = dim
        self.Avg = nn.AvgPool2d(kernel_size=3, stride=2, padding=1)
        self.TopNet1 = TopNet(channels, 11, nResblock)
        self.BottomNet1 = BottomNet(channels, 9, nResblock)
        self.BottomNet2 = BottomNet(channels, 7, nResblock)
        self.BottomNet3 = BottomNet(channels, 5, nResblock)
        self.BottomNet4 = BottomNet(channels, 3, nResblock)
        self.DeBottomNet1 = DeBottomNet(channels)
        self.DeTopNet1 = DeTopNet(channels, True, nResblock)
        self.DeTopNet2 = DeTopNet(channels, True, nResblock)
        self.DeTopNet3 = DeTopNet(channels, True, nResblock)
        self.DeTopNet4 = DeTopNet(channels, False, nResblock)
        self.convFinal6 = nn.Conv2d(channels, 3, kernel_size=3, stride=1, padding=1)
    def forward(self, x):
        x2 = self.Avg(x)
        x4 = self.Avg(x2)
        x8 = self.Avg(x4)
        x16 = self.Avg(x8)
        x0_ = self.TopNet1(x)
        x2_ = self.BottomNet1(self.Avg(x0_), x2)
        x4_ = self.BottomNet2(self.Avg(x2_), x4)
        x8_ = self.BottomNet3(self.Avg(x4_), x8)
        x16_ = self.BottomNet4(self.Avg(x8_), x16)
        x16for8 = self.DeBottomNet1(x16_)
        x8for4 = self.DeTopNet1(x8_, x16for8)
        x4for2 = self.DeTopNet2(x4_, x8for4)
        x2for0 = self.DeTopNet3(x2_, x4for2)
        x0_final = self.DeTopNet4(x0_, x2for0)
        x0_FinalRes = self.convFinal6(x0_final)
        return x0_FinalRes
