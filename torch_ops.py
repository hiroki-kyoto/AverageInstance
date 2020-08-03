# torch_ops.py
import os
import argparse
import torch
import torch.nn as nn
import torchvision.transforms.functional as F


def process_args():
    parser = argparse.ArgumentParser()
    parser.description = 'specify gpus, example: 0,2'
    parser.add_argument("-g", "--gpu", 
                        help="gpu id", 
                        dest="gpu", 
                        type=str, 
                        required=True)
    parser.add_argument("-b", "--batch",
                        help="batch size",
                        dest="batch",
                        type=int,
                        required=True)
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    print('using gpus: %s' % args.gpu)
    print('batch size: %d' % args.batch)
    return args


def conv(depth, nfilter, ksize=3, stride=1, 
         padding=0, dilation=1, groups=1, 
         bias=True, lrelu=None):
    """A wrapper for torch.nn.Conv2d.
    
    :param depth: number of channels of inputs
    :param nfilter: number of filters
    :param ksize: kernel size
    :param stride: same as in torch
    :param padding: same as in torch
    :param dilation: same as in torch
    :param groups: same as in torch
    :param bias: add bias or not
    :param lrelu: leaky_relu parameter, set as None to disable it
    :return an instance of torch.nn.Conv2d operation
        the output tensor has shape of:
        h(or w) = (h(or w) - kesize + 2*padding) / stride + 1
    :rtype: nn.Module
    """
    assert (depth>0 and nfilter>0 and ksize>0 and ksize%2==1 and 
            stride>0 and padding>=0 and dilation>=1 and groups>=1)
    conv_ = nn.Conv2d(depth, nfilter, ksize, stride, 
                      padding, dilation, groups, bias)
    if lrelu is not None:
        conv_ = nn.Sequential(conv_, 
                              nn.LeakyReLU(lrelu, inplace=True))
    return conv_


def dense(depth_in, depth_out, bias=True, lrelu=None):
    """A wrapper for torch.nn.Linear
    
    :param depth_in: number of channels of inputs
    :param depth_out: number of channels of outputs
    :param bias: add bias or not
    :param lrelu: leaky_relu parameter, set as None to disable it
    :rtype: nn.Module
    """
    assert (depth_in>0 and depth_out>0)
    linr_ = nn.Linear(depth_in, depth_out, bias)
    if lrelu is not None:
        linr_ = nn.Sequential(linr_, 
                              nn.LeakyReLU(lrelu, inplace=True))
    return linr_


def deconv(depth, nfilter, ksize=3, stride=1, 
           pad_in=0, pad_out=0, groups=1,
           dilation=1, pad_mode='zeros',
           bias=True, lrelu=None):
    """A wrapper for torch.nn.ConvTranspose2d
    
    :param depth: number of channels of inputs
    :param nfilter: number of channels of outputs
    :param ksize: kernel size
    :param stride: step size for a receptive field
    :param pad_in: input padding
    :param pad_out: output padding, as input padding in conv2d
    :param groups: number of groups for both input and outpt
    :param dilation: input dilation
    :param pad_mode: padding region filling strategy specification
    :param bias: adding bias or not
    :param lrelu: leaky relu parameter, set None to disable it
    :rtype: nn.Module
    """
    assert (depth>0 and nfilter>0 and ksize>0 and ksize%2==1 and 
            stride>0 and pad_in>=0 and pad_out>=0 and dilation>=1 and
            groups>=1 and depth%groups==0 and nfilter%groups==0)
    deconv_ = nn.ConvTranspose2d(depth, nfilter, ksize, stride, 
                      pad_in, pad_out, groups, bias, dilation,
                      pad_mode)
    if lrelu is not None:
        deconv_ = nn.Sequential(deconv_, 
                              nn.LeakyReLU(lrelu, inplace=True))
    return deconv_
    
