# osi_mnist.py
from collections.abc import Iterable
import numpy as np
import torchvision
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from torch_ops import *


plt.rcParams['font.sans-serif']=['SimHei']
plt.rcParams['axes.unicode_minus'] = False
select_gpus_from_user_input()


# Requirement: 
# As resize/crop all other transform ops are done on CPUs, 
# they left the GPUs SLEEP!!! This is not suggested.
# Prepare your training data in advance, such as 
#    cropping, padding, resizing
#    normalization can be done later on GPUs, 
#    never, ever do these on CPUs!
def load_mnist(is_train: bool):
    root = '../Datasets/'
    data_loader = torch.utils.data.DataLoader(
        torchvision.datasets.MNIST(
            root,
            train=is_train,
            download=True,
            transform=torchvision.transforms.ToTensor()))
    return data_loader


def load_image_folder(image_dir: str):
    dataset = torchvision.datasets.ImageFolder(
        root=image_dir,
        transform=torchvision.transforms.ToTensor())
    return dataset


def set_freeze_by_names(model, layer_names, freeze=True):
    if not isinstance(layer_names, Iterable):
        layer_names = [layer_names]
    for name, child in model.named_children():
        if name not in layer_names:
            continue
        for param in child.parameters():
            param.requires_grad = not freeze


def freeze_by_names(model, layer_names):
    set_freeze_by_names(model, layer_names, True)


def unfreeze_by_names(model, layer_names):
    set_freeze_by_names(model, layer_names, False)


def tensor2array(tensor):
    return np.array(tensor.tolist())


# An Auto Encoder is decomposed into encoder and decoder
class MNIST_Encoder(nn.Module):
    def __init__(self):
        super(MNIST_Encoder, self).__init__()
        # encode an image into a latent instance
        self.encoder_conv = nn.Sequential(
            conv(1, 8, 3, 1, lrelu=0.2),
            conv(8, 8, 3, 2, lrelu=0.2),
            conv(8, 16, 3, 1, lrelu=0.2),
            conv(16, 16, 3, 2, lrelu=0.2),
            conv(16, 32, 1, 1, lrelu=0.2),
            conv(32, 8, 1, 1, lrelu=0.2)
        ) # make it NCHW=[N,8,7,7]
        self.encoder_dense = nn.Sequential(
            dense(128, 16, lrelu=0.2),
            dense(16, 64, lrelu=0.2),
            dense(64, 8)
        ) # make it NC=[N, 8]

    def forward(self, x):
        feats = self.encoder_conv(x)
        # print(feats.shape)
        z = torch.flatten(feats, 1)
        z = self.encoder_dense(z)
        return z

    def load_params(self, model_path, device):
        state_dict = torch.load(model_path, map_location=device)
        self.load_state_dict(state_dict)


class MNIST_Decoder(nn.Module):
    def __init__(self):
        super(MNIST_Decoder, self).__init__()
        # decode a latent instance into an image
        self.decoder_dense = nn.Sequential(
            dense(8, 64, lrelu=0.2),
            dense(64, 16, lrelu=0.2),
            dense(16, 4 * 7 * 7, lrelu=0.2)
        ) # make it NC=[N, 4*7*7]
        self.decoder_deconv = nn.Sequential(
            deconv(4, 16, 3, 1, lrelu=0.2, pad_in=1, pad_out=0),
            deconv(16, 8, 3, 2, pad_in=1, pad_out=1, lrelu=0.2),
            deconv(8, 8, 3, 1, lrelu=0.2, pad_in=1, pad_out=0),
            deconv(8, 4, 3, 2, pad_in=1, pad_out=1, lrelu=0.2),
            deconv(4, 1, 3, 1, pad_in=1, pad_out=0)
        )

    def forward(self, z):
        feats = self.decoder_dense(z)
        # print(feats.shape)
        feats = torch.reshape(feats, [feats.shape[0], 4, 7, 7])
        # print(feats.shape)
        x = self.decoder_deconv(feats)
        return x

    def load_params(self, model_path, device):
        state_dict = torch.load(model_path, map_location=device)
        self.mobilenet.load_state_dict(state_dict)


# A Latent Auto Encoder is decomposed into encoder and decoder
class MNIST_LatentEncoder(nn.Module):
    def __init__(self):
        super(MNIST_LatentEncoder, self).__init__()
        # encode latent code into restricted one
        self.encoder = nn.Sequential(
            dense(8, 64, lrelu=0.2),
            dense(64, 16, lrelu=0.2),
            dense(16, 32, lrelu=0.2),
            dense(32, 8)
        )

    def forward(self, x):
        return self.encoder(x)

    def load_params(self, model_path, device):
        state_dict = torch.load(model_path, map_location=device)
        self.load_state_dict(state_dict)


class MNIST_LatentDecoder(nn.Module):
    def __init__(self):
        super(MNIST_LatentDecoder, self).__init__()
        # decode a restricted latent code into plain latent code
        self.decoder = nn.Sequential(
            dense(8, 64, lrelu=0.2),
            dense(64, 16, lrelu=0.2),
            dense(16, 32, lrelu=0.2),
            dense(32, 8)
        )

    def forward(self, z):
        return self.decoder(z)

    def load_params(self, model_path, device):
        state_dict = torch.load(model_path, map_location=device)
        self.mobilenet.load_state_dict(state_dict)


def MNIST_TrainAutoEncoder():
    """
    train an auto encoder on MNIST dataset
    """
    # build model
    encoder = MNIST_Encoder()
    decoder = MNIST_Decoder()

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print('train_device:{}'.format(device.type))
    encoder.to(device)
    decoder.to(device)

    loss_r = nn.L1Loss(reduction='mean')
    opt_enc = torch.optim.Adam(encoder.parameters(), lr=1E-4)
    opt_dec = torch.optim.Adam(decoder.parameters(), lr=1E-4)

    train_dataloader = load_mnist(True)

    # training procedure
    encoder.train()
    decoder.train()
    num_epochs = 500

    for epoch in range(num_epochs):
        running_loss_r = 0
        for i, sample_batch in enumerate(train_dataloader):
            x = sample_batch[0]
            # im_ = tensor2array(x)[0][0]
            # print(im_.shape)
            # plt.imshow(im_)
            # plt.show()
            x = x.to(device)
            # foward
            z = encoder(x)
            x_r = decoder(z)
            loss_r_ = loss_r(x_r, x)
            # backward
            opt_enc.zero_grad()
            opt_dec.zero_grad()
            loss_r_.backward()
            opt_enc.step()
            opt_dec.step()
            running_loss_r += loss_r_.item()
            every_n_batch = 10
            if not (i + 1) % every_n_batch:
                print('[{}, {}] loss_r={:.5f}'.format(
                    epoch + 1,
                    i + 1,
                    running_loss_r / every_n_batch))
                running_loss_r = 0.0
        # save models
        print('saving models...')
        torch.save(encoder.state_dict(),
                   '../Models/ClassifierEstimator/mnist_encoder.pth')
        torch.save(decoder.state_dict(),
                   '../Models/ClassifierEstimator/mnist_decoder.pth')
        print('models saved at epoch #%d' % (epoch + 1))
    print('training finished!')


if __name__ == '__main__':
    MNIST_TrainAutoEncoder()
