# mobilenet_v2.py
from collections.abc import Iterable
import numpy as np
import torch
import torchvision
from torch import nn as nn
from PIL import Image
from torch import Tensor
import imagenet_classes as imagenet
from torch.utils.data import DataLoader


def load_dataset(data_dir):
    dataset = torchvision.datasets.ImageFolder(
        root=data_dir,
        transform=torchvision.transforms.Compose([
            torchvision.transforms.Resize(224),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225))
        ]))
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


def test_single_image():
    mobilenet = torchvision.models.mobilenet_v2(pretrained=True)
    mobilenet.eval()
    im = Image.open("./demo/test2.png")
    im = im.resize([224, 224])
    x = np.array(im, dtype=np.float32)/255.0
    if len(x.shape)==1: # gray image
        x = np.stack([x, x, x], axis=-1)
    elif len(x.shape)==3 and x.shape[2]==4: # RGBA image
        x = x[:, :, 0:3]
    elif len(x.shape)==3 and x.shape[2]==3: # RGB image
        pass
    else:
        print('Error: invalid input image format!')
        exit(-1)
    x = x.transpose([2, 0, 1])
    x_t = torch.from_numpy(np.expand_dims(x, 0))
    y_t = mobilenet(x_t)
    y_t = torch.softmax(y_t, -1)
    y = np.array(y_t.tolist())[0]
    res = np.argmax(y)
    print('predicted class [%s] with confidence of [%6.3f]' % (imagenet.class_names[res], y[res]))


class TrustedMobileNetV2(nn.Module):
    def __init__(self):
        super(TrustedMobileNetV2, self).__init__()
        self.mobilenet = torchvision.models.mobilenet_v2(pretrained=True)
        freeze_by_names(self.mobilenet, ['features'])
        self.mobilenet.classifier = nn.Linear(self.mobilenet.last_channel, 2)
        # add decoder to restore input images, features=1280 for mobilenet-v2
        decoder_ = []
        layer_ = nn.Sequential(
            nn.Linear(1280, 256, False),
            nn.Linear(256, 256, True),
            nn.LeakyReLU(0.2)
        )
        decoder_.append(layer_)
        layer_ = nn.Sequential(
            nn.ConvTranspose2d(256, 128, (3, 3), (1, 1), padding=0, output_padding=0, bias=True),  # make it 3x3x128
            nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(128, 64, (3, 3), (2, 2), padding=0, output_padding=0, bias=True),  # make it 7x7x64
            nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(64, 32, (1, 1), (2, 2), padding=0, output_padding=1, bias=True),  # make it 14x14x32
            nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(32, 16, (1, 1), (2, 2), padding=0, output_padding=1, bias=True),  # make it 28x28x16
            nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(16, 8, (1, 1), (2, 2), padding=0, output_padding=1, bias=True),  # make it 56x56x8
            nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(8, 4, (1, 1), (2, 2), padding=0, output_padding=1, bias=True),  # make it 112x112x4
            nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(4, 4, (1, 1), (2, 2), padding=0, output_padding=1, bias=True),  # make it 224x224x4
            nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(4, 3, (1, 1), (1, 1), padding=0, output_padding=0, bias=True)  # make it 224x224x3
        )
        decoder_.append(layer_)
        decoder_ = nn.Sequential(*decoder_)
        self.mobilenet.add_module('decoder', decoder_)

    def forward(self, x):
        feats = self.features(x)
        x = feats.mean([2, 3])
        pred_class = self.mobilenet.classifier(x)
        pred_image = self.mobilenet.decoder(x)
        return pred_class, pred_image

    def to(self, *args, **kwargs):
        return self.mobilenet.to(*args, **kwargs)

    def train(self, mode=True):
        return self.mobilenet.train(mode)

    def eval(self):
        return self.mobilenet.eval()


def main():
    # test_single_image()
    # return
    net = TrustedMobileNetV2()
    print(net.eval())
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print('train_device:{}'.format(device.type))
    net = net.to(device)
    loss_c = nn.CrossEntropyLoss()
    loss_r = nn.L1Loss(reduction='mean')
    opt = torch.optim.Adam(net.parameters(), lr=1E-4)

    # setup dataset
    train_dataset = load_dataset('../Datasets/ClassifierEstimator/train/')
    train_dataloader = DataLoader(
        dataset=train_dataset,
        batch_size=1,
        shuffle=True)
    # training procedure
    num_epochs = 500
    for epoch in range(num_epochs):
        running_loss = 0.0
        for i, sample_batch in enumerate(train_dataloader):
            inputs = sample_batch[0]
            labels = sample_batch[1]
            net.train()
            # GPU/CPU
            inputs = inputs.to(device)
            labels = labels.to(device)
            opt.zero_grad()
            # foward
            class_, image_ = net(inputs)
            loss_c_ = loss_c(class_, labels)
            loss_r_ = loss_r(image_, inputs)
            loss = loss_c_ + loss_r_
            # backward: as features are frozen, no grad will merge in features, use detach for safety
            loss.backward()
            opt.step()
            running_loss_c = loss_c_.item()
            running_loss_r = loss_r_.item()
            running_loss += loss.item()
            every_n_batch = 10
            if not (i+1) % every_n_batch:
                print('[{}, {}] loss_c={:.5f} loss_r={:.5f} loss={:.5f}'.format(
                    epoch + 1,
                    i + 1,
                    running_loss_c / every_n_batch,
                    running_loss_r / every_n_batch,
                    running_loss / every_n_batch))
                running_loss = 0.0
        # check training accuracy
        correct = 0
        total = 0
        net.eval()
        for images_train, labels_train in train_dataloader:
            images_train = images_train.to(device)
            labels_train = labels_train.to(device)
            class_, image_ = net(images_train)
            _, prediction = torch.max(class_, 1)
            correct += (torch.sum((prediction == labels_train))).item()
            total += labels_train.size(0)
        print('#{} train accuracy={:.5f}'.format(epoch+1, 1.0*correct/total))

        print('saving models...')
        torch.save(net.state_dict(), '../Models/cherry-strawberry.pth')
        print('models saved at epoch #%d' % (epoch+1))
    print('training finished !')


if __name__ == '__main__':
    main()
