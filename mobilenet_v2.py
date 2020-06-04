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


def main():
    # test_single_image()
    # return

    mobilenet = torchvision.models.mobilenet_v2(pretrained=True)
    freeze_by_names(mobilenet, ['features'])
    mobilenet.classifier = nn.Linear(mobilenet.last_channel, 2)

    # add decoder to restore input images, features=1280 for mobilenetv2
    decoder = nn.Sequential([
        nn.Linear(1280, 256, False),
        nn.Linear(256, 256, True),
        nn.LeakyReLU(0.2),
        torch.reshape(mobilenet.features, [1, 1, 16, 16]),  # make it 16x16x1
        nn.Conv2d(1, 4, (3, 3), (1, 1), (1, 1), bias=False),  # make it 16x16x4
        nn.ConvTranspose2d(4, 1, (3, 3), (2, 2), bias=True),  # make it 17x17x1
        nn.LeakyReLU(0.2),
    ])
    mobilenet.add_module('decoder', decoder)
    mobilenet.decoder = []

    graph = mobilenet.eval()
    print(graph)
    return

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print('trian_device:{}'.format(device.type))
    mobilenet = mobilenet.to(device)
    loss_func = nn.CrossEntropyLoss()
    opt = torch.optim.Adam(mobilenet.parameters(), lr=1E-4)

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
            mobilenet.train()
            # GPU/CPU
            inputs = inputs.to(device)
            labels = labels.to(device)
            opt.zero_grad()
            # foward
            outputs = mobilenet(inputs)
            loss = loss_func(outputs, labels)
            # backward
            loss.backward()
            opt.step()
            running_loss += loss.item()
            if not (i+1) % 6:
                print('[{}, {}] loss={:.5f}'.format(epoch+1, i+1, running_loss / 10))
                running_loss = 0.0
        # check training accuracy
        correct = 0
        total = 0
        mobilenet.eval()
        for images_train, labels_train in train_dataloader:
            images_train = images_train.to(device)
            labels_train = labels_train.to(device)
            outputs_train = mobilenet(images_train)
            _, prediction = torch.max(outputs_train, 1)
            correct += (torch.sum((prediction == labels_train))).item()
            total += labels_train.size(0)
        print('#{} train accuracy={:.5f}'.format(epoch+1, 1.0*correct/total))

        print('saving models...')
        torch.save(mobilenet.state_dict(), '../Models/cherry-strawberry.pth')
        print('models saved at epoch #%d' % (epoch+1))
    print('training finished !')


if __name__ == '__main__':
    main()
