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
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif']=['SimHei']
plt.rcParams['axes.unicode_minus'] = False


def load_dataset(data_dir):
    dataset = torchvision.datasets.ImageFolder(
        root=data_dir,
        transform=torchvision.transforms.Compose([
            torchvision.transforms.Resize((224, 224)),
            torchvision.transforms.ToTensor(),
            # torchvision.transforms.Normalize(
            #     mean=(0.485, 0.456, 0.406),
            #     std=(0.229, 0.224, 0.225))
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

    train_dataset = load_dataset('../Datasets/ClassifierEstimator/train/')
    train_dataloader = DataLoader(dataset=train_dataset, batch_size=1, shuffle=True)
    for i, sample_batch in enumerate(train_dataloader):
        inputs = sample_batch[0]
        labels = sample_batch[1]
        y_t = mobilenet(inputs)
        y_t = torch.softmax(y_t, -1)
        y = tensor2array(y_t)[0]
        res = np.argmax(y)
        print('predicted class [%s] with confidence of [%6.3f]' %
              (imagenet.class_names[res], y[res]))
        im_ = tensor2array(inputs)[0]
        im_ = im_.transpose([1, 2, 0])
        plt.title(['cherry', 'strawberry'][labels[0].item()]
                  + ' <---> ' + imagenet.class_names[res])
        plt.imshow(im_)
        plt.show()

    # im = Image.open("./demo/test1.png")
    # im = im.resize([224, 224])
    # x = np.array(im, dtype=np.float32)/255.0
    # if len(x.shape)==1: # gray image
    #     x = np.stack([x, x, x], axis=-1)
    # elif len(x.shape)==3 and x.shape[2]==4: # RGBA image
    #     x = x[:, :, 0:3]
    # elif len(x.shape)==3 and x.shape[2]==3: # RGB image
    #     pass
    # else:
    #     print('Error: invalid input image format!')
    #     exit(-1)
    # x = x.transpose([2, 0, 1])
    # x_t = torch.from_numpy(np.expand_dims(x, 0))
    # y_t = mobilenet(x_t)
    # y_t = torch.softmax(y_t, -1)
    # y = np.array(y_t.tolist())[0]
    # res = np.argmax(y)
    # print('predicted class [%s] with confidence of [%6.3f]' %
    #       (imagenet.class_names[res], y[res]))


def tensor2array(tensor):
    return np.array(tensor.tolist())


class TrustedMobileNetV2(nn.Module):
    def __init__(self, pretrained=True):
        super(TrustedMobileNetV2, self).__init__()
        self.mobilenet = torchvision.models.mobilenet_v2(pretrained=pretrained)
        freeze_by_names(self.mobilenet, ['features'])
        self.mobilenet.classifier = nn.Sequential(
            nn.Linear(self.mobilenet.last_channel, 32, bias=True),
            nn.Linear(32, 64, bias=True),
            nn.LeakyReLU(0.2),
            nn.Linear(64, 32, bias=True),
            nn.Linear(32, 32, bias=True),
            nn.LeakyReLU(0.2),
            nn.Linear(32, 2, bias=True)
        )
        # add decoder to restore input images, features=1280 for mobilenet-v2
        decoder_ = []
        layer_ = nn.Sequential(
            nn.ConvTranspose2d(1280, 64,
                               (3, 3), (2, 2),
                               padding=1,
                               output_padding=1,
                               bias=True),  # make it 14x14x32
            nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(64, 32,
                               (3, 3), (2, 2),
                               padding=1,
                               output_padding=1,
                               bias=True),  # make it 28x28x16
            nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(32, 16,
                               (3, 3), (2, 2),
                               padding=1,
                               output_padding=1,
                               bias=True),  # make it 56x56x8
            nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(16, 8,
                               (3, 3), (2, 2),
                               padding=1,
                               output_padding=1,
                               bias=True),  # make it 112x112x4
            nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(8, 4,
                               (3, 3), (2, 2),
                               padding=1,
                               output_padding=1,
                               bias=True),  # make it 224x224x4
            nn.LeakyReLU(0.2),
            nn.Conv2d(4, 3,
                      (1, 1), (1, 1),
                      padding=0,
                      bias=True)  # make it 224x224x3
        )
        decoder_.append(layer_)
        decoder_ = nn.Sequential(*decoder_)
        self.mobilenet.add_module('decoder', decoder_)

    def forward(self, x):
        feats = self.mobilenet.features(x)
        x = feats.mean([2, 3])
        pred_class = self.mobilenet.classifier(x)
        pred_image = self.mobilenet.decoder(feats)
        #print(feats.shape)
        return pred_class, pred_image

    def to(self, *args, **kwargs):
        self.mobilenet.to(*args, **kwargs)
        return self

    def train(self, mode=True):
        self.mobilenet.train(mode)
        return self

    def eval(self):
        self.mobilenet.eval()
        return self

    def load_params(self, model_path):
        state_dict = torch.load(model_path)
        state_dict_new = {}
        for name_ in state_dict:
            data = tensor2array(state_dict[name_])
            print("%s\t%s" % (name_, data.shape))
            name_new = name_[len('mobilenet.'):]
            state_dict_new[name_new] = state_dict[name_]
        self.mobilenet.load_state_dict(state_dict_new)


def train():
    # test_single_image()
    # return
    net = TrustedMobileNetV2()
    # print(net.eval())
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print('train_device:{}'.format(device.type))
    net.to(device)
    loss_c = nn.CrossEntropyLoss(reduction='mean')
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
        running_loss_c = 0
        running_loss_r = 0
        for i, sample_batch in enumerate(train_dataloader):
            inputs = sample_batch[0]
            labels = sample_batch[1]
            # print(tensor2array(inputs)[0].shape)
            # im_ = tensor2array(inputs)[0]
            # im_ = im_.transpose([1, 2, 0])
            # plt.title(['cherry', 'strawberry'][labels[0].item()])
            # plt.imshow(im_)
            # plt.show()
            net.eval()
            # GPU/CPU
            inputs.to(device)
            labels.to(device)
            # foward
            class_, image_ = net(inputs)
            loss_c_ = loss_c(class_, labels)
            loss_r_ = loss_r(image_, inputs)
            # backward: as features are frozen,
            # no grad will merge in features,
            # use detach if unfreeze feature layers
            opt.zero_grad()
            loss_c_.backward()
            loss_r_.backward()
            opt.step()
            running_loss_c += loss_c_.item()
            running_loss_r += loss_r_.item()
            # running_loss += loss.item()
            every_n_batch = 10
            if not (i+1) % every_n_batch:
                print('[{}, {}] loss_c={:.5f} loss_r={:.5f}'.format(
                    epoch + 1,
                    i + 1,
                    running_loss_c / every_n_batch,
                    running_loss_r / every_n_batch))
                running_loss_c = 0.0
                running_loss_r = 0.0
        # check training accuracy
        correct = 0
        total = 0
        net.eval()
        for i, sample_batch in enumerate(train_dataloader):
            inputs = sample_batch[0]
            labels = sample_batch[1]
            class_, image_ = net(inputs)
            # im_1 = tensor2array(inputs[0])
            # im_2 = np.maximum(np.minimum(1.0, tensor2array(image_[0])), 0.0)
            # im_ = np.concatenate([im_1, im_2], axis=2)
            # im_ = im_.transpose([1, 2, 0])
            # plt.title(['cherry', 'strawberry'][labels[0].item()])
            # plt.imshow(im_)
            # plt.show()
            class_ = torch.softmax(class_, 1, torch.float32)
            # print('class_=%s' % tensor2array(class_))
            _, prediction = torch.max(class_, 1)
            # print('pred=%s, gt=%s' % (prediction, labels[0].item()))
            correct += (torch.sum((prediction == labels))).item()
            total += labels.size(0)
        print('#{:6d} train accuracy={:.5f}'.format(epoch+1, 1.0*correct/total))

        print('saving models...')
        torch.save(net.state_dict(), '../Models/ClassifierEstimator/cherry-strawberry.pth')
        print('models saved at epoch #%d' % (epoch+1))
    print('training finished!')


def test():
    net = TrustedMobileNetV2(pretrained=False)
    net.load_params('../Models/ClassifierEstimator/cherry-strawberry.pth')
    # print(net.eval())
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print('test_device:{}'.format(device.type))
    net.to(device)
    loss_r = nn.L1Loss(reduction='mean')

    # setup dataset
    test_dataset = load_dataset('../Datasets/ClassifierEstimator/test/')
    test_dataloader = DataLoader(
        dataset=test_dataset,
        batch_size=1,
        shuffle=False)
    # test procedure
    loss_r_pos = []
    loss_r_neg = []
    for i, sample_batch in enumerate(test_dataloader):
        inputs = sample_batch[0]
        labels = sample_batch[1]
        net.eval()
        # GPU/CPU
        inputs.to(device)
        labels.to(device)
        # foward
        class_, image_ = net(inputs)
        class_ = torch.softmax(class_, 1, torch.float32)
        loss_r_ = loss_r(image_, inputs)
        _, prediction = torch.max(class_, 1)
        if prediction[0].item() == labels[0].item():
            loss_r_pos.append(loss_r_.item())
        else:
            loss_r_neg.append(loss_r_.item())
        im_1 = tensor2array(inputs[0])
        im_2 = np.maximum(np.minimum(1.0, tensor2array(image_[0])), 0.0)
        im_ = np.concatenate([im_1, im_2], axis=2)
        im_ = im_.transpose([1, 2, 0])
        plt.clf()
        plt.title('标签:%s 预测:%s' %
                  (['樱桃', '草莓'][labels[0].item()],
                   ["错误", "正确"][int(prediction[0].item() == labels[0].item())]))
        plt.imshow(im_)
        plt.pause(0.01)
    loss_r_pos = np.array(loss_r_pos)
    loss_r_neg = np.array(loss_r_neg)
    total = loss_r_pos.shape[0] + loss_r_neg.shape[0]
    print('test accuracy: %6.3f' % (loss_r_pos.shape[0]*1.0 / total))
    print('rec error on positive: %6.3f+/-%6.3f' % (loss_r_pos.mean(), loss_r_pos.std()))
    print('rec error on negative: %6.3f+/-%6.3f' % (loss_r_neg.mean(), loss_r_neg.std()))


def predict():
    net = TrustedMobileNetV2(pretrained=False)
    net.load_params('../Models/ClassifierEstimator/cherry-strawberry.pth')
    # print(net.eval())
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print('test_device:{}'.format(device.type))
    net.to(device)
    loss_r = nn.L1Loss(reduction='mean')

    # setup dataset
    test_dataset = load_dataset('../Datasets/ClassifierEstimator/predict/')
    test_dataloader = DataLoader(
        dataset=test_dataset,
        batch_size=1,
        shuffle=False)
    # test procedure
    classes = test_dataset.classes
    losses = [list() for i in range(len(test_dataset.classes))]
    for i, sample_batch in enumerate(test_dataloader):
        inputs = sample_batch[0]
        labels = sample_batch[1]
        net.eval()
        # GPU/CPU
        inputs.to(device)
        labels.to(device)
        # foward
        _, image_ = net(inputs)
        loss_r_ = loss_r(image_, inputs)
        losses[labels[0].item()].append(loss_r_.item())
        im_1 = tensor2array(inputs[0])
        im_2 = np.maximum(np.minimum(1.0, tensor2array(image_[0])), 0.0)
        im_ = np.concatenate([im_1, im_2], axis=2)
        im_ = im_.transpose([1, 2, 0])
        plt.clf()
        plt.title('class=%s' % classes[labels[0].item()])
        plt.imshow(im_)
        plt.pause(0.01)
    for i, cid in enumerate(classes):
        loss_ = np.array(losses[i])
        print('rec error on %s: %6.3f+/-%6.3f' % (cid, loss_.mean(), loss_.std()))
    loss_all = np.concatenate(losses)
    print('total rec error: %6.3f+/-%6.3f' % (loss_all.mean(), loss_all.std()))


if __name__ == '__main__':
    #train()
    #test()
    predict()
