# mobilenet_v2.py
from collections.abc import Iterable
import numpy as np
import torch
import torchvision
from torch import nn as nn
import imagenet_classes as imagenet
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "3"

plt.rcParams['font.sans-serif']=['SimHei']
plt.rcParams['axes.unicode_minus'] = False


def load_mnist(is_train: bool):
    root = '../Datasets/'
    data_loader = torch.utils.data.DataLoader(
        torchvision.datasets.MNIST(
            root,
            train=is_train,
            download=True,
            transform=torchvision.transforms.Compose([
                torchvision.transforms.Resize(224),
                torchvision.transforms.ToTensor()
            ])))
    return data_loader


def load_dataset(data_dir):
    dataset = torchvision.datasets.ImageFolder(
        root=data_dir,
        transform=torchvision.transforms.Compose([
            torchvision.transforms.Resize((224, 224)),
            torchvision.transforms.ToTensor()
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

    def load_params(self, model_path, device):
        state_dict = torch.load(model_path, map_location=device)
        state_dict_new = {}
        for name_ in state_dict:
            data = tensor2array(state_dict[name_])
            print("%s\t%s" % (name_, data.shape))
            name_new = name_[len('mobilenet.'):]
            state_dict_new[name_new] = state_dict[name_]
        self.mobilenet.load_state_dict(state_dict_new)


def train_mnist_split(split_id: int, split_class: int):
    """
    train independent MobileNet-v2 based classifier on training split,
    every 2 figure as a group to discriminate
    :param split_class: number of category in classification
    :param split_id: the split index in 5 groups of figures in [0, 9]
            i.e., 0 falls in split#0{0,1}, 5 falls in split#2{4,5}.
            the split index can be formulated as floor(n/2)
    :return: null
    """
    # build model
    assert split_class == 2
    net = TrustedMobileNetV2()
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print('train_device:{}'.format(device.type))
    net.to(device)
    loss_c = nn.CrossEntropyLoss(reduction='mean')
    loss_r = nn.L1Loss(reduction='mean')
    opt = torch.optim.Adam(net.parameters(), lr=1E-4)
    train_dataloader = load_mnist(True)

    # training procedure
    num_epochs = 500
    for epoch in range(num_epochs):
        running_loss_c = 0
        running_loss_r = 0
        for i, sample_batch in enumerate(train_dataloader):
            inputs = sample_batch[0]
            labels = sample_batch[1]
            if np.floor(labels.detach().numpy()[0] / 2) != split_id:
                continue
            if inputs.shape[1] == 1:
                inputs = torch.cat([inputs, inputs, inputs], dim=1)
            # print(labels.detach().numpy())
            labels = torch.from_numpy(labels.detach().numpy() - split_id * split_class)
            # print(labels.detach().numpy())
            # print(inputs.shape)
            # im_ = tensor2array(inputs)[0]
            # im_ = im_.transpose([1, 2, 0])
            # plt.title(str(labels.detach().numpy()[0]))
            # plt.imshow(im_)
            # plt.show()

            net.eval()  # very important!!! As this disables batch norm in mobilenet-v2
            inputs = inputs.to(device)
            labels = labels.to(device)
            # foward
            class_, image_ = net(inputs)
            loss_c_ = loss_c(class_, labels)
            loss_r_ = loss_r(image_, inputs)
            # backward: as features are frozen,
            # no grad will flow back into features
            opt.zero_grad()
            loss_c_.backward()
            loss_r_.backward()
            opt.step()
            running_loss_c += loss_c_.item()
            running_loss_r += loss_r_.item()
            every_n_batch = 10
            if not (i + 1) % every_n_batch:
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
        sample_rate = 0.01
        for i, sample_batch in enumerate(train_dataloader):
            inputs = sample_batch[0]
            labels = sample_batch[1]
            if np.floor(labels.detach().numpy()[0] / 2) != split_id:
                continue
            if np.random.rand() > sample_rate:
                continue
            if inputs.shape[1] == 1:
                inputs = torch.cat([inputs, inputs, inputs], dim=1)
            # print(labels.detach().numpy())
            labels = torch.from_numpy(labels.detach().numpy() - split_id * split_class)
            # print(labels.detach().numpy())
            # print(inputs.shape)
            # im_ = tensor2array(inputs)[0]
            # im_ = im_.transpose([1, 2, 0])
            # plt.title(str(labels.detach().numpy()[0]))
            # plt.imshow(im_)
            # plt.show()
            inputs = inputs.to(device)
            labels = labels.to(device)
            class_, image_ = net(inputs)
            class_ = torch.softmax(class_, 1, torch.float32)
            _, prediction = torch.max(class_, 1)
            correct += (torch.sum((prediction == labels))).item()
            total += labels.size(0)
            print('*', end='')
        print('#{:6d} train accuracy={:.5f}'.format(epoch + 1, 1.0 * correct / total))
        # save models
        print('saving models...')
        torch.save(net.state_dict(), '../Models/ClassifierEstimator/SplitMNIST/mnist-split-%d.pth' % split_id)
        print('models saved at epoch #%d' % (epoch + 1))
    print('training finished!')


def test_mnist_split(split_id: int, split_class: int):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print('test_device:{}'.format(device.type))
    net = TrustedMobileNetV2(pretrained=False)
    net.load_params('../Models/ClassifierEstimator/SplitMNIST/mnist-split-%d.pth' % split_id, device)
    net.to(device)
    loss_r = nn.L1Loss(reduction='mean')

    # setup dataset
    test_dataloader = load_mnist(is_train=False)
    # test procedure
    loss_r_pos = []
    loss_r_neg = []
    for i, sample_batch in enumerate(test_dataloader):
        inputs = sample_batch[0]
        labels = sample_batch[1]
        inputs = sample_batch[0]
        labels = sample_batch[1]
        if np.floor(labels.detach().numpy()[0] / 2) != split_id:
            continue
        if inputs.shape[1] == 1:
            inputs = torch.cat([inputs, inputs, inputs], dim=1)
        labels = torch.from_numpy(labels.detach().numpy() - split_id * split_class)

        # print(labels.detach().numpy())
        # print(inputs.shape)
        # im_ = tensor2array(inputs)[0]
        # im_ = im_.transpose([1, 2, 0])
        # plt.title(str(labels.detach().numpy()[0]))
        # plt.imshow(im_)
        # plt.show()

        # foward
        inputs = inputs.to(device)
        labels = labels.to(device)
        net.eval()
        class_, image_ = net(inputs)
        class_ = torch.softmax(class_, 1, torch.float32)
        loss_r_ = loss_r(image_, inputs)
        _, prediction = torch.max(class_, 1)
        if prediction[0].item() == labels[0].item():
            loss_r_pos.append(loss_r_.item())
        else:
            loss_r_neg.append(loss_r_.item())
        #im_1 = tensor2array(inputs[0])
        #im_2 = np.maximum(np.minimum(1.0, tensor2array(image_[0])), 0.0)
        #im_ = np.concatenate([im_1, im_2], axis=2)
        #im_ = im_.transpose([1, 2, 0])
        #plt.clf()
        #plt.title('GT:%d    %s' %
        #          (labels.numpy()[0] + split_id*split_class,
        #           ["WRONG", "CORRECT"][int(prediction[0].item() == labels[0].item())]))
        #plt.imshow(im_)
        #plt.pause(0.01)
    loss_r_pos = np.array(loss_r_pos)
    loss_r_neg = np.array(loss_r_neg)
    total = loss_r_pos.shape[0] + loss_r_neg.shape[0]
    print('correct: %d\t total: %d' % (loss_r_pos.shape[0], total))
    print('test accuracy: %6.5f' % (loss_r_pos.shape[0]*1.0 / total))
    if loss_r_pos.shape[0] > 0:
        print('rec error on positive: %6.5f +/- %6.5f' % (loss_r_pos.mean(), loss_r_pos.std()))
    if loss_r_neg.shape[0] > 0:
        print('rec error on negative: %6.5f +/- %6.5f' % (loss_r_neg.mean(), loss_r_neg.std()))


def predict_mnist_split(all_class: int, split_class: int):
    assert all_class % split_class == 0
    num_split = int(all_class / split_class)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print('test_device:{}'.format(device.type))
    nets = []
    for i in range(num_split):
        nets.append(TrustedMobileNetV2(pretrained=False))
        nets[i].load_params('../Models/ClassifierEstimator/SplitMNIST/mnist-split-%d.pth' % i, device)
        nets[i].to(device)
    loss_r = nn.L1Loss(reduction='mean')

    # setup dataset
    test_dataloader = load_mnist(is_train=False)
    # test procedure
    correct_num = 0
    all_num = 0
    for i, sample_batch in enumerate(test_dataloader):
        inputs = sample_batch[0]
        labels = sample_batch[1]
        if inputs.shape[1] == 1:
            inputs = torch.cat([inputs, inputs, inputs], dim=1)
        # foward
        inputs = inputs.to(device)
        #labels = labels.to(device)

        pred = np.zeros([num_split], dtype=np.int32)
        err_ = np.zeros([num_split], dtype=np.float32)
        #prob = np.zeros([num_split], dtype=np.float32)
        #imgs = np.zeros([num_split, 3, inputs.shape[2], inputs.shape[3]], dtype=np.float32)

        for split_id in range(num_split):
            nets[split_id].eval()
            class_, image_ = nets[split_id](inputs)
            loss_r_ = loss_r(image_, inputs)
            class_ = torch.softmax(class_, -1)
            pred[split_id] = np.argmax(class_.detach().cpu().numpy()[0]) + split_class * split_id
            err_[split_id] = loss_r_.detach().cpu().numpy()
            #prob[split_id] = np.max(class_.detach().cpu().numpy()[0])
            #imgs[split_id] = np.maximum(np.minimum(image_.detach().cpu().numpy()[0], 1.0), 0.0)
        correct_num += (pred[np.argmin(err_)] == labels.detach().cpu().numpy()[0])
        #correct_num += (pred[np.argmax(prob)] == labels.detach().cpu().numpy()[0])
        all_num += 1
        print('%d / %d' % (correct_num, all_num))
        # if not pred[np.argmin(err_)] == labels.detach().cpu().numpy()[0]:
        #     print('pred: %d gt: %d' % (pred[np.argmin(err_)], labels.detach().cpu().numpy()[0]))
        #     im_gt = tensor2array(inputs[0])
        #     im_rc = imgs[np.argmin(err_)]
        #     im_ep = imgs[labels.detach().cpu().numpy()[0]//split_class]
        #     im_ = np.concatenate([im_gt, im_rc, im_ep], axis=2)
        #     im_ = im_.transpose([1, 2, 0])
        #     plt.imshow(im_)
        #     plt.title('L-M-R: Groundtruth-Predicted-Expected')
        #     plt.show()
    print('correct: %d\t total: %d' % (correct_num, all_num))
    print('test accuracy: %6.5f' % (1.0*correct_num / all_num))


if __name__ == '__main__':
    total_class = 10
    split_class = 2
    assert total_class % split_class == 0
    # for split_id in range(int(total_class/split_class)):
    #     train_mnist_split(split_id, split_class)
    #train_mnist_split(2, split_class)
    #test_mnist_split(1, split_class)
    predict_mnist_split(total_class, split_class)
