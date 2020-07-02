# continual_learning_svhn.py
from collections.abc import Iterable
import numpy as np
import torch
import torchvision
from torch import nn as nn
import imagenet_classes as imagenet
from torch.utils.data import DataLoader, sampler
#import matplotlib.pyplot as plt
#import matplotlib.patches as patches
import h5py
from PIL import Image
import random
import os

#plt.rcParams['font.sans-serif']=['SimHei']
#plt.rcParams['axes.unicode_minus'] = False


os.environ["CUDA_VISIBLE_DEVICES"] = "3"


def make_weights_for_balanced_classes(
        images: list,
        nclasses: int,
        num_split: int,
        split_id: int):
    assert nclasses % num_split == 0
    group_size = nclasses//num_split
    count = [0] * group_size
    for item in images:
        if item[1]//group_size == split_id:
            count[item[1] % group_size] += 1
    weight_per_class = [0.] * group_size
    N = float(sum(count))
    for i in range(group_size):
        weight_per_class[i] = N/float(count[i])
    weight = [0] * len(images)
    for idx, val in enumerate(images):
        if val[1]//group_size == split_id:
            weight[idx] = weight_per_class[val[1] % group_size]
    return weight


def load_svhn(is_train: bool, n_split: int, split_id: int):
    root = '../Datasets/SVHN/' + ['test-svhn', 'train-svhn'][is_train]
    dataset = torchvision.datasets.ImageFolder(
        root=root,
        transform=torchvision.transforms.Compose([
            torchvision.transforms.Resize(224), # w/h ratio = 1:1
            torchvision.transforms.ToTensor()
        ]))
    if is_train:
        w_ = make_weights_for_balanced_classes(dataset.imgs, len(dataset.classes), n_split, split_id)
        data_loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=1,
            sampler=[None, sampler.WeightedRandomSampler(w_, len(w_))][is_train])
    else:
        data_loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=1,
            shuffle=False)

    return data_loader


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


def train_svhn_split(split_id: int, split_class: int):
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
    train_dataloader = load_svhn(True, 10//split_class, split_id)

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
        torch.save(net.state_dict(), '../Models/ClassifierEstimator/SplitSVHN/svhn-split-%d.pth' % split_id)
        print('models saved at epoch #%d' % (epoch + 1))
    print('training finished!')


def test_svhn_split(split_id: int, split_class: int):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print('test_device:{}'.format(device.type))
    net = TrustedMobileNetV2(pretrained=False)
    net.load_params('../Models/ClassifierEstimator/SplitSVHN/svhn-split-%d.pth' % split_id, device)
    net.to(device)
    loss_r = nn.L1Loss(reduction='mean')

    # setup dataset
    test_dataloader = load_svhn(False, 10//split_class, split_id)
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


def predict_svhn_split(all_class: int, split_class: int):
    assert all_class % split_class == 0
    num_split = int(all_class / split_class)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print('test_device:{}'.format(device.type))
    nets = []
    for i in range(num_split):
        nets.append(TrustedMobileNetV2(pretrained=False))
        nets[i].load_params('../Models/ClassifierEstimator/SplitSVHN/svhn-split-%d.pth' % i, device)
        nets[i].to(device)
    loss_r = nn.L1Loss(reduction='mean')

    # setup dataset
    test_dataloader = load_svhn(False, 10//split_class, -1)
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
        prob = np.zeros([num_split], dtype=np.float32)

        for split_id in range(num_split):
            nets[split_id].eval()
            class_, image_ = nets[split_id](inputs)
            loss_r_ = loss_r(image_, inputs)
            pred[split_id] = np.argmax(class_.detach().cpu().numpy()[0]) + split_class * split_id
            prob[split_id] = loss_r_.detach().cpu().numpy()
        correct_num += (pred[np.argmin(prob)] == labels.detach().cpu().numpy()[0])
        all_num += 1
        #print('pred: %d gt: %d' %(pred[np.argmin(prob)], labels.detach().cpu().numpy()[0]))

    print('correct: %d\t total: %d' % (correct_num, all_num))
    print('test accuracy: %6.5f' % (1.0*correct_num / all_num))


def load_hdf5(path, subdir):
    print('process folder : %s' % subdir)
    filenames = []
    dir = os.path.join(path, subdir)
    for filename in os.listdir(dir):
        filenameParts = os.path.splitext(filename)
        if filenameParts[1] != '.png':
            continue
        filenames.append(filenameParts)
    svhnMat = h5py.File(name=os.path.join(dir, 'digitStruct.mat'), mode='r')
    datasets = []
    filecounts = len(filenames)
    for idx, file in enumerate(filenames):
        boxes = {}
        filenameNum = file[0]
        item = svhnMat['digitStruct']['bbox'][int(filenameNum) - 1].item()
        for key in ['label', 'left', 'top', 'width', 'height']:
            attr = svhnMat[item][key]
            values = [svhnMat[attr[()][i].item()][()][0][0]
                      for i in range(len(attr))] if len(attr) > 1 else [attr[()][0][0]]
            boxes[key] = values
        datasets.append({'dir': dir, 'file': file, 'boxes': boxes})
        if idx % 10 == 0: print('-- loading %d / %d' % (idx, filecounts))
    return datasets


def make_svhn_dataset(root:str, is_train:bool):
    #fn = 'svhn-annotation.npy'
    sub_ = ['test', 'train'][int(is_train)]
    svhn = load_hdf5(root, sub_)
    #np.save(fn, svhn)
    #exit(0)
    #svhn = np.load(fn, allow_pickle=True)
    print('SVHN matlab formatted annotations are loaded!')
    print('SVHN %s dataset contains images: %d' % (sub_, len(svhn)))

    #cmap = plt.get_cmap('tab20b')
    #colors = [cmap(i) for i in np.linspace(0, 1, 20)]

    for i in range(len(svhn)):
        im_ = np.array(Image.open('/'.join([svhn[i]['dir'], ''.join(list(svhn[i]['file']))])), np.float32)/255.0
        # plt.clf()
        # ax = plt.axes()
        # ax.imshow(im_)
        assert len(im_.shape) == 3 and im_.shape[2] == 3
        im_w = im_.shape[1]
        im_h = im_.shape[0]
        labels = np.int32([x % 10 for x in svhn[i]['boxes']['label']])
        y1 = np.maximum(np.int32(svhn[i]['boxes']['top']), 0)
        x1 = np.maximum(np.int32(svhn[i]['boxes']['left']), 0)
        box_w = np.minimum(np.int32(svhn[i]['boxes']['width']), im_w - x1 - 1)
        box_h = np.minimum(np.int32(svhn[i]['boxes']['height']), im_h - y1 - 1)
        # bbox_colors = random.sample(colors, labels.shape[0])
        # for ind in range(labels.shape[0]):
        #     bbox = patches.Rectangle((x1[ind], y1[ind]),
        #                              box_w[ind], box_h[ind],
        #                              linewidth=2,
        #                              edgecolor=bbox_colors[ind],
        #                              facecolor='none')
        #     ax.add_patch(bbox)
        #     plt.text(x1[ind], y1[ind],
        #              s=str(labels[ind]),
        #              color='white',
        #              verticalalignment='top',
        #              bbox={'color': bbox_colors[ind], 'pad': 0})
        # plt.pause(1)
        # save images of single figure
        for ind in range(labels.shape[0]):
            patch_ = np.uint8(im_[y1[ind]:y1[ind]+box_h[ind], x1[ind]:x1[ind]+box_w[ind]] * 255)
            # padding for ratio holding 1/1
            if patch_.shape[0] > patch_.shape[1]:
                max_ = patch_.shape[0]
                min_ = patch_.shape[1]
                pad_left = (max_ - min_) // 2
                pad_right = max_ - min_ - pad_left
                patch_new = np.zeros([max_, max_, patch_.shape[2]], np.uint8)
                patch_new[:, pad_left:max_-pad_right, :] = patch_[:, :, :]
            else:
                min_ = patch_.shape[0]
                max_ = patch_.shape[1]
                pad_up = (max_ - min_) // 2
                pad_down = max_ - min_ - pad_up
                patch_new = np.zeros([max_, max_, patch_.shape[2]], np.uint8)
                patch_new[pad_up:max_ - pad_down, :, :] = patch_[:, :, :]

            file_ = list(svhn[i]['file'])
            file_.insert(1, '_%d' % ind)
            file_ = ''.join(file_)
            path_ = '/'.join([svhn[i]['dir'] + '-svhn/%d' % labels[ind], file_])
            Image.fromarray(patch_new).save(path_)
    print('SVHN dataset conversion completed!')
    exit(0)


if __name__ == '__main__':
    #make_svhn_dataset('../Datasets/SVHN/', is_train=False)
    #exit(0)

    total_class = 10
    split_class = 2
    assert total_class % split_class == 0
    # for split_id in range(int(total_class/split_class)):
    #     train_svhn_split(split_id, split_class)
    #train_svhn_split(1, split_class)
    test_svhn_split(1, split_class)
    #predict_svhn_split(total_class, split_class)
