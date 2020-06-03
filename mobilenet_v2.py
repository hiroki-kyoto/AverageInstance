from urllib import request

from PIL import Image
import numpy as np
import torch
import torchvision
from requests import models
from tensorflow.python.ops import nn

import imagenet_classes as imagenet
import os
from torch.utils.data import DataLoader


def load_dataset(data_dir):
    dataset = torchvision.datasets.ImageFolder(
        root=os.path.join(data_dir, 'train'),
        transform=torchvision.transforms.Compose([
            torchvision.transforms.Resize(224),
            torchvision.transforms.RandomRotation(180, center=True),
            torchvision.transforms.RandomHorizontalFlip(),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225))
        ]))
    return dataset


class MobileNet(torch.nn.Module):
    def __init__(self, num_classes=685):   # num_classes，此处为 二分类值为2
        super(MobileNet, self).__init__()
        net = models.mobilenet_v2(pretrained=True)   # 从预训练模型加载VGG16网络参数
        net.classifier = nn.Sequential()  # 将分类层置空，下面将改变我们的分类层
        self.features = net  # 保留VGG16的特征层
        self.classifier = nn.Sequential(    # 定义自己的分类层
                nn.Linear(1280, 1000),  #512 * 7 * 7不能改变 ，由VGG16网络决定的，第二个参数为神经元个数可以微调
                nn.ReLU(True),
                nn.Dropout(0.5),
#                 nn.Linear(1024, 1024),
#                 nn.ReLU(True),
#                 nn.Dropout(0.3),
                nn.Linear(1000, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

def main():
    mobilenet = torchvision.models.mobilenet_v2(pretrained=True)
    num_fc_in = mobilenet.fc.in_features
    mobilenet.fc = torch.nn.Linear(num_fc_in, 2)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print('trian_device:{}'.format(device.type))
    mobilenet = mobilenet.to(device)
    loss_func = torch.nn.CrossEntropyLoss()
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
            # if i % 20 == 19:
            #     correct = 0
            #     total = 0
            #     model.eval()
            #     for images_test, labels_test in val_dataloader:
            #         images_test = images_test.to(device)
            #         labels_test = labels_test.to(device)
            #
            #         outputs_test = model(images_test)
            #         _, prediction = torch.max(outputs_test, 1)
            #         correct += (torch.sum((prediction == labels_test))).item()
            #         # print(prediction, labels_test, correct)
            #         total += labels_test.size(0)
            #     print('[{}, {}] running_loss = {:.5f} accurcay = {:.5f}'.format(epoch + 1, i + 1, running_loss / 20,
            #                                                                     correct / total))
            #     running_loss = 0.0

            # if i % 10 == 9:
            #     print('[{}, {}] loss={:.5f}'.format(epoch+1, i+1, running_loss / 10))
            #     running_loss = 0.0

        print('saving models...')
        torch.save(mobilenet.state_dict(), '../Models/cherry-strawberry.pth')
    print('training finished !')

    # test settings
    # mobilenet.eval()
    # im = Image.open("./demo/test2.png")
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
    # x_t = torch.Tensor(np.expand_dims(x, 0))
    # y_t = mobilenet(x_t)
    # y_t = torch.softmax(y_t, -1)
    # y = np.array(y_t.data[0])
    # res = np.argmax(y)
    # print('predicted class [%s] with confidence of [%6.3f]' % (imagenet.class_names[res], y[res]))


if __name__ == '__main__':
    main()

# sample code from [https://www.jianshu.com/p/d04c17368922]