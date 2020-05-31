from PIL import Image
import numpy as np
import torch
import torchvision.models as models
import imagenet_classes as imagenet


def main():
    mobilenet = models.mobilenet_v2(pretrained=True)
    #mobilenet.eval()
    num_fc_in = mobilenet.fc.in_features
    mobilenet.fc = torch.nn.Linear(num_fc_in, 2)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print('trian_device:{}'.format(device.type))
    mobilenet = mobilenet.to(device)
    loss = torch.nn.CrossEntropyLoss()
    opt = torch.optim.Adam(mobilenet.parameters(), lr=0.0001)

    # training procedure
    num_epochs = 500
    

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