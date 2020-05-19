from PIL import Image
import numpy as np
import torch
import torchvision.models as models
import imagenet_classes as imagenet


def main():
    mobilenet = models.mobilenet_v2(pretrained=True)
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
    x_t = torch.Tensor(np.expand_dims(x, 0))
    y_t = mobilenet(x_t)
    y_t = torch.softmax(y_t, -1)
    y = np.array(y_t.data[0])
    res = np.argmax(y)
    print('predicted class [%s] with confidence of [%6.3f]' % (imagenet.class_names[res], y[res]))


if __name__ == '__main__':
    main()
