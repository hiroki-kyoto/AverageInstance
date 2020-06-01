from PIL import Image
import numpy as np
import torch
import torchvision
import imagenet_classes as imagenet
import os


def load_dataset(data_dir):
    dataset = torchvision.datasets.ImageFolder(root=os.path.join(data_dir, 'train'),
                                                     transform=torchvision.transforms.Compose(
                                                         [
                                                             torchvision.transforms.Resize(224),
                                                             torchvision.transforms.RandomRotation(180, center=True),
                                                             torchvision.transforms.RandomHorizontalFlip(),
                                                             torchvision.transforms.ToTensor(),
                                                             torchvision.transforms.Normalize(
                                                                 mean=(0.485, 0.456, 0.406),
                                                                 std=(0.229, 0.224, 0.225))
                                                         ]))
    return dataset


def main():
    mobilenet = torchvision.models.mobilenet_v2(pretrained=True)
    num_fc_in = mobilenet.fc.in_features
    mobilenet.fc = torch.nn.Linear(num_fc_in, 2)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print('trian_device:{}'.format(device.type))
    mobilenet = mobilenet.to(device)
    loss = torch.nn.CrossEntropyLoss()
    opt = torch.optim.Adam(mobilenet.parameters(), lr=1E-4)

    # setup dataset
    train_dataset = load_dataset('../Datasets/ClassifierEstimator/train/')
    train_dataloader = torchvision.DataLoader(dataset=train_dataset, batch_size=1, shuffle=1) #### the real shuffle?
    #[https://www.cnblogs.com/LOSKI/p/11815670.html]

    # training procedure
    num_epochs = 500
    for epoch in range(num_epochs):
        running_loss = 0.0
        for i, sample_batch in enumerate(train_dataloader):
            inputs = sample_batch[0]
            labels = sample_batch[1]

            model.train()

            # GPU/CPU
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            # foward
            outputs = model(inputs)

            # loss
            loss = loss_fc(outputs, labels)

            # loss求导，反向
            loss.backward()

            # 优化
            optimizer.step()

            #
            running_loss += loss.item()

            # 測試
            if i % 20 == 19:
                correct = 0
                total = 0
                model.eval()
                for images_test, labels_test in val_dataloader:
                    images_test = images_test.to(device)
                    labels_test = labels_test.to(device)

                    outputs_test = model(images_test)
                    _, prediction = torch.max(outputs_test, 1)
                    correct += (torch.sum((prediction == labels_test))).item()
                    # print(prediction, labels_test, correct)
                    total += labels_test.size(0)
                print('[{}, {}] running_loss = {:.5f} accurcay = {:.5f}'.format(epoch + 1, i + 1, running_loss / 20,
                                                                                correct / total))
                running_loss = 0.0

            # if i % 10 == 9:
            #     print('[{}, {}] loss={:.5f}'.format(epoch+1, i+1, running_loss / 10))
            #     running_loss = 0.0

    print('training finish !')
    torch.save(model.state_dict(), './model/model_2.pth')
    

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