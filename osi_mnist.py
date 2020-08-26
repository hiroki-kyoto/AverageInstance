# osi_mnist.py
from collections.abc import Iterable
import numpy as np
import torchvision
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from torch_ops import *
from PIL import Image

plt.rcParams['font.sans-serif']=['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# gpu and batch size are set up here
args = process_args()


# Requirement: 
# As resize/crop all other transform ops are done on CPUs, 
# they left the GPUs SLEEP!!! This is not suggested.
# Prepare your training data in advance, such as 
#    cropping, padding, resizing
#    normalization can be done later on GPUs, 
#    never, ever do these on CPUs!
def load_mnist(is_train: bool, batch_size: int):
    root = '../Datasets/'
    data_loader = torch.utils.data.DataLoader(
        torchvision.datasets.MNIST(
            root,
            train=is_train,
            download=True,
            transform=torchvision.transforms.ToTensor()),
        batch_size=batch_size)
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
        self.load_state_dict(state_dict)


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
    encoder = encoder.to(device)
    decoder = decoder.to(device)

    loss_r = nn.L1Loss(reduction='mean')
    opt_enc = torch.optim.Adam(encoder.parameters(), lr=1E-4)
    opt_dec = torch.optim.Adam(decoder.parameters(), lr=1E-4)

    train_dataloader = load_mnist(True, args.batch)

    # training procedure
    encoder.train()
    decoder.train()
    num_epochs = 10000

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
            every_n_batch = 100
            if not (i + 1) % every_n_batch:
                print('[{}, {}] loss_r={:.5f}'.format(
                    epoch + 1,
                    i + 1,
                    running_loss_r / every_n_batch))
                running_loss_r = 0.0
        # save models
        print('saving models...')
        model_dir = '../Models/ClassifierEstimator/osi-mnist'
        torch.save(encoder.state_dict(), model_dir + '/mnist_encoder.pth')
        torch.save(decoder.state_dict(), model_dir + '/mnist_decoder.pth')
        print('models saved at epoch #%d' % (epoch + 1))
    print('training finished!')


def MNIST_TestAutoEncoder():
    """
    test an pretrained auto-encoder on MNIST dataset
    """
    # build model
    encoder = MNIST_Encoder()
    decoder = MNIST_Decoder()

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print('test_device:{}'.format(device.type))
    encoder = encoder.to(device)
    decoder = decoder.to(device)

    # restore parameters
    model_dir = '../Models/ClassifierEstimator/osi-mnist/'
    encoder.load_params(model_dir + '/mnist_encoder.pth', device)
    decoder.load_params(model_dir + '/mnist_decoder.pth', device)

    loss_r = nn.L1Loss(reduction='mean')
    test_dataloader = load_mnist(False, 1)

    encoder.eval()
    decoder.eval()
    all_loss_r = np.zeros([len(test_dataloader.dataset)])
    print(all_loss_r.shape)

    for i, sample_batch in enumerate(test_dataloader):
        x = sample_batch[0]
        # # add noise to input
        # noise = torch.Tensor(np.random.normal(0.0, 0.3, x.shape))
        # noise = noise * (1-x)
        # x = x + noise

        # # add a unknown pattern
        # bg_ = Image.open('test/osi-mnist/demo.jpg', 'r')
        # bg_ = np.array(bg_, dtype=np.float32) / 255
        # bg_ = torch.Tensor(bg_)
        # x = torch.min(x + bg_,
        #               torch.Tensor(np.ones(bg_.shape, dtype=np.float32)))

        # foward
        x = x.to(device)
        z = encoder(x)
        x_r = decoder(z)

        # # visualize
        # im_ = tensor2array(x)[0][0]
        # im_r = tensor2array(x_r)[0][0]
        # im_ = np.concatenate([im_, im_r], axis=1)
        # plt.imshow(im_)
        # plt.show()

        loss_r_ = loss_r(x_r, x)
        all_loss_r[i] = loss_r_.item()
    # show overall information
    print('test error: %6.5f +/- %6.5f' % (all_loss_r.mean() , all_loss_r.std()))
    # draw the error histogram


def MNIST_SaveTrainingLatentCodes(path: str):
    """
    prepare training dataset for the restricted Auto Encoder
    :param path: the path of training dataset to save
    :return: nul
    """
    # build model
    encoder = MNIST_Encoder()

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print('test_device:{}'.format(device.type))
    encoder.to(device)

    # restore parameters
    model_dir = '../Models/ClassifierEstimator/osi-mnist'
    encoder.load_params(model_dir + '/mnist_encoder.pth', device)

    batch_size = 16
    test_dataloader = load_mnist(True, batch_size)

    # training procedure
    encoder.eval()
    latents = np.zeros([len(test_dataloader.dataset), 8], dtype=np.float32)
    labels = np.zeros([latents.shape[0]], dtype=np.int32)
    for i, sample_batch in enumerate(test_dataloader):
        x = sample_batch[0]
        y = sample_batch[1]
        # foward
        x = x.to(device)
        z = encoder(x)
        # append to the dataset
        offset = batch_size * i
        latents[offset:offset+batch_size, :] = z.cpu().detach().numpy()
        labels[offset:offset+batch_size] = y.cpu().detach().numpy()
    # save the numpy array
    np.save(path, {'latents':latents, 'labels':labels})


# normalized clustering loss
# L2 instance distance to center / mean L2 distance to center
def NormalizedClusterLoss(x, center, radius):
    x = x - center
    return torch.mean(torch.sum(x*x, dim=1))


def MNIST_TrainRestrictedAutoEncoder(data_path: str):
    """
    train a restricted auto encoder on MNIST Latent vectors
    """
    # build model
    encoder = MNIST_LatentEncoder()
    decoder = MNIST_LatentEncoder()

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print('train_device:{}'.format(device.type))
    encoder = encoder.to(device)
    decoder = decoder.to(device)

    opt_enc = torch.optim.Adam(encoder.parameters(), lr=1E-4)
    opt_dec = torch.optim.Adam(decoder.parameters(), lr=1E-4)

    # load latent codes
    dataset = np.load(data_path, allow_pickle=True)
    latents = dataset.item()['latents']
    labels = dataset.item()['labels']
    latents = torch.Tensor(latents).to(device)

    # training procedure
    encoder.train()
    decoder.train()
    num_epochs = 10000
    num_samples = labels.shape[0]
    save_n_epoch = 10

    # explicit memory
    n_class = 10
    latent_dim = 8
    batch_size = 32
    cluster_centers = np.zeros([n_class, latent_dim], dtype=np.float32)
    cluster_radius = np.zeros([n_class], dtype=np.float32)
    centers = np.zeros([num_samples, latent_dim], dtype=np.float32)

    for epoch in range(num_epochs):
        print('distribution evaluation')
        for i in range(num_samples//batch_size):
            idx = i*batch_size
            m = encoder(latents[idx:idx+batch_size, :])
            m = m.cpu().detach().numpy()
            centers[idx:idx+batch_size, :] = m[:, :]
        for i in range(n_class):
            mask = np.expand_dims(labels == i, axis=-1)
            masked_centers = centers * mask
            cluster_centers[i, :] = np.sum(masked_centers, axis=0) / np.sum(mask)
            radius_ = np.sqrt(np.sum(np.square(masked_centers - cluster_centers[i]), axis=-1))
            cluster_radius[i] = np.sum(mask.squeeze(1) * radius_) / np.sum(mask)

        print('explicit memory loss: %6.3f +/- %6.3f' % (cluster_radius.mean(), cluster_radius.std()))

        # training
        running_loss = 0.0
        running_loss_c = 0.0
        running_loss_r = 0.0
        every_n_batch = 100

        seq = np.random.permutation(num_samples)
        latents_perm = latents[seq]
        labels_perm = labels[seq]
        centers_perm = torch.Tensor(cluster_centers[labels_perm])
        radius_perm = torch.Tensor(cluster_radius[labels_perm])
        centers_perm = centers_perm.to(device)
        radius_perm = radius_perm.to(device)

        for i in range(num_samples // batch_size):
            # forward
            idx = i * batch_size
            z = latents_perm[idx:idx+batch_size, :]
            m = encoder(z)
            loss_c = NormalizedClusterLoss(
                m, centers_perm[idx:idx+batch_size], radius_perm[idx:idx+batch_size])
            z_r = decoder(m)
            loss_r = z_r - z
            loss_r = torch.mean(torch.sum(loss_r*loss_r, dim=1))
            loss = loss_c + loss_r

            # backward
            opt_enc.zero_grad()
            opt_dec.zero_grad()
            loss.backward()
            opt_enc.step()
            opt_dec.step()
            running_loss += loss.item()
            running_loss_c += loss_c.item()
            running_loss_r += loss_r.item()

            if (i + 1) % every_n_batch == 0:
                print('[{}, {}] loss={:.5f} loss_c={:.5f} loss_r={:.5f}'.format(
                    epoch + 1,
                    i + 1,
                    running_loss / every_n_batch,
                    running_loss_c / every_n_batch,
                    running_loss_r / every_n_batch
                ))
                running_loss = 0.0
                running_loss_c = 0.0
                running_loss_r = 0.0
        # save models
        if (epoch + 1) % save_n_epoch == 0:
            print('saving models...')
            model_dir = '../Models/ClassifierEstimator/osi-mnist'
            torch.save(encoder.state_dict(), model_dir + '/mnist_latent_encoder.pth')
            torch.save(decoder.state_dict(), model_dir + '/mnist_latent_decoder.pth')
            print('models saved at epoch #%d' % (epoch + 1))

    print('training finished!')


def MNIST_TestRestrictedAutoEncoder():
    """
    test a restricted auto encoder on MNIST test dataset
    """
    # build model
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print('train_device:{}'.format(device.type))
    encoder = MNIST_Encoder().to(device)
    decoder = MNIST_Decoder().to(device)
    latent_encoder = MNIST_LatentEncoder().to(device)
    latent_decoder = MNIST_LatentEncoder().to(device)

    # restore parameters
    model_dir = '../Models/ClassifierEstimator/osi-mnist'
    encoder.load_params(model_dir + '/mnist_encoder.pth', device)
    decoder.load_params(model_dir + '/mnist_decoder.pth', device)
    latent_encoder.load_params(model_dir + '/mnist_latent_encoder.pth', device)
    latent_decoder.load_params(model_dir + '/mnist_latent_decoder.pth', device)

    # training procedure
    encoder.eval()
    decoder.eval()
    latent_encoder.eval()
    latent_decoder.eval()

    # explicit memory
    n_class = 10
    latent_dim = 8

    batch_size = 16
    dataset = load_mnist(is_train=True, batch_size=batch_size)
    num_samples = len(dataset.dataset)

    cluster_centers = np.zeros([n_class, latent_dim], dtype=np.float32)
    #cluster_radius = np.zeros([n_class], dtype=np.float32)
    cluster_radius = np.zeros([n_class, latent_dim], dtype=np.float32)
    latents = np.zeros([num_samples, latent_dim], dtype=np.float32)
    labels = np.zeros([num_samples], dtype=np.int32)

    print('distribution evaluation')
    for i, batch in enumerate(dataset):
        idx = i*batch_size
        x = batch[0].to(device)
        labels[idx:idx+batch_size] = batch[1].cpu().detach().numpy()
        z = encoder(x)
        m = latent_encoder(z)
        m = m.cpu().detach().numpy()
        latents[idx:idx+batch_size, :] = m[:, :]
    for i in range(n_class):
        mask = np.expand_dims(labels==i, axis=-1)
        masked_latents = latents * mask
        cluster_centers[i, :] = np.sum(masked_latents, axis=0) / np.sum(mask)
        delta = (latents - cluster_centers[i]) * mask
        cluster_radius[i, :] = np.sqrt(np.sum(np.square(delta), axis=0) / np.sum(mask))

        radius_ = np.sqrt(np.sum(np.square(latents - cluster_centers[i]), axis=-1))
        radius_ = np.sum(mask.squeeze(1) * radius_) / np.sum(mask)
        print("%6.5f =?= %6.5f" % (np.sqrt(np.sum(np.square(cluster_radius[i]))), radius_))

    #print(cluster_centers)
    #print(cluster_radius)
    #print('explicit memory loss: %6.5f +/- %6.5f' % (cluster_radius.mean(), cluster_radius.std()))


    # test the generator(explicit memory)
    for ii in range(n_class):
        for _ in np.arange(10):
            w = cluster_radius[ii]
            noise = np.random.rand(8) - 0.5
            noise_norm = np.sqrt(np.sum(np.square(noise)))
            noise = noise / noise_norm
            sample = cluster_centers[ii] + w * noise
            m_c = torch.Tensor(sample).to(device)
            m_c = torch.unsqueeze(m_c, dim=0)
            z_c = latent_decoder(m_c)
            x_c = decoder(z_c)
            im_ = x_c.cpu().detach().numpy()[0]
            plt.imshow(im_[0])
            plt.show()
    exit(0)

    '''
    # test the latent auto encoder
    batch_size = 16
    dataset = load_mnist(is_train=False, batch_size=batch_size)
    num_samples = len(dataset.dataset)

    running_loss = 0.0
    running_loss_c = 0.0
    running_loss_r = 0.0

    for i, batch in enumerate(dataset):
        # forward
        x = batch[0].to(device)
        y = batch[1].cpu().detach().numpy()
        z = encoder(x)
        m = latent_encoder(z)
        centers = torch.Tensor(cluster_centers[y]).to(device)
        radius = torch.Tensor(cluster_radius[y]).to(device)
        loss_c = NormalizedClusterLoss(m, centers, radius)
        z_r = latent_decoder(m)

        # # test auto encoder
        # x_r2 = decoder(z_r)
        # x_r1 = decoder(z)
        # im_ = np.concatenate(
        #     (x.cpu().detach().numpy()[0],
        #      x_r1.cpu().detach().numpy()[0],
        #      x_r2.cpu().detach().numpy()[0]), axis=1)
        # plt.imshow(im_[0])
        # plt.show()

        loss_r = z_r - z
        loss_r = torch.mean(torch.sum(loss_r*loss_r, dim=1))
        loss = loss_c + loss_r

        running_loss += loss.item()
        running_loss_c += loss_c.item()
        running_loss_r += loss_r.item()
    print('loss=%6.5f loss_c=%6.5f loss_r=%6.5f' %
          (running_loss/(num_samples/batch_size),
           running_loss_c/(num_samples/batch_size),
           running_loss_r/(num_samples/batch_size)))
    '''


    # test the classifier (explicit memory)
    batch_size = 16 # 16
    dataset = load_mnist(is_train=False, batch_size=batch_size)
    num_samples = len(dataset.dataset)

    n_corr = 0
    for i, batch in enumerate(dataset):
        # forward
        x = batch[0].to(device)
        y = batch[1].cpu().detach().numpy()
        z = encoder(x)
        m = latent_encoder(z)

        dis = np.stack(
            [np.sqrt(
                np.sum(
                    np.square(
                        m.cpu().detach().numpy() - cluster_centers[ii]
                    ), axis=-1
                )
            ) for ii in range(n_class)],
            axis=0
        )
        ids = np.argmin(dis, axis=0)
        assert len(ids)==len(y)
        n_corr += np.sum(ids == y)

        # # debug the error prediction
        # if ids[0]!=y[0]:
        #     x_r1 = decoder(latent_decoder(m))
        #     m2 = latent_encoder(encoder(x_r1))
        #     x_r2 = decoder(latent_decoder(m2))
        #     m3 = latent_encoder(encoder(x_r2))
        #     x_r3 = decoder(latent_decoder(m3))
        #
        #     dis = np.stack(
        #         [np.sqrt(
        #             np.sum(
        #                 np.square(
        #                     m3.cpu().detach().numpy() - cluster_centers[ii]
        #                 ), axis=-1
        #             )
        #         ) for ii in range(n_class)],
        #         axis=0
        #     )
        #     ids2 = np.argmin(dis, axis=0)
        #
        #     im_ = np.concatenate(
        #         (x.cpu().detach().numpy()[0],
        #          x_r1.cpu().detach().numpy()[0],
        #          x_r2.cpu().detach().numpy()[0],
        #          x_r3.cpu().detach().numpy()[0]), axis=-1)
        #     plt.imshow(im_[0])
        #     plt.title("Pred: %d   Truth: %d   Again: %d" % (ids[0], y[0], ids2[0]))
        #     plt.show()

    acc = n_corr * 1.0 / num_samples
    print('classification accuracy: %6.5f' % acc)


    print('test finished!')


def train_RAE_separately():
    pass


if __name__ == '__main__':
    #MNIST_TrainAutoEncoder()
    #MNIST_TestAutoEncoder()

    #latent_path = '../Datasets/MNIST/latent/latent_codes.npy'
    #MNIST_SaveTrainingLatentCodes(latent_path)
    #MNIST_TrainRestrictedAutoEncoder(latent_path)
    MNIST_TestRestrictedAutoEncoder()

