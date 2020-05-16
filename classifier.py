# A part of hualiaoshi
# test of zhishi chuandi
import tensorflow.compat.v1 as tf
import numpy as np
import glob
import platform
from PIL import Image
import matplotlib.pyplot as plt
import os

tf.disable_eager_execution()
plt.rcParams['font.sans-serif']=['SimHei']
plt.rcParams['axes.unicode_minus'] = False


def __windows__():
    return bool(platform.system() == 'Windows')


# Observation
# Abstract the unvaried features: learning from only positive samples
# Plan:
# 1. Train an Auto Encoder with spatial and temporal consistency loss.


def make_initializer():
    return tf.initializers.orthogonal()


# encode(x): the last layer is the hash code for sample [x].
def encode(x):
    if len(x.shape.as_list()) != 4:
        raise AssertionError('encode() expect a 4D tensor input!')
    name = 'encoder'
    layers = [x]
    with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
        # group #1 256x256x3 => 128x128x8
        conv_ = tf.layers.conv2d(
            inputs=layers[-1],
            filters=8,
            kernel_size=3,
            strides=(1, 1),
            padding='same',
            activation=tf.nn.leaky_relu,
            use_bias=True,
            kernel_initializer=make_initializer()
        )
        layers.append(conv_)
        conv_ = tf.layers.conv2d(
            inputs=layers[-1],
            filters=8,
            kernel_size=3,
            strides=(2, 2),
            padding='same',
            activation=tf.nn.leaky_relu,
            use_bias=True,
            kernel_initializer=make_initializer()
        )
        layers.append(conv_)

        # residual links to add up features: 64x64x8
        pool_ = tf.layers.average_pooling2d(
            inputs=layers[-1],
            pool_size=(3, 3),
            strides=(2, 2),
            padding='same'
        )

        # group #2 128x128x8 => 64x64x8
        conv_ = tf.layers.conv2d(
            inputs=layers[-1],
            filters=8,
            kernel_size=3,
            strides=(1, 1),
            padding='same',
            activation=tf.nn.leaky_relu,
            use_bias=True,
            kernel_initializer=make_initializer()
        )
        layers.append(conv_)
        conv_ = tf.layers.conv2d(
            inputs=layers[-1],
            filters=8,
            kernel_size=3,
            strides=(2, 2),
            padding='same',
            activation=tf.nn.leaky_relu,
            use_bias=True,
            kernel_initializer=make_initializer()
        )
        # add up features: 8+8=>16 channels of 64x64
        merge_ = tf.concat((conv_, pool_), axis=-1)
        layers.append(merge_)

        # residual links: 16 channels of 32x32
        pool_ = tf.layers.average_pooling2d(
            inputs=layers[-1],
            pool_size=(3, 3),
            strides=(2, 2),
            padding='same'
        )

        # group #3 64x64x16 => 32x32x16
        conv_ = tf.layers.conv2d(
            inputs=layers[-1],
            filters=16,
            kernel_size=3,
            strides=(1, 1),
            padding='same',
            activation=tf.nn.leaky_relu,
            use_bias=True,
            kernel_initializer=make_initializer()
        )
        layers.append(conv_)
        conv_ = tf.layers.conv2d(
            inputs=layers[-1],
            filters=16,
            kernel_size=3,
            strides=(2, 2),
            padding='same',
            activation=tf.nn.leaky_relu,
            use_bias=True,
            kernel_initializer=make_initializer()
        )
        # add up features: 16+16=>32 channels of 32x32
        merge_ = tf.concat((conv_, pool_), axis=-1)
        layers.append(merge_)

        # residual links: 32 channels of 16x16
        pool_ = tf.layers.average_pooling2d(
            inputs=layers[-1],
            pool_size=(3, 3),
            strides=(2, 2),
            padding='same'
        )

        # group #4 32x32x32 => 16x16x32
        conv_ = tf.layers.conv2d(
            inputs=layers[-1],
            filters=32,
            kernel_size=3,
            strides=(1, 1),
            padding='same',
            activation=tf.nn.leaky_relu,
            use_bias=True,
            kernel_initializer=make_initializer()
        )
        layers.append(conv_)
        conv_ = tf.layers.conv2d(
            inputs=layers[-1],
            filters=32,
            kernel_size=3,
            strides=(2, 2),
            padding='same',
            activation=tf.nn.leaky_relu,
            use_bias=True,
            kernel_initializer=make_initializer()
        )
        # add up features: 32+32=>64 channels of 16x16
        merge_ = tf.concat((conv_, pool_), axis=-1)
        layers.append(merge_)

        # residual links: 64 channels of 8x8
        pool_ = tf.layers.average_pooling2d(
            inputs=layers[-1],
            pool_size=(3, 3),
            strides=(2, 2),
            padding='same'
        )

        # group #5 16x16x64 => 8x8x64
        conv_ = tf.layers.conv2d(
            inputs=layers[-1],
            filters=64,
            kernel_size=3,
            strides=(1, 1),
            padding='same',
            activation=tf.nn.leaky_relu,
            use_bias=True,
            kernel_initializer=make_initializer()
        )
        layers.append(conv_)
        conv_ = tf.layers.conv2d(
            inputs=layers[-1],
            filters=64,
            kernel_size=3,
            strides=(2, 2),
            padding='same',
            activation=tf.nn.leaky_relu,
            use_bias=True,
            kernel_initializer=make_initializer()
        )
        # add up features: 64+64=>128 channels of 8x8
        merge_ = tf.concat((conv_, pool_), axis=-1)
        layers.append(merge_)

        # convert into hash code with 8192 units
        vec_ = tf.layers.flatten(layers[-1])
        layers.append(vec_)
        fc_ = tf.layers.dense(
            layers[-1],
            256,
            activation=tf.nn.leaky_relu,
            use_bias=True,
            kernel_initializer=make_initializer()
        )
        layers.append(fc_)
        fc_ = tf.layers.dense(
            layers[-1],
            1024,
            activation=tf.nn.leaky_relu,
            use_bias=False,
            kernel_initializer=make_initializer()
        )
        layers.append(fc_)
        fc_ = tf.layers.dense(
            layers[-1],
            256,
            activation=tf.nn.leaky_relu,
            use_bias=True,
            kernel_initializer=make_initializer()
        )
        layers.append(fc_)
        return layers


def decode(y):
    if len(y.shape.as_list()) != 2:
        raise AssertionError('decode() expects a 2D tensor input!')
    name = 'decoder'
    layers = [y]
    with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
        fc_ = tf.layers.dense(
            layers[-1],
            1024,
            activation=tf.nn.leaky_relu,
            use_bias=True,
            kernel_initializer=make_initializer()
        )
        layers.append(fc_)
        fc_ = tf.layers.dense(
            layers[-1],
            256,
            activation=tf.nn.leaky_relu,
            use_bias=True,
            kernel_initializer=make_initializer()
        )
        layers.append(fc_)
        fc_ = tf.layers.dense(
            layers[-1],
            8192,
            activation=tf.nn.leaky_relu,
            use_bias=True,
            kernel_initializer=make_initializer()
        )
        layers.append(fc_)
        # convert into a 4D tensor
        base = tf.reshape(layers[-1], shape=[-1, 8, 8, 128])
        layers.append(base)

        # split tensor into control layers of different levels
        ctl_layers = tf.split(base, 2, axis=-1)
        assert(len(ctl_layers)==2)
        layers.append(ctl_layers[0])
        # update base layer
        base = ctl_layers[1] # 8x8x64

        # upsample onto higher resolution
        # group #1 8x8x64 -> 16x16x32
        deconv_ = tf.layers.conv2d_transpose(
            inputs=layers[-1],
            filters=32,
            kernel_size=(3, 3),
            strides=(2, 2),
            padding='same',
            activation=tf.nn.leaky_relu,
            use_bias=True,
            kernel_initializer=make_initializer()
        )
        layers.append(deconv_)
        deconv_ = tf.layers.conv2d_transpose(
            inputs=layers[-1],
            filters=32,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding='same',
            activation=tf.nn.leaky_relu,
            use_bias=True,
            kernel_initializer=make_initializer()
        )
        layers.append(deconv_)

        # upsample control layer: 16x16x64
        new_size = (base.shape.as_list()[1] * 2,
                    base.shape.as_list()[2] * 2)
        base = tf.image.resize_bilinear(base, new_size)

        # split base again: 2 x 16x16x32
        ctl_layers = tf.split(base, 2, axis=-1)
        assert (len(ctl_layers) == 2)

        # update input layer: 32+32=64 of 16x16
        merge_ = tf.concat((layers[-1], ctl_layers[0]), axis=-1)
        layers.append(merge_)

        # update control layer: 16x16x32
        base = ctl_layers[1]

        # group #2 16x16x64 -> 32x32x16
        deconv_ = tf.layers.conv2d_transpose(
            inputs=layers[-1],
            filters=16,
            kernel_size=(3, 3),
            strides=(2, 2),
            padding='same',
            activation=tf.nn.leaky_relu,
            use_bias=True,
            kernel_initializer=make_initializer()
        )
        layers.append(deconv_)
        deconv_ = tf.layers.conv2d_transpose(
            inputs=layers[-1],
            filters=16,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding='same',
            activation=tf.nn.leaky_relu,
            use_bias=True,
            kernel_initializer=make_initializer()
        )
        layers.append(deconv_)

        # upsample control layer: 32x32x32
        new_size = (base.shape.as_list()[1] * 2,
                    base.shape.as_list()[2] * 2)
        base = tf.image.resize_bilinear(base, new_size)

        # split base again: 2 x 32x32x16
        ctl_layers = tf.split(base, 2, axis=-1)
        assert (len(ctl_layers) == 2)

        # update input layer: 16+16=32 of 32x32
        merge_ = tf.concat((layers[-1], ctl_layers[0]), axis=-1)
        layers.append(merge_)

        # update control layer: 32x32x16
        base = ctl_layers[1]

        # group #3 32x32x32 -> 64x64x8
        deconv_ = tf.layers.conv2d_transpose(
            inputs=layers[-1],
            filters=8,
            kernel_size=(3, 3),
            strides=(2, 2),
            padding='same',
            activation=tf.nn.leaky_relu,
            use_bias=True,
            kernel_initializer=make_initializer()
        )
        layers.append(deconv_)
        deconv_ = tf.layers.conv2d_transpose(
            inputs=layers[-1],
            filters=8,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding='same',
            activation=tf.nn.leaky_relu,
            use_bias=True,
            kernel_initializer=make_initializer()
        )
        layers.append(deconv_)

        # upsample control layer: 64x64x16
        new_size = (base.shape.as_list()[1] * 2,
                    base.shape.as_list()[2] * 2)
        base = tf.image.resize_bilinear(base, new_size)

        # split base again: 2 x 64x64x8
        ctl_layers = tf.split(base, 2, axis=-1)
        assert (len(ctl_layers) == 2)

        # update input layer: 8+8=16 of 64x64
        merge_ = tf.concat((layers[-1], ctl_layers[0]), axis=-1)
        layers.append(merge_)

        # update control layer: 64x64x8
        base = ctl_layers[1]

        # group #4 64x64x16 -> 128x128x8
        deconv_ = tf.layers.conv2d_transpose(
            inputs=layers[-1],
            filters=8,
            kernel_size=(3, 3),
            strides=(2, 2),
            padding='same',
            activation=tf.nn.leaky_relu,
            use_bias=True,
            kernel_initializer=make_initializer()
        )
        layers.append(deconv_)
        deconv_ = tf.layers.conv2d_transpose(
            inputs=layers[-1],
            filters=8,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding='same',
            activation=tf.nn.leaky_relu,
            use_bias=True,
            kernel_initializer=make_initializer()
        )
        layers.append(deconv_)

        # upsample control layer: 128x128x8
        new_size = (base.shape.as_list()[1] * 2,
                    base.shape.as_list()[2] * 2)
        base = tf.image.resize_bilinear(base, new_size)

        # update input layer: 8+8=16 of 128x128
        merge_ = tf.concat((layers[-1], base), axis=-1)
        layers.append(merge_)

        # group #5 128x128x16 -> 256x256x3
        deconv_ = tf.layers.conv2d_transpose(
            inputs=layers[-1],
            filters=8,
            kernel_size=(3, 3),
            strides=(2, 2),
            padding='same',
            activation=tf.nn.leaky_relu,
            use_bias=True,
            kernel_initializer=make_initializer()
        )
        layers.append(deconv_)
        deconv_ = tf.layers.conv2d_transpose(
            inputs=layers[-1],
            filters=3,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding='same',
            activation=tf.nn.leaky_relu,
            use_bias=True,
            kernel_initializer=make_initializer()
        )
        layers.append(deconv_)
        return layers


def check_encoder():
    x = tf.placeholder(dtype=tf.float32, shape=[None, 256, 256, 3])
    layers = encode(x)
    y = layers[-1]
    print(y.shape.as_list())
    print('encode(): pass')


def check_decoder():
    y = tf.placeholder(dtype=tf.float32, shape=[None, 16])
    layers = decode(y)
    x = layers[-1]
    print(x.shape.as_list())
    print('decode(): pass')


def check():
    check_encoder()
    check_decoder()


def build_network():
    # x: input RGB image,
    # z: latent code for RGB image,
    # y: output RGB image
    x = tf.placeholder(dtype=tf.float32, shape=[None, 256, 256, 3])
    encoder = encode(x)
    z = encoder[-1]
    decoder = decode(z)
    y = decoder[-1]
    return x, y, z


def make_unified_loss(a, b):
    # norm of a and b should be ignored in the loss
    a_unified = a / tf.sqrt(tf.reduce_sum(a * a, axis=-1) + 1E-7)
    b_unified = b / tf.sqrt(tf.reduce_sum(b * b, axis=-1) + 1E-7)
    return tf.reduce_mean(tf.abs(a_unified - b_unified))


def make_loss(x1, y1, z1, x2, y2, z2):
    loss_chongjian = tf.reduce_mean(tf.abs(x1 - y1)) \
                     + tf.reduce_mean(tf.abs(x2 - y2)) / 2.0
    loss_lianxu = make_unified_loss(z1, z2)
    alpha = 1 - tf.minimum(5.0*loss_chongjian, 1.0)
    alpha = tf.stop_gradient(alpha)
    loss = loss_chongjian * (1 - alpha) + loss_lianxu * alpha
    return loss, loss_chongjian, loss_lianxu


def get_ticker():
    step = tf.get_variable(
        name='step',
        shape=[],
        dtype=tf.int32,
        initializer=tf.initializers.zeros)
    step_next = tf.assign_add(step, 1)
    return step, step_next


def load_dataset(path):
    fixed_size = [256, 256]
    classes = glob.glob(path + '/*')
    data = {}
    data['images'] = []
    data['classes'] = []
    data['sizes'] = []
    for i in range(len(classes)):
        if not os.path.isdir(classes[i]):
            continue
        if __windows__():
            seg = classes[i].rfind('\\')
            data['classes'].append(classes[i][seg + 1:])
        else:
            seg = classes[i].rfind('/')
            data['classes'].append(classes[i][seg + 1:])
        files = glob.glob(classes[i] + '/*.jpg')
        # print(files)
        images = [np.array(Image.open(fp).resize(fixed_size), np.float32) / 255.0 for fp in files]
        print("%16s:\t%d" % (data['classes'][-1], len(images)))
        data['images'].extend(images)
        data['sizes'].append(len(images))
    return data


def check_dataset():
    data = load_dataset('../../Datasets/ClassifierEstimator/train/')
    print('图片总量: %d' % len(data['images']))
    print('类别个数: %d' % len(data['classes']))
    print('图片尺寸: %s' % str(data['images'][0].shape))
    assert len(data['classes']) == len(data['sizes'])
    for i in range(len(data['classes'])):
        print('%16s:%8d' % (data['classes'][i], data['sizes'][i]))


def train(ckpt_dir, data_path, log_dir, max_epoc=1000):
    model_save_freq = 1 # unit: epoch
    with tf.Graph().as_default():
        step, next = get_ticker()
        x1, y1, z1 = build_network()
        x2, y2, z2 = build_network()
        vars = tf.trainable_variables()
        for v in vars:
            print("%24s:%18s" % (v.name, str(v.shape)))
        loss, loss_chongjian, loss_lianxu = \
            make_loss(x1, y1, z1, x2, y2, z2)
        opt = tf.train.AdamOptimizer(learning_rate=1e-4)\
            .minimize(loss, global_step=step)
        tf.summary.scalar('Total Loss', loss)
        tf.summary.scalar('Reconstruction Loss', loss_chongjian)
        tf.summary.scalar('Continuity Loss', loss_lianxu)
        logs = tf.summary.merge_all()
        # add IO
        logger = tf.summary.FileWriter(log_dir, tf.get_default_graph())
        sess = tf.Session()
        saver = tf.train.Saver()
        ckpt_file = tf.train.latest_checkpoint(ckpt_dir)
        if ckpt_file:
            print('发现模型参数文件，正在恢复模型。')
            saver.restore(sess, ckpt_file)
        else:
            ans = input("未找到模型参数文件，是否从头开始训练? [ Y | N ]\n")
            if ans == 'N' or ans == 'n':
                print("训练已终止。")
                return
            else:
                print("训练将重头开始。")
            sess.run(tf.global_variables_initializer())
            print("模型参数已经初始化完成。")
        # training loop
        data = load_dataset(data_path)
        for epoc in range(max_epoc):
            episode_id = np.random.randint(0, len(data['sizes']))
            offset = 0
            for _epid in range(episode_id):
                offset += data['sizes'][_epid]
            size_ = data['sizes'][episode_id]
            train_idx = np.random.permutation(size_-1) + offset
            for idx in train_idx:
                # step is updated by the Adam Optimizer!!!
                _, loss_, step_, logs_ = sess.run(
                    [opt, loss, step, logs],
                    feed_dict={
                        x1: np.expand_dims(data['x'][idx], 0),
                        x2: np.expand_dims(data['x'][idx+1], 0)
                    })
                print('step#%6d\tepoch#%6d\titer#%6d\tloss=%5.5f' %
                      (step_, epoc, idx, loss_))
                logger.add_summary(logs_, global_step=step_)
            if (epoc + 1) % model_save_freq == 0:
                saver.save(sess, ckpt_dir, global_step=step_)
                print('模型保存完毕.')


def predict(ckpt_dir, data_path):
    with tf.Graph().as_default():
        x, y, z = build_network()
        sess = tf.Session()
        saver = tf.train.Saver()
        ckpt_file = tf.train.latest_checkpoint(ckpt_dir)
        if ckpt_file:
            print('发现模型参数文件，正在恢复模型。')
            saver.restore(sess, ckpt_file)
        else:
            print("未找到模型参数文件，无法加载模型，推断终止！")
            return
        # training loop
        data = load_dataset(data_path)
        for i in range(2, len(data['x'])-2):
            z_1 = sess.run(z, feed_dict={x: np.expand_dims(data['x'][i-2], 0)})
            z_2 = sess.run(z, feed_dict={x: np.expand_dims(data['x'][i], 0)})
            z_3 = sess.run(z, feed_dict={x: np.expand_dims(data['x'][i+2], 0)})
            y_2 = sess.run(y, feed_dict={z: z_2})
            z_2_i = (z_1 + z_3) / 2.0
            y_2_i = sess.run(y, feed_dict={z: z_2_i})
            y_2 = np.maximum(np.minimum(y_2[0], 1.0), 0.0)
            y_2_i = np.maximum(np.minimum(y_2_i[0], 1.0), 0.0)
            y_2 = np.concatenate((data['x'][i], y_2, y_2_i), axis=1)
            plt.clf()
            plt.title('图片#%06d' % i)
            plt.imshow(y_2)
            plt.pause(0.01)


if __name__ == '__main__':
    print('==== RUNNING FROM AUTO ENCODER ====')
    check()
    check_dataset()

    # train('../../Models/ClassifierEstimator/',  # model saving path
    #       '../../Datasets/ClassifierEstimator/train/',  # dataset loading path
    #       '../../Logs/ClassifierEstimator/',  # logging path
    #       500)  # the maximum number of epoch to run

    # predict('../../Models/AE/', '../../Datasets/shikonglianxu/episodes')
    print('===================================')