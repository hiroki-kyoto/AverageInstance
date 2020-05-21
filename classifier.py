# classifier
import tensorflow
if tensorflow.__version__ >= '2.0':
    import tensorflow.compat.v1 as tf
    tf.disable_eager_execution()
else:
    import tensorflow as tf
import numpy as np
import glob
import platform
from PIL import Image
import matplotlib.pyplot as plt
import os


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
        # fc_ = tf.layers.dense(
        #     layers[-1],
        #     256,
        #     activation=tf.nn.leaky_relu,
        #     use_bias=True,
        #     kernel_initializer=make_initializer()
        # )
        # layers.append(fc_)
        # fc_ = tf.layers.dense(
        #     layers[-1],
        #     1024,
        #     activation=tf.nn.leaky_relu,
        #     use_bias=False,
        #     kernel_initializer=make_initializer()
        # )
        # layers.append(fc_)
        # fc_ = tf.layers.dense(
        #     layers[-1],
        #     256,
        #     activation=tf.nn.leaky_relu,
        #     use_bias=True,
        #     kernel_initializer=make_initializer()
        # )
        # layers.append(fc_)
        return layers


def decode(z):
    if len(z.shape.as_list()) != 2:
        raise AssertionError('decode() expects a 2D tensor input!')
    name = 'decoder'
    layers = [z]
    with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
        # fc_ = tf.layers.dense(
        #     layers[-1],
        #     1024,
        #     activation=tf.nn.leaky_relu,
        #     use_bias=True,
        #     kernel_initializer=make_initializer()
        # )
        # layers.append(fc_)
        # fc_ = tf.layers.dense(
        #     layers[-1],
        #     256,
        #     activation=tf.nn.leaky_relu,
        #     use_bias=True,
        #     kernel_initializer=make_initializer()
        # )
        # layers.append(fc_)
        # fc_ = tf.layers.dense(
        #     layers[-1],
        #     8192,
        #     activation=tf.nn.leaky_relu,
        #     use_bias=True,
        #     kernel_initializer=make_initializer()
        # )
        # layers.append(fc_)
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


def classify(z, n_classes):
    layers = [z]
    fc_ = tf.layers.dense(
        layers[-1],
        n_classes,
        activation=None,
        use_bias=False,
        kernel_initializer=make_initializer()
    )
    layers.append(fc_)
    layers.append(tf.nn.softmax(fc_))
    return layers


def check_encoder():
    x = tf.placeholder(dtype=tf.float32, shape=[None, 256, 256, 3])
    layers = encode(x)
    z= layers[-1]
    print(z.shape.as_list())
    print('encoder: pass')


def check_decoder():
    z = tf.placeholder(dtype=tf.float32, shape=[None, 8192])
    layers = decode(z)
    x_r = layers[-1]
    print(x_r.shape.as_list())
    print('decoder: pass')


def check_classifier():
    z = tf.placeholder(dtype=tf.float32, shape=[None, 8192])
    layers = classify(z, 2)
    y = layers[-1]
    print(y.shape.as_list())
    print('classifier: pass')


def check():
    check_encoder()
    check_decoder()
    check_classifier()


def build_network(n_classes, inference_only=True):
    # x: input RGB image
    # z: latent code for RGB image
    # y: output label (association)
    # x_r : output reconstruction of x(reconstruction)
    x = tf.placeholder(dtype=tf.float32, shape=[None, 256, 256, 3])
    encoder = encode(x)
    z = encoder[-1]
    classifier = classify(z, n_classes)
    if inference_only:
        y = classifier[-1]
    else:
        y = classifier[-2]
    z_no_grad = tf.stop_gradient(z)
    decoder = decode(z_no_grad)
    x_r = decoder[-1]
    return x, y, z, x_r


def make_loss(x, y, x_r, y_g):
    loss_ass = tf.losses.softmax_cross_entropy(y_g, y)
    loss_rec = tf.reduce_mean(tf.abs(x - x_r))
    loss = loss_rec + loss_ass
    return loss, loss_ass, loss_rec


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
        images = []
        for fp in files:
            im_ = Image.open(fp).resize(fixed_size)
            im_ = np.array(im_, np.float32) / 255.0
            if len(im_.shape) == 1: # duplicate gray channels into RGB ones
                im_ = np.stack([im_, im_, im_], axis=-1)
                images.append(im_)
            elif len(im_.shape) == 3:
                if im_.shape[2] == 4: # rgba, kick off alpha channel
                    im_ = im_[:, :, 0:3]
                    images.append(im_)
                elif im_.shape[2] == 3:
                    images.append(im_)
                else:
                    raise ValueError("image has invalid format: %s" % fp)
            else:
                raise ValueError("image has invalid format: %s" % fp)
        print("%16s:\t%d" % (data['classes'][-1], len(images)))
        data['images'].extend(images)
        data['sizes'].append(len(images))
    return data


def check_dataset():
    data = load_dataset('../Datasets/ClassifierEstimator/train/')
    print('图片总量: %d' % len(data['images']))
    print('类别个数: %d' % len(data['classes']))
    print('图片尺寸: %s' % str(data['images'][0].shape))
    assert len(data['classes']) == len(data['sizes'])
    for i in range(len(data['classes'])):
        print('%16s:%8d' % (data['classes'][i], data['sizes'][i]))


def train(ckpt_dir, data_path, log_dir, max_epoc=1000):
    print('Loading dataset...')
    data = load_dataset(data_path)
    print('dataset loaded!')

    model_save_freq = 1 # unit: epoch
    with tf.Graph().as_default():
        step, next = get_ticker()
        n_classes = len(data['classes'])
        x, y, z, x_r = build_network(n_classes, False)
        vars_ = tf.trainable_variables()
        for v in vars_:
            print("%24s:%18s" % (v.name, str(v.shape)))
        # groundtruth label
        y_g = tf.placeholder(dtype=tf.float32, shape=[None, n_classes])
        loss, loss_ass, loss_rec = make_loss(x, y, x_r, y_g)
        opt = tf.train.AdamOptimizer(learning_rate=1e-4)\
            .minimize(loss_rec, global_step=step)
        tf.summary.scalar('Total Loss', loss)
        tf.summary.scalar('Association Loss', loss_ass)
        tf.summary.scalar('Reconstruction Loss', loss_rec)
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
        for epoc in range(max_epoc):
            for itr in range(len(data['images'])):
                # rebalance the samples
                class_id = np.random.randint(0, len(data['classes']))
                offset = 0
                for _cid in range(class_id):
                    offset += data['sizes'][_cid]
                size_ = data['sizes'][class_id]
                train_idx = np.random.randint(size_) + offset
                y_g_ = np.zeros([len(data['classes'])], dtype=np.float32)
                y_g_[class_id] = 1.0
                # step is updated by the Adam Optimizer!!!
                _, loss_, loss_ass_, loss_rec_, step_, logs_ = sess.run(
                    [opt, loss, loss_ass, loss_rec, step, logs],
                    feed_dict={
                        x: np.expand_dims(data['images'][train_idx], 0),
                        y_g: np.expand_dims(y_g_, 0)
                    })
                print('step#%6d epoch#%6d iter#%6d loss=%5.5f loss_ass=%5.5f loss_rec=%5.5f'%
                      (step_, epoc, itr, loss_, loss_ass_, loss_rec_))
                logger.add_summary(logs_, global_step=step_)
            if (epoc + 1) % model_save_freq == 0:
                saver.save(sess, ckpt_dir, global_step=step_)
                print('模型保存完毕.')


def test(ckpt_dir, data_path):
    print('Loading dataset...')
    data = load_dataset(data_path)
    print('dataset loaded!')
    n_classes = len(data['classes'])
    with tf.Graph().as_default():
        x, y, z, x_r = build_network(n_classes)
        sess = tf.Session()
        saver = tf.train.Saver()
        ckpt_file = tf.train.latest_checkpoint(ckpt_dir)
        if ckpt_file:
            print('发现模型参数文件，正在恢复模型。')
            saver.restore(sess, ckpt_file)
        else:
            print("未找到模型参数文件，无法加载模型，推断终止！")
            return
        # check accuracy
        pos_num = 0.0
        neg_num = 0.0
        rec_err_pos = np.zeros([len(data['images'])], dtype=np.float32)
        rec_err_neg = np.zeros([len(data['images'])], dtype=np.float32)
        for class_id in range(len(data['classes'])):
            offset = 0
            for _cid in range(class_id):
                offset += data['sizes'][_cid]
            for _id in range(data['sizes'][class_id]):
                _id += offset
                x_ = data['images'][_id]
                y_, x_r_ = sess.run([y, x_r], feed_dict={x: np.expand_dims(x_, 0)})
                if np.argmax(y_[0]) == class_id:
                    rec_err_pos[int(pos_num)] = np.mean(np.abs(x_ - x_r_[0]))
                    pos_num += 1.0
                else:
                    rec_err_neg[int(neg_num)] = np.mean(np.abs(x_ - x_r_[0]))
                    neg_num += 1.0
                im_rec = np.maximum(np.minimum(x_r_[0], 1.0), 0.0)
                im_rec = np.concatenate((data['images'][_id], im_rec), axis=1)
                plt.clf()
                plt.title('图片#%06d 预测%s' % (_id, ["错误", "正确"][np.argmax(y_[0]) == class_id]))
                plt.imshow(im_rec)
                plt.pause(1)

        print('Test Accuracy: %6.3f' % (pos_num / len(data['images'])))
        if pos_num > 0:
            mean_rec_err_pos = np.sum(rec_err_pos) / pos_num
            std_rec_err_pos = np.sqrt(np.sum(np.square(rec_err_pos - mean_rec_err_pos)) / pos_num)
            print('Reconstruction Error on Positive Samples: %6.3f' % mean_rec_err_pos)
            print('Reconstruction Stddev on Positive Samples: %6.3f' % std_rec_err_pos)
        if neg_num > 0:
            mean_rec_err_neg = np.sum(rec_err_neg) / neg_num
            std_rec_err_neg = np.sqrt(np.sum(np.square(rec_err_neg - mean_rec_err_neg)) / neg_num)
            print('Reconstruction Error on Negative Samples: %6.3f' % mean_rec_err_neg)
            print('Reconstruction Stddev on Negative Samples: %6.3f' % std_rec_err_neg)


def predict(ckpt_dir, data_path):
    print('Loading dataset...')
    data = load_dataset(data_path)
    print('dataset loaded!')
    n_classes = len(data['classes'])
    with tf.Graph().as_default():
        x, y, z, x_r = build_network(n_classes)
        sess = tf.Session()
        saver = tf.train.Saver()
        ckpt_file = tf.train.latest_checkpoint(ckpt_dir)
        if ckpt_file:
            print('发现模型参数文件，正在恢复模型。')
            saver.restore(sess, ckpt_file)
        else:
            print("未找到模型参数文件，无法加载模型，推断终止！")
            return
        rec_err = np.zeros([len(data['images'])], dtype=np.float32)
        for _id in range(len(data['images'])):
            x_ = data['images'][_id]
            y_, x_r_ = sess.run([y, x_r], feed_dict={x: np.expand_dims(x_, 0)})
            rec_err[_id] = np.mean(np.abs(x_ - x_r_[0]))
            im_rec = np.maximum(np.minimum(x_r_[0], 1.0), 0.0)
            im_rec = np.concatenate((data['images'][_id], im_rec), axis=1)
            plt.clf()
            plt.title('图片#%06d' % _id)
            plt.imshow(im_rec)
            plt.pause(1)
        mean_rec_err = np.sum(rec_err) / float(len(data['images']))
        std_rec_err = np.sqrt(np.sum(np.square(rec_err - mean_rec_err)) / float(len(data['images'])))
        print('Reconstruction Error: %6.3f' % mean_rec_err)
        print('Reconstruction Stddev: %6.3f' % std_rec_err)


if __name__ == '__main__':
    print('==== RUNNING FROM AUTO ENCODER ====')
    #check()
    #check_dataset()

    # train('../Models/ClassifierEstimator/',  # model saving path
    #       '../Datasets/ClassifierEstimator/train/',  # dataset loading path
    #       '../Logs/ClassifierEstimator/',  # logging path
    #       500)  # the maximum number of epoch to run

    #test('../Models/ClassifierEstimator/', '../Datasets/ClassifierEstimator/test/')
    predict('../Models/ClassifierEstimator/', '../Datasets/ClassifierEstimator/test-single/')
    print('===================================')