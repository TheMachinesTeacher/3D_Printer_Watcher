import tensorflow as tf
from tf_gen_models.ops import *

def mnist_discriminator(x, batchSize, is_training=True, reuse=False):
    # Network Architecture is exactly same as in infoGAN (https://arxiv.org/abs/1606.03657)
    # Architecture : (64)4c2s-(128)4c2s_BL-FC1024_BL-FC1_S
    with tf.variable_scope("discriminator", reuse=reuse):

        net = lrelu(conv2d(x, 64, 4, 4, 2, 2, name='d_conv1'))
        net = lrelu(bn(conv2d(net, 128, 4, 4, 2, 2, name='d_conv2'), is_training=is_training, scope='d_bn2'))
        net = tf.reshape(net, [batchSize, -1])
        net = lrelu(bn(linear(net, 1024, scope='d_fc3'), is_training=is_training, scope='d_bn3'))
        out_logit = linear(net, 1, scope='d_fc4')
        out = tf.nn.sigmoid(out_logit)

        return out, out_logit, net

def mnist_generator(z, batchSize, is_training=True, reuse=False):
    # Network Architecture is exactly same as in infoGAN (https://arxiv.org/abs/1606.03657)
    # Architecture : FC1024_BR-FC7x7x128_BR-(64)4dc2s_BR-(1)4dc2s_S
    with tf.variable_scope("generator", reuse=reuse):
        net = tf.nn.relu(bn(linear(z, 1024, scope='g_fc1'), is_training=is_training, scope='g_bn1'))
        net = tf.nn.relu(bn(linear(net, 128 * 7 * 7, scope='g_fc2'), is_training=is_training, scope='g_bn2'))
        net = tf.reshape(net, [batchSize, 7, 7, 128])
        net = tf.nn.relu(
            bn(deconv2d(net, [batchSize, 14, 14, 64], 4, 4, 2, 2, name='g_dc3'), is_training=is_training, scope='g_bn3'))
        out = tf.nn.sigmoid(deconv2d(net, [batchSize, 28, 28, 1], 4, 4, 2, 2, name='g_dc4'))

        return out

def printerGenerator(x, batchSize, is_training=True, reuse=False):
    with tf.variable_scope("generator", reuse=reuse):
        net = tf.nn.relu(bn(linear(z, 65536, scope='g_fc1'), is_training=is_training, scope='g_bn1'))
        net = tf.nn.relu(bn(linear(net, 128*45*80, scope='g_fc2'), is_training=is_training, scope='g_bn2'))
        net = tf.reshape(net, [batchSize, 45, 80, 128])
        net = tf.nn.relu(bn(deconv2d(net, [batchSize, 90, 160, 64], 4, 4, 2, 2, name='g_dc3'), is_training=is_training, scope='g_bn3'))
        out = tf.nn.sigmoid(deconv2d(net, [batchSize, 180, 320, 1], 4, 4, 2, 2, name='g_dc4'))
        return out
