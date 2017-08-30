#!/usr/bin/env python3
from tf_gen_models.WGAN import WGAN
from tf_gen_models.utils import show_all_variables
from Models import *
import tensorflow as tf

def main():
    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
        gan = WGAN(sess, epoch=10, batch_size=64, dataset_name='mnist', checkpoint_dir='checkpoint', result_dir='results', log_dir='logs')
        gan.discriminator = mnist_discriminator
        gan.generator = mnist_generator
        gan.build_model()
        show_all_variables()
        gan.train()
        print(" [*] Training finished!")

        gan.visualize_results(9)
        print(" [*] Testing finished!")

if __name__ == '__main__':
    main()
