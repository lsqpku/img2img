import os
import scipy.misc
import numpy as np

from model6 import DCGAN
from utils import pp, visualize, to_json, show_all_variables

import tensorflow as tf

flags = tf.app.flags
flags.DEFINE_integer("epoch", 50, "Epoch to train [25]")
flags.DEFINE_float("learning_rate", 0.0002, "Learning rate of for adam [0.0002]")
flags.DEFINE_float("beta1", 0.5, "Momentum term of adam [0.5]")
flags.DEFINE_integer("train_size", np.inf, "The size of train images [np.inf]")
flags.DEFINE_integer("batch_size", 64, "The size of batch images [64]")
flags.DEFINE_integer("input_height", 64, "The size of image to use (will be center cropped). [108]")
flags.DEFINE_integer("input_width", None,
                     "The size of image to use (will be center cropped). If None, same value as input_height [None]")
flags.DEFINE_integer("output_height", 64, "The size of the output images to produce [64]")
flags.DEFINE_integer("output_width", None,
                     "The size of the output images to produce. If None, same value as output_height [None]")
flags.DEFINE_string("input_fname_pattern", "*.png", "Glob pattern of filename of input images [*]")
flags.DEFINE_boolean("train", True, "True for training, False for testing [False]")
flags.DEFINE_boolean("crop", False, "True for training, False for testing [False]")

flags.DEFINE_string("dataset", "wiki_blurred/", "The name of dataset [celebA, mnist, lsun]")
flags.DEFINE_string("checkpoint_dir", "checkpoint6/", "Directory name to save the checkpoints [checkpoint]")
flags.DEFINE_string("sample_dir", "samples6/", "Directory name to save the image samples [samples]")
flags.DEFINE_string("dataset_test", "wikitest_blurred/", "Directory name of testing samples")
flags.DEFINE_string("dataset_target", "wiki/", "Directory name of target(generated) images")
flags.DEFINE_string("logdir", "logs6/", "logdir")

FLAGS = flags.FLAGS


def main(_):
    pp.pprint(flags.FLAGS.__flags)

    if FLAGS.input_width is None:
        FLAGS.input_width = FLAGS.input_height
    if FLAGS.output_width is None:
        FLAGS.output_width = FLAGS.output_height

    if not os.path.exists(FLAGS.checkpoint_dir):
        os.makedirs(FLAGS.checkpoint_dir)
    if not os.path.exists(FLAGS.sample_dir):
        os.makedirs(FLAGS.sample_dir)

    # gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.333)
    run_config = tf.ConfigProto()
    run_config.gpu_options.allow_growth = True
    run_config.gpu_options.per_process_gpu_memory_fraction = 0.3

    with tf.Session(config=run_config) as sess:

        dcgan = DCGAN(
            sess,
            input_width=FLAGS.input_width,
            input_height=FLAGS.input_height,
            output_width=FLAGS.output_width,
            output_height=FLAGS.output_height,
            batch_size=FLAGS.batch_size,
            sample_num=FLAGS.batch_size,
            dataset_name=FLAGS.dataset,
            input_fname_pattern=FLAGS.input_fname_pattern,
            crop=FLAGS.crop,
            checkpoint_dir=FLAGS.checkpoint_dir,
            sample_dir=FLAGS.sample_dir,
            dataset_test=FLAGS.dataset_test,
            dataset_target=FLAGS.dataset_target,
            logdir=FLAGS.logdir)


        show_all_variables()

        if FLAGS.train:
            dcgan.train(FLAGS)
        else:
            if not dcgan.load(FLAGS.checkpoint_dir)[0]:
                raise Exception("[!] Train a model first, then run test mode")
            dcgan.test(FLAGS)


if __name__ == '__main__':
    tf.app.run()
