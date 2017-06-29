import tensorflow as tf
import train
import numpy as np
import pprint
import copy

flags = tf.app.flags
flags.DEFINE_string("datafile", "data/cosmo_primary_64_1k_train.npy", "Input data file for cosmo")
flags.DEFINE_integer("epoch", 1, "Epochs to train [1]")
flags.DEFINE_float("learning_rate", 0.0002, "Learning rate of for adam [0.0002]")
flags.DEFINE_float("beta1", 0.5, "Momentum term of adam [0.5]")
flags.DEFINE_float("flip_labels", 0, "Probability of flipping labels [0]")
flags.DEFINE_integer("z_dim", 100, "Dimension of noise vector z [100]")
flags.DEFINE_integer("nd_layers", 4, "Number of discriminator conv2d layers. [4]")
flags.DEFINE_integer("ng_layers", 4, "Number of generator conv2d_transpose layers. [4]")
flags.DEFINE_integer("gf_dim", 64, "Dimension of gen filters in last conv layer. [64]")
flags.DEFINE_integer("df_dim", 64, "Dimension of discrim filters in first conv layer. [64]")
flags.DEFINE_integer("batch_size", 64, "The size of batch images [64]")
flags.DEFINE_integer("output_size", 64, "The size of the output images to produce [64]")
flags.DEFINE_integer("c_dim", 1, "Dimension of image color. [1]")
flags.DEFINE_string("data_format", "NHWC", "data format [NHWC]")
flags.DEFINE_boolean("transpose_matmul_b", False, "Transpose matmul B matrix for performance [False]")
flags.DEFINE_string("checkpoint_dir", "checkpoints", "Directory name to save the checkpoints [checkpoint]")
flags.DEFINE_string("experiment", "run_0", "Tensorboard run directory name [run_0]")
flags.DEFINE_boolean("save_checkpoint", False, "Save a checkpoint every epoch [False]")
flags.DEFINE_boolean("verbose", True, "print loss on every step [False]")
flags.DEFINE_string("arch", "default", "Architecture, default, KNL or HSW")
config = flags.FLAGS

data_format = ["NHWC"]
batch_size = [32, 64, 128, 256]

def main(_):

    pprint.PrettyPrinter().pprint(config.__flags)

    for df in data_format:
        config.data_format = df
        avg_time = train.train_dcgan(get_data(), config)
        print("\ndata_format = %s. batch_size = %i"%(config.data_format, config.batch_size))
        print("Average time per batch = %3.3f +- %3.5f (s)\n"%(avg_time[0], avg_time[1]))

    for bs in batch_size:
        config.batch_size = bs
        avg_time = train.train_dcgan(get_data(), config)
        print("\ndata_format = %s. batch_size = %i"%(config.data_format, config.batch_size))
        print("Average time per batch = %3.3f +- %3.5f (s)\n"%(avg_time[0], avg_time[1]))

def get_data():
    data = np.load(config.datafile, mmap_mode='r')

    if config.data_format == 'NHWC':
        data = np.expand_dims(data, axis=-1)
    else: # 'NCHW'
        data = np.expand_dims(data, axis=1)

    return data

if __name__ == '__main__':
    tf.app.run()
