import os
import numpy as np
import time
import tensorflow as tf
import dcgan
from utils import save_checkpoint, load_checkpoint
from config_device import config_device

def train_dcgan(data, config):

    training_graph = tf.Graph()

    with training_graph.as_default():

        tf.set_random_seed(1)

        gan = dcgan.dcgan(output_size=config.output_size,
                          batch_size=config.batch_size,
                          nd_layers=config.nd_layers,
                          ng_layers=config.ng_layers,
                          df_dim=config.df_dim,
                          gf_dim=config.gf_dim,
                          c_dim=config.c_dim,
                          z_dim=config.z_dim,
                          flip_labels=config.flip_labels,
                          data_format=config.data_format,
                          transpose_b=config.transpose_matmul_b)

        gan.training_graph()
        update_op = gan.optimizer(config.learning_rate, config.beta1)

        checkpoint_dir = os.path.join(config.checkpoint_dir, config.experiment)

        sess_config = config_device(config.arch)
        with tf.Session(config=sess_config) as sess:
            writer = tf.summary.FileWriter('./logs/'+config.experiment+'/train', sess.graph)

            init_op = tf.global_variables_initializer()
            sess.run(init_op)

            if config.save_checkpoint:
                load_checkpoint(sess, gan.saver, 'dcgan', checkpoint_dir)

            epoch = sess.run(gan.increment_epoch)
            start_time = time.time()
            time_logs = []
            for epoch in range(epoch, epoch + config.epoch):

                num_batches = data.shape[0] // config.batch_size
                for idx in range(0, num_batches):
                    batch_images = data[idx*config.batch_size:(idx+1)*config.batch_size]

                    _, g_sum, d_sum = sess.run([update_op, gan.g_summary, gan.d_summary], 
                                               feed_dict={gan.images: batch_images})

                    global_step = gan.global_step.eval()
                    writer.add_summary(g_sum, global_step)
                    writer.add_summary(d_sum, global_step)

                    if global_step > 4:
                        if global_step > 5:
                            time_logs.append(time.time() - time_ref)
                        time_ref = time.time()

                    if config.verbose:
                        errD_fake = gan.d_loss_fake.eval()
                        errD_real = gan.d_loss_real.eval({gan.images: batch_images})
                        errG = gan.g_loss.eval()

                        print("Epoch: [%2d] Step: [%4d/%4d] time: %4.4f, d_loss: %.8f, g_loss: %.8f" \
                                % (epoch, idx, num_batches, time.time() - start_time, errD_fake+errD_real, errG))

                    elif global_step%100 == 0:
                        print("Epoch: [%2d] Step: [%4d/%4d] time: %4.4f"%(epoch, idx, num_batches, time.time() - start_time))

                # save a checkpoint every epoch
                if config.save_checkpoint:
                    save_checkpoint(sess, gan.saver, 'dcgan', checkpoint_dir, epoch)
                sess.run(gan.increment_epoch)

    tf.reset_default_graph()
    return np.average(time_logs), np.std(time_logs)/np.sqrt(len(time_logs))
