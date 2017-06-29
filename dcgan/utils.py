import os
import tensorflow as tf

def save_checkpoint(sess, saver, tag, checkpoint_dir, counter):

    model_name = tag + '.model-epoch'

    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    saver.save(sess, os.path.join(checkpoint_dir, model_name), global_step=counter)

def load_checkpoint(sess, saver, tag, checkpoint_dir, counter=None):
    print(" [*] Reading checkpoints...")

    counter_name = 'epoch'

    ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
    if ckpt and ckpt.model_checkpoint_path:
        ckpt_name = os.path.basename(ckpt.model_checkpoint_path)

        if not counter==None:
            ckpt_name_epoch = ckpt_name[:ckpt_name.find(counter_name)] + counter_name + '-%i'%epoch
            if os.path.exists(os.path.join(checkpoint_dir, ckpt_name_epoch+'.index')):
                ckpt_name = ckpt_name_epoch
            else:
                print("Checkpoint for ", counter_name , counter_name, "doesn't exist. Using latest checkpoint instead!")

        saver.restore(sess, os.path.join(checkpoint_dir, ckpt_name))
        print(" [*] Success to read {}".format(ckpt_name))
        return True
    else:
        print(" [*] Failed to find a checkpoint")
        return False
