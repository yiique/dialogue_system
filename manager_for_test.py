__author__ = 'liushuman'

'''
# get TF logger

# create formatter and add it to the handlers
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# create file handler which logs even debug messages
fh = logging.FileHandler('tensorflow.log')
fh.setLevel(logging.DEBUG)
fh.setFormatter(formatter)
log.addHandler(fh)'''


import json
import logging
import numpy as np
import random
import sys
import tensorflow as tf
import time


SEED = 88


log = logging.getLogger('tensorflow')
log.setLevel(logging.DEBUG)

FLAGS = tf.flags.FLAGS

tf.flags.DEFINE_integer("GPU_num", 4, """""")

tf.flags.DEFINE_integer("batch_size", 40, """""")
tf.flags.DEFINE_integer("dia_max_len", 10, """""")
tf.flags.DEFINE_integer("sen_max_len", 60, """""")
tf.flags.DEFINE_integer("candidate_num", 300,
                        """
                            candidate triples number that been scored, weight of others is zero
                        """)
tf.flags.DEFINE_integer("common_vocab", 8603, """""")
tf.flags.DEFINE_integer("entities", 51590, """""")
tf.flags.DEFINE_integer("relations", 3996, """""")
tf.flags.DEFINE_integer("start_token", 8601, """""")
tf.flags.DEFINE_integer("end_token", 8600, """""")
tf.flags.DEFINE_integer("unk", 8602, """""")

tf.flags.DEFINE_float("grad_clip", 5.0, """""")
tf.flags.DEFINE_float("learning_rate", 0.001, """""")
tf.flags.DEFINE_integer("epoch", 100, """""")

tf.flags.DEFINE_string("weight_path", "./data/corpus1/weight.save", """""")


from models import model_bi_gate
# TODO: highway, dropout


def average_gradients(tower_grads):
    average_grads = []
    for grad_and_vars in zip(*tower_grads):
        # Note that each grad_and_vars looks like the following:
        #   ((grad0_gpu0, var0_gpu0), ... , (grad0_gpuN, var0_gpuN))
        grads = [g for g, _ in grad_and_vars]
        # Average over the 'tower' dimension.
        grad = tf.stack(grads, 0)
        grad = tf.reduce_mean(grad, 0)
        # Keep in mind that the Variables are redundant because they are shared
        # across towers. So .. we will just return the first tower's pointer to
        # the Variable.
        v = grad_and_vars[0][1]
        grad_and_var = (grad, v)
        average_grads.append(grad_and_var)
    return average_grads


def main_simple():
    ################################
    # step1: Init
    ################################
    random.seed(SEED)
    np.random.seed(SEED)

    tf.logging.info("STEP1: Init...")
    f = open("./data/corpus1/mul_dia.index", 'r')

    hyper_params = {
        "common_vocab": FLAGS.common_vocab,
        "kb_vocab": FLAGS.entities + FLAGS.relations,

        "emb_dim": 512,
        "encoder_layer_num": 1,
        "encoder_h_dim": 512,
        "hred_h_dim": 1024,
        "decoder_gen_layer_num": 1,
        "decoder_gen_h_dim": 512,
        "decoder_mlp_layer_num": 3,
        "decoder_mlp_h_dim": 512
    }

    with tf.device('/cpu:0'):
        model = model_bi_gate.BiScorerGateDecoderModel(hyper_params=hyper_params)
        s_d, t_d, turn_m, s_m, t_m, loss, update = model.build_tower()
        # update = model.optimizer.apply_gradients(grad)

        config = tf.ConfigProto(allow_soft_placement=True)
        config.gpu_options.allow_growth = True
        sess = tf.Session(config=config)
        sess.run(tf.global_variables_initializer())

        try:
            model.load_weight(sess)
        except:
            tf.logging.warning("NO WEIGHT FILE, INIT FROM BEGINNING...")

        tf.logging.info("STEP2: Training...")
        losses = []
        count = 0
        for _ in range(FLAGS.epoch):
            while True:
                try:
                    batch = []
                    for j in range(FLAGS.batch_size):
                        line = f.readline()[:-1]
                        batch.append(json.loads(line))
                except:
                    tf.logging.info("============================================================")
                    tf.logging.info("avg loss: " + str(np.mean(losses)))
                    model.save_weight(sess)
                    f.seek(0)
                    count = 0
                    losses = []
                    break

                count += 1
                if count % 50 == 0:
                    model.save_weight(sess)

                feed_dict = {}
                src_dialogue = np.transpose([sample["src_dialogue"] for sample in batch], [1, 2, 0])
                tgt_dialogue = np.transpose([sample["tgt_dialogue"] for sample in batch], [1, 2, 0])
                turn_mask = np.transpose([sample["turn_mask"] for sample in batch], [1, 0])
                src_mask = np.transpose([sample["src_mask"] for sample in batch], [1, 2, 0])
                tgt_mask = np.transpose([sample["tgt_mask"] for sample in batch], [1, 2, 0])
                feed_dict[s_d] = src_dialogue
                feed_dict[t_d] = tgt_dialogue
                feed_dict[turn_m] = turn_mask
                feed_dict[s_m] = src_mask
                feed_dict[t_m] = tgt_mask

                outputs = sess.run([loss, update], feed_dict=feed_dict)
                loss_value = outputs[0]

                tf.logging.info("---------------------count-------------------")
                tf.logging.info(str(_) + "-" + str(count) + "    " + time.ctime())
                tf.logging.info("---------------------loss-------------------")
                tf.logging.info(loss_value)
                losses.append(loss_value)
                break

        model.save_weight(sess)


if __name__ == "__main__":
    main_simple()