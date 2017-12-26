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

tf.flags.DEFINE_integer("batch_size", 1,
                        """batch_size for eval should be 1 at a time""")
tf.flags.DEFINE_integer("beam_size", 10, """""")
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
tf.flags.DEFINE_float("penalty_factor", 0.6, """""")
tf.flags.DEFINE_integer("epoch", 1, """""")

tf.flags.DEFINE_string("weight_path", "./data/corpus1/weight.save", """""")


from models import model_bi_gate
# TODO: highway, dropout


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

    with tf.device('/gpu:0'):
        model = model_bi_gate.BiScorerGateDecoderModel(hyper_params=hyper_params)
        s_d, s_m, t_m, prob, pred = model.build_eval()

        config = tf.ConfigProto(allow_soft_placement=True)
        config.gpu_options.allow_growth = True
        sess = tf.Session(config=config)
        sess.run(tf.global_variables_initializer())

        try:
            model.load_weight(sess)
        except:
            tf.logging.warning("NO WEIGHT FILE, INIT FROM BEGINNING...")

        tf.logging.info("STEP2: Evaling...")
        count = 0
        for _ in f:
            sample = json.loads(_[:-1])

            count += 1
            if count == 10:
                break

            feed_dict = {}
            src_dialogue = np.transpose([sample["src_dialogue"]], [1, 2, 0])
            tgt_dialogue = np.transpose([sample["tgt_dialogue"]], [1, 2, 0])
            src_mask = np.transpose([sample["src_mask"]], [1, 2, 0])
            turn_mask = np.transpose([sample["turn_mask"]], [1, 0])
            feed_dict[s_d] = src_dialogue
            feed_dict[s_m] = src_mask
            feed_dict[t_m] = turn_mask

            outputs = sess.run([prob, pred], feed_dict=feed_dict)
            # pred_tgt = outputs[0]

            tf.logging.info("---------------------tgt-------------------")
            print tgt_dialogue[sum(turn_mask[0])-1]
            tf.logging.info("---------------------pred-------------------")
            print outputs[0]
            print outputs[1]


if __name__ == "__main__":
    main_simple()