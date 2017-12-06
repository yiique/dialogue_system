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

tf.flags.DEFINE_integer("grad_clip", 5.0, """""")
tf.flags.DEFINE_integer("learning_rate", 0.01, """""")
tf.flags.DEFINE_integer("epoch", 3, """""")

tf.flags.DEFINE_string("weight_path", "./data/corpus1/weight.save", """""")


from models import model_bi_gate
# TODO: highway, dropout


def main_simple():
    ################################
    # step1: Init
    ################################
    tf.logging.info("STEP1: Init...")
    random.seed(SEED)
    np.random.seed(SEED)

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
    model = model_bi_gate.BiScorerGateDecoderModel(hyper_params=hyper_params)

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    sess.run(tf.global_variables_initializer())

    try:
        model.load_weight(sess)
    except:
        tf.logging.warning("NO WEIGHT FILE, INIT FROM BEGINNING...")

    losses = []
    f = open("./data/corpus1/mul_dia.index", 'r')
    count = 0
    for _ in range(FLAGS.epoch):
        try:
            batch = []
            for i in range(FLAGS.batch_size):
                line = f.readline()[:-1]
                batch.append(json.loads(line))
        except:
            tf.logging.info("============================================================")
            tf.logging.info("avg loss: " + str(np.mean(losses)))
            model.save_weight(sess)
            f.seek(0)
            losses = []
            continue

        count += 1
        if count % 3 == 0:
            continue

        src_dialogue = np.transpose([sample["src_dialogue"] for sample in batch], [1, 2, 0])
        tgt_dialogue = np.transpose([sample["tgt_dialogue"] for sample in batch], [1, 2, 0])
        turn_mask = np.transpose([sample["turn_mask"] for sample in batch], [1, 0])
        src_mask = np.transpose([sample["src_mask"] for sample in batch], [1, 2, 0])
        tgt_mask = np.transpose([sample["tgt_mask"] for sample in batch], [1, 2, 0])

        output = model.train_batch_simple(sess, src_dialogue, tgt_dialogue, turn_mask, src_mask, tgt_mask)
        loss = output[0]

        tf.logging.info("---------------------count-------------------")
        tf.logging.info(str(_) + "    " + time.ctime())
        tf.logging.info("---------------------loss-------------------")
        tf.logging.info(loss)
        losses.append(loss)

    model.save_weight(sess)


if __name__ == "__main__":
    main_simple()