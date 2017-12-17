__author__ = 'liushuman'


import json
import logging
import numpy as np
import os
import random
import sys
import tensorflow as tf
import time

from tensorflow.python.ops import tensor_array_ops, control_flow_ops


SEED = 88


log = logging.getLogger('tensorflow')
log.setLevel(logging.DEBUG)

FLAGS = tf.flags.FLAGS

tf.flags.DEFINE_integer("GPU_num", 4, """""")

tf.flags.DEFINE_integer("batch_size", 40, """""")
tf.flags.DEFINE_integer("beam_size", 10, """""")
tf.flags.DEFINE_integer("dia_max_len", 10, """""")
tf.flags.DEFINE_integer("sen_max_len", 60, """""")
tf.flags.DEFINE_integer("candidate_num", 0,
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
tf.flags.DEFINE_integer("epoch", 200, """""")

tf.flags.DEFINE_string("weight_path", "./../data/corpus1/weight.test", """""")


sys.path.append("..")
from nn_components import graph_base
from nn_components import encoder
from nn_components import HRED
from nn_components import KBscorer
from nn_components import decoder


HYPER_PARAMS = {
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


class TestModel(graph_base.GraphBase):
    """
    """
    def __init__(self, hyper_params=None, params=None):
        graph_base.GraphBase.__init__(self, hyper_params, params)

        # network design
        tf.logging.info("============================================================")
        tf.logging.info("BUILDING NETWORK...")

        self.hyper_params["model_type"] = "Test HREDDECODER"
        self.embedding = graph_base.get_params([self.hyper_params["common_vocab"] + self.hyper_params["kb_vocab"],
                                                self.hyper_params["emb_dim"]])
        self.encoder = encoder.Encoder(
            self.hyper_params["encoder_layer_num"], self.hyper_params["emb_dim"], self.hyper_params["encoder_h_dim"])
        self.kb_scorer = KBscorer.BiKBScorer(
            self.hyper_params["emb_dim"] + self.hyper_params["hred_h_dim"], FLAGS.candidate_num)
        self.hred = HRED.HRED(self.hyper_params["encoder_h_dim"],
                              self.hyper_params["hred_h_dim"], self.hyper_params["emb_dim"])
        self.decoder = decoder.Decoder(
            [self.hyper_params["decoder_gen_layer_num"], self.hyper_params["emb_dim"],
             self.hyper_params["decoder_gen_h_dim"], self.hyper_params["hred_h_dim"] + FLAGS.candidate_num,
             self.hyper_params["common_vocab"]],
            [], [self.hyper_params["decoder_mlp_layer_num"],
                 self.hyper_params["emb_dim"] + FLAGS.candidate_num * 2 +
                 self.hyper_params["hred_h_dim"] + self.hyper_params["decoder_gen_h_dim"],
                 self.hyper_params["decoder_mlp_h_dim"], 2],
            d_type="MASK", hyper_params=None, params=None)

        self.print_params()
        self.encoder.print_params()
        self.kb_scorer.print_params()
        self.hred.print_params()
        self.decoder.print_params()

        self.params = [self.embedding] + self.encoder.params + self.hred.params + \
                      self.kb_scorer.params + self.decoder.params
        params_dict = {}
        for i in range(0, len(self.params)):
            params_dict[str(i)] = self.params[i]
        self.saver = tf.train.Saver(params_dict)

        self.optimizer = self.get_optimizer()

        self.params_simple = [self.embedding] + self.encoder.params + self.hred.params + self.decoder.params

    def build_tower(self):
        # placeholder
        src_dialogue = tf.placeholder(dtype=tf.float32, shape=[
            FLAGS.dia_max_len, FLAGS.sen_max_len, FLAGS.batch_size])
        src_mask = tf.placeholder(dtype=tf.float32, shape=[
            FLAGS.dia_max_len, FLAGS.sen_max_len, FLAGS.batch_size])
        turn_mask = tf.placeholder(dtype=tf.float32, shape=[FLAGS.dia_max_len, FLAGS.batch_size])
        tgt_dialogue = tf.placeholder(dtype=tf.float32, shape=[
            FLAGS.dia_max_len, FLAGS.sen_max_len, FLAGS.batch_size])
        tgt_mask = tf.placeholder(dtype=tf.float32, shape=[
            FLAGS.dia_max_len, FLAGS.sen_max_len, FLAGS.batch_size])

        prob_simple = self.train_forward_simple(src_dialogue, src_mask, turn_mask, tgt_dialogue, tgt_mask)
        train_loss_simple = -tf.reduce_mean(
            tf.reduce_sum(tf.reduce_sum(
                tf.one_hot(tf.to_int32(tgt_dialogue), FLAGS.common_vocab + FLAGS.candidate_num, 1.0, 0.0) *
                tf.log(tf.clip_by_value(prob_simple, 1e-20, 1.0)), -1) * tgt_mask, [0, 1])
        )
        train_grad_simple = self.optimizer.compute_gradients(train_loss_simple, self.params_simple)

        return src_dialogue, src_mask, tgt_dialogue, tgt_mask, train_loss_simple, train_grad_simple

    def train_forward_simple(self, src_dialogue, src_mask, turn_mask, tgt_dialogue, tgt_mask):
        src_index_ta = tensor_array_ops.TensorArray(dtype=tf.float32, size=0, dynamic_size=True).\
            unstack(src_dialogue)
        tgt_index_ta = tensor_array_ops.TensorArray(dtype=tf.float32, size=0, dynamic_size=True).\
            unstack(tgt_dialogue)
        turn_mask_ta = tensor_array_ops.TensorArray(dtype=tf.float32, size=0, dynamic_size=True).\
            unstack(turn_mask)
        src_mask_ta = tensor_array_ops.TensorArray(dtype=tf.float32, size=0, dynamic_size=True).unstack(src_mask)
        tgt_mask_ta = tensor_array_ops.TensorArray(dtype=tf.float32, size=0, dynamic_size=True).unstack(tgt_mask)
        prob_ta = tensor_array_ops.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)

        _, _, _, _, _, _, prob_ta, _ = control_flow_ops.while_loop(
            cond=lambda i, _1, _2, _3, _4, _5, _6, _7: i < src_index_ta.size(),
            body=self._train_step_simple,
            loop_vars=(
                tf.constant(0, tf.int32),
                src_index_ta, tgt_index_ta, turn_mask_ta, src_mask_ta, tgt_mask_ta, prob_ta,
                [tf.zeros([FLAGS.batch_size, self.hyper_params["hred_h_dim"]]),
                 tf.zeros([FLAGS.batch_size, self.hyper_params["hred_h_dim"]])],
            )
        )
        return prob_ta.stack()

    def _train_step_simple(self, i,
                           src_index_ta, tgt_index_ta, turn_mask_ta, src_mask_ta, tgt_mask_ta, prob_ta,
                           hred_cell_tm1):
        src_index = src_index_ta.read(i)
        tgt_index = tgt_index_ta.read(i)
        turn_mask = turn_mask_ta.read(i)
        src_mask = src_mask_ta.read(i)
        tgt_mask = tgt_mask_ta.read(i)
        src_emb = tf.nn.embedding_lookup(self.embedding, tf.to_int32(src_index))
        tgt_emb = tf.nn.embedding_lookup(self.embedding, tf.to_int32(tgt_index))

        src_utterance = self.encoder.forward(src_emb, src_mask)[-1]
        relevant_score = tf.zeros([FLAGS.batch_size, FLAGS.candidate_num])
        weighted_sum_content = tf.zeros([FLAGS.batch_size, self.hyper_params["emb_dim"]])
        tgt_utterance, hred_memory = self.hred.lstm.step_with_content(
            src_utterance,
            tf.expand_dims(turn_mask, -1),
            weighted_sum_content, hred_cell_tm1)
        prob = self.decoder.forward(tf.expand_dims(tgt_index, -1),
                                    tgt_emb, tf.expand_dims(tgt_mask, -1),
                                    tgt_utterance, weighted_sum_content, relevant_score)

        prob_ta = prob_ta.write(i, prob)

        return i+1, src_index_ta, tgt_index_ta, turn_mask_ta, src_mask_ta, tgt_mask_ta, prob_ta, \
               [tgt_utterance, hred_memory]

    def build_eval(self):
        # placeholder
        src_dialogue = tf.placeholder(dtype=tf.float32, shape=[
            FLAGS.dia_max_len, FLAGS.sen_max_len, 1])
        src_mask = tf.placeholder(dtype=tf.float32, shape=[
            FLAGS.dia_max_len, FLAGS.sen_max_len, 1])
        prob, pred = self._test_step(
            tf.unstack(src_dialogue)[0], tf.unstack(src_mask)[0],
            [tf.zeros([FLAGS.batch_size, self.hyper_params["hred_h_dim"]]),
             tf.zeros([FLAGS.batch_size, self.hyper_params["hred_h_dim"]])]
        )
        return src_dialogue, src_mask, prob, pred

    def _test_step(self, src_index, src_mask, hred_cell_tm1):
        turn_mask = tf.ones([1, 1])
        src_emb = tf.nn.embedding_lookup(self.embedding, tf.to_int32(src_index))

        src_utterance = self.encoder.forward(src_emb, src_mask)[-1]
        relevanct_score = tf.zeros([1, FLAGS.candidate_num])
        weighted_sum_content = tf.zeros([1, self.hyper_params["emb_dim"]])
        tgt_utterance, hred_memory = self.hred.lstm.step_with_content(
            src_utterance, turn_mask, weighted_sum_content, hred_cell_tm1)
        prob, pred = self.decoder.forward_with_beam(
            tgt_utterance, weighted_sum_content, relevanct_score, self.embedding)
        return prob, pred

    def get_optimizer(self, *args, **kwargs):
        # return tf.train.AdadeltaOptimizer()
        # ada should be initialize before use, so it should be written in __init__
        # return tf.train.GradientDescentOptimizer(FLAGS.learning_rate)
        # return tf.train.RMSPropOptimizer(FLAGS.learning_rate)
        return tf.train.AdamOptimizer(FLAGS.learning_rate)

    def save_weight(self, session, idx=None):
        if idx:
            self.saver.save(session, FLAGS.weight_path + str(idx))
        else:
            self.saver.save(session, FLAGS.weight_path)

    def load_weight(self, session):
        self.saver.restore(session, FLAGS.weight_path)


def main():
    ################################
    # step1: Init
    ################################
    random.seed(SEED)
    np.random.seed(SEED)

    tf.logging.info("STEP1: Init...")
    f = open("./../data/corpus1/mul_dia.index", 'r')

    with tf.device('/gpu:0'):
        model = TestModel(hyper_params=HYPER_PARAMS)
        s_d, s_m, t_d, t_m, loss, grad = model.build_tower()
        update = model.optimizer.apply_gradients(grad)

        t_sd, t_sm, prob, pred = model.build_eval()

        config = tf.ConfigProto(allow_soft_placement=True)
        config.gpu_options.allow_growth = True
        sess = tf.Session(config=config)
        sess.run(tf.global_variables_initializer())

        try:
            model.load_weight(sess)
        except:
            tf.logging.warning("NO WEIGHT FILE, INIT FROM BEGINNING...")

        stop_control = 2

        tf.logging.info("STEP2: Training...")
        losses = []
        count = 0
        for _ in range(0, FLAGS.epoch):
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
                if count == stop_control:
                    f.seek(0)
                    count = 0
                    print "==avg: ", np.mean(losses)
                    losses = []
                    break

                feed_dict = {}
                src_dialogue = np.transpose([sample["src_dialogue"] for sample in batch], [1, 2, 0])
                src_mask = np.transpose([sample["src_mask"] for sample in batch], [1, 2, 0])
                tgt_dialogue = np.transpose([sample["tgt_dialogue"] for sample in batch], [1, 2, 0])
                tgt_mask = np.transpose([sample["tgt_mask"] for sample in batch], [1, 2, 0])
                feed_dict[s_d] = src_dialogue
                feed_dict[s_m] = src_mask
                feed_dict[t_d] = tgt_dialogue
                feed_dict[t_m] = tgt_mask

                outputs = sess.run([loss, update], feed_dict=feed_dict)
                loss_value = outputs[0]

                tf.logging.info("---------------------count-------------------")
                tf.logging.info(str(_) + "-" + str(count) + "    " + time.ctime())
                tf.logging.info("---------------------loss-------------------")
                tf.logging.info(loss_value)
                losses.append(loss_value)

        model.save_weight(sess)

        tf.logging.info("STEP3: Evaluating...")
        count = 0
        for _ in range(0, stop_control-1):
            try:
                batch = []
                for j in range(1):
                    line = f.readline()[:-1]
                    batch.append(json.loads(line))
            except:
                tf.logging.ERROR("TESTING LOADING ERROR!")

            count += 1

            feed_dict = {}
            src_dialogue = np.transpose([sample["src_dialogue"] for sample in batch], [1, 2, 0])
            src_mask = np.transpose([sample["src_mask"] for sample in batch], [1, 2, 0])
            tgt_dialogue = np.transpose([sample["tgt_dialogue"] for sample in batch], [1, 2, 0])
            feed_dict[t_sd] = src_dialogue
            feed_dict[t_sm] = src_mask

            outputs = sess.run([prob, pred], feed_dict=feed_dict)

            tf.logging.info("---------------------count-------------------")
            tf.logging.info(str(_) + "-" + str(count) + "    " + time.ctime())
            tf.logging.info("---------------------src-------------------")
            tf.logging.info(src_dialogue)
            tf.logging.info("---------------------tgt-------------------")
            tf.logging.info(tgt_dialogue)
            tf.logging.info("---------------------pred-------------------")
            tf.logging.info(outputs[0])
            tf.logging.info(outputs[1])


if __name__ == "__main__":
    main()