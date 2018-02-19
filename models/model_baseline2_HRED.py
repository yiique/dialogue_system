__author__ = 'liushuman'


import os
import sys
import tensorflow as tf

from tensorflow.python.ops import tensor_array_ops, control_flow_ops

sys.path.append("..")
from nn_components import graph_base
from nn_components import encoder
from nn_components import HRED
from nn_components import decoder


FLAGS = tf.flags.FLAGS


class BaselineModel(graph_base.GraphBase):
    """
    """
    def __init__(self, hyper_params=None, params=None):
        graph_base.GraphBase.__init__(self, hyper_params, params)

        # network design
        tf.logging.info("============================================================")
        tf.logging.info("BUILDING NETWORK...")

        self.hyper_params["model_type"] = "Baseline2 HRED"
        # Note: order in embedding cannot change!
        self.embedding = graph_base.get_params([FLAGS.vocab_size, self.hyper_params["emb_dim"]])
        self.encoder = encoder.Encoder(
            self.hyper_params["encoder_layer_num"], self.hyper_params["emb_dim"],
            self.hyper_params["encoder_h_dim"], norm=FLAGS.norm)
        self.hred = HRED.HRED(self.hyper_params["encoder_h_dim"],
                              self.hyper_params["hred_h_dim"], 0, norm=FLAGS.norm)
        self.decoder = decoder.AuxDecoder([
            self.hyper_params["decoder_layer_num"], self.hyper_params["emb_dim"], self.hyper_params["decoder_h_dim"],
            self.hyper_params["hred_h_dim"], FLAGS.vocab_size
        ])

        self.print_params()
        self.encoder.print_params()
        self.hred.print_params()
        self.decoder.print_params()

        self.params = [self.embedding] + self.encoder.params + self.hred.params + self.decoder.params
        params_dict = {}
        for i in range(0, len(self.params)):
            params_dict[str(i)] = self.params[i]
        self.saver = tf.train.Saver(params_dict)

        self.optimizer = self.get_optimizer()

    def build_tower(self):
        """
        training tower for gpus
        :parameter: src: source indices in sen_len * batch_size
        :parameter: src_mask: indices mask in 0/1 in sen_len * batch_size
        :parameter: tgt_indices: target true indices in sen_len * batch_size(for embedding lookup)
        :parameter: tgt_mask: indices mask in 0/1 in sen_len * batch_size
        :parameter: turn_mask: turn finish mask in 0/1 batch_size
        :return:
        """
        # placeholder
        src = tf.placeholder(dtype=tf.int32, shape=[FLAGS.sen_max_len, FLAGS.batch_size])
        src_mask = tf.placeholder(dtype=tf.float32, shape=[FLAGS.sen_max_len, FLAGS.batch_size])
        tgt_indices = tf.placeholder(dtype=tf.int32, shape=[FLAGS.sen_max_len, FLAGS.batch_size])
        tgt_mask = tf.placeholder(dtype=tf.float32, shape=[FLAGS.sen_max_len, FLAGS.batch_size])
        turn_mask = tf.placeholder(dtype=tf.float32, shape=[FLAGS.batch_size])

        hred_hidden_tm1 = tf.placeholder(dtype=tf.float32, shape=[FLAGS.batch_size, self.hyper_params["hred_h_dim"]])
        hred_memory_tm1 = tf.placeholder(dtype=tf.float32, shape=[FLAGS.batch_size, self.hyper_params["hred_h_dim"]])

        # supervise_training
        decoder_prob, tgt_utterance, hred_memory = self._train_step(
            src, src_mask, tgt_indices, tgt_mask, turn_mask,
            hred_hidden_tm1, hred_memory_tm1
        )

        train_loss_decoder = -tf.reduce_mean(
            tf.reduce_sum(
                tf.reduce_sum(
                    tf.one_hot(tf.to_int32(tgt_indices), FLAGS.common_vocab + FLAGS.candidate_num, 1.0, 0.0) *
                    tf.log(tf.clip_by_value(decoder_prob, 1e-20, 1.0)), -1) * tgt_mask,
                0)
        )
        train_grad = self.optimizer.compute_gradients(train_loss_decoder, self.params)

        return src, src_mask, tgt_indices, tgt_mask, turn_mask, \
               hred_hidden_tm1, hred_memory_tm1, \
               train_loss_decoder, train_grad, tgt_utterance, hred_memory

    def _train_step(self,
                    src, src_mask, tgt, tgt_mask, turn_mask,
                    hred_hidden_tm1, hred_memory_tm1):

        src_emb = tf.nn.embedding_lookup(self.embedding, src)                                   # len * size * emb_dim
        tgt_emb = tf.nn.embedding_lookup(self.embedding, tgt)

        src_utterance = self.encoder.forward(src_emb, src_mask)[-1]
        tgt_utterance, hred_memory = self.hred.lstm.step_with_content(
            src_utterance, tf.expand_dims(turn_mask, -1),
            tf.zeros([FLAGS.batch_size, 0]), [hred_hidden_tm1, hred_memory_tm1])
        prob = self.decoder.forward(tgt_emb, tf.expand_dims(tgt_mask, -1), tgt_utterance, 1)

        return prob, tgt_utterance, hred_memory

    def build_eval(self):
        # placeholder
        src = tf.placeholder(dtype=tf.int32, shape=[FLAGS.sen_max_len, 1])
        src_mask = tf.placeholder(dtype=tf.float32, shape=[FLAGS.sen_max_len, 1])
        turn_mask = tf.placeholder(dtype=tf.float32, shape=[1])
        hred_hidden_tm1 = tf.placeholder(dtype=tf.float32, shape=[1, self.hyper_params["hred_h_dim"]])
        hred_memory_tm1 = tf.placeholder(dtype=tf.float32, shape=[1, self.hyper_params["hred_h_dim"]])

        prob, pred = self._test_step(
            src, src_mask, turn_mask,
            hred_hidden_tm1, hred_memory_tm1)

        return src, src_mask, turn_mask, \
               hred_hidden_tm1, hred_memory_tm1, \
               prob, pred

    def _test_step(self, src, src_mask, turn_mask, hred_hidden_tm1, hred_memory_tm1):
        src_emb = tf.nn.embedding_lookup(self.embedding, src)                                   # len * size * emb_dim
        src_utterance = self.encoder.forward(src_emb, src_mask, 1)[-1]

        tgt_utterance, hred_memory = self.hred.lstm.step_with_content(
            src_utterance, tf.expand_dims(turn_mask, -1),
            tf.zeros([FLAGS.beam_size, 0]), [hred_hidden_tm1, hred_memory_tm1])

        prob, pred = self.decoder.test_forward(tgt_utterance, self.embedding, 1)

        return prob, pred

    def get_optimizer(self, *args, **kwargs):
        # return tf.train.GradientDescentOptimizer(FLAGS.learning_rate)
        # return tf.train.RMSPropOptimizer(FLAGS.learning_rate)
        # return tf.train.AdadeltaOptimizer()
        return tf.train.AdamOptimizer(FLAGS.learning_rate)

    def save_weight(self, session, suffix=None):
        if suffix:
            self.saver.save(session, FLAGS.weight_path + suffix)
        else:
            self.saver.save(session, FLAGS.weight_path)

    def load_weight(self, session):
        self.saver.restore(session, FLAGS.weight_path)
