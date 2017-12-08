__author__ = 'liushuman'


import os
import sys
import tensorflow as tf

from tensorflow.python.ops import tensor_array_ops, control_flow_ops

sys.path.append("..")
from nn_components import graph_base
from nn_components import encoder
from nn_components import KBscorer
from nn_components import HRED
from nn_components import decoder


FLAGS = tf.flags.FLAGS


class BiScorerGateDecoderModel(graph_base.GraphBase):
    """
    """
    def __init__(self, hyper_params=None, params=None):
        graph_base.GraphBase.__init__(self, hyper_params, params)

        # network design
        tf.logging.info("============================================================")
        tf.logging.info("BUILDING NETWORK...")

        self.hyper_params["model_type"] = "Bilinear Scorer & Gate Decoder"
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

        self.params = [self.embedding] + self.encoder.params + self.kb_scorer.params + \
                       self.hred.params + self.decoder.params
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
        tgt_dialogue = tf.placeholder(dtype=tf.float32, shape=[
            FLAGS.dia_max_len, FLAGS.sen_max_len, FLAGS.batch_size])
        turn_mask = tf.placeholder(dtype=tf.float32, shape=[FLAGS.dia_max_len, FLAGS.batch_size])
        src_mask = tf.placeholder(dtype=tf.float32, shape=[
            FLAGS.dia_max_len, FLAGS.sen_max_len, FLAGS.batch_size])
        tgt_mask = tf.placeholder(dtype=tf.float32, shape=[
            FLAGS.dia_max_len, FLAGS.sen_max_len, FLAGS.batch_size])
        # candidates = tf.placeholder(dtype=tf.float32, shape=[
        #     FLAGS.dia_max_len, FLAGS.batch_size, FLAGS.candidate_num, 2])

        prob_simple = self.train_forward_simple(src_dialogue, tgt_dialogue,
                                                turn_mask, src_mask, tgt_mask)
        train_loss_simple = -tf.reduce_mean(
            tf.reduce_sum(tf.reduce_sum(
                tf.one_hot(tf.to_int32(tgt_dialogue), FLAGS.common_vocab + FLAGS.candidate_num, 1.0, 0.0) *
                tf.log(tf.clip_by_value(prob_simple, 1e-20, 1.0)), -1) * tgt_mask, [0, 1])
        )
        # train_grad_simple = tf.clip_by_global_norm(self.optimizer.compute_gradients(train_loss_simple),
        #                                            FLAGS.grad_clip)
        # train_grad_simple = self.optimizer.compute_gradients(train_loss_simple, self.params_simple)
        train_grad_simple = tf.gradients(train_loss_simple, self.params_simple)
        train_update_simple = self.optimizer.apply_gradients(zip(train_grad_simple, self.params_simple))
        return src_dialogue, tgt_dialogue, turn_mask, src_mask, tgt_mask, \
               train_loss_simple, train_update_simple

    '''def train_batch_simple(self, sess,
                           batch_src_dialogue, batch_tgt_dialogue, batch_turn_mask, batch_src_mask, batch_tgt_mask):
        """
        training process1 with multi turn dialog only
        :return:
        """
        outputs = sess.run([self.train_loss_simple, self.train_updates_simple],
                           feed_dict={self.src_dialogue: batch_src_dialogue,
                                      self.tgt_dialogue: batch_tgt_dialogue,
                                      self.turn_mask: batch_turn_mask,
                                      self.src_mask: batch_src_mask,
                                      self.tgt_mask: batch_tgt_mask})
        return outputs'''

    def train_forward_simple(self, src_dialogue, tgt_dialogue, turn_mask, src_mask, tgt_mask):
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

    def get_optimizer(self, *args, **kwargs):
        # return tf.train.AdadeltaOptimizer()
        # ada should be initialize before use, so it should be written in __init__
        # return tf.train.GradientDescentOptimizer(FLAGS.learning_rate)
        return tf.train.RMSPropOptimizer(FLAGS.learning_rate)

    def save_weight(self, session, idx=None):
        if idx:
            self.saver.save(session, FLAGS.weight_path + str(idx))
        else:
            self.saver.save(session, FLAGS.weight_path)

    def load_weight(self, session):
        self.saver.restore(session, FLAGS.weight_path)