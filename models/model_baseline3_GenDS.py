__author__ = 'liushuman'


import os
import sys
import tensorflow as tf

from tensorflow.python.ops import tensor_array_ops, control_flow_ops

sys.path.append("..")
from nn_components import graph_base
from nn_components import encoder
from nn_components import KBscorer
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

        self.hyper_params["model_type"] = "Baseline3 GenDS"
        # Note: order in embedding cannot change!
        self.embedding = graph_base.get_params([FLAGS.common_vocab + FLAGS.entities +
                                                FLAGS.relations + FLAGS.sen_max_len,
                                                self.hyper_params["emb_dim"]])
        self.encoder = encoder.Encoder(
            self.hyper_params["encoder_layer_num"], self.hyper_params["emb_dim"],
            self.hyper_params["encoder_h_dim"], norm=FLAGS.norm)
        self.kb_retriever = KBscorer.KBRetriever(
            self.hyper_params["emb_dim"], self.hyper_params["encoder_h_dim"], self.hyper_params["hred_h_dim"],
            self.hyper_params["k_cnn_layer_num"], self.hyper_params["k_cnn_kernel_size"],
            self.hyper_params["k_cnn_h_dim"],
            self.hyper_params["enquirer_mlp_layer_num"], self.hyper_params["enquirer_mlp_h_dim"],
            self.hyper_params["diffuser_mlp_layer_num"], self.hyper_params["diffuser_mlp_h_dim"],
            self.hyper_params["scorer_mlp_layer_num"], self.hyper_params["scorer_mlp_h_dim"])
        self.decoder = decoder.Decoder(
            [self.hyper_params["decoder_gen_layer_num"], self.hyper_params["emb_dim"],
             self.hyper_params["decoder_gen_h_dim"], self.hyper_params["encoder_h_dim"] + FLAGS.candidate_num,
             FLAGS.common_vocab + FLAGS.candidate_num],
            [self.hyper_params["hred_h_dim"] + self.hyper_params["emb_dim"] * 2 + FLAGS.candidate_num * 2 +
             self.hyper_params["emb_dim"],
             FLAGS.candidate_num],
            [self.hyper_params["decoder_mlp_layer_num"],
             self.hyper_params["emb_dim"] + FLAGS.candidate_num * 2 +
             self.hyper_params["encoder_h_dim"] + self.hyper_params["decoder_gen_h_dim"],
             self.hyper_params["decoder_mlp_h_dim"], 2],
            d_type="GATE", norm=FLAGS.norm, hyper_params=None, params=None)

        self.print_params()
        self.encoder.print_params()
        self.kb_retriever.print_params()
        self.decoder.print_params()

        self.params = [self.embedding] + self.encoder.params + self.kb_retriever.params + self.decoder.params
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
        :parameter: tgt: target indices in sen_len * batch_size(here indices range from 0-common_vocab + can_num)
        :parameter: tgt_mask: indices mask in 0/1 in sen_len * batch_size
        :parameter: turn_mask: turn finish mask in 0/1 batch_size
        :parameter: enquire_strings: char indices of string match candidates in src in batch_size * e_c_num * sen_len
        :parameter: enquire_entities: entity+relation/entity indices of string match cands in batch_size * e_c_num * 2
        :parameter: enquire_mask: indices mask for enquire strings in 0/1 in size * e_c_num * sen_len
        :parameter: enquire_score_golden: golden score for enquire cands in 0/1 in size * e_c_num
        :return:
        """
        # placeholder
        src = tf.placeholder(dtype=tf.int32, shape=[FLAGS.sen_max_len, FLAGS.batch_size])
        src_mask = tf.placeholder(dtype=tf.float32, shape=[FLAGS.sen_max_len, FLAGS.batch_size])
        tgt_indices = tf.placeholder(dtype=tf.int32, shape=[FLAGS.sen_max_len, FLAGS.batch_size])
        tgt = tf.placeholder(dtype=tf.float32, shape=[FLAGS.sen_max_len, FLAGS.batch_size])
        tgt_mask = tf.placeholder(dtype=tf.float32, shape=[FLAGS.sen_max_len, FLAGS.batch_size])
        enquire_strings = tf.placeholder(dtype=tf.int32,
                                         shape=[FLAGS.batch_size, FLAGS.enquire_can_num, FLAGS.sen_max_len])
        enquire_entities = tf.placeholder(dtype=tf.int32, shape=[FLAGS.batch_size, FLAGS.enquire_can_num, 2])
        enquire_mask = tf.placeholder(dtype=tf.float32,
                                      shape=[FLAGS.batch_size, FLAGS.enquire_can_num, FLAGS.sen_max_len])
        enquire_score_golden = tf.placeholder(dtype=tf.float32,
                                              shape=[FLAGS.batch_size, FLAGS.enquire_can_num])

        # supervise_training
        enquire_score, decoder_prob = self._train_step(
            src, src_mask, tgt_indices, tgt, tgt_mask,
            enquire_strings, enquire_entities, enquire_mask, enquire_score_golden)

        train_loss_alpha = tf.reduce_mean(tf.reduce_sum(
            tf.square(enquire_score - enquire_score_golden), -1
        ))
        train_loss_decoder = -tf.reduce_mean(
            tf.reduce_sum(
                tf.reduce_sum(
                    tf.one_hot(tf.to_int32(tgt), FLAGS.common_vocab + FLAGS.candidate_num, 1.0, 0.0) *
                    tf.log(tf.clip_by_value(decoder_prob, 1e-20, 1.0)), -1) * tgt_mask,
                0)
        )
        train_loss = FLAGS.loss_alpha * train_loss_alpha + FLAGS.loss_decoder * train_loss_decoder
        train_grad = self.optimizer.compute_gradients(train_loss, self.params)

        return src, src_mask, tgt_indices, tgt, tgt_mask, \
               enquire_strings, enquire_entities, enquire_mask, enquire_score_golden, \
               train_loss_alpha, train_loss_decoder, train_loss, train_grad

    def _train_step(self,
                    src, src_mask, tgt_indices, tgt, tgt_mask,
                    enquire_strings, enquire_entities, enquire_mask, enquire_score_golden):
        # TODO: try a smaller type embedding

        src_emb = tf.nn.embedding_lookup(self.embedding, src)                                   # len * size * emb_dim

        enquire_strings_emb = tf.nn.embedding_lookup(self.embedding, enquire_strings)       # size * e_num * max_l * emb
        enquire_entities_emb = tf.nn.embedding_lookup(self.embedding, enquire_entities)     # size * e_num * 2 * emb

        enquire_strings_avg = tf.reduce_sum(enquire_strings_emb * tf.expand_dims(enquire_mask, -1), 2)\
                              / tf.clip_by_value(tf.expand_dims(tf.reduce_sum(enquire_mask, -1), -1),
                                                 1, FLAGS.sen_max_len)             # size * e_num * emb(0 for null)
        enquire_entities_avg = tf.reduce_mean(enquire_entities_emb, 2)                      # size * e_num * emb_dim
        enquire_entities_sum = \
            tf.reduce_sum(enquire_entities_avg * tf.expand_dims(enquire_score_golden, -1), 1)\
                / tf.clip_by_value(tf.reduce_sum(enquire_score_golden, 1, keep_dims=True),
                                   1.0, FLAGS.enquire_can_num)                                   # size * emb_dim

        sum_content = enquire_entities_sum
        candidate_emb = enquire_entities_avg

        pred_embedding = tf.concat([
            tf.tile(tf.expand_dims(
                tf.slice(self.embedding, [0, 0], [FLAGS.common_vocab, self.hyper_params["emb_dim"]]), 0),
                [FLAGS.batch_size, 1, 1]),
            candidate_emb], 1)                                                              # size * v+c_num * e_dim
        emb_ta = tensor_array_ops.TensorArray(dtype=tf.float32, size=0, dynamic_size=True).unstack(pred_embedding)
        idx_ta = tensor_array_ops.TensorArray(dtype=tf.int32, size=0, dynamic_size=True).\
            unstack(tf.transpose(tf.to_int32(tgt)))
        tgt_ta = tensor_array_ops.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)
        def _loop_body(i, emb_ta, idx_ta, tgt_e_ta):
            emb_t = emb_ta.read(i)                                                          # v+c_num * e_dim
            idx_t = idx_ta.read(i)                                                          # max_len
            tgt_e = tf.nn.embedding_lookup(emb_t, idx_t)                                    # max_len * emb_dim
            tgt_e_ta = tgt_e_ta.write(i, tgt_e)
            return i+1, emb_ta, idx_ta, tgt_e_ta
        _, _, _, tgt_ta = control_flow_ops.while_loop(
            cond=lambda i, _1, _2, _3: i < FLAGS.batch_size,
            body=_loop_body,
            loop_vars=(tf.constant(0, tf.int32),
                       emb_ta, idx_ta, tgt_ta)
        )
        tgt_emb = tf.transpose(tgt_ta.stack(), [1, 0, 2])

        src_utterance = self.encoder.forward(src_emb, src_mask)[-1]
        knowledge_utterance, enquire_score = self.kb_retriever.enquirer_unit(
            src_emb, src_mask, enquire_strings_avg, tf.zeros([FLAGS.batch_size, self.hyper_params["hred_h_dim"]]))
        prob = self.decoder.forward(tf.expand_dims(tgt, -1),
                                    tgt_emb, tf.expand_dims(tgt_mask, -1),
                                    src_utterance, sum_content,
                                    enquire_score_golden)

        return enquire_score, prob

    def build_eval(self):
        # placeholder
        src = tf.placeholder(dtype=tf.int32, shape=[FLAGS.sen_max_len, 1])
        src_mask = tf.placeholder(dtype=tf.float32, shape=[FLAGS.sen_max_len, 1])
        enquire_strings = tf.placeholder(dtype=tf.int32, shape=[1, FLAGS.enquire_can_num, FLAGS.sen_max_len])
        enquire_entities = tf.placeholder(dtype=tf.int32, shape=[1, FLAGS.enquire_can_num, 2])
        enquire_mask = tf.placeholder(dtype=tf.float32, shape=[1, FLAGS.enquire_can_num, FLAGS.sen_max_len])

        enquire_score, prob, pred = self._test_step(
            src, src_mask, enquire_strings, enquire_entities, enquire_mask)

        return src, src_mask, enquire_strings, enquire_entities, enquire_mask, \
               enquire_score, prob, pred

    def _test_step(self, src, src_mask,
                   enquire_strings, enquire_entities, enquire_mask):
        # TODO: entity embedding has difference here, should be changed to obj embedding
        src_emb = tf.nn.embedding_lookup(self.embedding, src)                                   # len * size * emb_dim

        enquire_strings_emb = tf.nn.embedding_lookup(self.embedding, enquire_strings)       # size * e_num * max_l * emb
        enquire_entities_emb = tf.nn.embedding_lookup(self.embedding, enquire_entities)     # size * e_num * 2 * emb

        enquire_strings_avg = tf.reduce_sum(enquire_strings_emb * tf.expand_dims(enquire_mask, -1), 2)\
                              / tf.clip_by_value(tf.expand_dims(tf.reduce_sum(enquire_mask, -1), -1),
                                                 1, FLAGS.sen_max_len)             # size * e_num * emb(0 for null)
        enquire_entities_avg = tf.reduce_mean(enquire_entities_emb, 2)                      # size * e_num * emb_dim
        enquire_entity_mask = tf.clip_by_value(
            tf.reduce_sum(enquire_mask, -1, keep_dims=True), 0.0, 1.0)                      # size * e_c_num * 1

        src_utterance = self.encoder.forward(src_emb, src_mask, 1)[-1]

        knowledge_utterance, enquire_score = self.kb_retriever.enquirer_unit(
            src_emb, src_mask, enquire_strings_avg, tf.zeros([1, self.hyper_params["hred_h_dim"]]), 1)
        enquire_entities_sum = \
            tf.reduce_sum(enquire_entities_avg * tf.expand_dims(enquire_score, -1) * enquire_entity_mask, 1)\
                / tf.clip_by_value(tf.reduce_sum(tf.reduce_sum(enquire_entity_mask, -1) * enquire_score, -1,
                                                 keep_dims=True),
                                   1.0, FLAGS.enquire_can_num)                                   # size * emb_dim
        candidate_emb = enquire_entities_avg          # size * can_num * e_dim
        sum_content = tf.reduce_sum(enquire_entities_avg * tf.expand_dims(enquire_score, -1), 1) \
            / tf.clip_by_value(tf.reduce_sum(enquire_score, -1, keep_dims=True), 1.0, FLAGS.enquire_can_num)

        pred_embedding = tf.concat([
            tf.slice(self.embedding, [0, 0], [FLAGS.common_vocab, self.hyper_params["emb_dim"]]),
            tf.squeeze(candidate_emb)], 0)
        prob, pred = self.decoder.forward_with_beam(
            src_utterance, sum_content, enquire_score, pred_embedding)

        return enquire_score, prob, pred

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
