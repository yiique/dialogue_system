__author__ = 'liushuman'

import graph_base
import numpy as np
import sys
import tensorflow as tf


sys.path.append("..")


from tensorflow.python.ops import tensor_array_ops, control_flow_ops


from inference import beam_search


FLAGS = tf.flags.FLAGS


class Decoder(graph_base.GraphBase):
    def __init__(self, gen_params, score_params, mlp_params, d_type="MASK", norm=False,
                 hyper_params=None, params=None):
        graph_base.GraphBase.__init__(self, hyper_params, params)

        self.hyper_params["decoder_type"] = d_type
        self.hyper_params["norm"] = norm

        self.hyper_params["gen_nn_layer_num"] = gen_params[0]
        self.hyper_params["gen_nn_in_dim"] = gen_params[1]
        self.hyper_params["gen_nn_h_dim"] = gen_params[2]
        self.hyper_params["gen_nn_c_dim"] = gen_params[3]
        self.hyper_params["gen_nn_o_dim"] = gen_params[4]

        self.hyper_params["mlp_layer_num"] = mlp_params[0]
        self.hyper_params["mlp_in_dim"] = mlp_params[1]
        self.hyper_params["mlp_h_dim"] = mlp_params[2]
        self.hyper_params["mlp_o_dim"] = 2                                  # generate by kb or common

        self.hyper_params["score_nn_layer_num"] = 1
        self.hyper_params["score_nn_h_dim"] = 1
        if self.hyper_params["decoder_type"] == "MASK":
            pass
        elif self.hyper_params["decoder_type"] == "GATE":
            self.hyper_params["score_in_dim"] = score_params[0]
            self.hyper_params["score_o_dim"] = score_params[1]
        elif self.hyper_params["decoder_type"] == "JOINT":
            self.hyper_params["score_nn_layer_num"] = score_params[0]
            self.hyper_params["score_nn_in_dim"] = score_params[1]
            self.hyper_params["score_nn_h_dim"] = score_params[2]
            self.hyper_params["score_nn_c_dim"] = score_params[3]
            self.hyper_params["score_nn_o_dim"] = score_params[4]

        self.score_unit, self.gen_unit, self.latent_unit, self.predict_unit = self.create_unit()

    def forward(self, x_pred, x_emb, x_m,
                utterance, weighted_sum_content, relevant_score, size=FLAGS.batch_size):
        """
        forward with golden including start and end token
        :param x_pred: max_len * size * 1
        :param x_emb: max_len * size * e_dim
                    Note emb according to true indices, x_pred here is fake indices
        :param x_m: max_len * size * 1
        :param utterance: size * hred_h_dim
        :param weighted_sum_content: size * e_dim
        :param relevant_score: size * can
        :return: max_len * size * vocab_size(with start_token pre written)
        """
        x_pred_ta = tensor_array_ops.TensorArray(dtype=tf.int32, size=0, dynamic_size=True).unstack(tf.to_int32(x_pred))
        x_emb_ta = tensor_array_ops.TensorArray(dtype=tf.float32, size=0, dynamic_size=True).unstack(x_emb)
        x_m_ta = tensor_array_ops.TensorArray(dtype=tf.float32, size=0, dynamic_size=True).unstack(x_m)
        prob_ta = tensor_array_ops.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)
        tgt_start_token = tf.one_hot(tf.ones([size], dtype=tf.int32) * FLAGS.start_token,
                                     FLAGS.common_vocab + FLAGS.candidate_num, 1.0, 0.0)
        prob_ta = prob_ta.write(0, tgt_start_token)

        _, _, _, _, _, _, _, _, _, _, prob_ta, _ = control_flow_ops.while_loop(
            cond=lambda i, _1, _2, _3, _4, _5, _6, _7, _8, _9, _10, _11: i < x_pred_ta.size()-1,
            body=self._step,
            loop_vars=(
                tf.constant(0, dtype=tf.int32),
                tf.nn.softmax(relevant_score),
                tf.stack([[tf.zeros([size, self.hyper_params["score_nn_h_dim"]]),
                           tf.zeros([size, self.hyper_params["score_nn_h_dim"]])]
                          for _ in range(self.hyper_params["score_nn_layer_num"])]),
                tf.stack([[tf.zeros([size, self.hyper_params["gen_nn_h_dim"]]),
                           tf.zeros([size, self.hyper_params["gen_nn_h_dim"]])]
                          for _ in range(self.hyper_params["gen_nn_layer_num"])]),
                utterance, weighted_sum_content, relevant_score,
                x_pred_ta, x_emb_ta, x_m_ta, prob_ta, size
            )
        )

        prob_ta = prob_ta.stack()
        return prob_ta

    def _step(self, i,
              score_tm1,
              score_cell_tm1, gen_cell_tm1,
              utterance, weighted_sum_content, relevant_score,
              x_pred_ta, x_emb_ta, x_m_ta, prob_ta,
              size=FLAGS.batch_size):
        """
        step with golden
        :param score_tm1: size * can
        :param score_cell_tm1: score hidden with memory in layer_num * 2 * size * h_dim
        :param gen_cell_tm1: gen hidden with memory in layer_num * 2 * size * h_dim
        :param utterance: size * hred_h_dim
        :param weighted_sum_content: size * emb_dim
        :param relevant_score: size * can
        """
        x_pred_t = x_pred_ta.read(i)
        x_emb_t = x_emb_ta.read(i)
        x_m_t = x_m_ta.read(i)

        score_logits, score_cell_ts, gen_cell_ts, prob = self._loop_body(
            x_pred_t, x_emb_t, score_tm1, score_cell_tm1, gen_cell_tm1,
            utterance, weighted_sum_content, relevant_score, 1. - x_m_t, size)
        prob_ta = prob_ta.write(i+1, prob)

        score_logits = tf.reshape(score_logits, [FLAGS.batch_size, FLAGS.candidate_num])
        score_cell_ts = tf.reshape(tf.stack(score_cell_ts), [self.hyper_params["score_nn_layer_num"], 2,
                                                             FLAGS.batch_size, self.hyper_params["score_nn_h_dim"]])
        gen_cell_ts = tf.reshape(tf.stack(gen_cell_ts), [self.hyper_params["gen_nn_layer_num"], 2,
                                                         FLAGS.batch_size, self.hyper_params["gen_nn_h_dim"]])

        return i+1, score_logits, score_cell_ts, gen_cell_ts, \
               utterance, weighted_sum_content, relevant_score, \
               x_pred_ta, x_emb_ta, x_m_ta, prob_ta, size

    def forward_with_beam(self, utterance, weighted_sum_content, relevant_score, knowledge_embedding,
                          size=FLAGS.beam_size):
        """
        :param utterance: 1 * hred_h_dim
        :param weighted_sum_content: 1 * emb_dim
        :param relevant_score: 1 * can
        :param knowledge_embedding: (common_vocab + can) * emb_dim
        """
        init_score, init_x_pred, init_finished_mask, init_finished_len, init_score_cells, init_gen_cells, \
        init_utterance, init_weighted_sum_content, init_relevant_score, init_prob_records, init_pred_records = \
            self._init_with_beam(utterance, weighted_sum_content, relevant_score, knowledge_embedding)

        _, _, _, _, _, _, _, _, _, _, _, prob_records, pred_records, _ = control_flow_ops.while_loop(
            cond=lambda i, _1, _2, _3, _4, _5, _6, _7, _8, _9, _10, _11, _12, _13: i < FLAGS.sen_max_len-2,
            body=self._step_with_beam,
            loop_vars=(
                tf.constant(1, dtype=tf.int32),
                init_score, init_x_pred, init_finished_mask, init_finished_len,
                init_score_cells, init_gen_cells,
                init_utterance, init_weighted_sum_content, init_relevant_score, knowledge_embedding,
                init_prob_records, init_pred_records, size
            )
        )

        return prob_records, pred_records

    def _init_with_beam(self, utterance, weighted_sum_content, relevant_score, knowledge_embedding, size=1):
        tgt_start_token = tf.expand_dims(tf.ones(shape=[1], dtype=tf.int32) * FLAGS.start_token, -1)
        tgt_start_emb = tf.reshape(tf.nn.embedding_lookup(knowledge_embedding, tgt_start_token), [1, -1])
        score_tm1 = tf.nn.softmax(relevant_score)
        tgt_start_mask = tf.ones([1])

        score_cell_tm1 = tf.stack([[tf.zeros([1, self.hyper_params["score_nn_h_dim"]]),
                                    tf.zeros([1, self.hyper_params["score_nn_h_dim"]])]
                                   for _ in range(self.hyper_params["score_nn_layer_num"])])
        gen_cell_tm1 = tf.stack([[tf.zeros([1, self.hyper_params["gen_nn_h_dim"]]),
                                  tf.zeros([1, self.hyper_params["gen_nn_h_dim"]])]
                                 for _ in range(self.hyper_params["gen_nn_layer_num"])])

        score_logits, score_cell_ts, gen_cell_ts, prob = self._loop_body(
            tgt_start_token, tgt_start_emb, score_tm1, score_cell_tm1, gen_cell_tm1,
            utterance, weighted_sum_content, relevant_score, 1. - tgt_start_mask, size
        )

        init_score = tf.tile(score_logits, [FLAGS.beam_size, 1])
        prob_top_k, x_top_k = tf.nn.top_k(tf.squeeze(prob), k=FLAGS.beam_size, sorted=True)
        init_x_pred = tf.expand_dims(x_top_k, -1)
        init_finished_mask = tf.to_float(tf.equal(init_x_pred, tf.ones([size, 1], dtype=tf.int32) * FLAGS.end_token))
        init_finished_len = tf.ones([FLAGS.beam_size, 1]) * 2
        init_score_cells = tf.tile(tf.stack(score_cell_ts), [1, 1, FLAGS.beam_size, 1])
        init_gen_cells = tf.tile(tf.stack(gen_cell_ts), [1, 1, FLAGS.beam_size, 1])
        init_utterance = tf.tile(utterance, [FLAGS.beam_size, 1])
        init_weighted_sum_content = tf.tile(weighted_sum_content, [FLAGS.beam_size, 1])
        init_relevant_score = tf.tile(relevant_score, [FLAGS.beam_size, 1])
        init_prob_records = tf.expand_dims(prob_top_k, -1)
        init_pred_records = tf.concat(
            [tf.tile(tgt_start_token, [FLAGS.beam_size, 1]), init_x_pred,
             tf.ones([FLAGS.beam_size, FLAGS.sen_max_len-2], dtype=tf.int32) * FLAGS.end_token], 1)
        return init_score, init_x_pred, init_finished_mask, init_finished_len, init_score_cells, init_gen_cells, \
               init_utterance, init_weighted_sum_content, init_relevant_score, \
               init_prob_records, init_pred_records

    def _step_with_beam(self, i,
                        score_tm1, x_pred_t, finished_mask, finished_len,
                        score_cell_tm1, gen_cell_tm1,
                        utterance, weighted_sum_content, relevant_score, knowledge_embedding,
                        prob_records, pred_records,
                        size=FLAGS.beam_size):
        """
        step with beam search
        :param score_tm1: size * can
        :param x_pred_t: size * 1
        :param finished_mask: beam_size * 1
        :param finished_len: beam_size * 1
        :param utterance: size * hred_h_dim
        :param weighted_sum_content: size * emb_dim
        :param relevant_score: size * can
        :param knowledge_embedding: special embedding in (common_words+can) * e_dim extracted for each dialogue turn
        :param prob_records: total prob for each beam in beam_size * 1
        :param pred_records: total pred record for each beam in beam_size * sen_max_len
        """
        x_emb_t = tf.nn.embedding_lookup(knowledge_embedding, tf.squeeze(x_pred_t, -1))

        score_logits, score_cell_ts, gen_cell_ts, prob = self._loop_body(
            x_pred_t, x_emb_t, score_tm1, score_cell_tm1, gen_cell_tm1,
            utterance, weighted_sum_content, relevant_score, finished_mask, size)

        # beam step
        new_pred, new_finished_mask, new_finished_len, \
        new_scores, new_score_cells, new_gen_cells, new_prob_records, new_pred_records = \
            beam_search.beam_step(i, prob, finished_mask, finished_len,
                                  score_logits, tf.stack(score_cell_ts), tf.stack(gen_cell_ts),
                                  prob_records, pred_records, size)

        new_scores = tf.reshape(new_scores, [FLAGS.beam_size, FLAGS.candidate_num])
        new_pred = tf.reshape(new_pred, [FLAGS.beam_size, 1])
        new_finished_mask = tf.reshape(new_finished_mask, [FLAGS.beam_size, 1])
        new_finished_len = tf.reshape(new_finished_len, [FLAGS.beam_size, 1])
        new_score_cells = tf.reshape(new_score_cells, [self.hyper_params["score_nn_layer_num"], 2,
                                                       FLAGS.beam_size, self.hyper_params["score_nn_h_dim"]])
        new_gen_cells = tf.reshape(new_gen_cells, [self.hyper_params["gen_nn_layer_num"], 2,
                                                   FLAGS.beam_size, self.hyper_params["gen_nn_h_dim"]])
        new_prob_records = tf.reshape(new_prob_records, [FLAGS.beam_size, 1])
        new_pred_records = tf.reshape(new_pred_records, [FLAGS.beam_size, FLAGS.sen_max_len])

        return i+1, new_scores, new_pred, new_finished_mask, new_finished_len, \
               new_score_cells, new_gen_cells, \
               utterance, weighted_sum_content, relevant_score, knowledge_embedding, \
               new_prob_records, new_pred_records, size

    def _loop_body(self,
                   x_pred_t, x_emb_t, score_tm1,
                   score_cell_tm1, gen_cell_tm1,
                   utterance, weighted_sum_content, relevant_score, finished_mask,
                   size=FLAGS.batch_size):
        """
        loop body for each decoder step
        :param score_cell_tm1: stacked  list for memory
        :param gen_cell_tm1: stacked list for memory
        :param finished_mask: finished_mark for one sample(should be 1. - x_mask for training)
        :param size: batch_size fore training, beam_size for eval
        """
        # score
        score_cell_ts = [tf.unstack(cell) for cell in tf.unstack(score_cell_tm1)]
        if self.hyper_params["decoder_type"] == "MASK":
            score_logits = self.score_unit(score_tm1, x_pred_t, size)
        elif self.hyper_params["decoder_type"] == "GATE":
            score_content = tf.concat([
                utterance, weighted_sum_content, relevant_score, score_tm1, x_emb_t], 1)
            score_logits = self.score_unit(score_content, score_tm1, x_pred_t, size)
        elif self.hyper_params["decoder_type"] == "JOINT":
            score_content = tf.concat([
                utterance, weighted_sum_content, relevant_score], 1)
            score_x = tf.concat([score_tm1, x_emb_t], 1)
            score_cell_ts, score_logits = self.score_unit(
                score_x, x_pred_t, score_tm1, score_content, score_cell_tm1, size)
        else:
            tf.logging.ERROR("raise decoder scorer key error!")
            raise KeyError

        # gen
        gen_content = tf.concat([utterance, score_logits], 1)
        gen_cell_ts, gen_logits = self.gen_unit(x_emb_t, 1. - finished_mask, gen_content, gen_cell_tm1)

        # latent
        if self.hyper_params["decoder_type"] == "MASK" or self.hyper_params["decoder_type"] == "GATE":
            latent_hidden = tf.concat([x_emb_t, score_logits, utterance, relevant_score, gen_cell_ts[-1][0]], 1)
        elif self.hyper_params["decoder_type"] == "JOINT":
            latent_hidden = tf.concat([
                x_emb_t, score_logits, utterance, relevant_score, gen_cell_ts[-1][0], score_cell_ts[-1][0]], 1)
        else:
            tf.logging.ERROR("raise decoder scorer key error!")
            raise KeyError
        latent_logits = self.latent_unit(latent_hidden)

        # predict
        prob = self.predict_unit(gen_logits, score_logits, latent_logits, size)     # size * (common_words+can)
        return score_logits, score_cell_ts, gen_cell_ts, prob

    def create_unit(self):
        # scorer
        if self.hyper_params["decoder_type"] == "MASK":
            pass
        elif self.hyper_params["decoder_type"] == "GATE":
            self.w_score_lr = graph_base.get_params([self.hyper_params["score_in_dim"],
                                                     self.hyper_params["score_o_dim"]])
            self.b_score_lr = graph_base.get_params([self.hyper_params["score_o_dim"]])
            self.params.extend([self.w_score_lr, self.b_score_lr])
        elif self.hyper_params["decoder_type"] == "JOINT":
            self.score_lstms = []
            for i in range(self.hyper_params["score_nn_layer_num"]):
                if i == 0:
                    lstm = graph_base.LSTM(
                        self.hyper_params["score_nn_in_dim"], self.hyper_params["score_nn_h_dim"],
                        self.hyper_params["score_nn_c_dim"], norm=self.hyper_params["norm"])
                else:
                    lstm = graph_base.LSTM(
                        self.hyper_params["score_nn_h_dim"], self.hyper_params["score_nn_h_dim"],
                        self.hyper_params["score_nn_c_dim"], norm=self.hyper_params["norm"])
                self.score_lstms.append(lstm)
                self.params.extend(lstm.params)
            self.w_score_lr = graph_base.get_params([self.hyper_params["score_nn_h_dim"],
                                                     self.hyper_params["score_nn_o_dim"]])
            self.b_score_lr = graph_base.get_params([self.hyper_params["score_nn_o_dim"]])
            self.params.extend([self.w_score_lr, self.b_score_lr])

        # gen lstm
        self.gen_lstms = []
        for i in range(self.hyper_params["gen_nn_layer_num"]):
            if i == 0:
                lstm = graph_base.LSTM(
                    self.hyper_params["gen_nn_in_dim"], self.hyper_params["gen_nn_h_dim"],
                    self.hyper_params["gen_nn_c_dim"], norm=self.hyper_params["norm"])
            else:
                lstm = graph_base.LSTM(
                    self.hyper_params["gen_nn_h_dim"], self.hyper_params["gen_nn_h_dim"],
                    self.hyper_params["gen_nn_c_dim"], norm=self.hyper_params["norm"])
            self.gen_lstms.append(lstm)
            self.params.extend(lstm.params)
        self.w_gen_lr = graph_base.get_params([self.hyper_params["gen_nn_h_dim"], self.hyper_params["gen_nn_o_dim"]])
        self.b_gen_lr = graph_base.get_params([self.hyper_params["gen_nn_o_dim"]])
        self.params.extend([self.w_gen_lr, self.b_gen_lr])

        # latent preceptrons
        self.perceptrons = []
        for i in range(self.hyper_params["mlp_layer_num"]):
            if i == 0:
                w = graph_base.get_params([self.hyper_params["mlp_in_dim"], self.hyper_params["mlp_h_dim"]])
            elif i == self.hyper_params["mlp_layer_num"] - 1:
                w = graph_base.get_params([self.hyper_params["mlp_h_dim"], self.hyper_params["mlp_o_dim"]])
            else:
                w = graph_base.get_params([self.hyper_params["mlp_h_dim"], self.hyper_params["mlp_h_dim"]])
            self.perceptrons.append([w])
            self.params.extend([w])

        def mask_score_unit(score_tm1, word_pred_t, size=FLAGS.batch_size):
            """
            :param score_tm1: score tm1 in size * can
            :param word_pred: word pred in size * 1 (relative index)
            :return:
            """
            score_tm1_expand = tf.concat(
                [tf.zeros([size, FLAGS.common_vocab], dtype=tf.float32), score_tm1], 1)
            score_mask = tf.one_hot(tf.to_int32(tf.squeeze(word_pred_t)),
                                    FLAGS.common_vocab + FLAGS.candidate_num, 0.0, 1.0)         # reverse one hot
            score_logits = tf.nn.softmax(tf.slice(score_tm1_expand * score_mask,
                                                  [0, FLAGS.common_vocab], [size, FLAGS.candidate_num]))
            return score_logits

        def gate_score_unit(content, score_tm1, word_pred_t, size=FLAGS.batch_size):
            """
            :param content: content for score_gate in size * in_dim
                (utterance + weighted sum content + relevant score + scoretm1 + tripletm1(x_t))
            :param word_pred_t: size * 1 (relative index)
            :return: score prob in size * can
            """
            score_gate = tf.sigmoid(tf.matmul(content, self.w_score_lr) + self.b_score_lr)
            score_mask = tf.expand_dims(tf.reduce_sum(
                tf.one_hot(tf.to_int32(tf.squeeze(word_pred_t)), FLAGS.common_vocab + FLAGS.candidate_num, 1.0, 0.0) *
                tf.concat([tf.zeros(size, FLAGS.common_vocab), tf.ones(size, FLAGS.candidate_num)], 1), 1), -1)
            score_logits = tf.nn.softmax(score_gate * score_tm1) * score_mask + (1. - score_mask) * score_tm1
            return score_logits

        def joint_score_unit(x, word_pred_t, score_tm1, content, cell_tm1s, size=FLAGS.batch_size):
            """
            :param x: input for first layer in size * in_dim
                (scoretm1 + tripletm1(x_t))
            :param word_pred_t: size * 1(control whether to update the score)
            :param score_tm1: size * can for recover
            :param content: content for all layer in size * c_dim
                (utterance + weighted sum content + relevant score)
            :param cell_tm1s: hidden with memory for each layer in layer_num * [size * h_dim]
            :return: hidden with memory in layer_num * [size * h_dim], score softmax in size * can
            """
            hidden_t = x
            cell_ts = []
            cell_tm1s = [tf.unstack(cell) for cell in tf.unstack(cell_tm1s)]
            # both hidden and logits do not update when mask is zero
            score_mask = tf.expand_dims(tf.reduce_sum(
                tf.one_hot(tf.to_int32(tf.squeeze(word_pred_t)), FLAGS.common_vocab + FLAGS.candidate_num, 1.0, 0.0) *
                tf.concat([tf.zeros(size, FLAGS.common_vocab), tf.ones(size, FLAGS.candidate_num)], 1), 1), -1)
            for i in range(self.hyper_params["score_nn_layer_num"]):
                lstm = self.score_lstms[i]
                cell_tm1 = cell_tm1s[i]

                hidden_t, memory_t = lstm.step_with_content(hidden_t, score_mask, content, cell_tm1)
                cell_ts.append([hidden_t, memory_t])
            score_logits = tf.nn.softmax(tf.matmul(hidden_t, self.w_score_lr) + self.b_score_lr)
            score_logits = score_mask * score_logits + (1. - score_mask) * score_tm1
            return cell_ts, score_logits

        def gen_unit(x, x_mask, content, cell_tm1s):
            """
            common word prob generate
            :param x: input for first layer in size * in_dim
            :param x_mask: mask in size * 1
            :param content: content for all layer in size * c_dim(utterance + score)
            :param cell_tm1s: hidden with memory for each layer in layer_num * size * h_dim
            :return: hidden with memory in layer_num * [size * h_dim], word softmax in size * common_vocab
            """
            hidden_t = x
            cell_ts = []
            cell_tm1s = [tf.unstack(cell) for cell in tf.unstack(cell_tm1s)]
            for i in range(self.hyper_params["gen_nn_layer_num"]):
                lstm = self.gen_lstms[i]
                cell_tm1 = cell_tm1s[i]

                hidden_t, memory_t = lstm.step_with_content(hidden_t, x_mask, content, cell_tm1)
                cell_ts.append([hidden_t, memory_t])
            gen_logits = tf.nn.softmax(tf.matmul(hidden_t, self.w_gen_lr) + self.b_gen_lr)
            return cell_ts, gen_logits

        def latent_unit(hidden):
            """
            :param hidden: hidden in size * mlp_in_dim(wordt + scoret + utterance + relevant score + hidden * 1/2)
            :return: latent prediction in size * 2
            """
            for i in range(self.hyper_params["mlp_layer_num"]):
                layer = self.perceptrons[i]
                hidden = tf.tanh(tf.matmul(hidden, layer[0]))
            latent_logits = tf.nn.softmax(hidden)
            return latent_logits

        def predict_unit(gen_logits, score_logits, latent_logits, size=FLAGS.batch_size):
            """
            :param gen_logits: word softmax in size * vocab
            :param score_logits: score softmax in size * can
            :param latent_logits: latent softmax in size * 2
            :return: total prob in size * (vocab + can)
            """
            gen_logits_extend = tf.concat([
                gen_logits, tf.zeros([size, FLAGS.candidate_num], dtype=tf.float32)], 1)
            score_logits_extend = tf.concat([
                tf.zeros([size, FLAGS.common_vocab], dtype=tf.float32), score_logits], 1)

            prob = (gen_logits_extend * tf.slice(latent_logits, [0, 0], [size, 1])) + \
                   (score_logits_extend * tf.slice(latent_logits, [0, 1], [size, 1]))
            return prob

        if self.hyper_params["decoder_type"] == "MASK":
            return mask_score_unit, gen_unit, latent_unit, predict_unit
        elif self.hyper_params["decoder_type"] == "GATE":
            return gate_score_unit, gen_unit, latent_unit, predict_unit
        elif self.hyper_params["decoder_type"] == "JOINT":
            return joint_score_unit, gen_unit, latent_unit, predict_unit
        raise KeyError


class AuxDecoder(graph_base.GraphBase):
    def __init__(self, gen_params, norm=False, hyper_params=None, params=None):
        graph_base.GraphBase.__init__(self, hyper_params, params)

        self.hyper_params["gen_nn_layer_num"] = gen_params[0]
        self.hyper_params["gen_nn_in_dim"] = gen_params[1]
        self.hyper_params["gen_nn_h_dim"] = gen_params[2]
        self.hyper_params["gen_nn_c_dim"] = gen_params[3]
        self.hyper_params["gen_nn_o_dim"] = gen_params[4]
        self.hyper_params["norm"] = norm

        self.gen_unit = self.create_unit()

    def forward(self, x_emb, x_m,
                utterance, size=FLAGS.batch_size):
        """
        forward with golden including start and end token
        :param x_emb: max_len * size * e_dim
        :param x_m: max_len * size * 1
        :param utterance: size * hred_h_dim
        :return: max_len * size * vocab_size(with start_token pre written)
        """
        x_emb_ta = tensor_array_ops.TensorArray(dtype=tf.float32, size=0, dynamic_size=True).unstack(x_emb)
        x_m_ta = tensor_array_ops.TensorArray(dtype=tf.float32, size=0, dynamic_size=True).unstack(x_m)
        prob_ta = tensor_array_ops.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)
        tgt_start_token = tf.one_hot(tf.ones([size], dtype=tf.int32) * FLAGS.start_token,
                                     FLAGS.common_vocab + FLAGS.candidate_num, 1.0, 0.0)
        prob_ta = prob_ta.write(0, tgt_start_token)

        _, _, _, _, _, prob_ta, _ = control_flow_ops.while_loop(
            cond=lambda i, _1, _2, _3, _4, _5, _6: i < x_emb_ta.size()-1,
            body=self._step,
            loop_vars=(
                tf.constant(0, dtype=tf.int32),
                tf.stack([[tf.zeros([size, self.hyper_params["gen_nn_h_dim"]]),
                           tf.zeros([size, self.hyper_params["gen_nn_h_dim"]])]
                          for _ in range(self.hyper_params["gen_nn_layer_num"])]),
                utterance, x_emb_ta, x_m_ta, prob_ta, size
            )
        )

        prob_ta = prob_ta.stack()
        return prob_ta

    def _step(self, i, gen_cell_tm1,
              utterance,
              x_emb_ta, x_m_ta, prob_ta,
              size=FLAGS.batch_size):
        """
        step with golden
        :param gen_cell_tm1: gen hidden with memory in layer_num * 2 * size * h_dim
        :param utterance: size * encoder_h_dim
        """
        x_emb_t = x_emb_ta.read(i)
        x_m_t = x_m_ta.read(i)

        gen_cell_ts, gen_logits = self.gen_unit(x_emb_t, x_m_t, utterance, gen_cell_tm1)
        prob_ta = prob_ta.write(i+1, gen_logits)

        gen_cell_ts = tf.reshape(tf.stack(gen_cell_ts), [self.hyper_params["gen_nn_layer_num"], 2,
                                                         FLAGS.batch_size, self.hyper_params["gen_nn_h_dim"]])

        return i+1, gen_cell_ts, \
               utterance, \
               x_emb_ta, x_m_ta, prob_ta, size

    def test_forward(self, utterance, embedding, size=FLAGS.batch_size):
        prob_ta = tensor_array_ops.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)
        pred_ta = tensor_array_ops.TensorArray(dtype=tf.int32, size=0, dynamic_size=True)
        tgt_start_token = tf.ones([FLAGS.batch_size, 1], dtype=tf.int32) * FLAGS.start_token

        _, _, _, _, _, prob_ta, pred_ta, _ = control_flow_ops.while_loop(
            cond=lambda i, _1, _2, _3, _4, _5, _6, _7: i < FLAGS.sen_max_len-1,
            body=self._test_step,
            loop_vars=(
                tf.constant(0, dtype=tf.int32), tgt_start_token,
                tf.stack([[tf.zeros([size, self.hyper_params["gen_nn_h_dim"]]),
                           tf.zeros([size, self.hyper_params["gen_nn_h_dim"]])]
                          for _ in range(self.hyper_params["gen_nn_layer_num"])]),
                utterance, embedding, prob_ta, pred_ta, size
            )
        )
        return prob_ta.stack(), pred_ta.stack()

    def _test_step(self, i, x_pred, gen_cell_tm1,
                   utterance, embedding, prob_ta, pred_ta,
                   size=FLAGS.batch_size):
        x_emb_t = tf.nn.embedding_lookup(embedding, tf.reshape(x_pred, [-1]))
        x_m = tf.ones([size, 1])

        gen_cell_ts, gen_logits = self.gen_unit(x_emb_t, x_m, utterance, gen_cell_tm1)
        gen_pred = tf.to_int32(tf.reshape(tf.arg_max(gen_logits, 1), [FLAGS.batch_size, 1]))
        prob_ta = prob_ta.write(i, gen_logits)
        pred_ta = pred_ta.write(i, gen_pred)

        return i+1, gen_pred, \
               tf.reshape(tf.stack(gen_cell_ts),
                          [self.hyper_params["gen_nn_layer_num"], 2,
                           FLAGS.batch_size, self.hyper_params["gen_nn_h_dim"]]), \
               utterance, embedding, prob_ta, pred_ta, size

    def create_unit(self):
        # gen lstm
        self.gen_lstms = []
        for i in range(self.hyper_params["gen_nn_layer_num"]):
            if i == 0:
                lstm = graph_base.LSTM(
                    self.hyper_params["gen_nn_in_dim"], self.hyper_params["gen_nn_h_dim"],
                    self.hyper_params["gen_nn_c_dim"], norm=self.hyper_params["norm"])
            else:
                lstm = graph_base.LSTM(
                    self.hyper_params["gen_nn_h_dim"], self.hyper_params["gen_nn_h_dim"],
                    self.hyper_params["gen_nn_c_dim"], norm=self.hyper_params["norm"])
            self.gen_lstms.append(lstm)
            self.params.extend(lstm.params)
        self.w_gen_lr = graph_base.get_params([self.hyper_params["gen_nn_h_dim"], self.hyper_params["gen_nn_o_dim"]])
        self.b_gen_lr = graph_base.get_params([self.hyper_params["gen_nn_o_dim"]])
        self.params.extend([self.w_gen_lr, self.b_gen_lr])

        def gen_unit(x, x_mask, content, cell_tm1s):
            """
            common word prob generate
            :param x: input for first layer in size * in_dim
            :param x_mask: mask in size * 1
            :param cell_tm1s: hidden with memory for each layer in layer_num * size * h_dim
            :return: hidden with memory in layer_num * [size * h_dim], word softmax in size * common_vocab
            """
            hidden_t = x
            cell_ts = []
            cell_tm1s = [tf.unstack(cell) for cell in tf.unstack(cell_tm1s)]
            for i in range(self.hyper_params["gen_nn_layer_num"]):
                lstm = self.gen_lstms[i]
                cell_tm1 = cell_tm1s[i]

                hidden_t, memory_t = lstm.step_with_content(hidden_t, x_mask, content, cell_tm1)
                cell_ts.append([hidden_t, memory_t])
            gen_logits = tf.nn.softmax(tf.matmul(hidden_t, self.w_gen_lr) + self.b_gen_lr)
            return cell_ts, gen_logits
        return gen_unit
