__author__ = 'liushuman'

import encoder
import graph_base
import tensorflow as tf


FLAGS = tf.flags.FLAGS


class BiKBScorer(graph_base.GraphBase):
    def __init__(self, in_dim, out_dim, hyper_params=None, params=None):
        graph_base.GraphBase.__init__(self, hyper_params, params)

        self.hyper_params["score_type"] = "Bilinear KB Scorer"
        self.hyper_params["in_dim"] = in_dim
        self.hyper_params["out_dim"] = out_dim
        self.scorer = self.create_bi_scorer()

    def create_bi_scorer(self):
        self.S = graph_base.get_params([self.hyper_params["in_dim"], self.hyper_params["out_dim"]])
        self.params.extend([self.S])

        def unit(x, triples, hidden_tm1, x_m):
            """
            :param x: input questions in max_len * batch_size * emb_dim(without position)
            :param triples: input (entity, entity) or (entity, relation) in batch_size * candidate_num * 2 * emb_dim
            :param hidden_tm1: hred hidden in batch_size * h_dim
            :param x_m: mask in max_len * batch_size
            :return: score in batch_size * candidate_num
            """
            x_average = tf.reduce_sum(x, 0)/tf.expand_dims(tf.reduce_sum(x_m, 0), 1)    # batch_size * emb_dim
            triple_average = tf.reduce_mean(triples, 2)
            triple_average = tf.transpose(triple_average, perm=[0, 2, 1])               # batch_size * emb_dim * can

            score = tf.matmul(tf.expand_dims(tf.matmul(tf.concat([x_average, hidden_tm1], 1), self.S), 1),
                              triple_average)                                           # batch_size * 1 * can

            sum_content = tf.reduce_sum(
                tf.transpose(score, perm=[0, 2, 1]) * tf.transpose(triple_average, perm=[0, 2, 1]),
                axis=1)                                                                 # batch_size * emb
            score = tf.squeeze(score)                                                   # batch_size * can

            return score, sum_content

        return unit


class CNNKBScorer(graph_base.GraphBase):
    def __init__(self, layer_num, kernel_size, in_dim, h_dim,
                 mlp_layer_num, mlp_in_dim, mlp_h_dim,
                 hyper_params=None, params=None):
        graph_base.GraphBase.__init__(self, hyper_params, params)

        self.hyper_params["score_type"] = "CNN KB Scorer"
        self.hyper_params["nn_layer_num"] = layer_num
        self.hyper_params["kernel_size"] = kernel_size
        self.hyper_params["nn_in_dim"] = in_dim
        self.hyper_params["nn_h_dim"] = h_dim
        self.hyper_params["mlp_layer_num"] = mlp_layer_num
        self.hyper_params["mlp_in_dim"] = mlp_in_dim
        self.hyper_params["mlp_h_dim"] = mlp_h_dim
        self.scorer = self.create_cnn_scorer()

    def create_cnn_scorer(self):
        self.cnn_encoder = encoder.CNNEncoder(layer_num=self.hyper_params["nn_layer_num"],
                                              kernel_size=self.hyper_params["kernel_size"],
                                              in_dim=self.hyper_params["nn_in_dim"],
                                              h_dim=self.hyper_params["nn_h_dim"])
        self.params.extend(self.cnn_encoder.params)
        self.perceptrons = []
        for i in range(self.hyper_params["mlp_layer_num"]):
            if i == 0:
                w = graph_base.get_params([self.hyper_params["mlp_in_dim"], self.hyper_params["mlp_h_dim"]])
            elif i == self.hyper_params["mlp_layer_num"] - 1:
                w = graph_base.get_params([self.hyper_params["mlp_h_dim"], 1])
            else:
                w = graph_base.get_params([self.hyper_params["mlp_h_dim"], self.hyper_params["mlp_h_dim"]])
            self.perceptrons.append([w])
            self.params.extend([w])
        
        def unit(x, triples, hidden_tm1, x_m=None):
            """
            cnn representation with mlp scorer
            :param x: input questions in max_len * batch_size * emb_dim(with position)
            :param triples: input (entity, entity) or (entity, relation) in batch_size * candidate_num * 2 * emb_dim
            :param hidden_tm1: hred hidden in batch_size * h_dim
            :return: score in batch_size * candidate_num
            """
            representation = tf.tile(
                tf.expand_dims(self.cnn_encoder.forward(x), 1),
                [1, FLAGS.candidate_num, 1])
            hidden_tm1 = tf.tile(
                hidden_tm1,
                [1, FLAGS.candidate_num, 1])
            triple_average = tf.reduce_mean(triples, 2)

            hidden = tf.concat([representation, hidden_tm1, triple_average], 2)  # batch_size * can * in_dim
            for i in range(self.hyper_params["mlp_layer_num"]):
                layer = self.perceptrons[i]
                hidden = tf.tanh(tf.matmul(hidden, tf.expand_dims(layer[0], 0)))

            hidden = tf.squeeze(hidden)
            return hidden

        return unit
