__author__ = 'liushuman'

import encoder
import graph_base
import tensorflow as tf


from tensorflow.python.ops import tensor_array_ops, control_flow_ops


FLAGS = tf.flags.FLAGS


class CandidateSeeker():
    def __init__(self, emb_dim):
        self.emb_dim = emb_dim
        self.seeker = self.create_seeker()

    def create_seeker(self):

        def unit(representation, embedding, size=FLAGS.batch_size):
            """
            :param representation: size * emb_dim
            :param embedding:
            :return: can indices in batch_size * emb_candidate_num
            """
            knowledge_embedding = tf.slice(embedding, [FLAGS.common_vocab, 0],
                                           [FLAGS.entities, self.emb_dim])
            embedding_distance = tf.reduce_sum(
                tf.square(
                    tf.tile(
                        tf.expand_dims(representation, 1),
                        [1, FLAGS.entities, 1]) - knowledge_embedding),
                -1)                                                         # size * entities
            similarity = 0. - embedding_distance
            _, top_indices = tf.nn.top_k(similarity, k=FLAGS.emb_candidate_num)
            entity_indices = top_indices + FLAGS.common_vocab

            return entity_indices

        return unit


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


class KBRetriever(graph_base.GraphBase):
    def __init__(self, emb_dim, o_encoder_h_dim, hred_h_dim,
                 encoder_layer_num, encoder_kernel_size, encoder_h_dim,
                 enquirer_mlp_layer_num, enquirer_mlp_h_dim,
                 diffuser_mlp_layer_num, diffuser_mlp_h_dim,
                 scorer_mlp_layer_num, scorer_mlp_h_dim,
                 hyper_params=None, params=None):
        graph_base.GraphBase.__init__(self, hyper_params, params)

        self.hyper_params["model_type"] = "Diffuse Kb Retriever"
        self.hyper_params["emb_dim"] = emb_dim
        self.hyper_params["o_encoder_h_dim"] = o_encoder_h_dim
        self.hyper_params["hred_h_dim"] = hred_h_dim

        self.hyper_params["encoder_layer_num"] = encoder_layer_num
        self.hyper_params["encoder_kernel_size"] = encoder_kernel_size
        self.hyper_params["encoder_h_dim"] = encoder_h_dim

        self.hyper_params["enquirer_mlp_layer_num"] = enquirer_mlp_layer_num
        self.hyper_params["enquirer_mlp_in_dim"] = \
            self.hyper_params["emb_dim"] + self.hyper_params["encoder_h_dim"] + self.hyper_params["hred_h_dim"]
        self.hyper_params["enquirer_mlp_h_dim"] = enquirer_mlp_h_dim

        self.hyper_params["diffuser_mlp_layer_num"] = diffuser_mlp_layer_num
        self.hyper_params["diffuser_mlp_in_dim"] = \
            self.hyper_params["emb_dim"] + self.hyper_params["hred_h_dim"] + self.hyper_params["o_encoder_h_dim"] \
            + self.hyper_params["encoder_h_dim"] + FLAGS.enquire_can_num + self.hyper_params["emb_dim"]
        self.hyper_params["diffuser_mlp_h_dim"] = diffuser_mlp_h_dim

        self.hyper_params["scorer_mlp_layer_num"] = scorer_mlp_layer_num
        self.hyper_params["scorer_mlp_in_dim"] = \
            self.hyper_params["hred_h_dim"] + self.hyper_params["o_encoder_h_dim"] \
            + FLAGS.enquire_can_num + FLAGS.diffuse_can_num
        self.hyper_params["scorer_mlp_h_dim"] = scorer_mlp_h_dim

        self.enquirer_unit, self.diffuser_unit, self.scorer_unit = self.create_retriever()

    def create_retriever(self):
        self.knowledge_encoder = encoder.Encoder(
            self.hyper_params["encoder_layer_num"],
            self.hyper_params["emb_dim"], self.hyper_params["encoder_h_dim"], FLAGS.norm)
        self.params.extend(self.knowledge_encoder.params)

        self.enquirer_perceptrons = []
        for i in range(self.hyper_params["enquirer_mlp_layer_num"]):
            if i == 0:
                w = graph_base.get_params(
                    [self.hyper_params["enquirer_mlp_in_dim"], self.hyper_params["enquirer_mlp_h_dim"]])
            elif i == self.hyper_params["enquirer_mlp_layer_num"] - 1:
                w = graph_base.get_params(
                    [self.hyper_params["enquirer_mlp_h_dim"], 1])
            else:
                w = graph_base.get_params(
                    [self.hyper_params["enquirer_mlp_h_dim"], self.hyper_params["enquirer_mlp_h_dim"]])
            self.enquirer_perceptrons.append([w])
            self.params.extend([w])

        self.diffuse_perceptrons = []
        for i in range(self.hyper_params["diffuser_mlp_layer_num"]):
            if i == 0:
                w = graph_base.get_params(
                    [self.hyper_params["diffuser_mlp_in_dim"], self.hyper_params["diffuser_mlp_h_dim"]])
            elif i == self.hyper_params["diffuser_mlp_layer_num"] - 1:
                w = graph_base.get_params([
                    self.hyper_params["diffuser_mlp_h_dim"], 1])
            else:
                w = graph_base.get_params(
                    [self.hyper_params["diffuser_mlp_h_dim"], self.hyper_params["diffuser_mlp_h_dim"]])
            self.diffuse_perceptrons.append([w])
            self.params.extend([w])

        self.score_perceptrons = []
        for i in range(self.hyper_params["scorer_mlp_layer_num"]):
            if i == 0:
                w = graph_base.get_params(
                    [self.hyper_params["scorer_mlp_in_dim"], self.hyper_params["scorer_mlp_h_dim"]])
            elif i == self.hyper_params["scorer_mlp_layer_num"] - 1:
                w = graph_base.get_params(
                    [self.hyper_params["scorer_mlp_h_dim"], FLAGS.enquire_can_num + FLAGS.diffuse_can_num])
            else:
                w = graph_base.get_params(
                    [self.hyper_params["scorer_mlp_h_dim"], self.hyper_params["scorer_mlp_h_dim"]])
            self.score_perceptrons.append([w])
            self.params.extend([w])

        def enquirer_unit(src_emb, src_mask, enquire_strings_avg, hred_h_tim1, size=FLAGS.batch_size):
            """
            :param src_emb: src embedding with position in max_len * size * emb_dim
            :param enquire_strings_avg: e_c_embedding avg in size * enquire_can_num * emb_dim
                                        note its char embeddings rather than entity embedding here
            :return:
            """
            knowledge_utterance = self.knowledge_encoder.forward(src_emb, src_mask, size)[-1]         # size * h_dim
            hidden = tf.concat([tf.tile(tf.expand_dims(knowledge_utterance, 1), [1, FLAGS.enquire_can_num, 1]),
                                tf.tile(tf.expand_dims(hred_h_tim1, 1), [1, FLAGS.enquire_can_num, 1]),
                                enquire_strings_avg], 2)                                    # size * e_c_num * e+h_dim
            for i in range(self.hyper_params["enquirer_mlp_layer_num"]):
                layer = self.enquirer_perceptrons[i][0]
                hidden = tf.sigmoid(tf.matmul(hidden, tf.tile(tf.expand_dims(layer, 0), [size, 1, 1])))
            enquire_score = tf.reduce_sum(hidden, -1)                                              # size * e_c_num

            return knowledge_utterance, enquire_score

        def diffuse_unit(hred_hidden_tm1, src_utterance, knowledge_utterance,
                         enquire_score, enquire_entities_sum, embedding, size=FLAGS.batch_size):
            """
            use batch loop rather than embedding loop for faster training
            :param hred_hidden_tm1: size * hred_h_dim
            :param knowledge_utterance: size * encoder_dim
            :param enquire_entities_sum: size * emb_dim
                                        its entity embeddings avg here
            :param embedding: total_embedding in (words + entities + relations + positions)
            :return:
            """
            entity_embedding = tf.slice(embedding, [FLAGS.common_vocab, 0],
                                        [FLAGS.entities, self.hyper_params["emb_dim"]])  # entities_num * emb_dim

            hidden = tf.concat([
                hred_hidden_tm1, src_utterance, knowledge_utterance, enquire_score, enquire_entities_sum], 1)
            hidden_ta = tensor_array_ops.TensorArray(dtype=tf.float32, size=0, dynamic_size=True).unstack(hidden)
            prob_ta = tensor_array_ops.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)

            def _loop_body(i, prob_ta, hidden_ta, entity_embedding):
                hidden_t = hidden_ta.read(i)
                hidden_t = tf.concat([tf.tile(tf.expand_dims(hidden_t, 0), [FLAGS.entities, 1]), entity_embedding],
                                     1)                                                 # entities_num * (sum_len)

                for j in range(self.hyper_params["diffuser_mlp_layer_num"]):
                    layer = self.diffuse_perceptrons[j][0]
                    hidden_t = tf.sigmoid(tf.matmul(hidden_t, layer))

                prob_ta = prob_ta.write(i, tf.reduce_sum(hidden_t, -1))
                return i+1, prob_ta, hidden_ta, entity_embedding

            _, prob_ta, _, _ = control_flow_ops.while_loop(
                cond=lambda i, _1, _2, _3: i < hidden_ta.size(),
                body=_loop_body,
                loop_vars=(
                    tf.constant(0, tf.int32),
                    prob_ta, hidden_ta, entity_embedding)
            )

            prob = prob_ta.stack()                                                      # size * entity_num
            prob_topk, indices_topk = tf.nn.top_k(prob, FLAGS.diffuse_can_num, sorted=True)    # size * d_c_num
            return prob, prob_topk, indices_topk + FLAGS.common_vocab

        def score_unit(hred_hidden_tm1, src_utterance, enquirer_score, diffuse_score):
            hidden = tf.concat([hred_hidden_tm1, src_utterance, enquirer_score, diffuse_score], 1)
            for i in range(self.hyper_params["scorer_mlp_layer_num"]):
                layer = self.score_perceptrons[i][0]
                hidden = tf.sigmoid(tf.matmul(hidden, layer))

            return hidden                                       # size * e+d_c_num

        return enquirer_unit, diffuse_unit, score_unit


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
