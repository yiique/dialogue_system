__author__ = 'liushuman'


import tensorflow as tf


from collections import namedtuple


FLAGS = tf.flags.FLAGS


class BeamLeaf():
    def __init__(self):
        self.parent = 0
        self.score = ""
        self.predict = 0



class BeamSearchStepOutput(namedtuple("BeamSearchStepOutput", ["scores", "predicted_ids", "beam_parent_ids"])):
    """Outputs for a single step of beam search.
    Args:
    scores: Score for each beam, a float32 vector
    predicted_ids: predictions for this step step, an int32 vector
    beam_parent_ids: an int32 vector containing the beam indices of the
        continued beams from the previous step
    """
    pass


class BeamSearch(object):
    def __init__(self):
        self.beam_size = FLAGS.beam_size
        self.alive_num = 0
        self.finish_num = 0
        self.beam_tree = []

    def build_beam(self):
        return BeamSearchState(
            log_probs=tf.zeros([self.beam_size]),
            finished=tf.zeros([self.beam_size], dtype=tf.bool),
            lengths=tf.zeros([self.beam_size], dtype=tf.int32))

    def _beam_step(self):
        pass

    def gather(self):
        pass


def beam_step(i, prob,
         finished_mask, finished_len,
         beam_scores, beam_score_cells, beam_gen_cells,
         beam_prob_records, beam_pred_records,
         size=FLAGS.beam_size):
    """
    beam search step according to log prob
    :param prob: predict prob for previous beam
    :param finished_mask: finished mask in size * 1
    :param finished_len: sentence length till now(invariant for finished sentence) in size * 1
    :return:
    """
    # finished beam prob replace
    prob = tf.one_hot(tf.ones([size], dtype=tf.int32) * FLAGS.end_token,
                      FLAGS.common_vocab + FLAGS.candidate_num, 1.0, 0.0) * finished_mask + \
           prob * (1. - finished_mask)

    # cal score
    log_probs = prob * beam_prob_records
    length_penalty = tf.div((5. + tf.to_float(finished_len))**FLAGS.penalty_factor, (5. + 1.)**FLAGS.penalty_factor)
    score = log_probs / length_penalty

    # cal indices
    flatten_score = tf.reshape(score, [1])
    flatten_indices = tf.nn.top_k(flatten_score, k=size, sorted=True)           # size
    beam_indices = tf.div(flatten_indices, FLAGS.common_vocab + FLAGS.candidate_num)    # size
    word_indices = tf.mod(flatten_indices, FLAGS.common_vocab + FLAGS.candidate_num)    # size

    # gather
    new_beam_scores = tf.gather(beam_scores, beam_indices)
    new_beam_score_cells = tf.gather(beam_score_cells, beam_indices)
    new_beam_gen_cells = tf.gather(beam_gen_cells, beam_indices)
    new_beam_prob_records = tf.nn.embedding_lookup(tf.reshape(log_probs, [-1, 1]), flatten_indices)
    new_beam_pred_records = tf.gather(beam_pred_records, beam_indices) * \
                            tf.one_hot(tf.ones([size], dtype=tf.int32) * (i+1), FLAGS.sen_max_len, 0.0, 1.0) + \
                            tf.one_hot(tf.ones([size], dtype=tf.int32) * (i+1), FLAGS.sen_max_len, 1.0, 0.0) * \
                            tf.expand_dims(word_indices, -1)

    step_finished = tf.expand_dims(
        tf.to_int32(tf.equal(word_indices,
                             tf.ones([size], dtype=tf.int32) * FLAGS.end_token)), -1)
    new_finished_mask = tf.maximum(tf.gather(finished_mask, beam_indices), step_finished)
    new_finished_len = tf.gather(finished_len, beam_indices) + (1 - step_finished)

    return tf.expand_dims(word_indices, -1), \
           new_finished_mask, new_finished_len, \
           new_beam_scores, new_beam_score_cells, new_beam_gen_cells, \
           new_beam_prob_records, new_beam_pred_records