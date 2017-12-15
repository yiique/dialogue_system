__author__ = 'liushuman'


import tensorflow as tf


from collections import namedtuple


FLAGS = tf.flags.FLAGS


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
    flatten_score = tf.reshape(score, [-1])
    _, flatten_indices = tf.nn.top_k(flatten_score, k=size, sorted=True)           # size
    beam_indices = tf.div(flatten_indices, FLAGS.common_vocab + FLAGS.candidate_num)
    word_indices = tf.mod(flatten_indices, FLAGS.common_vocab + FLAGS.candidate_num)

    # gather
    new_beam_scores = tf.gather(beam_scores, beam_indices)
    new_beam_score_cells = tf.transpose(tf.gather(
        tf.transpose(beam_score_cells, perm=[2, 0, 1, 3]), beam_indices), perm=[1, 2, 0, 3])
    new_beam_gen_cells = tf.transpose(tf.gather(
        tf.transpose(beam_gen_cells, perm=[2, 0, 1, 3]), beam_indices), perm=[1, 2, 0, 3])
    new_beam_prob_records = tf.nn.embedding_lookup(tf.reshape(log_probs, [-1, 1]), flatten_indices)
    new_beam_pred_records = tf.gather(beam_pred_records, beam_indices) * \
                            tf.one_hot(tf.ones([size], dtype=tf.int32) * (i+1), FLAGS.sen_max_len, 0, 1) + \
                            tf.one_hot(tf.ones([size], dtype=tf.int32) * (i+1), FLAGS.sen_max_len, 1, 0) * \
                            tf.expand_dims(word_indices, -1)

    step_finished = tf.expand_dims(
        tf.to_float(tf.equal(word_indices,
                             tf.ones([size], dtype=tf.int32) * FLAGS.end_token)), -1)
    new_finished_mask = tf.maximum(tf.gather(finished_mask, beam_indices), step_finished)
    new_finished_len = tf.gather(finished_len, beam_indices) + (1 - step_finished)

    return tf.expand_dims(word_indices, -1), \
           new_finished_mask, new_finished_len, \
           new_beam_scores, new_beam_score_cells, new_beam_gen_cells, \
           new_beam_prob_records, new_beam_pred_records