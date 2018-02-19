__author__ = 'liushuman'


import logging
import tensorflow as tf


log = logging.getLogger('tensorflow')
log.setLevel(logging.DEBUG)

FLAGS = tf.flags.FLAGS

tf.flags.DEFINE_integer("GPU_num", 4, """""")

tf.flags.DEFINE_integer("batch_size", 40, """""")
tf.flags.DEFINE_integer("beam_size", 10, """""")
tf.flags.DEFINE_integer("dia_max_len", 8, """""")
tf.flags.DEFINE_integer("sen_max_len", 80, """""")

tf.flags.DEFINE_integer("vocab_size", 55356, """""")

tf.flags.DEFINE_integer("start_token", 0, """""")
tf.flags.DEFINE_integer("end_token", 1, """""")
tf.flags.DEFINE_integer("unk", 2, """""")

tf.flags.DEFINE_float("grad_clip", 5.0, """""")
tf.flags.DEFINE_float("learning_rate", 0.001, """""")
tf.flags.DEFINE_float("penalty_factor", 0.6, """""")
tf.flags.DEFINE_boolean("norm", False, """""")
tf.flags.DEFINE_integer("epoch", 221, """""")

tf.flags.DEFINE_string("multi_bleu_path", "./data/multi-bleu.perl", """""")

tf.flags.DEFINE_string("dictionary_path", "./data/corpus4/dictionary", """""")
tf.flags.DEFINE_string("training_path", "./data/corpus4/data.index.train", """""")
tf.flags.DEFINE_string("valid_path", "./data/corpus4/data.index.valid", """""")
tf.flags.DEFINE_string("valid_hypothesis_path", "./data/corpus4/valid.hypothesis", """""")
tf.flags.DEFINE_string("valid_reference_path", "./data/corpus4/valid.reference", """""")
tf.flags.DEFINE_string("valid_demo_path", "./data/corpus4/valid.demo", """""")
tf.flags.DEFINE_string("test_path", "./data/corpus4/data.index.test", """""")
tf.flags.DEFINE_string("weight_path", "./data/corpus4/weight.save", """""")


HYPER_PARAMS = {
    "emb_dim": 1024,
    "encoder_layer_num": 1,
    "encoder_h_dim": 1024,
    "decoder_layer_num": 1,
    "decoder_h_dim": 1024,
}
