__author__ = 'liushuman'


import logging
import tensorflow as tf


log = logging.getLogger('tensorflow')
log.setLevel(logging.DEBUG)

FLAGS = tf.flags.FLAGS

tf.flags.DEFINE_integer("GPU_num", 4, """""")

tf.flags.DEFINE_integer("batch_size", 10, """""")
tf.flags.DEFINE_integer("beam_size", 10, """""")
tf.flags.DEFINE_integer("dia_max_len", 8, """""")
tf.flags.DEFINE_integer("sen_max_len", 80, """""")


tf.flags.DEFINE_integer("enquire_can_num", 250, """""")
tf.flags.DEFINE_integer("diffuse_can_num", 50, """""")
tf.flags.DEFINE_integer("candidate_num", 300, """""")
tf.flags.DEFINE_integer("common_vocab", 2894, """""")
tf.flags.DEFINE_integer("entities", 29460, """""")
tf.flags.DEFINE_integer("relations", 2, """""")

tf.flags.DEFINE_integer("start_token", 0, """""")
tf.flags.DEFINE_integer("end_token", 1, """""")
tf.flags.DEFINE_integer("unk", 2, """""")

tf.flags.DEFINE_float("loss_alpha", 0.15, """""")
tf.flags.DEFINE_float("loss_beta", 0.35, """""")
tf.flags.DEFINE_float("loss_gamma", 0.15, """""")
tf.flags.DEFINE_float("loss_decoder", 0.35, """""")
tf.flags.DEFINE_float("grad_clip", 5.0, """""")
tf.flags.DEFINE_float("learning_rate", 0.001, """""")
tf.flags.DEFINE_float("penalty_factor", 0.6, """""")
tf.flags.DEFINE_boolean("norm", False, """""")
tf.flags.DEFINE_integer("epoch", 80, """""")

tf.flags.DEFINE_string("multi_bleu_path", "./data/multi-bleu.perl", """""")

tf.flags.DEFINE_string("dictionary_path", "./data/corpus3/dictionary", """""")
tf.flags.DEFINE_string("training_path", "./data/corpus3/part.index.train", """""")
tf.flags.DEFINE_string("valid_path", "./data/corpus3/part.index.valid", """""")
tf.flags.DEFINE_string("valid_hypothesis_path", "./data/corpus3/valid.hypothesis", """""")
tf.flags.DEFINE_string("valid_reference_path", "./data/corpus3/valid.reference", """""")
tf.flags.DEFINE_string("valid_demo_path", "./data/corpus3/valid.demo", """""")
tf.flags.DEFINE_string("test_path", "./data/corpus3/part.index.test", """""")
tf.flags.DEFINE_string("weight_path", "./data/corpus3/weight.save", """""")


HYPER_PARAMS = {
    "emb_dim": 512,
    "encoder_layer_num": 1,
    "encoder_h_dim": 512,
    "hred_h_dim": 1024,
    "k_cnn_layer_num": 5,
    "k_cnn_kernel_size": 3,
    "k_cnn_h_dim": 512,
    "enquirer_mlp_layer_num": 2,
    "enquirer_mlp_h_dim": FLAGS.enquire_can_num,
    "diffuser_mlp_layer_num": 2,
    "diffuser_mlp_h_dim": FLAGS.diffuse_can_num,
    "scorer_mlp_layer_num": 2,
    "scorer_mlp_h_dim": FLAGS.candidate_num,
    "decoder_gen_layer_num": 1,
    "decoder_gen_h_dim": 512,
    "decoder_mlp_layer_num": 3,
    "decoder_mlp_h_dim": 512
}