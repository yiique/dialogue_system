__author__ = 'liushuman'


import json
import numpy as np
import random
import re
import subprocess
import sys
sys.path.append("../..")
import tensorflow as tf
import time


SEED = 88
FLAGS = tf.flags.FLAGS


from configurations.configs import config_corpus3 as model_config
from models.model_bi_gate import BiScorerGateDecoderModel as Model
# TODO: highway, dropout


def main_eval():
    random.seed(SEED)
    np.random.seed(SEED)

    tf.logging.info("STEP1: Init...")
    f = open(FLAGS.valid_path, 'r')

    f_h = open(FLAGS.valid_hypothesis_path, 'w')
    f_r = open(FLAGS.valid_reference_path, 'w')

    hyper_params = model_config.HYPER_PARAMS
    dictionary = json.loads(open(FLAGS.dictionary).readline())

    with tf.device('/gpu:0'):
        model = Model(hyper_params=hyper_params)
        s_d, s_m, t_m, prob, pred = model.build_eval()

        config = tf.ConfigProto(allow_soft_placement=True)
        config.gpu_options.allow_growth = True
        sess = tf.Session(config=config)
        sess.run(tf.global_variables_initializer())

        try:
            model.load_weight(sess)
        except:
            tf.logging.warning("NO WEIGHT FILE, INIT FROM BEGINNING...")

        tf.logging.info("STEP2: Evaling...")
        count = 0
        for _ in f:
            sample = json.loads(_[:-1])

            count += 1
            # if count == 10:
            #     break

            feed_dict = {}
            src_dialogue = np.transpose([sample["src_dialogue"]], [1, 2, 0])
            tgt_dialogue = np.transpose([sample["tgt_dialogue"]], [1, 2, 0])
            src_mask = np.transpose([sample["src_mask"]], [1, 2, 0])
            turn_mask = np.transpose([sample["turn_mask"]], [1, 0])
            feed_dict[s_d] = src_dialogue
            feed_dict[s_m] = src_mask
            feed_dict[t_m] = turn_mask

            outputs = sess.run([prob, pred], feed_dict=feed_dict)
            pred_dialogue = outputs[1]

            tgt_flatten = np.transpose(tgt_dialogue, [0, 2, 1])
            tgt_flatten = [x[0] for x in tgt_flatten][0: int(sum([y[0] for y in turn_mask]))]
            pred_flatten = [x[0] for x in pred_dialogue]             # turn_mask * 80

            if len(tgt_flatten) != len(pred_flatten):
                raise AssertionError

            for i in range(len(tgt_flatten)):
                tgt_sentence = tgt_flatten[i]
                pred_sentence = pred_flatten[i]
                te_index = tgt_sentence.index(FLAGS.end_token)
                if FLAGS.end_token not in pred_sentence:
                    pe_index = FLAGS.sen_max_len
                else:
                    pe_index = pred_sentence.index(FLAGS.end_token)

                tgt_sentence = tgt_sentence[0: te_index+1]
                pred_sentence = pred_sentence[0: pe_index+1]

                tgt_tokens = [dictionary[x].encode('utf-8') for x in tgt_sentence]
                pred_tokens = [dictionary[x].encode('utf-8') for x in pred_sentence]

                f_r.write(" ".join(tgt_tokens) + "\n")
                f_h.write(" ".join(pred_tokens) + "\n")

        f_r.close()
        f_h.close()

        tf.logging.info("STEP3: Calculating BLEU...")
        with open(FLAGS.valid_hypothesis_path, 'r') as read_pred:
            bleu_cmd = [FLAGS.multi_bleu_path]
            bleu_cmd += [FLAGS.valid_reference_path]
            try:
                bleu_out = subprocess.check_output(
                    bleu_cmd, stdin=read_pred, stderr=subprocess.STDOUT
                )
                bleu_out = bleu_out.decode("utf-8")
                bleu_score = re.search(r"BLEU = (.+?),", bleu_out).group(1)
                bleu_score = float(bleu_score)
                print "BLEU:"
            except subprocess.CalledProcessError as error:
                if error.output is not None:
                    print "ERROR IN CAL BLEU:", error.output
                bleu_score = np.float32(0.0)

        print "BLEU score: ", np.float32(bleu_score)


if __name__ == "__main__":
    main_eval()