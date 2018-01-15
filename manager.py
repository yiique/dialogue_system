__author__ = 'liushuman'


import json
import numpy as np
import os
import random
import sys
sys.path.append("../..")
import tensorflow as tf
import time


SEED = 88
FLAGS = tf.flags.FLAGS


# Change config in configs and model in models to judge model
from configurations.configs import config_corpus3 as model_config
from models.model_bi_gate import BiScorerGateDecoderModel as Model
# TODO: highway, dropout


def average_gradients(tower_grads):
    average_grads = []
    for grad_and_vars in zip(*tower_grads):
        # Note that each grad_and_vars looks like the following:
        #   ((grad0_gpu0, var0_gpu0), ... , (grad0_gpuN, var0_gpuN))
        grads = [g for g, _ in grad_and_vars]
        # Average over the 'tower' dimension.
        grad = tf.stack(grads, 0)
        grad = tf.reduce_mean(grad, 0)
        # Keep in mind that the Variables are redundant because they are shared
        # across towers. So .. we will just return the first tower's pointer to
        # the Variable.
        v = grad_and_vars[0][1]
        grad_and_var = (grad, v)
        average_grads.append(grad_and_var)
    return average_grads


def train_iter(ep_no, sess, model, tower_records, avg_loss, tower_losses, update):
    count = 0
    losses = []
    f_train = open(FLAGS.training_path, 'r')
    while True:
        try:
            batches = [[] for _ in range(FLAGS.GPU_num)]
            for i in range(FLAGS.GPU_num):
                for j in range(FLAGS.batch_size):
                    line = f_train.readline()[:-1]
                    batches[i].append(json.loads(line))
        except:
            break

        count += 1
        if count % 500 == 0:
            model.save_weight(sess)

        feed_dict = {}
        for i in range(FLAGS.GPU_num):
            src_dialogue = np.transpose([sample["src_dialogue"] for sample in batches[i]], [1, 2, 0])
            tgt_dialogue = np.transpose([sample["tgt_dialogue"] for sample in batches[i]], [1, 2, 0])
            turn_mask = np.transpose([sample["turn_mask"] for sample in batches[i]], [1, 0])
            src_mask = np.transpose([sample["src_mask"] for sample in batches[i]], [1, 2, 0])
            tgt_mask = np.transpose([sample["tgt_mask"] for sample in batches[i]], [1, 2, 0])
            feed_dict[tower_records[i][0]] = src_dialogue
            feed_dict[tower_records[i][1]] = src_mask
            feed_dict[tower_records[i][2]] = turn_mask
            feed_dict[tower_records[i][3]] = tgt_dialogue
            feed_dict[tower_records[i][4]] = tgt_mask

        outputs = sess.run([avg_loss, tower_losses, update], feed_dict=feed_dict)

        tf.logging.info("---------------------count-------------------")
        tf.logging.info(str(ep_no) + "-" + str(count) + "    " + time.ctime())
        tf.logging.info("---------------------loss-------------------")
        tf.logging.info(outputs[0])
        tf.logging.info(outputs[1])
        losses.append(outputs[0])

    tf.logging.info("============================================================")
    tf.logging.info("avg loss: " + str(np.mean(losses)))
    model.save_weight(sess)


def valid_iter(ep_no, sess, valid_sd, valid_sm, valid_tm, valid_prob, valid_pred, dictionary):
    count = 0
    f_valid = open(FLAGS.valid_path, 'r')
    f_h = open(FLAGS.valid_hypothesis_path, 'w')
    f_r = open(FLAGS.valid_reference_path, 'w')
    f_demo = open(FLAGS.valid_demo_path, 'w')

    for _ in f_valid:
        sample = json.loads(_[:-1])

        count += 1

        feed_dict = {}
        src_dialogue = np.transpose([sample["src_dialogue"]], [1, 2, 0])
        tgt_dialogue = np.transpose([sample["tgt_dialogue"]], [1, 2, 0])
        src_mask = np.transpose([sample["src_mask"]], [1, 2, 0])
        turn_mask = np.transpose([sample["turn_mask"]], [1, 0])
        feed_dict[valid_sd] = src_dialogue
        feed_dict[valid_sm] = src_mask
        feed_dict[valid_tm] = turn_mask

        outputs = sess.run([valid_prob, valid_pred], feed_dict=feed_dict)
        pred_dialogue = outputs[1]

        src_flatten = np.transpose(src_dialogue, [0, 2, 1])
        src_flatten = [x[0] for x in src_flatten][0: int(sum([y[0] for y in turn_mask]))]
        tgt_flatten = np.transpose(tgt_dialogue, [0, 2, 1])
        tgt_flatten = [x[0] for x in tgt_flatten][0: int(sum([y[0] for y in turn_mask]))]
        pred_flatten = [x[0] for x in pred_dialogue]             # turn_mask * 80

        if len(tgt_flatten) != len(pred_flatten) or len(src_flatten) != len(tgt_flatten):
            raise AssertionError

        if count % 50 == ep_no % 50:
            tf.logging.info("---------------------<sample>-------------------")
        for i in range(len(tgt_flatten)):
            src_sentence = src_flatten[i].tolist()
            tgt_sentence = tgt_flatten[i].tolist()
            pred_sentence = pred_flatten[i].tolist()
            se_index = src_sentence.index(FLAGS.end_token)
            te_index = tgt_sentence.index(FLAGS.end_token)
            if FLAGS.end_token not in pred_sentence:
                pe_index = FLAGS.sen_max_len
            else:
                pe_index = pred_sentence.index(FLAGS.end_token)

            src_sentence = src_sentence[0: se_index+1]
            tgt_sentence = tgt_sentence[0: te_index+1]
            pred_sentence = pred_sentence[0: pe_index+1]

            src_tokens = [dictionary[x].encode('utf-8') for x in src_sentence]
            tgt_tokens = [dictionary[x].encode('utf-8') for x in tgt_sentence]
            pred_tokens = [dictionary[x].encode('utf-8') for x in pred_sentence]

            f_r.write(" ".join(tgt_tokens) + "\n")
            f_h.write(" ".join(pred_tokens) + "\n")

            f_demo.write("<src>" + " ".join(src_tokens) + "</src>\n")
            f_demo.write("<tgt>" + " ".join(tgt_tokens) + "</tgt>\n")
            f_demo.write("<pred>" + " ".join(pred_tokens) + "</pred>\n")

            if count % 50 == ep_no % 50:
                tf.logging.info("<src>" + " ".join(src_tokens) + "</src>\n")
                tf.logging.info("<tgt>" + " ".join(tgt_tokens) + "</tgt>\n")
                tf.logging.info("<pred>" + " ".join(pred_tokens) + "</pred>\n")
        if count % 50 == ep_no % 50:
            tf.logging.info("---------------------</sample>-------------------")

    f_valid.close()
    f_r.close()
    f_h.close()
    f_demo.close()

    tf.logging.info("Calculating BLEU...")
    os.system("perl " + FLAGS.multi_bleu_path + " " + FLAGS.valid_reference_path +
              " < " + FLAGS.valid_hypothesis_path)


def main_simple():
    random.seed(SEED)
    np.random.seed(SEED)

    hyper_params = model_config.HYPER_PARAMS
    dictionary = json.loads(open(FLAGS.dictionary_path).readline())
    dictionary = {dictionary[key]: key for key in dictionary}

    with tf.device('/cpu:0'):
        tf.logging.info("STEP1: Init...")
        model = Model(hyper_params=hyper_params)

        tower_records = []
        tf.logging.info("STEP2: Map/Reduce...")
        for gpu_id in range(FLAGS.GPU_num):
            with tf.device('/gpu:%d' % gpu_id):
                tf.logging.info('Building tower:%d...' % gpu_id)
                with tf.name_scope('tower_%d' % gpu_id):
                    with tf.variable_scope('cpu_variables', reuse=gpu_id > 0):
                        s_d, s_m, turn_m, t_d, t_m, _, loss_simple, grad_simple, _, _ = model.build_tower()
                        tower_records.append(
                            (s_d, s_m, turn_m, t_d, t_m, loss_simple, grad_simple))

        _, _, _, _, _, tower_losses, tower_grads = zip(*tower_records)
        avg_loss = tf.reduce_mean(tower_losses)
        update = model.optimizer.apply_gradients(average_gradients(tower_grads))

        valid_sd, valid_sm, valid_tm, valid_prob, valid_pred = model.build_eval()

        config = tf.ConfigProto(allow_soft_placement=True)
        config.gpu_options.allow_growth = True
        sess = tf.Session(config=config)
        sess.run(tf.global_variables_initializer())

        try:
            model.load_weight(sess)
        except:
            tf.logging.warning("NO WEIGHT FILE, INIT FROM BEGINNING...")

        tf.logging.info("STEP3: Training...")
        for ep in range(FLAGS.epoch):
            train_iter(ep, sess, model, tower_records, avg_loss, tower_losses, update)

            if ep % 5 == 0:
                tf.logging.info("STEP4: Evaluating...")
                valid_iter(ep, sess, valid_sd, valid_sm, valid_tm, valid_prob, valid_pred, dictionary)


if __name__ == "__main__":
    main_simple()