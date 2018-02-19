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
from configurations.configs import config_diffuse_corpus4 as model_config
from models.model_diffuse_mask import DiffuseModel as Model
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


def train_iter(ep, sess, model, tower_records,
               avg_losses_alpha, avg_losses_decoder, avg_losses,
               tower_losses, update):
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

        dialogue_losses = []

        tf.logging.info("---------------------count-------------------")
        tf.logging.info(str(ep) + "-" + str(count) + "    " + time.ctime())
        for i in range(FLAGS.dia_max_len):
            feed_dict = {}
            for j in range(FLAGS.GPU_num):
                src = np.transpose([sample[i]["src"] for sample in batches[j]])
                src_mask = np.transpose([sample[i]["src_mask"] for sample in batches[j]])
                tgt_indices = np.transpose([sample[i]["tgt_indices"] for sample in batches[j]])
                tgt = np.transpose([sample[i]["tgt"] for sample in batches[j]])
                tgt_mask = np.transpose([sample[i]["tgt_mask"] for sample in batches[j]])
                enquire_strings = [sample[i]["enquire_strings"] for sample in batches[j]]
                enquire_entities = [sample[i]["enquire_entities"] for sample in batches[j]]
                enquire_mask = [sample[i]["enquire_mask"] for sample in batches[j]]
                enquire_score_golden = [sample[i]["enquire_score_golden"] for sample in batches[j]]

                feed_dict[tower_records[j][0]] = src
                feed_dict[tower_records[j][1]] = src_mask
                feed_dict[tower_records[j][2]] = tgt_indices
                feed_dict[tower_records[j][3]] = tgt
                feed_dict[tower_records[j][4]] = tgt_mask
                feed_dict[tower_records[j][5]] = enquire_strings
                feed_dict[tower_records[j][6]] = enquire_entities
                feed_dict[tower_records[j][7]] = enquire_mask
                feed_dict[tower_records[j][8]] = enquire_score_golden

            outputs = sess.run([
                avg_losses_alpha, avg_losses_decoder, avg_losses,
                tower_losses, update], feed_dict=feed_dict)

            tf.logging.info("-  -  -  -  -  -  sentence_loss %d -  -  -  -  -  -" % i)
            tf.logging.info(str(outputs[0]) + "\t/\t" + str(outputs[1]) + "\t/\t" + str(outputs[2]) +
                            "\t/\t" + str(outputs[3]))
            dialogue_losses.append(outputs[2])

        dialogue_loss = np.mean(dialogue_losses)
        tf.logging.info("---------------------dialogue_loss-------------------")
        tf.logging.info(dialogue_loss)
        losses.append(dialogue_loss)

    tf.logging.info("============================================================")
    tf.logging.info("avg loss: " + str(np.mean(losses)))
    model.save_weight(sess)


def valid_iter(ep_no, sess, valid_params, dictionary):
    count = 0
    f_valid = open(FLAGS.valid_path, 'r')
    f_demo = open(FLAGS.valid_demo_path, 'w')
    kb_dict = json.loads(open("./data/corpus4/kb.experiment").readline())
    kb2alias_dict = {}
    for key in kb_dict["movie"]:
        kb2alias_dict[key] = kb_dict["movie"][key]["title"]
    for key in kb_dict["celebrity"]:
        kb2alias_dict[key] = kb_dict["celebrity"][key]["name"]

    for _ in f_valid:
        sample = json.loads(_.strip())

        count += 1

        f_demo.write("<dialogue>\n")
        if count % 200 == ep_no % 200:
            tf.logging.info("---------------------<sample>-------------------\n")
        for i in range(FLAGS.dia_max_len):
            feed_dict = {}

            src = np.transpose([sample[i]["src"]])
            src_mask = np.transpose([sample[i]["src_mask"]])
            tgt_indices = sample[i]["tgt_indices"]
            tgt_mask = sample[i]["tgt_mask"]
            turn_mask = [sample[i]["turn_mask"]]
            enquire_strings = [sample[i]["enquire_strings"]]
            enquire_entities = [sample[i]["enquire_entities"]]
            enquire_mask = [sample[i]["enquire_mask"]]
            enquire_objs = sample[i]["enquire_objs"]
            enquire_score_golden = sample[i]["enquire_score_golden"]

            if int(turn_mask[0]) == 0:
                break

            feed_dict[valid_params[0]] = src
            feed_dict[valid_params[1]] = src_mask
            feed_dict[valid_params[2]] = enquire_strings
            feed_dict[valid_params[3]] = enquire_entities
            feed_dict[valid_params[4]] = enquire_mask

            outputs = sess.run([valid_params[5], valid_params[6], valid_params[7]], feed_dict=feed_dict)
            pred_enquire_score = outputs[0][0]
            pred_sentence = outputs[2]          # beam_size * max_len

            src_flatten = np.transpose(src).tolist()[0][0: int(sum(x[0] for x in src_mask))]
            tgt_flatten = tgt_indices[0: int(sum(tgt_mask))]
            pred_flatten = pred_sentence[0].tolist()
            for j in range(len(pred_flatten)):
                if FLAGS.common_vocab <= pred_flatten[j] < FLAGS.common_vocab + FLAGS.enquire_can_num:
                    pred_flatten[j] = enquire_objs[pred_flatten[j] - FLAGS.common_vocab][0]
            if FLAGS.end_token not in pred_flatten:
                pass
            else:
                pred_flatten = pred_flatten[0: pred_flatten.index(FLAGS.end_token)]
            src_tokens = [dictionary[x].encode('utf-8') for x in src_flatten]
            tgt_tokens = [dictionary[x] for x in tgt_flatten]
            tgt_tokens = [kb2alias_dict[x].encode('utf-8') if x in kb2alias_dict else x.encode('utf-8')
                          for x in tgt_tokens]
            pred_tokens = [dictionary[x] for x in pred_flatten]
            pred_tokens = [kb2alias_dict[x].encode('utf-8') if x in kb2alias_dict else x.encode('utf-8')
                           for x in pred_tokens]

            enquire_loss = [float(x-y) * float(x-y) for x, y in zip(pred_enquire_score, enquire_score_golden)]

            f_demo.write("\t<src>" + " ".join(src_tokens) + "</src>\n")
            f_demo.write("\t<tgt>" + " ".join(tgt_tokens) + "</tgt>\n")
            f_demo.write("\t<pred>" + " ".join(pred_tokens) + "</pred>\n")
            f_demo.write("\t<enquire_loss>" + str(sum(enquire_loss)))
            f_demo.write("\t<pred_enquire>" + " ".join([str(x) for x in pred_enquire_score]) + "\n")
            f_demo.write("\t<enquire_golden>" + " ".join([str(x) for x in enquire_score_golden]) + "\n")

            if count % 200 == ep_no % 200:
                tf.logging.info("<src>" + " ".join(src_tokens) + "</src>\n")
                tf.logging.info("<tgt>" + " ".join(tgt_tokens) + "</tgt>\n")
                tf.logging.info("<pred>" + " ".join(pred_tokens) + "</pred>\n")
        if count % 200 == ep_no % 200:
            tf.logging.info("---------------------</sample>-------------------\n")

    f_valid.close()
    f_demo.close()


def main_simple():
    dictionary = json.loads(open(FLAGS.dictionary_path).readline())
    dictionary = {dictionary[key]: key for key in dictionary}
    random.seed(SEED)
    np.random.seed(SEED)

    hyper_params = model_config.HYPER_PARAMS

    with tf.device('/cpu:0'):
        print "STEP1: Test Init..."
        model = Model(hyper_params=hyper_params)

        tower_records = []
        tf.logging.info("STEP2: Map/Reduce...")
        for gpu_id in range(FLAGS.GPU_num):
            with tf.device('/gpu:%d' % gpu_id):
                tf.logging.info('Building tower: %d...' % gpu_id)
                with tf.name_scope('tower_%d' % gpu_id):
                    with tf.variable_scope('cpu_variables', reuse=gpu_id > 0):
                        t_src, t_src_mask, t_tgt_indices, t_tgt, t_tgt_mask, \
                        t_enquire_strings, t_enquire_entities, t_enquire_mask, t_enquire_score_golden, \
                        t_loss_alpha, t_loss_decoder, t_loss, t_grad = model.build_tower()

                        tower_records.append(
                            (t_src, t_src_mask, t_tgt_indices, t_tgt, t_tgt_mask,
                             t_enquire_strings, t_enquire_entities, t_enquire_mask, t_enquire_score_golden,
                             t_loss_alpha, t_loss_decoder, t_loss, t_grad))
        _, _, _, _, _, _, _, _, _, \
        tower_losses_alpha, tower_losses_decoder, tower_losses, tower_grads = zip(*tower_records)
        avg_losses_alpha = tf.reduce_mean(tower_losses_alpha)
        avg_losses_decoder = tf.reduce_mean(tower_losses_decoder)
        avg_losses = tf.reduce_mean(tower_losses)
        update = model.optimizer.apply_gradients(average_gradients(tower_grads))

        valid_params = model.build_eval()

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
            train_iter(ep, sess, model, tower_records,
                       avg_losses_alpha, avg_losses_decoder, avg_losses,
                       tower_losses, update)

            if ep % 5 == 0:
                tf.logging.info("STEP4: Evaluating...")
                valid_iter(ep, sess, valid_params, dictionary)


if __name__ == "__main__":
    main_simple()
