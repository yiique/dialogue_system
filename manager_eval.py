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
# NKD
from configurations.configs import config_diffuse_corpus4 as model_config
from models.model_diffuse_mask import DiffuseModel as Model
# TODO: highway, dropout
# baseline1
from configurations.configs import config_baseline1_LSTM as model_config_baseline1
from models.model_baseline1_LSTM import BaselineModel as BaselineModel1
# baseline2
from configurations.configs import config_baseline2_HRED as model_config_baseline2
from models.model_baseline2_HRED import BaselineModel as BaselineModel2
# baseline3
from configurations.configs import config_baseline3_GenDS as model_config_baseling3
from models.model_baseline3_GenDS import BaselineModel as BaselineModel3




def valid_NKD(sess, valid_params, dictionary):
    count = 0
    f_valid = open(FLAGS.valid_path, 'r')
    f_test = open(FLAGS.test_path, 'r')
    kb_dict = json.loads(open("./data/corpus4/kb.experiment").readline())
    kb2alias_dict = {}
    for key in kb_dict["movie"]:
        kb2alias_dict[key] = kb_dict["movie"][key]["title"]
    for key in kb_dict["celebrity"]:
        kb2alias_dict[key] = kb_dict["celebrity"][key]["name"]

    f_hyp = open(FLAGS.valid_hypothesis_path, 'w')
    f_ref = open(FLAGS.valid_reference_path, 'w')
    entity_accuracy = []
    entity_recall = []
    cosine_list = []
    f_human = open("./data/corpus4/demo.human", 'w')

    f_file = f_valid
    while True:
        line = ""
        try:
            line = f_file.readline()
        except:
            if f_file == f_valid:
                f_file = f_test
                continue
            elif f_file == f_test:
                break

        sample = json.loads(line.strip())
        count += 1

        hred_hiddens = [[0.0 for _ in range(model_config.HYPER_PARAMS["hred_h_dim"])]]
        hred_memorys = [[0.0 for _ in range(model_config.HYPER_PARAMS["hred_h_dim"])]]

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
            diffuse_golden = sample[i]["diffuse_golden"]
            diffuse_mask = sample[i]["diffuse_mask"]

            if int(turn_mask[0]) == 0:
                break

            feed_dict[valid_params[0]] = src
            feed_dict[valid_params[1]] = src_mask
            feed_dict[valid_params[2]] = turn_mask
            feed_dict[valid_params[3]] = enquire_strings
            feed_dict[valid_params[4]] = enquire_entities
            feed_dict[valid_params[5]] = enquire_mask
            feed_dict[valid_params[6]] = hred_hiddens
            feed_dict[valid_params[7]] = hred_memorys

            outputs = sess.run([valid_params[8], valid_params[9], valid_params[10], valid_params[11], valid_params[12], valid_params[13]], feed_dict=feed_dict)

            pred_enquire_score = outputs[0][0]
            pred_diffuse_score = outputs[1][0]
            pred_diffuse_indices = outputs[2][0]
            pred_sentence = outputs[5]          # beam_size * max_len

            src_flatten = np.transpose(src).tolist()[0][0: int(sum(x[0] for x in src_mask))]
            tgt_flatten = tgt_indices[0: int(sum(tgt_mask))]
            pred_flatten = pred_sentence[0].tolist()
            pred_enquired_entities = []
            pred_diffused_entities = []
            for j in range(len(pred_flatten)):
                if FLAGS.common_vocab <= pred_flatten[j] < FLAGS.common_vocab + FLAGS.enquire_can_num:
                    pred_flatten[j] = enquire_objs[pred_flatten[j] - FLAGS.common_vocab][0]
                    pred_enquired_entities.append(pred_flatten[j])
                elif FLAGS.common_vocab + FLAGS.enquire_can_num <= pred_flatten[j]:
                    pred_flatten[j] = pred_diffuse_indices[pred_flatten[j] - FLAGS.common_vocab - FLAGS.enquire_can_num]
                    pred_diffused_entities.append(pred_flatten[j])

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
            diffuse_topk = [kb2alias_dict[dictionary[x]].encode('utf-8')
                            if dictionary[x] in kb2alias_dict else dictionary[x].encode('utf-8')
                            for x in pred_diffuse_indices]
            diffuse_golden_topk = [kb2alias_dict[dictionary[x]].encode('utf-8')
                                   if dictionary[x] in kb2alias_dict else dictionary[x].encode('utf-8')
                                   for x in diffuse_golden]

            # BLEU
            f_ref.write(" ".join([x.encode('utf-8') for x in tgt_tokens]) + "\n")
            f_hyp.write(" ".join([x.encode('utf-8') for x in pred_tokens]) + "\n")
            # entity
            if sum(enquire_mask) > 0:
                count_acc = 0.0
                if len(pred_enquired_entities) > 0:
                    for entity in pred_enquired_entities:
                        if [entity] in enquire_objs:
                            count_acc += 1.0
                    count_acc = float(count_acc)/float(len(pred_enquired_entities))
                entity_accuracy.append(count_acc)
                count_re = 0.0
                for j in range(len(enquire_mask)):
                    if enquire_mask[j] == 1:
                        if enquire_objs[j][0] in pred_enquired_entities:
                            count_re += 1.0
                count_re /= float(sum(enquire_mask))
                entity_recall.append(count_re)
            # cosine
            if sum(diffuse_mask) > 0:
                positive_cosine = 0.0
                negative_cosine = 0.0
                for j in range(FLAGS.diffuse_can_num):
                    if pred_diffuse_indices[j] in pred_flatten:
                        positive_cosine += pred_diffuse_score[j]
                    else:
                        negative_cosine += pred_diffuse_score[j]
                if len(pred_diffuse_indices) > 0:
                    positive_cosine /= float(len(pred_diffuse_indices))
                if FLAGS.diffuse_can_num - len(pred_diffused_entities) > 0:
                    negative_cosine /= float(FLAGS.diffuse_can_num - len(pred_diffused_entities))
                cosine_list.append(positive_cosine - negative_cosine)
            # human
            f_human.write("<dialogue " + str(count) + ">\n")
            f_human.write("\t<src>" + " ".join([x.encode('utf-8') for x in src_tokens]) + "\n")
            f_human.write("\t<tgt>" + " ".join([x.encode('utf-8') for x in tgt_tokens]) + "\n")
            f_human.write("\t<pred>" + " ".join([x.encode('utf-8') for x in pred_tokens]) + "\n")

    # f_valid.close()
    os.system("perl " + FLAGS.multi_bleu_path + " " + FLAGS.valid_reference_path +
              " < " + FLAGS.valid_hypothesis_path)
    print "acc: ", np.mean(entity_accuracy)
    print "recall: ", np.mean(entity_recall)
    print "cos: ", np.mean(cosine_list)


def main_simple():
    dictionary = json.loads(open(FLAGS.dictionary_path).readline())
    dictionary = {dictionary[key]: key for key in dictionary}
    random.seed(SEED)
    np.random.seed(SEED)

    hyper_params = model_config.HYPER_PARAMS

    with tf.device('/cpu:0'):
        print "STEP1: Test Init..."
        model = Model(hyper_params=hyper_params)

        valid_params = model.build_eval()

        config = tf.ConfigProto(allow_soft_placement=True)
        config.gpu_options.allow_growth = True
        sess = tf.Session(config=config)
        sess.run(tf.global_variables_initializer())

        tf.logging.info("STEP2: Evaluating...")
        valid_NKD(sess, valid_params, dictionary)


if __name__ == "__main__":
    main_simple()
