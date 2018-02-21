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
COMMON_VOCAB = 5271
FLAGS = tf.flags.FLAGS


# Change config in configs and model in models to judge model
# NKD
from configurations.configs import config_diffuse_corpus4 as model_config
from models.model_diffuse_mask import DiffuseModel as Model
# TODO: highway, dropout
# baseline1
# from configurations.configs import config_baseline1_LSTM as model_config
# from models.model_baseline1_LSTM import BaselineModel as Model
# baseline2
# from configurations.configs import config_baseline2_HRED as model_config
# from models.model_baseline2_HRED import BaselineModel as Model
# baseline3
# from configurations.configs import config_baseline3_GenDS as model_config
# from models.model_baseline3_GenDS import BaselineModel as Model


def valid_LSTM(sess, valid_params, dictionary):
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
    QA_entity_accuracy_list = []
    QA_entity_recall_list = []
    entire_entity_accuracy_list = []
    entire_entity_recall_list = []
    number_of_entity_list = []
    f_human = open("./data/corpus4/demo.human", 'w')

    f_file = f_valid
    while True:
        line = ""
        try:
            line = f_file.readline()
            if line == "":
                raise
        except:
            if f_file == f_valid:
                f_file = f_test
                continue
            elif f_file == f_test:
                break

        sample = json.loads(line.strip())
        count += 1

        if count % 10 == 0:
            print count, time.ctime()
        if count == 30:
            break

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
            enquire_golden = sample[i]["enquire_score_golden"]
            enquire_objs = sample[i]["enquire_objs"]
            diffuse_mask = sample[i]["diffuse_mask"]
            diffuse_golden = sample[i]["diffuse_golden"]

            if int(turn_mask[0]) == 0:
                break

            feed_dict[valid_params[0]] = src
            feed_dict[valid_params[1]] = src_mask

            outputs = sess.run([valid_params[2], valid_params[3]], feed_dict=feed_dict)
            pred_sentence = outputs[1]          # beam_size * max_len

            src_flatten = np.transpose(src).tolist()[0][0: int(sum(x[0] for x in src_mask))]
            tgt_flatten = tgt_indices[0: int(sum(tgt_mask))]
            pred_flatten = pred_sentence[0].tolist()

            golden_enquired_entities = []
            golden_diffused_entities = []
            for tgt_index in tgt_flatten:
                if tgt_index in enquire_objs and enquire_golden[enquire_objs.index(tgt_index)] == 1:
                    golden_enquired_entities.append(tgt_index)
                elif tgt_index in diffuse_golden:
                    golden_diffused_entities.append(tgt_index)

            pred_entities = []
            for indice in tgt_flatten:
                if indice >= COMMON_VOCAB:
                    pred_entities.append(indice)

            if FLAGS.end_token not in pred_flatten:
                pass
            else:
                pred_flatten = pred_flatten[0: pred_flatten.index(FLAGS.end_token)]
            src_tokens = [dictionary[x].encode('utf-8') for x in src_flatten]
            tgt_tokens = [dictionary[x] for x in tgt_flatten]
            tgt_tokens = [kb2alias_dict[x].encode('utf-8') if x in kb2alias_dict else x.encode('utf-8')
                          for x in tgt_tokens]
            pred_tokens = [dictionary[x[0]] for x in pred_flatten]
            pred_tokens = [kb2alias_dict[x].encode('utf-8') if x in kb2alias_dict else x.encode('utf-8')
                           for x in pred_tokens]

            # BLEU
            f_ref.write(" ".join([x for x in tgt_tokens]) + "\n")
            f_hyp.write(" ".join([x for x in pred_tokens]) + "\n")
            # number
            number_of_entity_list.append(float(len(pred_entities)))
            # part entity
            if len(golden_enquired_entities) > 0:
                # acc
                count_acc = 0.0
                if len(pred_entities) > 0:
                    for entity in pred_entities:
                        if entity in golden_enquired_entities:
                            count_acc += 1.0
                    count_acc /= float(len(pred_entities))
                # recall
                count_re = 0.0
                for entity in golden_enquired_entities:
                    if entity in pred_entities:
                        count_re += 1.0
                count_re /= float(len(golden_enquired_entities))
                QA_entity_accuracy_list.append(count_acc)
                QA_entity_recall_list.append(count_re)
            # entire entity
            golden_entities = golden_enquired_entities + golden_diffused_entities
            if len(golden_entities) > 0:
                # acc
                count_acc = 0.0
                if len(pred_entities) > 0:
                    for entity in pred_entities:
                        if entity in golden_entities:
                            count_acc += 1.0
                    count_acc /= float(len(pred_entities))
                # recall
                count_re = 0.0
                for entity in golden_entities:
                    if entity in pred_entities:
                        count_re += 1.0
                count_re /= float(len(golden_entities))
                entire_entity_accuracy_list.append(count_acc)
                entire_entity_recall_list.append(count_re)
            # cosine
            # human
            f_human.write("<dialogue " + str(count) + ">\n")
            f_human.write("\t<src>" + " ".join([x for x in src_tokens]) + "\n")
            f_human.write("\t<tgt>" + " ".join([x for x in tgt_tokens]) + "\n")
            f_human.write("\t<pred>" + " ".join([x for x in pred_tokens]) + "\n")

    f_valid.close()
    f_test.close()
    f_hyp.close()
    f_ref.close()
    f_human.close()
    os.system("perl " + FLAGS.multi_bleu_path + " " + FLAGS.valid_reference_path +
              " < " + FLAGS.valid_hypothesis_path)
    print "part acc: ", np.mean(QA_entity_accuracy_list)
    print "part re: ", np.mean(QA_entity_recall_list)
    print "entire acc: ", np.mean(entire_entity_accuracy_list)
    print "entire re: ", np.mean(entire_entity_recall_list)
    print "numbers: ", np.mean(number_of_entity_list)


def valid_HRED(sess, valid_params, dictionary):
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
    QA_entity_accuracy_list = []
    QA_entity_recall_list = []
    entire_entity_accuracy_list = []
    entire_entity_recall_list = []
    number_of_entity_list = []
    f_human = open("./data/corpus4/demo.human", 'w')

    f_file = f_valid
    while True:
        line = ""
        try:
            line = f_file.readline()
            if line == "":
                raise
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

        if count % 10 == 0:
            print count, time.ctime()
        if count == 30:
            break

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
            enquire_golden = sample[i]["enquire_score_golden"]
            enquire_objs = [x[0] for x  in sample[i]["enquire_objs"]]
            diffuse_mask = sample[i]["diffuse_mask"]
            diffuse_golden = sample[i]["diffuse_golden"]

            if int(turn_mask[0]) == 0:
                break

            feed_dict[valid_params[0]] = src
            feed_dict[valid_params[1]] = src_mask
            feed_dict[valid_params[2]] = turn_mask
            feed_dict[valid_params[3]] = hred_hiddens
            feed_dict[valid_params[4]] = hred_memorys

            outputs = sess.run([valid_params[5], valid_params[6]], feed_dict=feed_dict)
            pred_sentence = outputs[1]          # beam_size * max_len

            src_flatten = np.transpose(src).tolist()[0][0: int(sum(x[0] for x in src_mask))]
            tgt_flatten = tgt_indices[0: int(sum(tgt_mask))]
            pred_flatten = pred_sentence[0].tolist()

            golden_enquired_entities = []
            golden_diffused_entities = []
            for tgt_index in tgt_flatten:
                if tgt_index in enquire_objs and enquire_golden[enquire_objs.index(tgt_index)] == 1:
                    golden_enquired_entities.append(tgt_index)
                elif tgt_index in diffuse_golden:
                    golden_diffused_entities.append(tgt_index)

            pred_entities = []
            for indice in tgt_flatten:
                if indice >= COMMON_VOCAB:
                    pred_entities.append(indice)

            if FLAGS.end_token not in pred_flatten:
                pass
            else:
                pred_flatten = pred_flatten[0: pred_flatten.index(FLAGS.end_token)]

            src_tokens = [dictionary[x].encode('utf-8') for x in src_flatten]
            tgt_tokens = [dictionary[x] for x in tgt_flatten]
            tgt_tokens = [kb2alias_dict[x].encode('utf-8') if x in kb2alias_dict else x.encode('utf-8')
                          for x in tgt_tokens]
            pred_tokens = [dictionary[x[0]] for x in pred_flatten]
            pred_tokens = [kb2alias_dict[x].encode('utf-8') if x in kb2alias_dict else x.encode('utf-8')
                           for x in pred_tokens]

            # BLEU
            f_ref.write(" ".join([x for x in tgt_tokens]) + "\n")
            f_hyp.write(" ".join([x for x in pred_tokens]) + "\n")
            # number
            number_of_entity_list.append(float(len(pred_entities)))
            # part entity
            if len(golden_enquired_entities) > 0:
                # acc
                count_acc = 0.0
                if len(pred_entities) > 0:
                    for entity in pred_entities:
                        if entity in golden_enquired_entities:
                            count_acc += 1.0
                    count_acc /= float(len(pred_entities))
                # recall
                count_re = 0.0
                for entity in golden_enquired_entities:
                    if entity in pred_entities:
                        count_re += 1.0
                count_re /= float(len(golden_enquired_entities))
                QA_entity_accuracy_list.append(count_acc)
                QA_entity_recall_list.append(count_re)
            # entire entity
            golden_entities = golden_enquired_entities + golden_diffused_entities
            if len(golden_entities) > 0:
                # acc
                count_acc = 0.0
                if len(pred_entities) > 0:
                    for entity in pred_entities:
                        if entity in golden_entities:
                            count_acc += 1.0
                    count_acc /= float(len(pred_entities))
                # recall
                count_re = 0.0
                for entity in golden_entities:
                    if entity in pred_entities:
                        count_re += 1.0
                count_re /= float(len(golden_entities))
                entire_entity_accuracy_list.append(count_acc)
                entire_entity_recall_list.append(count_re)
            # cosine
            # human
            f_human.write("<dialogue " + str(count) + ">\n")
            f_human.write("\t<src>" + " ".join([x for x in src_tokens]) + "\n")
            f_human.write("\t<tgt>" + " ".join([x for x in tgt_tokens]) + "\n")
            f_human.write("\t<pred>" + " ".join([x for x in pred_tokens]) + "\n")

    f_valid.close()
    f_test.close()
    f_hyp.close()
    f_ref.close()
    f_human.close()
    os.system("perl " + FLAGS.multi_bleu_path + " " + FLAGS.valid_reference_path +
              " < " + FLAGS.valid_hypothesis_path)
    print "part acc: ", np.mean(QA_entity_accuracy_list)
    print "part re: ", np.mean(QA_entity_recall_list)
    print "entire acc: ", np.mean(entire_entity_accuracy_list)
    print "entire re: ", np.mean(entire_entity_recall_list)
    print "numbers: ", np.mean(number_of_entity_list)


def valid_GenDS(sess, valid_params, dictionary):
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
    QA_entity_accuracy_list = []
    QA_entity_recall_list = []
    entire_entity_accuracy_list = []
    entire_entity_recall_list = []
    number_of_entity_list = []
    f_human = open("./data/corpus4/demo.human", 'w')

    f_file = f_valid
    while True:
        line = ""
        try:
            line = f_file.readline()
            if line == "":
                raise
        except:
            if f_file == f_valid:
                f_file = f_test
                continue
            elif f_file == f_test:
                break

        sample = json.loads(line.strip())
        count += 1

        if count % 10 == 0:
            print count, time.ctime()
        if count == 30:
            break

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
            enquire_golden = sample[i]["enquire_score_golden"]
            enquire_objs = [x[0] for x in sample[i]["enquire_objs"]]
            diffuse_mask = sample[i]["diffuse_mask"]
            diffuse_golden = sample[i]["diffuse_golden"]

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

            golden_enquired_entities = []
            golden_diffused_entities = []
            for tgt_index in tgt_flatten:
                if tgt_index in enquire_objs and enquire_golden[enquire_objs.index(tgt_index)] == 1:
                    golden_enquired_entities.append(tgt_index)
                elif tgt_index in diffuse_golden:
                    golden_diffused_entities.append(tgt_index)

            pred_enquired_entities = []
            pred_diffused_entities = []
            for j in range(len(pred_flatten)):
                if FLAGS.common_vocab <= pred_flatten[j] < FLAGS.common_vocab + FLAGS.enquire_can_num:
                    pred_flatten[j] = enquire_objs[pred_flatten[j] - FLAGS.common_vocab]
                    pred_enquired_entities.append(pred_flatten[j])
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

            # BLEU
            f_ref.write(" ".join([x for x in tgt_tokens]) + "\n")
            f_hyp.write(" ".join([x for x in pred_tokens]) + "\n")
            # number
            number_of_entity_list.append(float(len(pred_enquired_entities) + len(pred_diffused_entities)))
            # part entity
            if len(golden_enquired_entities) > 0:
                # acc
                count_acc = 0.0
                if len(pred_enquired_entities) > 0:
                    for entity in pred_enquired_entities:
                        if entity in golden_enquired_entities:
                            count_acc += 1.0
                    count_acc /= float(len(pred_enquired_entities))
                # recall
                count_re = 0.0
                for entity in golden_enquired_entities:
                    if entity in pred_enquired_entities:
                        count_re += 1.0
                count_re /= float(len(golden_enquired_entities))
                QA_entity_accuracy_list.append(count_acc)
                QA_entity_recall_list.append(count_re)
            # entire entity
            golden_entities = golden_enquired_entities + golden_diffused_entities
            pred_entities = pred_enquired_entities + pred_diffused_entities
            if len(golden_entities) > 0:
                # acc
                count_acc = 0.0
                if len(pred_entities) > 0:
                    for entity in pred_entities:
                        if entity in golden_entities:
                            count_acc += 1.0
                    count_acc /= float(len(pred_entities))
                # recall
                count_re = 0.0
                for entity in golden_entities:
                    if entity in pred_entities:
                        count_re += 1.0
                count_re /= float(len(golden_entities))
                entire_entity_accuracy_list.append(count_acc)
                entire_entity_recall_list.append(count_re)

            # human
            f_human.write("<dialogue " + str(count) + ">\n")
            f_human.write("\t<src>" + " ".join([x for x in src_tokens]) + "\n")
            f_human.write("\t<tgt>" + " ".join([x for x in tgt_tokens]) + "\n")
            f_human.write("\t<pred>" + " ".join([x for x in pred_tokens]) + "\n")

    f_valid.close()
    f_test.close()
    f_hyp.close()
    f_ref.close()
    f_human.close()
    os.system("perl " + FLAGS.multi_bleu_path + " " + FLAGS.valid_reference_path +
              " < " + FLAGS.valid_hypothesis_path)
    print "part acc: ", np.mean(QA_entity_accuracy_list)
    print "part re: ", np.mean(QA_entity_recall_list)
    print "entire acc: ", np.mean(entire_entity_accuracy_list)
    print "entire re: ", np.mean(entire_entity_recall_list)
    print "numbers: ", np.mean(number_of_entity_list)


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
    QA_entity_accuracy_list = []
    QA_entity_recall_list = []
    entire_entity_accuracy_list = []
    entire_entity_recall_list = []
    number_of_entity_list = []
    positive_cosine_list = []
    negative_cosine_list = []
    reduce_cosine_list = []
    f_human = open("./data/corpus4/demo.human", 'w')

    f_file = f_valid
    while True:
        line = ""
        try:
            line = f_file.readline()
            if line == "":
                raise
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

        if count % 10 == 0:
            print count, time.ctime()
        if count == 30:
            break

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
            enquire_golden = sample[i]["enquire_score_golden"]
            enquire_objs = [x[0] for x in sample[i]["enquire_objs"]]
            diffuse_mask = sample[i]["diffuse_mask"]
            diffuse_golden = sample[i]["diffuse_golden"]

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

            outputs = sess.run([valid_params[8], valid_params[9], valid_params[10], valid_params[11], valid_params[12],
                                valid_params[13]], feed_dict=feed_dict)

            pred_diffuse_score = outputs[1][0]
            pred_diffuse_indices = outputs[2][0]
            pred_sentence = outputs[5]          # beam_size * max_len

            src_flatten = np.transpose(src).tolist()[0][0: int(sum(x[0] for x in src_mask))]
            tgt_flatten = tgt_indices[0: int(sum(tgt_mask))]
            pred_flatten = pred_sentence[0].tolist()

            golden_enquired_entities = []
            golden_diffused_entities = []
            for tgt_index in tgt_flatten:
                if tgt_index in enquire_objs and enquire_golden[enquire_objs.index(tgt_index)] == 1:
                    golden_enquired_entities.append(tgt_index)
                elif tgt_index in diffuse_golden:
                    golden_diffused_entities.append(tgt_index)

            pred_enquired_entities = []
            pred_diffused_entities = []
            for j in range(len(pred_flatten)):
                if FLAGS.common_vocab <= pred_flatten[j] < FLAGS.common_vocab + FLAGS.enquire_can_num:
                    pred_flatten[j] = enquire_objs[pred_flatten[j] - FLAGS.common_vocab]
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

            # BLEU
            f_ref.write(" ".join([x for x in tgt_tokens]) + "\n")
            f_hyp.write(" ".join([x for x in pred_tokens]) + "\n")
            # number
            number_of_entity_list.append(float(len(pred_enquired_entities) + len(pred_diffused_entities)))
            # part entity
            if len(golden_enquired_entities) > 0:
                # acc
                count_acc = 0.0
                if len(pred_enquired_entities) > 0:
                    for entity in pred_enquired_entities:
                        if entity in golden_enquired_entities:
                            count_acc += 1.0
                    count_acc /= float(len(pred_enquired_entities))
                # recall
                count_re = 0.0
                for entity in golden_enquired_entities:
                    if entity in pred_enquired_entities:
                        count_re += 1.0
                count_re /= float(len(golden_enquired_entities))
                QA_entity_accuracy_list.append(count_acc)
                QA_entity_recall_list.append(count_re)
            # entire entity
            golden_entities = golden_enquired_entities + golden_diffused_entities
            pred_entities = pred_enquired_entities + pred_diffused_entities
            if len(golden_entities) > 0:
                # acc
                count_acc = 0.0
                if len(pred_entities) > 0:
                    for entity in pred_entities:
                        if entity in golden_entities:
                            count_acc += 1.0
                    count_acc /= float(len(pred_entities))
                # recall
                count_re = 0.0
                for entity in golden_entities:
                    if entity in pred_entities:
                        count_re += 1.0
                count_re /= float(len(golden_entities))
                entire_entity_accuracy_list.append(count_acc)
                entire_entity_recall_list.append(count_re)
            # cosine
            if len(golden_diffused_entities) > 0:
                positive_cosine = 0.0
                negative_cosine = 0.0
                for j in range(FLAGS.diffuse_can_num):
                    if pred_diffuse_indices[j] in pred_flatten:
                        positive_cosine += pred_diffuse_score[j]
                    else:
                        negative_cosine += pred_diffuse_score[j]
                if len(pred_diffused_entities) > 0:
                    positive_cosine /= float(len(pred_diffused_entities))
                if FLAGS.diffuse_can_num - len(pred_diffused_entities) > 0:
                    negative_cosine /= float(FLAGS.diffuse_can_num - len(pred_diffused_entities))
                positive_cosine_list.append(positive_cosine)
                negative_cosine_list.append(negative_cosine)
                reduce_cosine_list.append(positive_cosine - negative_cosine)
            # human
            f_human.write("<dialogue " + str(count) + ">\n")
            f_human.write("\t<src>" + " ".join([x for x in src_tokens]) + "\n")
            f_human.write("\t<tgt>" + " ".join([x for x in tgt_tokens]) + "\n")
            f_human.write("\t<pred>" + " ".join([x for x in pred_tokens]) + "\n")

    f_valid.close()
    f_test.close()
    f_hyp.close()
    f_ref.close()
    f_human.close()
    os.system("perl " + FLAGS.multi_bleu_path + " " + FLAGS.valid_reference_path +
              " < " + FLAGS.valid_hypothesis_path)
    print "part acc: ", np.mean(QA_entity_accuracy_list)
    print "part re: ", np.mean(QA_entity_recall_list)
    print "entire acc: ", np.mean(entire_entity_accuracy_list)
    print "entire re: ", np.mean(entire_entity_recall_list)
    print "numbers: ", np.mean(number_of_entity_list)
    print "pos cos: ", np.mean(positive_cosine_list)
    print "neg cos: ", np.mean(negative_cosine_list)
    print "red cos: ", np.mean(reduce_cosine_list)


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

        model.load_weight(sess)

        tf.logging.info("STEP2: Evaluating...")
        valid_NKD(sess, valid_params, dictionary)


if __name__ == "__main__":
    main_simple()
