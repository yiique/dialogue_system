# too long/short
# dialog turn
# first sentence is a title(it can be removed)
#                   eg: 27, 33, 36
# punc
#
# fan ti zi replace
# special word
#   url             eg: 5, 19, 35
#   phone number    eg: 26
#   emoji           eg: 27
# lexical

import json
import sys
sys.path.append("../..")


from data import utils
from data import zh_wiki


kb_alias_file = "../corpus3/kb.alias"
kb_triples_file = "../corpus3/kb.triples"
multi_dia = "../corpus3/article.processed"
multi_index = "../corpus3/article.index"
actor_qa_prefix = "../corpus3/actor_qa.parsed-"

dictionary_file = "../corpus3/dictionary"


MAX_LEN = 80
MAX_TURN = 8


# Question, Answer, Subject, Predicate, Object, ConfidenceScore
def main_for_statistic():
    dictionary = {}
    entity = {}
    relation = {}
    count = [0, 0]

    f = open(multi_dia)
    for line in f:
        sens = line[:-1].split("</s>")
        for sen in sens:
            sen = sen.decode('utf-8')
            for char in sen:
                if char not in dictionary:
                    dictionary[char] = 0
                dictionary[char] += 1
    print "mul_dia chars: ", len(dictionary)
    f.close()

    num = 15
    for i in range(0, num):
        f = open(actor_qa_prefix + str(i))
        for line in f:
            info = json.loads(line[:-1])
            string = utils.com2sim(info["question"] + info["question_detail"] + info["answer"])

            for char in string:
                if char not in dictionary:
                    dictionary[char] = 0
                dictionary[char] += 1
        f.close()

    print "total char: ", len(dictionary)
    sort_dictionary = sorted(dictionary.iteritems(), key=lambda d: d[1], reverse=True)

    cut = False
    if cut:
        sort_dictionary = sort_dictionary[0:30000]

    f = open(kb_triples_file)
    for line in f:
        triple = line.strip().split("\t\t")
        entity[triple[0]] = 0
        entity[triple[2]] = 0
        relation[triple[1]] = 0
    f.close()

    count = 3
    dictionary = {"<START>": 0, "<END>": 1, "<UNK>": 2}
    for pair in sort_dictionary:
        dictionary[pair[0]] = count
        count += 1
    for en in entity:
        dictionary[en] = count
        count += 1
    for re in relation:
        dictionary[re] = count
        count += 1

    print "dict len: ", len(dictionary)
    f = open(dictionary_file, 'w')
    f.write(json.dumps(dictionary))


def main_for_index():
    dictionary = json.loads(open(dictionary_file).readline())

    f_new = open(multi_index, 'w')
    f_old = open(multi_dia, 'r')

    count = 0
    for line in f_old:
        count += 1
        # if count == 6:
        #     break
        sample = {
            "src_dialogue": [[dictionary["<UNK>"] for _ in range(MAX_LEN)] for __ in range(MAX_TURN)],
            "tgt_dialogue": [[dictionary["<UNK>"] for _ in range(MAX_LEN)] for __ in range(MAX_TURN)],
            "turn_mask": [0 for _ in range(MAX_TURN)],
            "src_mask": [[0 for _ in range(MAX_LEN)] for __ in range(MAX_TURN)],
            "tgt_mask": [[0 for _ in range(MAX_LEN)] for __ in range(MAX_TURN)]
        }
        sens = line[:-1].decode("utf-8").split("</s>")
        for i in range(len(sens)):
            if i % 2 == 0:
                dialogue_key = "src_dialogue"
                mask_key = "src_mask"
            else:
                dialogue_key = "tgt_dialogue"
                mask_key = "tgt_mask"
            sen = sens[i]
            sample[dialogue_key][int(i/2)][0] = dictionary["<START>"]
            sample[mask_key][int(i/2)][0] = 1
            for j in range(len(sen)):
                uchar = sen[j]
                index = dictionary[uchar]
                sample[dialogue_key][int(i/2)][j+1] = index
                sample[mask_key][int(i/2)][j+1] = 1
            sample[dialogue_key][int(i/2)][j+2] = dictionary["<END>"]
            sample[mask_key][int(i/2)][j+2] = 1
            sample["turn_mask"][int(i/2)] = 1

        if len(sens) % 2 != 0:
            sample["tgt_dialogue"][int(i/2)][0] = dictionary["<START>"]
            sample["tgt_dialogue"][int(i/2)][1] = dictionary["<END>"]
            sample["tgt_mask"][int(i/2)][0] = 1
            sample["tgt_mask"][int(i/2)][1] = 1

        '''if count % 5 == 0:
            print count
            print "=========================="
            print line
            print "src dialogue: "
            for _ in sample["src_dialogue"]:
                print _
            print "src mask: "
            for _ in sample["src_mask"]:
                print _
            print "tgt dialogue: "
            for _ in sample["tgt_dialogue"]:
                print _
            print "tgt mask: "
            for _ in sample["tgt_mask"]:
                print _
            print "turn mask: ", sample["turn_mask"]'''
        f_new.write(json.dumps(sample) + "\n")

    print "line", count
    f_new.close()


def file_split():
    count = 0
    f = open(multi_index)
    f_train = open(multi_index + ".train", 'w')
    f_valid = open(multi_index + ".valid", 'w')
    f_test = open(multi_index + ".test", 'w')
    for line in f:
        count += 1
        if count % 100 == 0:
            f_valid.write(line)
        elif count % 100 == 1:
            f_test.write(line)
        else:
            f_train.write(line)
    f.close()
    f_train.close()
    f_valid.close()
    f_test.close()


if __name__ == "__main__":
    # main_for_pre_process()
    main_for_statistic()
    main_for_index()
    file_split()