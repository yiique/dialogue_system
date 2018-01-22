import cPickle as pickle
import json
import sys
sys.path.append("../..")


from data import utils

prefix = "../baseline1/ubuntu.pkl"
dictionary_file = "../baseline1/Dataset.dict.pkl"


MAX_LEN = 80
MAX_TURN = 8
CUT_OFF = 30000
CUT = True
Dictionary = {x[0]: int(x[1]) for x in pickle.load(open(dictionary_file))}
Dictionary["<START>"] = len(Dictionary)
Dictionary["<END>"] = len(Dictionary)
open("../baseline1/dictionary", 'w').write(json.dumps(Dictionary))


def line_cleaner(indices):
    sens = ("</w>".join([str(x) for x in indices])).split("</w>1</w>")

    if len(sens) > MAX_TURN * 2:
        return ""
    new_sens = []
    for sen in sens:
        words = sen.split("</w>")
        if len(words) > MAX_LEN - 2:
            return ""
        else:
            new_sens.append(" ".join(words))
    new_line = "</s>".join(new_sens) + "\n"
    return new_line


def file_cleaner(file_name):
    f_old = pickle.load(open(file_name))
    f_new = open(file_name + ".clean", 'w')
    count = [0, 0]
    for line in f_old:
        count[0] += 1
        new_line = line_cleaner(line)
        if new_line == "":
            continue
        f_new.write(new_line)
        count[1] += 1
    # f_old.close()
    f_new.close()
    return count


def main_for_preprocess():
    print "training file cleaning: (total/succ)", file_cleaner(prefix + ".train")
    print "valid file cleaning: (total/succ)", file_cleaner(prefix + ".valid")
    print "test file cleaning: (total/succ)", file_cleaner(prefix + ".test")


def file_indexer(file_name):
    f_old = open(file_name)
    f_new = open(file_name + ".index", 'w')

    count = 0
    for line in f_old:
        sens = line.strip().split("</s>")
        sample = {
            "src_dialogue": [[Dictionary["**unknown**"] for _ in range(MAX_LEN)] for __ in range(MAX_TURN)],
            "tgt_dialogue": [[Dictionary["**unknown**"] for _ in range(MAX_LEN)] for __ in range(MAX_TURN)],
            "turn_mask": [0 for _ in range(MAX_TURN)],
            "src_mask": [[0 for _ in range(MAX_LEN)] for __ in range(MAX_TURN)],
            "tgt_mask": [[0 for _ in range(MAX_LEN)] for __ in range(MAX_TURN)]
        }

        for i in range(len(sens)):
            if i % 2 == 0:
                dialogue_key = "src_dialogue"
                mask_key = "src_mask"
            else:
                dialogue_key = "tgt_dialogue"
                mask_key = "tgt_mask"
            sen = sens[i]
            sample[dialogue_key][int(i/2)][0] = Dictionary["<START>"]
            sample[mask_key][int(i/2)][0] = 1
            words = sen.split(" ")
            for j in range(len(words)):
                index = int(words[j])
                sample[dialogue_key][int(i/2)][j+1] = index
                sample[mask_key][int(i/2)][j+1] = 1
            sample[dialogue_key][int(i/2)][j+2] = Dictionary["<END>"]
            sample[mask_key][int(i/2)][j+2] = 1
            sample["turn_mask"][int(i/2)] = 1
        if len(sens) % 2 != 0:
            sample["tgt_dialogue"][int(i/2)][0] = Dictionary["<START>"]
            sample["tgt_dialogue"][int(i/2)][1] = Dictionary["<END>"]
            sample["tgt_mask"][int(i/2)][0] = 1
            sample["tgt_mask"][int(i/2)][1] = 1
        f_new.write(json.dumps(sample) + "\n")
        count += 1

    f_old.close()
    f_new.close()
    print "index file: ", file_name, "len: ", count


def main_for_index():
    file_indexer(prefix + ".train.clean")
    file_indexer(prefix + ".valid.clean")
    file_indexer(prefix + ".test.clean")


if __name__ == "__main__":
    main_for_preprocess()
    main_for_index()
