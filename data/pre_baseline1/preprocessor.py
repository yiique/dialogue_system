import json
import sys
sys.path.append("../..")


from data import utils

prefix = ""
dictionary_file = "../baseline1/dictionary"


MAX_LEN = 80
MAX_TURN = 8
CUT_OFF = 30000
CUT = True


def line_cleaner(line):
    new_line = ""
    sens = line.strip().split("__eou__ __eot__")
    if len(sens) > MAX_TURN * 2:
        return new_line
    new_sens = []
    for sen in sens:
        sen = sen.replace("__eou__", ".")
        words = sen.split(" ")
        if len(words) > MAX_LEN - 2:
            return new_line
        else:
            new_sens.append(sen)
    new_line = "</s>".join(new_sens) + "\n"
    return new_line


def file_cleaner(file_name):
    f_old = open(file_name)
    f_new = open(file_name + ".clean", 'w')
    count = [0, 0]
    for line in f_old:
        count[0] += 1
        new_line = line_cleaner(line)
        if new_line == "":
            continue
        f_new.write(new_line)
        count[1] += 1
    f_old.close()
    f_new.close()
    return count


def main_for_preprocess():
    print "training file cleaning: (total/succ)", file_cleaner(prefix + ".train")
    print "valid file cleaning: (total/succ)", file_cleaner(prefix + ".valid")
    print "test file cleaning: (total/succ)", file_cleaner(prefix + ".test")


def main_for_statistic():
    dictionary = {}

    f = open(prefix + ".train.clean")
    for line in f:
        sens = line.strip().split("</s>")
        for sen in sens:
            words = sen.split(" ")
            for word in words:
                if word not in dictionary:
                    dictionary[word] = 0
                dictionary[word] += 1
    print "total words: ", len(dictionary)
    f.close()

    sort_dictionary = sorted(dictionary.iteritems(), key=lambda d: d[1], reverse=True)
    if CUT:
        sort_dictionary = sort_dictionary[0:CUT_OFF]

    count = 3
    dictionary = {"<START>": 0, "<END>": 1, "<UNK>": 2}
    for pair in sort_dictionary:
        dictionary[pair[0]] = count
        count += 1

    print "dict len: ", len(dictionary)
    f = open(dictionary_file, 'w')
    f.write(json.dumps(dictionary))


def file_indexer(file_name):
    f_old = open(file_name)
    f_new = open(file_name + ".index", 'w')

    dictionary = json.loads(open(dictionary_file).readline())

    count = 0
    for line in f_old:
        sens = line.strip().split("</s>")
        sample = {
            "src_dialogue": [[dictionary["<UNK>"] for _ in range(MAX_LEN)] for __ in range(MAX_TURN)],
            "tgt_dialogue": [[dictionary["<UNK>"] for _ in range(MAX_LEN)] for __ in range(MAX_TURN)],
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
            sample[dialogue_key][int(i/2)][0] = dictionary["<START>"]
            sample[mask_key][int(i/2)][0] = 1
            words = sen.split(" ")
            for j in range(len(words)):
                uchar = words[j]
                if uchar in dictionary:
                    index = dictionary[uchar]
                else:
                    index = dictionary["<UNK>"]
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
    main_for_statistic()
    main_for_index()
