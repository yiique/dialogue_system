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

import utils


import json


def main_for_pre_process():
    count = [0, 0, 0, 0, 0, 0]

    f_old = open("./corpus1/mul_dia.raw")
    f_new = open("./corpus1/mul_dia.processed", 'w')
    for line in f_old:
        sens = line[:-1].split("</s>")
        new_sens = []

        # special words
        for sen in sens:
            words = sen.split(" ")
            new_words = []

            url_mark = False
            for word in words:
                if "url" in word:
                    url_mark = True
                    new_words.append("<url>")
                    continue
                if url_mark is True:
                    url_mark = False
                    eng = True
                    for char in word:
                        if "a" <= char <= "z" or "A" <= char <= "Z":
                            pass
                        else:
                            eng = False
                            break
                    if eng:
                        continue
                try:
                    if 13000000000 < int(word) < 13999999999:
                        new_words.append("<phone_number>")
                        continue
                except:
                    pass
                new_words.append(word)
            new_sens.append(new_words)

        # first sentence/dialog turn
        # new_sens = new_sens[1:]
        if len(new_sens) < 2 or len(new_sens) > 20:
            count[1] += 1
            continue
        new_sens = ["".join(_) for _ in new_sens]

        # length
        correct_l = True
        correct_s = False
        for sen in new_sens:
            if len(sen.decode('utf-8')) > 58:
                correct_l = False
            if len(sen.decode('utf-8')) >= 2:
                correct_s = True
        if not correct_l or not correct_s:
            count[2] += 1
            continue

        # complicate & unk
        for i in range(0, len(new_sens)):
            sen = new_sens[i].decode("utf-8")
            new_sen = utils.com2sim(sen)
            # new_sen = utils.unk_replace(new_sen)
            new_sens[i] = new_sen.encode("utf-8")

        f_new.write("</s>".join(new_sens) + "\n")
        count[0] += 1
        if count[0] % 10000 == 0:
            print count[0], "</s>".join(new_sens)
    f_old.close()
    f_new.close()

    prefix = "./corpus1/single_qa/part-000"
    f_new = open("./corpus1/single_qa.processed", 'w')
    for i in range(0, 32):
        if i <= 9:
            f_old = open(prefix + "0" + str(i))
        else:
            f_old = open(prefix + str(i))

        for line in f_old:
            sens = utils.com2sim(line[:-1].decode('utf-8')).split("\t")
            if len(sens) != 6:
                count[4] += 1
                continue
            if float(sens[-1]) < 1.0:
                count[5] += 1

            f_new.write("\t".join(sens).encode("utf-8") + "\n")
            count[3] += 1
            if count[3] % 20000 == 0:
                print count[3], "\t".join(sens).encode("utf-8")
        f_old.close()
    f_new.close()

    print "count info(mul_num/turn error/len error/sqa_num/key error/low confidence): ", count


# Question, Answer, Subject, Predicate, Object, ConfidenceScore
def main_for_statistic():
    dictionary = {"<START>": 0, "<END>": 0, "<UNK>": 0}
    entity = {}
    relation = {}
    count = [0, 0]

    f = open("./corpus1/mul_dia.processed")
    for line in f:
        sens = line[:-1].split("</s>")
        for sen in sens:
            for char in sen.decode("utf-8"):
                # if not (utils.is_chinese(char) or utils.is_alphabet(char) or utils.is_number(char)):
                #     dictionary["<UNK>"] += 1
                #     continue
                if char not in dictionary:
                    dictionary[char] = 0
                dictionary[char] += 1
        count[0] += 1
    print "mul_dia chars: ", len(dictionary)
    f.close()

    f = open("./corpus1/single_qa.processed")
    for line in f:
        sens = line[:-1].split("\t")
        for char in sens[0].decode("utf-8") + sens[1].decode("utf-8"):
            # if not (utils.is_chinese(char) or utils.is_alphabet(char) or utils.is_number(char)):
            #     dictionary["<UNK>"] += 1
            #     continue
            if char not in dictionary:
                dictionary[char] = 0
            dictionary[char] += 1
        if sens[2] not in entity:
            entity[sens[2]] = 0
        entity[sens[2]] += 1
        if sens[4] not in entity:
            entity[sens[4]] = 0
        entity[sens[4]] += 1
        if sens[3] not in relation:
            relation[sens[3]] = 1
        relation[sens[3]] += 1
        count[1] += 1

    print "total char: ", len(dictionary)
    sort_dictionary = sorted(dictionary.iteritems(), key=lambda d: d[1], reverse=True)
    for pair in sort_dictionary[:10]:
        print pair[0], pair[1]
    index_count = 0
    for index_count in range(len(sort_dictionary)):
        if sort_dictionary[index_count][1] <= 50:
            break
    print "cut: ", index_count
    for pair in sort_dictionary[index_count-10: index_count]:
        print pair[0], pair[1]
    # print "unk: ", dictionary["<UNK>"]

    print "entities: ", len(entity)
    sort_entities = sorted(entity.iteritems(), key=lambda d: d[1], reverse=True)
    for pair in sort_entities[0:5]:
        print pair[0], pair[1]

    print "relations: ", len(relation)
    sort_relations = sorted(relation.iteritems(), key=lambda d: d[1], reverse=True)
    for pair in sort_relations[0:5]:
        print pair[0], pair[1]

    print "count(mul/sqa):", count

    count = 0
    dictionary = {}
    for pair in sort_dictionary:
        dictionary[pair[0]] = count
        count += 1

    print "dict len: ", len(dictionary)
    f = open("./corpus1/dictionary", 'w')
    f.write(json.dumps(dictionary))
    f = open("./corpus1/dictionary.txt", 'w')
    for key in dictionary:
        f.write(key.encode('utf-8') + "\n")


def main_for_index():
    dictionary = json.loads(open("./corpus1/dictionary").readline())

    # f_new = open("./corpus1/mul_dia.index", 'w')
    f_old = open("./corpus1/mul_dia.processed", 'r')

    count = 0
    for line in f_old:
        count += 1
        if count == 6:
            break
        sample = {
            "src_dialogue": [[dictionary["<UNK>"] for _ in range(60)] for __ in range(10)],
            "tgt_dialogue": [[dictionary["<UNK>"] for _ in range(60)] for __ in range(10)],
            "turn_mask": [0 for _ in range(10)],
            "src_mask": [[0 for _ in range(60)] for __ in range(10)],
            "tgt_mask": [[0 for _ in range(60)] for __ in range(10)]
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

        if count % 5 == 0:
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
            print "turn mask: ", sample["turn_mask"]
        # f_new.write(json.dumps(sample) + "\n")

    print "line", count
    # f_new.close()


if __name__ == "__main__":
    # main_for_pre_process()
    # main_for_statistic()
    main_for_index()