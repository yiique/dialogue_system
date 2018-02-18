#coding:utf-8
__author__ = 'liushuman'


import json
import random
import numpy as np


RAW_FILE = "../corpus4/data.experiment"
RAW_TRAIN_FILE = "../corpus4/data.experiment.train"
RAW_VALID_FILE = "../corpus4/data.experiment.valid"
RAW_TEST_FILE = "../corpus4/data.experiment.test"

KB_FILE = "../corpus4/kb.experiment"

DICTIONARY_FILE = "../corpus4/dictionary"
INDEX_TRAIN_FILE = "../corpus4/data.index.train"
INDEX_VALID_FILE = "../corpus4/data.index.valid"
INDEX_TEST_FILE = "../corpus4/data.index.test"


MAX_LEN = 80
MAX_TURN = 8
ENQUIRE_CAN_NUM = 80
DIFFUSE_CAN_NUM = 20


ENTITY_NUM = 50081


def file_split():
    count = 0
    f = open(RAW_FILE)
    f_train = open(RAW_TRAIN_FILE, 'w')
    f_valid = open(RAW_VALID_FILE, 'w')
    f_test = open(RAW_TEST_FILE, 'w')

    dataset = []
    for line in f:
        dataset.append(line)
    indices = [i for i in range(0, len(dataset))]
    random.shuffle(indices)

    for i in range(len(dataset)):
        line = dataset[indices[i]]
        count += 1
        if count % 80 == 0:
            f_valid.write(line)
        elif count % 80 == 1:
            f_test.write(line)
        else:
            f_train.write(line)
    f.close()
    f_train.close()
    f_valid.close()
    f_test.close()


def kb_init():
    kb_dict = json.loads(open(KB_FILE).readline())
    movie_kb_dict = kb_dict["movie"]
    celebrity_kb_dict = kb_dict["celebrity"]
    for id in celebrity_kb_dict:
        celebrity_kb_dict[id]["act_movie"] = []
    for id in movie_kb_dict:
        info = movie_kb_dict[id]
        for actor in info["actor"]:
            celebrity_kb_dict[actor]["act_movie"].append(id)

    movie_alias_dict = {}
    celebrity_alias_dict = {}

    for movie_id in movie_kb_dict:
        movie_alias = movie_kb_dict[movie_id]["title"]
        if len(movie_alias) <= 1:
            continue
        try:
            if 0 <= int(movie_alias) <= 99999:
                continue
        except:
            pass
        movie_alias_dict[movie_alias] = movie_id
    for celebrity_id in celebrity_kb_dict:
        celebrity_alias = celebrity_kb_dict[celebrity_id]["name"]
        if len(celebrity_alias) <= 1:
            continue
        try:
            if 0 <= int(celebrity_alias) <= 99999:
                continue
        except:
            pass
        eng = True
        for uchar in celebrity_alias:
            if 'a' <= uchar <= 'z' or 'A' <= uchar <= 'Z':
                pass
            else:
                eng = False
                break
        if eng and len(celebrity_alias) < 4:
            continue
        celebrity_alias_dict[celebrity_alias] = celebrity_id

    print "kb len: ", len(movie_kb_dict), len(celebrity_kb_dict)
    print "alias len: ", len(movie_alias_dict), len(celebrity_alias_dict)
    return movie_kb_dict, celebrity_kb_dict, movie_alias_dict, celebrity_alias_dict


def main_for_statistic(movie_kb_dict, celebrity_kb_dict):
    dictionary = {}
    relation = {
        "release_time_is": 0,
        "direct_by": 0, "act_by": 0,
        "act_movie": 0}

    f = open(RAW_TRAIN_FILE)
    for line in f:
        dialogue = json.loads(line.strip())
        for sentence in dialogue:
            for char in sentence["raw_sentence"]:
                if char not in dictionary:
                    dictionary[char] = 0
                dictionary[char] += 1
    f.close()

    print "total char: ", len(dictionary) + 3
    sort_dictionary = sorted(dictionary.iteritems(), key=lambda d: d[1], reverse=True)
    for pair in sort_dictionary[0: 5]:
        print pair[0].encode('utf-8'), pair[1]

    release_time_dict = {}
    for id in movie_kb_dict:
        release_time = movie_kb_dict[id]["release_time"]
        if release_time != "":
            release_time_dict[release_time] = 0

    count = 3
    dictionary = {"<START>": 0, "<END>": 1, "<UNK>": 2}
    for pair in sort_dictionary:
        dictionary[pair[0]] = count
        count += 1
    for movie in movie_kb_dict:
        dictionary[movie] = count
        count += 1
    for celebrity in celebrity_kb_dict:
        dictionary[celebrity] = count
        count += 1
    for release_time in release_time_dict:
        dictionary[release_time] = count
        count += 1
    for re in relation:
        dictionary[re] = count
        count += 1

    print "dict len: ", len(dictionary)
    print "entity len: ", len(movie_kb_dict) + len(celebrity_kb_dict) + len(release_time_dict)
    print "relation len: ", len(relation)
    f = open(DICTIONARY_FILE, 'w')
    f.write(json.dumps(dictionary))
    return dictionary


NAME_PUNC_LIST = [" ", "　", "·", "・"]
# NAME_PUNC_LIST = ["·", " ", "　", ".", "-", "—", "・", ":", "：", "'", ")", "(", "）", "（", "！", "!", "?",
#                   "？", "~", "`", "~", "@", "@", "#", "#", "$", "￥", "%", "%", "^", "……", "&", "&", "*", "*"]
NAME_PUNC_LIST = [x.decode('utf-8') for x in NAME_PUNC_LIST]
def sentence_processor(string):
    new_string = ""
    for i in range(0, len(string)):
        if string[i] in NAME_PUNC_LIST and i != 0 and 'a' <= string[i-1] <= 'z' \
                and i != len(string) - 1 and 'a' <= string[i+1] <= 'z':
            continue
        else:
            new_string += string[i]
    if len(new_string) == 0:
        return new_string
    return new_string


def enquire(string, dictionary,
            movie_alias_dict, celebrity_alias_dict, movie_kb_dict, celebrity_kb_dict,
            entity_history):
    # entity history is a list record history entity alias in the dialogue
    enquired_strings = []
    enquired_entities = []
    enquired_objs = []
    enquired_masks = []
    enquired_alias = []

    enquired_triples = []
    # type(entity/triple), entity_alias, entity, relation, entity
    new_history = []

    enquired_movies = []
    enquired_actors = []
    for movie_alias in movie_alias_dict:
        if movie_alias in string:
            movie_id = movie_alias_dict[movie_alias]
            enquired_movies.append(movie_alias)
            enquired_triples.append(
                ["entity", movie_alias, movie_id, movie_id, movie_id])
            new_history.append(movie_alias)
    for celebrity_alias in celebrity_alias_dict:
        if celebrity_alias in string:
            celebrity_id = celebrity_alias_dict[celebrity_alias]
            enquired_actors.append(celebrity_alias)
            enquired_triples.append(
                ["entity", celebrity_alias, celebrity_id, celebrity_id, celebrity_id])
            new_history.append(celebrity_alias)

    for movie_alias in enquired_movies:
        movie_id = movie_alias_dict[movie_alias]
        directors = movie_kb_dict[movie_id]["director"]
        for director in directors:
            enquired_triples.append(
                ["triple", movie_alias, movie_id, "direct_by", director])
    for movie_alias in enquired_movies:
        movie_id = movie_alias_dict[movie_alias]
        actors = movie_kb_dict[movie_id]["actor"]
        for actor in actors:
            enquired_triples.append(
                ["triple", movie_alias, movie_id, "act_by", actor])
    for movie_alias in enquired_movies:
        movie_id = movie_alias_dict[movie_alias]
        release_time = movie_kb_dict[movie_id]["release_time"]
        if release_time != "":
            enquired_triples.append(
                ["triple", movie_alias, movie_id, "release_time_is", release_time])
    for celebrity_alias in enquired_actors:
        celebrity_id = celebrity_alias_dict[celebrity_alias]
        movies = celebrity_kb_dict[celebrity_id]["act_movie"]
        for movie in movies:
            enquired_triples.append(
                ["triple", celebrity_alias, celebrity_id, "act_movie", movie])

    for entity in entity_history:
        if entity in movie_alias_dict:
            movie_id = movie_alias_dict[entity]
            enquired_triples.append(["entity", entity, movie_id, movie_id, movie_id])
        elif entity in celebrity_alias_dict:
            celebrity_id = celebrity_alias_dict[entity]
            enquired_triples.append(["entity", entity, celebrity_id, celebrity_id, celebrity_id])
    for entity in entity_history:
        if entity in movie_alias_dict:
            movie_id = movie_alias_dict[entity]
            directors = movie_kb_dict[movie_id]["director"]
            for director in directors:
                enquired_triples.append(
                    ["triple", entity, movie_id, "direct_by", director])
    for entity in entity_history:
        if entity in movie_alias_dict:
            movie_id = movie_alias_dict[entity]
            actors = movie_kb_dict[movie_id]["actor"]
            for actor in actors:
                enquired_triples.append(
                    ["triple", entity, movie_id, "act_by", actor])
    for entity in entity_history:
        if entity in movie_alias_dict:
            movie_id = movie_alias_dict[entity]
            release_time = movie_kb_dict[movie_id]["release_time"]
            if release_time != "":
                enquired_triples.append(
                    ["triple", entity, movie_id, "release_time_is", release_time])
    for entity in entity_history:
        if entity in celebrity_alias_dict:
            celebrity_id = celebrity_alias_dict[entity]
            movies = celebrity_kb_dict[celebrity_id]["act_movie"]
            for movie in movies:
                enquired_triples.append(
                    ["triple", entity, celebrity_id, "act_movie", movie])

    for i in range(ENQUIRE_CAN_NUM):
        e_string = [dictionary["<UNK>"] for _ in range(MAX_LEN)]
        e_entity = [dictionary["<UNK>"] for _ in range(2)]
        e_obj = [dictionary["<UNK>"]]
        e_mask = [0 for _ in range(MAX_LEN)]
        e_alias = "<UNK>"

        if i < len(enquired_triples):
            e_triple = enquired_triples[i]

            chars = []
            if e_triple[0] == "entity":
                chars.extend([x for x in e_triple[1]])
            elif e_triple[0] == "triple":
                chars.extend([x for x in e_triple[1]])
                chars.append(e_triple[3])

            for j in range(len(chars)):
                char = chars[j]
                if char not in dictionary:
                    char = "<UNK>"
                e_string[j] = dictionary[char]
                e_mask[j] = 1

            e_entity[0] = dictionary[e_triple[2]]
            e_entity[1] = dictionary[e_triple[3]]
            e_obj[0] = dictionary[e_triple[-1]]
            e_alias = e_triple[1]

        enquired_strings.append(e_string)
        enquired_entities.append(e_entity)
        enquired_objs.append(e_obj)
        enquired_masks.append(e_mask)
        enquired_alias.append(e_alias)

    # entity_history = new_history + entity_history

    return enquired_strings, enquired_entities, enquired_objs, enquired_masks, \
           new_history, enquired_alias


def main_for_index(raw_file, index_file):
    f_old = open(raw_file, 'r')
    f_new = open(index_file, 'w')

    count = 0
    drop_count = 0
    for line in f_old:
        count += 1
        # if count == 11:
        #     break
        dialogue = json.loads(line.strip())
        if len(dialogue) % 2 != 0:
            dialogue.append({
                "raw_sentence": '',
                "movie": [], "celebrity": [], "time": [], "triple": [], })
        dialogue_sample = []
        drop = False

        entity_history = []
        for i in range(len(dialogue)/2):
            src_sentence = dialogue[i * 2]
            tgt_sentence = dialogue[i * 2 + 1]
            sample = {
                "src": [dictionary["<UNK>"] for _ in range(MAX_LEN)],
                "src_mask": [0 for _ in range(MAX_LEN)],
                "tgt_indices": [dictionary["<UNK>"] for _ in range(MAX_LEN)],
                "tgt": [dictionary["<UNK>"] for _ in range(MAX_LEN)],
                "tgt_mask": [0 for _ in range(MAX_LEN)],
                "turn_mask": 0,
                "enquire_strings": [[dictionary["<UNK>"] for _ in range(MAX_LEN)] for __ in range(ENQUIRE_CAN_NUM)],
                "enquire_entities": [[dictionary["<UNK>"] for _ in range(2)] for __ in range(ENQUIRE_CAN_NUM)],
                "enquire_objs": [[dictionary["<UNK>"]] for _ in range(ENQUIRE_CAN_NUM)],
                "enquire_mask": [[0 for _ in range(MAX_LEN)] for __ in range(ENQUIRE_CAN_NUM)],
                "enquire_score_golden": [0 for _ in range(ENQUIRE_CAN_NUM)],
                "diffuse_golden": [dictionary["<UNK>"] for _ in range(DIFFUSE_CAN_NUM)],
                "diffuse_mask": [0 for _ in range(DIFFUSE_CAN_NUM)],
                "retriever_score_golden": [0 for _ in range(ENQUIRE_CAN_NUM + DIFFUSE_CAN_NUM)]
            }

            raw_src = src_sentence["raw_sentence"]
            src_chars = ["<START>"] + [x for x in raw_src] + ["<END>"]
            for j in range(len(src_chars)):
                char = src_chars[j]
                if char not in dictionary:
                    char = "<UNK>"
                sample["src"][j] = dictionary[char]
                sample["src_mask"][j] = 1

            raw_tgt = tgt_sentence["raw_sentence"]
            tgt_alias_dict = {}
            for movie in tgt_sentence["movie"]:
                raw_tgt = raw_tgt.replace(movie[0], "<ENTITY>" + movie[1] + "</ENTITY>")
                tgt_alias_dict[movie[1]] = movie[0]
            for celebrity in tgt_sentence["celebrity"]:
                raw_tgt = raw_tgt.replace(celebrity[0], "<ENTITY>" + celebrity[1] + "</ENTITY>")
                tgt_alias_dict[celebrity[1]] = celebrity[0]
            tgt_chars = []
            tgt_parts = raw_tgt.split("<ENTITY>")
            for part in tgt_parts:
                if part == "":
                    continue
                if "</ENTITY>" in part:
                    pair = part.split("</ENTITY>")
                    tgt_chars.append(pair[0])
                    if pair[1] != "":
                        tgt_chars.extend([x for x in pair[1]])
                else:
                    tgt_chars.extend([x for x in part])
            if len(tgt_chars) > MAX_LEN - 2:
                drop = True
                break
            tgt_chars = ["<START>"] + tgt_chars + ["<END>"]
            for j in range(len(tgt_chars)):
                char = tgt_chars[j]
                if char not in dictionary:
                    char = "<UNK>"
                sample["tgt_indices"][j] = dictionary[char]
                sample["tgt_mask"][j] = 1

            sample["turn_mask"] = 1

            e_strings, e_entities, e_objs, e_masks, new_history, e_alias = enquire(
                raw_src, dictionary, movie_alias_dict, celebrity_alias_dict, movie_kb_dict, celebrity_kb_dict,
                entity_history)
            sample["enquire_strings"] = e_strings
            sample["enquire_entities"] = e_entities
            sample["enquire_objs"] = e_objs
            sample["enquire_mask"] = e_masks
            for j in range(ENQUIRE_CAN_NUM):
                match = False
                e_can = [e_entities[j][0], e_entities[j][1], e_objs[j][0]]
                if e_can[0] == e_can[1] == e_can[2]:
                    for k in range(len(src_sentence["movie"])):
                        if dictionary[src_sentence["movie"][k][1]] == e_can[0]:
                            sample["enquire_score_golden"][j] = 1
                            match = True
                    if not match:
                        for k in range(len(src_sentence["celebrity"])):
                            if dictionary[src_sentence["celebrity"][k][1]] == e_can[0]:
                                sample["enquire_score_golden"][j] = 1
                                match = True
                else:
                    for k in range(len(src_sentence["triple"])):
                        tri = src_sentence["triple"][k]
                        if e_can[0] == dictionary[tri[0]] and e_can[1] == dictionary[tri[1]]\
                                and e_can[2] == dictionary[tri[2]]:
                            sample["enquire_score_golden"][j] = 1
                if e_alias[j] in entity_history and sample["enquire_score_golden"][j] == 0:
                    sample["enquire_score_golden"][j] = (1. - float(entity_history.index(e_alias[j]))
                                                         / float(len(entity_history))) * 0.5
            entity_history = new_history + entity_history

            cands = [x[0] for x in sample["enquire_objs"]]
            golden_objs = []
            for j in range(ENQUIRE_CAN_NUM):
                if sample["enquire_score_golden"][j] == 0:
                    continue
                golden_objs.append(sample["enquire_objs"][j][0])
            common_vocabs = len(dictionary) - 4 - ENTITY_NUM
            diffuse_count = 0
            for j in range(len(tgt_chars)):
                char = tgt_chars[j]
                if char not in dictionary:
                    char = "<UNK>"
                index = dictionary[char]
                if index < common_vocabs:
                    continue
                if index in golden_objs:
                    continue
                if index in sample["diffuse_golden"]:
                    continue
                sample["diffuse_golden"][diffuse_count] = index
                sample["diffuse_mask"][diffuse_count] = 1
                diffuse_count += 1
                if diffuse_count >= DIFFUSE_CAN_NUM:
                    break
                cands.append(index)
                if char in tgt_alias_dict:
                    char = tgt_alias_dict[char]
                entity_history.insert(0, char)

            for j in range(ENQUIRE_CAN_NUM):
                if sample["enquire_objs"][j][0] in sample["tgt_indices"] and sample["enquire_score_golden"][j] == 1:
                    sample["retriever_score_golden"][j] = sample["enquire_score_golden"][j]
            for j in range(DIFFUSE_CAN_NUM):
                sample["retriever_score_golden"][j+ENQUIRE_CAN_NUM] = sample["diffuse_mask"][j]

            for j in range(len(sample["tgt_indices"])):
                index = sample["tgt_indices"][j]
                if index < common_vocabs:
                    pass
                else:
                    index = cands.index(index) + common_vocabs
                sample["tgt"][j] = index

            '''print "######################################", count, "######################################"
            print "================src================"
            print raw_src.encode('utf-8')
            print sample["src"]
            print sample["src_mask"]
            print "================tgt================"
            print raw_tgt.encode('utf-8')
            print tgt_chars
            print sample["tgt_indices"]
            print sample["tgt_mask"]
            print "================enquire================"
            print "----strings----"
            print sample["enquire_strings"]
            print "----entities----"
            print sample["enquire_entities"]
            print "----objs----"
            print sample["enquire_objs"]
            print "----mask----"
            print sample["enquire_mask"]
            print "----score----"
            print sample["enquire_score_golden"]
            print "================diffuse================"
            print len(golden_objs)
            print "----golden----"
            print sample["diffuse_golden"]
            print "----mask----"
            print sample["diffuse_mask"]
            print "================retriever================"
            print sample["retriever_score_golden"]
            print "================final_tgt================"
            print sample["tgt_indices"]
            print sample["tgt"]'''

            dialogue_sample.append(sample)

        if drop:
            drop_count += 1
            continue

        while len(dialogue_sample) < MAX_TURN:
            dialogue_sample.append({
                "src": [dictionary["<UNK>"] for _ in range(MAX_LEN)],
                "src_mask": [0 for _ in range(MAX_LEN)],
                "tgt_indices": [dictionary["<UNK>"] for _ in range(MAX_LEN)],
                "tgt": [dictionary["<UNK>"] for _ in range(MAX_LEN)],
                "tgt_mask": [0 for _ in range(MAX_LEN)],
                "turn_mask": 0,
                "enquire_strings": [[dictionary["<UNK>"] for _ in range(MAX_LEN)] for __ in range(ENQUIRE_CAN_NUM)],
                "enquire_entities": [[dictionary["<UNK>"] for _ in range(2)] for __ in range(ENQUIRE_CAN_NUM)],
                "enquire_objs": [[dictionary["<UNK>"]] for _ in range(ENQUIRE_CAN_NUM)],
                "enquire_mask": [[0 for _ in range(MAX_LEN)] for __ in range(ENQUIRE_CAN_NUM)],
                "enquire_score_golden": [0 for _ in range(ENQUIRE_CAN_NUM)],
                "diffuse_golden": [dictionary["<UNK>"] for _ in range(DIFFUSE_CAN_NUM)],
                "diffuse_mask": [0 for _ in range(DIFFUSE_CAN_NUM)],
                "retriever_score_golden": [0 for _ in range(ENQUIRE_CAN_NUM + DIFFUSE_CAN_NUM)]
            })
        f_new.write(json.dumps(dialogue_sample) + "\n")

    print "line", count, "/", drop_count
    f_old.close()
    f_new.close()


if __name__ == "__main__":
    file_split()
    movie_kb_dict, celebrity_kb_dict, movie_alias_dict, celebrity_alias_dict = kb_init()
    dictionary = main_for_statistic(movie_kb_dict, celebrity_kb_dict)

    main_for_index(RAW_TRAIN_FILE, INDEX_TRAIN_FILE)
    main_for_index(RAW_VALID_FILE, INDEX_VALID_FILE)
    main_for_index(RAW_TEST_FILE, INDEX_TEST_FILE)
