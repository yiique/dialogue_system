#coding:utf-8
__author__ = 'liushuman'


import json


RAW_FILE = "../corpus3/part.raw"
RAW_TRAIN_FILE = "../corpus3/part.raw.train"
RAW_VALID_FILE = "../corpus3/part.raw.valid"
RAW_TEST_FILE = "../corpus3/part.raw.test"

ALIAS_FILE = "../corpus3/part.kb.alias"
KB_FILE = "../corpus3/part.kb.triples"

DICTIONARY_FILE = "../corpus3/dictionary"
INDEX_TRAIN_FILE = "../corpus3/part.index.train"
INDEX_VALID_FILE = "../corpus3/part.index.valid"
INDEX_TEST_FILE = "../corpus3/part.index.test"


MAX_LEN = 80
MAX_TURN = 8
ENQUIRE_CAN_NUM = 250
DIFFUSE_CAN_NUM = 50


def file_split():
    count = 0
    f = open(RAW_FILE)
    f_train = open(RAW_TRAIN_FILE, 'w')
    f_valid = open(RAW_VALID_FILE, 'w')
    f_test = open(RAW_TEST_FILE, 'w')
    for line in f:
        count += 1
        if count % 25 == 0:
            f_valid.write(line)
        elif count % 25 == 1:
            f_test.write(line)
        else:
            f_train.write(line)
    f.close()
    f_train.close()
    f_valid.close()
    f_test.close()


def kb_init():
    alias_dict = json.loads(open(ALIAS_FILE).readline())
    movie_alias_dict = alias_dict["movie"]
    actor_alias_dict = alias_dict["actor"]
    movie_alias_dict_filtered = {}
    actor_alias_dict_filtered = {}
    kb_dict = json.loads(open(KB_FILE).readline())
    movie_kb_dict = kb_dict["movie"]
    actor_kb_dict = kb_dict["actor"]
    director_kb_dict = kb_dict["director"]

    for alias in movie_alias_dict:
        if len(alias) <= 1:
            continue
        try:
            if 0 <= int(alias) <= 99999:
                continue
        except:
            pass
        movie_alias_dict_filtered[alias] = movie_alias_dict[alias]
    for alias in actor_alias_dict:
        if len(alias) <= 1:
            continue
        try:
            if 0 <= int(alias) <= 99999:
                continue
        except:
            pass
        eng = True
        for uchar in alias:
            if 'a' <= uchar <= 'z':
                pass
            else:
                eng = False
                break
        if eng and len(alias) < 4:
            continue
        actor_alias_dict_filtered[alias] = actor_alias_dict[alias]
    print "alias filter done: "
    print "movies(full/filtered): ", len(movie_alias_dict), len(movie_alias_dict_filtered)
    print "actors(full/filtered): ", len(actor_alias_dict), len(actor_alias_dict_filtered)
    return movie_kb_dict, actor_kb_dict, director_kb_dict, movie_alias_dict_filtered, actor_alias_dict_filtered


def main_for_statistic(movie_kb_dict, actor_kb_dict, director_kb_dict):
    dictionary = {}
    relation = {"direct_by": 0, "act_by": 0}

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

    count = 3
    dictionary = {"<START>": 0, "<END>": 1, "<UNK>": 2}
    for pair in sort_dictionary:
        dictionary[pair[0]] = count
        count += 1
    for movie in movie_kb_dict:
        dictionary[movie] = count
        count += 1
    for actor in actor_kb_dict:
        dictionary[actor] = count
        count += 1
    for director in director_kb_dict:
        dictionary[director] = count
        count += 1
    for re in relation:
        dictionary[re] = count
        count += 1

    print "dict len: ", len(dictionary)
    print "entity len: ", len(movie_kb_dict) + len(actor_kb_dict) + len(director_kb_dict)
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
            movie_alias_dict, actor_alias_dict,
            movie_kb_dict, actor_kb_dict, director_kb_dict):
    enquired_strings = []
    enquired_entities = []
    enquired_objs = []
    enquired_masks = []

    enquired_triples = []
    # type(entity/triple), entity_alias, entity, relation, entity

    enquired_movies = []
    enquired_actors = []
    enquired_directors = []
    for movie_alias in movie_alias_dict:
        if movie_alias in string:
            movie_id = movie_alias_dict[movie_alias]
            enquired_movies.append(movie_alias)
            enquired_triples.append(
                ["entity", movie_alias, movie_id, movie_id, movie_id])
    for actor_alias in actor_alias_dict:
        if actor_alias in string:
            actor_id = actor_alias_dict[actor_alias]
            enquired_actors.append(actor_alias)
            enquired_triples.append(
                ["entity", actor_alias, actor_id, actor_id, actor_id])
    for director in director_kb_dict:
        if director in string:
            enquired_directors.append(director)
            enquired_triples.append(
                ["entity", director, director, director, director])

    for movie_alias in enquired_movies:
        movie_id = movie_alias_dict[movie_alias]
        directors = movie_kb_dict[movie_id]["direct_by"]
        for director in directors:
            if director not in director_kb_dict:
                continue
            enquired_triples.append(
                ["triple", movie_alias, movie_id, "direct_by", director])
    '''for director in enquired_directors:
        movies = director_kb_dict[director]
        for movie in movies:
            if movie not in movie_kb_dict:
                continue
            enquired_triples.append(
                ["triple", director, director, "direct_movie", movie])'''
    for movie_alias in enquired_movies:
        movie_id = movie_alias_dict[movie_alias]
        actors = movie_kb_dict[movie_id]["act_by"]
        for actor in actors:
            if actor not in actor_kb_dict:
                continue
            enquired_triples.append(
                ["triple", movie_alias, movie_id, "act_by", actor])
    '''for actor_alias in enquired_actors:
        actor_id = actor_alias_dict[actor_alias]
        movies = actor_kb_dict[actor_id]
        for movie in movies:
            if movie not in movie_kb_dict:
                continue
            enquired_triples.append(
                ["triple", actor_alias, actor_id, "act_movie", movie])'''

    for i in range(ENQUIRE_CAN_NUM):
        e_string = [dictionary["<UNK>"] for _ in range(MAX_LEN)]
        e_entity = [dictionary["<UNK>"] for _ in range(2)]
        e_obj = [dictionary["<UNK>"]]
        e_mask = [0 for _ in range(MAX_LEN)]

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

        enquired_strings.append(e_string)
        enquired_entities.append(e_entity)
        enquired_objs.append(e_obj)
        enquired_masks.append(e_mask)

    return enquired_strings, enquired_entities, enquired_objs, enquired_masks


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
                "movie": [], "actor": [], "director": [], "triple": [], })
        dialogue_sample = []
        drop = False

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

            raw_src = sentence_processor(src_sentence["raw_sentence"])
            src_chars = ["<START>"] + [x for x in raw_src] + ["<END>"]
            for j in range(len(src_chars)):
                char = src_chars[j]
                if char not in dictionary:
                    char = "<UNK>"
                sample["src"][j] = dictionary[char]
                sample["src_mask"][j] = 1

            raw_tgt = sentence_processor(tgt_sentence["raw_sentence"])
            for movie in tgt_sentence["movie"]:
                raw_tgt = raw_tgt.replace(movie[0], "<ENTITY>" + movie[1] + "</ENTITY>")
            for director in tgt_sentence["director"]:
                raw_tgt = raw_tgt.replace(director, "<ENTITY>" + director + "</ENTITY>")
            for actor in tgt_sentence["actor"]:
                raw_tgt = raw_tgt.replace(actor[0], "<ENTITY>" + actor[1] + "</ENTITY>")
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

            e_strings, e_entities, e_objs, e_masks = enquire(
                raw_src, dictionary, movie_alias_dict, actor_alias_dict, movie_kb_dict, actor_kb_dict, director_kb_dict)
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
                        for k in range(len(src_sentence["actor"])):
                            if dictionary[src_sentence["actor"][k][1]] == e_can[0]:
                                sample["enquire_score_golden"][j] = 1
                                match = True
                    if not match:
                        for k in range(len(src_sentence["director"])):
                            if dictionary[src_sentence["director"][k]] == e_can[0]:
                                sample["enquire_score_golden"][j] = 1
                else:
                    for k in range(len(src_sentence["triple"])):
                        tri = src_sentence["triple"][k]
                        if e_can[0] == dictionary[tri[0]] and e_can[1] == dictionary[tri[1]]\
                                and e_can[2] == dictionary[tri[2]]:
                            sample["enquire_score_golden"][j] = 1

            cands = [x[0] for x in sample["enquire_objs"]]
            golden_objs = []
            for j in range(ENQUIRE_CAN_NUM):
                if sample["enquire_score_golden"][j] == 0:
                    continue
                golden_objs.append(sample["enquire_objs"][j][0])
            common_vocabs = len(dictionary) - 2 - len(movie_kb_dict) - len(actor_kb_dict) - len(director_kb_dict)
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
                sample["diffuse_golden"][diffuse_count] = index
                sample["diffuse_mask"][diffuse_count] = 1
                diffuse_count += 1
                cands.append(index)

            for j in range(ENQUIRE_CAN_NUM):
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
            print raw_src
            print sample["src"]
            print sample["src_mask"]
            print "================tgt================"
            print raw_tgt
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
    movie_kb_dict, actor_kb_dict, director_kb_dict, movie_alias_dict, actor_alias_dict = kb_init()
    dictionary = main_for_statistic(movie_kb_dict, actor_kb_dict, director_kb_dict)

    '''print len(actor_kb_dict)
    print actor_kb_dict["<actor39062>"]
    print dictionary["<actor39062>"]'''

    main_for_index(RAW_TRAIN_FILE, INDEX_TRAIN_FILE)
    main_for_index(RAW_VALID_FILE, INDEX_VALID_FILE)
    main_for_index(RAW_TEST_FILE, INDEX_TEST_FILE)
