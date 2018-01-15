#coding:utf-8
import json
import jieba.analyse
import jieba.posseg as pseg
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
        sens = line.strip().split("</s>")
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
            info = json.loads(line.strip())
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
    print "entity len: ", len(entity)
    print "relation len: ", len(relation)
    f = open(dictionary_file, 'w')
    f.write(json.dumps(dictionary))


MOVIE_ALIAS_DICT = {}
ACTOR_ALIAS_DICT = {}
def alias_init():
    alias_dict = json.loads(open(kb_alias_file).readline())
    movie_alias_dict = alias_dict["movies"]
    actor_alias_dict = alias_dict["actors"]

    for alias in movie_alias_dict:
        if len(alias) <= 1 or len(movie_alias_dict[alias]) != 1:
            continue
        MOVIE_ALIAS_DICT[alias] = movie_alias_dict[alias]
    for alias in actor_alias_dict:
        if len(alias) <= 2 or len(actor_alias_dict[alias]) != 1:
            continue
        ACTOR_ALIAS_DICT[alias] = actor_alias_dict[alias]
    print "alias filter done: "
    print "movies(full/filtered): ", len(movie_alias_dict), len(MOVIE_ALIAS_DICT)
    print "actors(full/filtered): ", len(actor_alias_dict), len(ACTOR_ALIAS_DICT)


NAME_PUNC_LIST = [" ", "　", "·", "・"]
# NAME_PUNC_LIST = ["·", " ", "　", ".", "-", "—", "・", ":", "：", "'", ")", "(", "）", "（", "！", "!", "?",
#                   "？", "~", "`", "~", "@", "@", "#", "#", "$", "￥", "%", "%", "^", "……", "&", "&", "*", "*"]
NAME_PUNC_LIST = [x.decode('utf-8') for x in NAME_PUNC_LIST]
def sentence_processor(string):
    new_string = ""
    for i in range(1, len(string)-1):
        if string[i] in NAME_PUNC_LIST and 'a' <= string[i-1] <= 'z' and 'A' <= string[i-1] <= 'Z':
            continue
        else:
            new_string += string[i]
    if len(new_string) == 0:
        return new_string
    return new_string


def tokenizer(sentence, entity_related=True):
    """
    :param sentence:  string in unicode
    :param entity_related:
    :return:
    """
    if not entity_related:
        return [x for x in sentence]
    else:
        sentence = sentence_processor(sentence)

        pos_words = pseg.cut(sentence)
        split_words = []
        split_flags = []
        for w, f in pos_words:
            split_words.append(w)
            split_flags.append(f)

        key_words = jieba.analyse.extract_tags(sentence, topK=10, withWeight=True)
        for k in range(len(key_words))[::-1]:
            if key_words[k][1] < 1.2 or key_words[k][0] not in split_words:
                del key_words[k]
            else:
                index = split_words.index(key_words[k][0])
                if "n" not in split_flags[index] or split_flags[index] == 'eng':
                    del key_words[k]
        key_words = [k[0] for k in key_words]

        # movie
        match_movies = []
        filtered_match_movies = []
        for movie_name in MOVIE_ALIAS_DICT:
            if movie_name in sentence:
                match_movies.append(movie_name)

        for movie_name in match_movies:
            substr = False
            for _ in match_movies:
                if movie_name != _ and movie_name in _:
                    substr = True
                    break
            if substr:
                continue
            if 1 < len(movie_name) <= 2:
                if movie_name == sentence:
                    pass
                elif movie_name in split_words:
                    index = split_words.index(movie_name)
                    if (index == 0 or split_flags[index-1] == 'x') and \
                            (index == len(split_words)-1 or split_flags[index+1] == 'x'):
                        pass
                    else:
                        continue
                else:
                    continue
            elif 2 < len(movie_name) <= 5:
                if movie_name == sentence:
                    pass
                else:
                    s_index = -1
                    e_index = -1
                    match = True
                    for split_word in split_words:
                        if movie_name.startswith(split_word):
                            s_index = split_words.index(split_word)
                        if movie_name.endswith(split_word):
                            e_index = split_words.index(split_word)
                    if s_index == -1 or e_index == -1 or e_index < s_index:
                        match = False
                    elif "".join(split_words[s_index:e_index+1]) != movie_name:
                        match = False
                    if match and (s_index == 0 or split_flags[s_index-1] == 'x') and \
                            (e_index == len(split_words)-1 or split_flags[e_index+1] == 'x'):
                        pass
                    else:
                        match = False

                    if match:
                        pass
                    else:
                        cover = 0
                        for word in key_words:
                            if word in movie_name:
                                cover += len(word)
                        if float(cover) / float(len(movie_name)) >= 0.4:
                            pass
                        else:
                            continue
            else:
                pass
            filtered_match_movies.append(movie_name)

        for movie_name in filtered_match_movies:
            movie_entity = MOVIE_ALIAS_DICT[movie_name][0]
            sentence = sentence.replace(movie_name, "<ENTITY>" + str(movie_entity) + "</ENTITY>")

        # actor
        match_actors = []
        filtered_match_actors = []
        for actor_name in ACTOR_ALIAS_DICT:
            if actor_name in sentence:
                match_actors.append(actor_name)

        for actor_name in match_actors:
            substr = False
            for _ in match_actors:
                if actor_name != _ and actor_name in _:
                    substr = True
                    break
            if substr:
                continue
            if len(actor_name) <= 2:
                if actor_name not in split_words:
                    continue
                else:
                    index = split_words.index(actor_name)
                    if split_flags[index] != "nr":
                        continue
            else:
                pass
            filtered_match_actors.append(actor_name)

        for actor_name in filtered_match_actors:
            actor_entity = ACTOR_ALIAS_DICT[actor_name][0]
            sentence = sentence.replace(actor_name, "<ENTITY>" + str(actor_entity) + "</ENTITY>")

        # split
        uchars = []
        sens = sentence.split("<ENTITY>")
        for sen in sens:
            if "</ENTITY>" not in sen:
                uchars.extend([x for x in sen])
            else:
                pair = sen.split("</ENTITY>")
                uchars.extend([pair[0]])
                uchars.extend([x for x in pair[1]])
        return uchars


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
        sens = line.strip().decode("utf-8").split("</s>")
        for i in range(len(sens)):
            if i % 2 == 0:
                dialogue_key = "src_dialogue"
                mask_key = "src_mask"
            else:
                dialogue_key = "tgt_dialogue"
                mask_key = "tgt_mask"
            sen = sens[i]
            sen = tokenizer(sen, True)
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
        if count % 50 == 0:
            f_valid.write(line)
        elif count % 50 == 1:
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