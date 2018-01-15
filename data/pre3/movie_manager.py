#coding:utf-8
import jieba.analyse
import jieba.posseg as pseg
import json
import time


import zh_wiki


# complicate
COM_DICT = {}
for key in zh_wiki.zh2Hant:
    COM_DICT[zh_wiki.zh2Hant[key].decode('utf-8')] = key.decode('utf-8')


def do_cut(string):
    if "。".decode('utf-8') in string:
        cut_punc = "。".decode('utf-8')
    elif "，".decode('utf-8') in string:
        cut_punc = "，".decode('utf-8')
    elif ",".decode('utf-8') in string:
        cut_punc = ",".decode('utf-8')
    elif "\n" in string:
        cut_punc = "\n"
    elif "\t" in string:
        cut_punc = "\t"
    elif " ".decode('utf-8') in string:
        cut_punc = " ".decode('utf-8')
    else:
        cut_punc = ""
    if cut_punc != "":
        cut_index = string.rindex(cut_punc)
        cut_string = string[0: cut_index+1]
    else:
        cut_string = ""
    return cut_string


def sense_judge(string):
    new_string = ""
    for ch in string:
        if u'\u4e00' <= ch <= u'\u9fa5' or 'a' <= ch <= 'z' or 'A' <= ch <= "Z":
            new_string += ch
    return new_string


def complicate_cleaner(string):
    new_string = ""
    for char in string:
        if char in COM_DICT:
            new_string += COM_DICT[char]
        else:
            new_string += char
    return new_string.lower()


MAX_LEN = 78
MIN_DIA_LEN = 2
MAX_DIA_LEN = 16


NAME_PUNC_LIST = ["·", " ", "　", ".", "-", "—", "・", ":", "：", "'", ")", "(", "）", "（", "！", "!", "?",
                  "？", "~", "`", "~", "@", "@", "#", "#", "$", "￥", "%", "%", "^", "……", "&", "&", "*", "*"]
NAME_PUNC_LIST = [x.decode('utf-8') for x in NAME_PUNC_LIST]


def sentence_processor(string):
    for punc in NAME_PUNC_LIST:
        string = string.replace(punc, "")
    if len(string) == 0:
        return string
    return string


def mul_kb_related():
    alias_file = "../../corpus/kb.alias"
    triple_file = "../../corpus/kb.triples"
    kb_transfer_list = ["../../corpus/movie_content.processed", "../../corpus/actor_content.processed"]
    processed_file = "../../corpus/article.processed.1w.clean"
    test_file = "../../corpus/article.related.1w.filter"
    test_f = open(test_file, 'w')

    alias_dictionary = {}
    # reversed_alias_dictionary = {}

    alias_dict = json.loads(open(alias_file).readline())
    movie_alias_dict = alias_dict["movies"]
    actor_alias_dict = alias_dict["actors"]
    movie_alias_dict_filtered = {}
    actor_alias_dict_filtered = {}

    for alias in movie_alias_dict:
        if len(alias) <= 1 or len(movie_alias_dict[alias]) != 1:
            continue
        movie_alias_dict_filtered[alias] = movie_alias_dict[alias]
        # alias_entity = movie_alias_dict[alias][0]
        # if alias_entity not in reversed_alias_dictionary:
        #     reversed_alias_dictionary[alias_entity] = ""
        # reversed_alias_dictionary[alias_entity] += alias.encode('utf-8') + "=="
    for alias in actor_alias_dict:
        if len(alias) <= 2 or len(actor_alias_dict[alias]) != 1:
            continue
        actor_alias_dict_filtered[alias] = actor_alias_dict[alias]
        # alias_entity = actor_alias_dict[alias][0]
        # if alias_entity not in reversed_alias_dictionary:
        #     reversed_alias_dictionary[alias_entity] = ""
        # reversed_alias_dictionary[alias_entity] += alias.encode('utf-8') + "=="
    print "movie_alias_len/actor_alias_len", len(movie_alias_dict), len(actor_alias_dict)
    movie_alias_dict = movie_alias_dict_filtered
    actor_alias_dict = actor_alias_dict_filtered
    print "movie_alias_filtered_len/actor_alias_filtered_len", len(movie_alias_dict), len(actor_alias_dict)


    count = [0, 0, 0, 0, 0, 0, 0, 0]
    for line in open(processed_file):
        if count[0] % 1000 == 0:
            pass
        if count[0] % 500 == 0:
            print count, time.ctime()
        # if count[0] == 200:
        #    break
        count[0] += 1
        sentences = line[:-1].split("</s>")

        test_f.write("<dialogue_begin>\n")
        for sentence in sentences:
            test_f.write("\t<sentence_begin>\n")

            test_f.write("\t" + sentence + "\n")
            key_words = jieba.analyse.extract_tags(sentence, topK=10, withWeight=True)
            pos_words = pseg.cut(sentence)
            full_words = []
            full_flags = []
            for w, f in pos_words:
                full_words.append(w)
                full_flags.append(f)
            for k in range(len(key_words))[::-1]:
                if key_words[k][1] < 1.2 or key_words[k][0] not in full_words:
                    del key_words[k]
                else:
                    index = full_words.index(key_words[k][0])
                    if "n" not in full_flags[index] or full_flags[index] == 'eng':
                        del key_words[k]
            key_words = [k[0] for k in key_words]
            test_f.write("\t<key_words>" + "==".join(x.encode('utf-8') for x in key_words) + "\n")
            test_f.write("\t<split_words>" + "==".join(x.encode('utf-8') for x in full_words) + "\n")
            test_f.write("\t<split_tags>" + "==".join(x.encode('utf-8') for x in full_flags) + "\n")
            sentence = sentence_processor(sentence.decode('utf-8'))

            test_f.write("\t\t<movie>\n")
            match_movies = []
            filtered_match_movies = []
            for movie_name in movie_alias_dict:
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
                    elif movie_name in full_words:
                        index = full_words.index(movie_name)
                        if (index == 0 or full_flags[index-1] == 'x') and \
                                (index == len(full_words)-1 or full_flags[index+1] == 'x'):
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
                        for split_word in full_words:
                            if movie_name.startswith(split_word):
                                s_index = full_words.index(split_word)
                            if movie_name.endswith(split_word):
                                e_index = full_words.index(split_word)
                        if s_index == -1 or e_index == -1 or e_index < s_index:
                            match = False
                        elif "".join(full_words[s_index:e_index+1]) != movie_name:
                            match = False
                        if match and (s_index == 0 or full_flags[s_index-1] == 'x') and \
                                (e_index == len(full_words)-1 or full_flags[e_index+1] == 'x'):
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
                movie_entitys = movie_alias_dict[movie_name]
                for movie_entity in movie_entitys:
                    test_f.write("\t\t\t" + movie_name.encode('utf-8') + "\t\t" + movie_entity.encode('utf-8') +
                                 "\n")
                count[1] += 1
            test_f.write("\t\t</movie>\n")
            test_f.write("\t\t<actor>\n")
            for actor_name in actor_alias_dict:
                if actor_name in sentence:
                    if len(actor_name) <= 2:
                        if actor_name not in full_words:
                            continue
                        else:
                            index = full_words.index(actor_name)
                            if full_flags[index] != "nr":
                                continue
                    else:
                        pass
                    actor_entitys = actor_alias_dict[actor_name]
                    for actor_entity in actor_entitys:
                        test_f.write("\t\t\t" + actor_name.encode('utf-8') + "\t\t" + actor_entity.encode('utf-8') + "\n")
                    count[2] += 1
            test_f.write("\t\t</actor>\n")


            test_f.write("\t<sentence_end>\n")
        test_f.write("<dialogue_end>\n")

    print len(alias_dictionary)

if __name__ == "__main__":
    # main()
    mul_kb_related()
