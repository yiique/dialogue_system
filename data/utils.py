# -*- coding:utf-8 -*-
__author__ = 'liushuman'


import zh_wiki


PUNCLIST = [
    "\t", "\n", " ", " ",
    "`", "~", "!", "@", "#", "$", "%", "^", "&", "*", "(", ")", "_", "-", "+", "=", "{", "}", "[", "]", "|", "\\",
    ":", ";", '"', "'", ",", ".", "<", ">", "?", "/",
    "～", "·", "！", "@", "#", "¥", "%", "……", "&", "*", "（", "）", "-", "——", "+", "=", "[", "]", "{", "}", "、",
    "|", "；", "：", "‘", "“", "，", "。", "《", "》", "？", "/"]


def is_chinese(uchar):
    if u'\u4e00' <= uchar <= u'\u9fa5':
        return True
    else:
        return False


def is_number(uchar):
    if u'\u0030' <= uchar <= u'\u0039':
        return True
    else:
        return False


def is_alphabet(uchar):
    if (u'\u0041' <= uchar <= u'\u005a') or (u'\u0061' <= uchar <= u'\u007a'):
        return True
    else:
        return False


COMDICT = {}
for key in zh_wiki.zh2Hant:
    COMDICT[zh_wiki.zh2Hant[key].decode('utf-8')] = key.decode('utf-8')


def com2sim(line):
    new_line = ""
    for uchar in line:
        if uchar in COMDICT:
            new_line += COMDICT[uchar]
        else:
            new_line += uchar
    return new_line


def unk_replace(line):
    new_line = ""
    for uchar in line:
        if not (is_chinese(uchar) or is_number(uchar) or is_alphabet(uchar) or uchar in PUNCLIST):
            new_line += "<UNK>"
        else:
            new_line += uchar
    return new_line
