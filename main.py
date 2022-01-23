#!/usr/bin/env python
# coding: utf-8

from collections import defaultdict
import os
import pathlib
import pickle
import random
import sys
import numpy as np
import pandas as pd

# random.seed(723)

CURR = "."
PATH = "english2.csv"  # 読み込む単語帳のパス
CHAPTER_START = 2  # 何章から練習の対象とするか
CHAPTER_END = 2  # 何章まで練習の対象とするか
REVERSE = False  # Falseなら和訳モード、Trueなら英訳モード
CNT = 10  # 挑戦する回数
LOAD = True  # 前回までの記録を読み込む
RECORD = True  # 結果を記録する  ※和訳モードのみ
PROB = True  # Trueなら過去の正解率に応じて出題確率を制御
LOSS = 0.2  # 正解数から目減りさせる数 (LOSS回正解するまで出題確率が下がらず、その後もLOSS回分目減りさせる)
MIN_PROB = 0.03  # 全問正解の場合に収束する出題確率
PENALTY = 0.5  # 不正解時のペナルティ (正解数を減らす)
ALL = False  # 全問出題モード　 ※CNTに関わらず指定chapの全問出題される
RANDOM = True  # ランダム出題モード

curr = pathlib.Path(CURR)
path = pathlib.Path(PATH)


def load(words, LOAD=LOAD):
    """過去の記録を読み込む
    """
    tmp_path = curr / "results" / (path.stem + "_dict_try.pickle")
    if tmp_path.exists() and LOAD:
        with open(tmp_path, "rb") as f:
            dict_try = pickle.load(f)
    else:
        dict_try = defaultdict(int)
    tmp_path = curr / "results" / (path.stem + "_dict_correct.pickle")
    if tmp_path.exists() and LOAD:
        with open(tmp_path, "rb") as f:
            dict_correct = pickle.load(f)
    else:
        dict_correct = defaultdict(int)
    return cleansing(dict_try, words), cleansing(dict_correct, words)


def cleansing(d, words):
    """単語帳に存在する単語の記録のみ抜粋する
    """
    dict_ret = defaultdict(int)
    for k, v in d.items():
        if k in words:
            dict_ret[k] = v
    return dict_ret


def dump(dict_try, dict_correct):
    """記録を保存する
    """
    with open(curr / "results" / (path.stem + "_dict_try.pickle"), "wb") as f:
        pickle.dump(dict_try, f)
    with open(curr / "results" / (path.stem + "_dict_correct.pickle"), "wb") as f:
        pickle.dump(dict_correct, f)
    get_accuracy()


def custom_input(dict_try, dict_correct):
    """文字列を受け取り色々する
    """
    inp = input()
    if inp in {"end", "えんｄ", "e", "え"}:
        if RECORD:
            dump(dict_try, dict_correct)
        sys.exit()
    else:
        return inp


def probability(df, dict_try, dict_correct, idx):
    """出題確率を制御する
    """
    word = df["word"][idx]
    accuracy = 0 if dict_try[word] == 0 else max(0, dict_correct[word] - LOSS) / dict_try[word]
    return 1 if dict_try[word] == 0 else 1 - accuracy * (1 - MIN_PROB)


def record(dict_try, dict_correct, idx, word):
    """記録するための処理
    """
    print("正解しましたか？ (y/n/覚えた/リセット)\n>>")
    inp = custom_input(dict_try, dict_correct)
    dict_try[word] += 1
    if inp in {"覚えた", "おぼえた"}:
        dict_correct[word] = dict_try[word]
    elif inp == "リセット":
        dict_correct[word] = 0
        dict_try[word] = 0
    elif inp in {"y", "yes", "はい", "正解", "1", "いぇｓ", "ｙ"}:
        dict_correct[word] += 1
    else:
        dict_correct[word] -= min(dict_correct[word], PENALTY)
    print("\n")


def practice(df, dict_try, dict_correct, i, idx):
    """単語の練習のための処理
    """
    if REVERSE:
        word = df["japanese"][idx]
        print("{}. 「{}」 を英語で言うと？\n>>".format(i + 1, word))
        inp = custom_input(dict_try, dict_correct)
        print("正解は 「{}」\n".format(df["word"][idx]))
    else:
        word = df["word"][idx]
        accuracy = 0 if dict_try[word] == 0 else dict_correct[word] / dict_try[word]
        print("{}. 「{}」 の品詞と日本語訳は？  ※これまでの正解率は{:.1f}%\n>>".format(i + 1, word, accuracy * 100))
        inp = custom_input(dict_try, dict_correct)
        print("正解は 「{}」 「{}」\n".format(
            df["part_of_speech"][idx], df["japanese"][idx]))
        record(dict_try, dict_correct, idx, word)


def get_data():
    """必要なデータを取得する
    """
    df_raw = pd.read_csv(curr / "dict" / PATH, encoding="utf-8")
    df = df_raw.copy()
    words = set(df["word"].values)
    df = df.loc[(df["chap"] >= CHAPTER_START) & (df["chap"] <= CHAPTER_END)]
    df.reset_index(drop=True, inplace=True)
    dict_try, dict_correct = load(words)
    return df, df_raw, words, dict_try, dict_correct


def get_accuracy(narrow_down=True):
    """正解率を取得する
    """
    df, _, _, dict_try, dict_correct = get_data()
    words = set(df["word"].values)
    result = []
    cnt_correct = cnt_try = cnt_word = perfect = 0
    for k, v in dict_try.items():
        if narrow_down:
            if k not in words:
                continue
        if v > 0:
            tmp_correct = dict_correct[k]
            result.append(((tmp_correct * 100) / v, k, str(tmp_correct) + "/" + str(v)))
            cnt_correct += tmp_correct
            cnt_try += v
            cnt_word += 1
            if tmp_correct == v:
                perfect += 1
    result.sort(reverse=True)
    flg_100 = False
    flg_50 = False
    print("\n---------- 以下、正解率100% ----------")
    for v, k, cnt in result:
        if not flg_100 and v < 100:
            print("\n---------- 以下、正解率100%未満 ----------")
            flg_100 = True
        elif not flg_50 and v < 50:
            print("\n---------- 以下、正解率50%未満 ----------")
            flg_50 = True
        print(" {}: {:.1f} %　…… ({})".format(k, v, cnt))
    else:
        print("\n---------- results ----------")
        print("　accuracy = {:.1f} %　…… ({}問/{}問)".format(cnt_correct * 100 / cnt_try, cnt_correct, cnt_try))
        print("　perfect = {:.1f} %　…… ({}words/{}words)".format(perfect / cnt_word * 100, perfect, cnt_word))


def main():
    global CURR
    global PATH
    global CHAPTER_START
    global CHAPTER_END
    global REVERSE
    global CNT
    global LOAD
    global RECORD
    global PROB
    global LOSS
    global MIN_PROB
    global PENALTY
    global ALL
    global RANDOM
    global curr
    global path

    df, df_raw, words, dict_try, dict_correct = get_data()

    nums = df.shape[0]
    idxs = [x for x in range(nums)]

    if RANDOM:
        random.shuffle(idxs)

    cnt = nums if ALL else CNT

    i = 0
    j = 0

    while i < cnt:
        idx = idxs[(i + j) % nums]
        if not ALL and PROB:
            if probability(df, dict_try, dict_correct, idx) < random.uniform(0, 1):
                j += 1
                continue
        # print(probability(df, dict_try, dict_correct, idx))
        practice(df, dict_try, dict_correct, i, idx)
        i += 1
    else:
        print("{}問終わり！\n".format(i))
        if RECORD:
            dump(dict_try, dict_correct)


if __name__ == '__main__':
    main()