from argparse import ArgumentError
from typing import List
from unidecode import unidecode


def segment_token(model, token, method, suffix_n=3):

    # method determines which morphemes we wish to keep
    # we wish to keep.
    # Available methods include:
    #   pure_morfessor: hablaba --> ha bl a ba
    #   countback_morfessor: hablaba --> ha3 bl2 a1 ba0
    #   suffix_morfessor: hablaba --> a_ba
    #   suffix: (suffix_n=3) hablaba --> aba

    if len(token) == 1:
        return token
    token = token.lower()
    token = unidecode(token)
    try:
        t = model.segment(token)
    except:
        t, _ = model.viterbi_segment(token, maxlen=5)

    if method == "pure_morfessor":
        res = pure_morfessor(t)
        return res
    elif method == "countback_morfessor":
        res = countback_morfessor(t)
        return res
    elif method == "suffix_morfessor":
        res = suffix_morfessor(t)
        return res

    elif method == "suffix1":
        res = suffix(t, suffix_n=1)
        return res
    elif method == "suffix2":
        res = suffix(t, suffix_n=2)
        return res
    elif method == "suffix3":
        res = suffix(t, suffix_n=3)
        return res
    elif method == "suffix4":
        res = suffix(t, suffix_n=4)
        return res

    elif method == "suffix5":
        res = suffix(t, suffix_n=5)
        return res
    elif method == "whole":
        res = "".join(t)
        return res
    else:
        print("method: ", method)
        raise Exception(f"method type not acceptable: {method}")


def pure_morfessor(segmented_token: List[str]):
    # label morphemes from end
    t = segmented_token
    new_t = []
    for i, morph in enumerate(t):
        new_morph = morph
        new_t.append(new_morph)
    t = " ".join(new_t)
    return t


def countback_morfessor(segmented_token: List[str]):
    # label morphemes from end
    t = segmented_token
    new_t = []
    for i, morph in enumerate(t):
        d_from_back = abs(i - len(t) + 1)
        new_morph = morph + str(d_from_back)
        new_t.append(new_morph)
    t = " ".join(new_t)
    return t


def suffix_morfessor(segmented_token: List[str]):
    t = segmented_token

    if len(t) >= 2:
        t = t[-2] + "__" + t[-1]
    else:
        t = "__" + t[-1]

    return t


def suffix(segmented_token: List[str], suffix_n=3):
    raw_token = "".join(segmented_token)
    if len(raw_token) <= suffix_n:
        return raw_token
    return raw_token[-suffix_n:]
