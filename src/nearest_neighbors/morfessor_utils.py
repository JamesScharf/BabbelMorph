from unidecode import unidecode


def segment_token(model, token):
    if len(token) == 1:
        return token
    token = token.lower()
    token = unidecode(token)
    try:
        t = model.segment(token)
    except:
        t, _ = model.viterbi_segment(token, maxlen=5)

    # label morphemes from end
    # new_t = []
    # for i, morph in enumerate(t):
    #    d_from_back = abs(i - len(t) + 1)
    #    new_morph = morph + str(d_from_back)
    #    new_t.append(new_morph)
    # t = " ".join(new_t)
    if len(t) >= 2:
        t = t[-2] + "__" + t[-1]
    else:
        t = "__" + t[-1]
    return t
