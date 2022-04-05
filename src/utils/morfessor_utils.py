from unidecode import unidecode


def segment_token(model, token):
    token = token.lower()
    token = unidecode(token)
    try:
        t = model.segment(token)
    except:
        t, _ = model.viterbi_segment(token, maxlen=5)
    t = "".join(t[0:-1]) + "__" + t[-1]
    return t
