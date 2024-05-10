# from preprocess_multiwave import interpolate


# import pickle
# with open('all_infection_origin.bin', 'rb') as f: t = pickle.load(f)
# res = interpolate(t)
# pass

import pickle

def getrange(t):
    if not hasattr(t, '__len__'): raise TypeError
    return range(len(t))

# with open('tmp.bin', 'rb') as f: t, _t =pickle.load(f)
def compare(t, _t):
    res = True
    for i in getrange(t):
        for j in getrange(t[i]):
            if isinstance(t[i][j], int): res = res and (t[i][j] == _t[i][j])
            else:
                for k in getrange(t[i][j]):
                    res = res and (t[i][j][k] == _t[i][j][k]).any()

    return res