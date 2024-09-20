def tmpsave(*v):
    if len(v) == 1: v = v[0]
    import pickle
    with open('/tmp/tmp_hbj.bin', 'wb') as f:
        pickle.dump(v, f)
    print('√')

def tmpload():
    import pickle
    with open('/tmp/tmp_hbj.bin', 'rb') as f:
        res = pickle.load(f)
    print('√')
    return res
    