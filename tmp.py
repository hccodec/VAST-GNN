from preprocess_multiwave import interpolate


import pickle
with open('all_infection_origin.bin', 'rb') as f: t = pickle.load(f)
res = interpolate(t)
pass