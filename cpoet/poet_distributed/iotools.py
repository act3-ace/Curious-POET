import json
import numpy as np
import pandas as pd
import os

# def serialize_random(stream):
#     if hasattr(stream, "get_state"):
#         state = list(stream.get_state())
#     else:
#         state = list(stream.getstate())
        
#     state[1] = [int(i) for i in state[1]]
#     return(state)

# def serialize_random(stream):
#     state = list(stream.get_state())
#     state[1] = [int(i) for i in state[1]]
#     return(state)

# def deserialize_random(stream):
#     random_state = stream
#     random_state[1] = np.array(random_state[1])
#     return(random_state)


## Recursivly encodes objects with a reprJSON function
# https://stackoverflow.com/questions/5160077/encoding-nested-python-object-in-json
#
# Encdoing numpy objects:
# https://stackoverflow.com/questions/26646362/numpy-array-is-not-json-serializable
# 
# Decoding objects
# https://stackoverflow.com/questions/48991911/how-to-write-a-custom-json-decoder-for-a-complex-object

# Usage
# with open(filename, 'w') as jsonfile:
#     json.dump(edge, jsonfile, cls=NumpyEncoder)
# with open(filename, 'r') as jsonfile:
#     edge1 = json.load(jsonfile, cls=NumpyDecoder)


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return {"np.integer": int(obj)}
        if isinstance(obj, np.floating):
            return {"np.floating": float(obj)}
        if isinstance(obj, np.ndarray):
            return {"np.array": obj.tolist()}
        return json.JSONEncoder.default(self, obj)


class NumpyDecoder(json.JSONDecoder):
    def __init__(self, *args, **kwargs):
        json.JSONDecoder.__init__(self, object_hook=self.object_hook, *args, **kwargs)
    def object_hook(self, dct):
        if 'np.integer' in dct:
            return np.int_(dct["np.integer"])
        if 'np.floating' in dct:
            return np.float_(dct["np.floating"])
        if 'np.array' in dct:
            return np.array(dct["np.array"])
        return dct

### Turn dictionary with keyed tuple into csv

def saveKeyedTuple(dct, file):
    l_keys = set()
    r_keys = set()

    for k in dct.keys():
        if not isinstance(k, tuple):
            raise Exception("Dictionary contains keys that are not tuples!") 
        if len(k)!=2:
            raise Exception("Dictionary contains keys that are not length 2")

        l_keys.add(k[0])
        r_keys.add(k[1])

    with open(file, "w") as f:
        f.write("")
        for r in r_keys:
            f.write(f",{r}")

        f.write("\n")

        for l in l_keys:
            f.write(f"{l},")
            for r in r_keys:
                s = dct.get((l,r),"")
                f.write(f"{s},")
            
            f.write("\n")

def loadKeyedTuple(file):
    dct = {}
    if os.path.isfile(file):
        df = pd.read_csv(file,index_col=0)
        tmp = df[df.columns[:-1]]
        tmp.columns = df.columns[1:]

        for i in tmp.columns:
            col = tmp[i]
            for j in tmp.index:
                dct[(j,i)] = col.loc[j]
                
    return(dct)