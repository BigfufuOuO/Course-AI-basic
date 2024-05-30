import numpy as np

import pandas as pd
import json

class JsonEncoder(json.JSONEncoder):
    """Convert numpy classes to JSON serializable objects."""

    def default(self, obj):
        if isinstance(obj, (np.integer, np.int64, np.int32)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64, np.float32)):
            return float(obj)
        elif isinstance(obj, np.bool_):
            return bool(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(JsonEncoder, self).default(obj)
# 创建 DataFrame

def convert_keys_to_strings(d):
    """Convert all keys in the dictionary to strings."""
    new_dict = {}
    for key, value in d.items():
        # Convert the key to string if it is not already a string
        if not isinstance(key, str):
            key = str(key)
        new_dict[key] = value
    return new_dict


a = {'a' : {np.int32(1):'c'}, np.int32(2): 'b'}
a = convert_keys_to_strings(a)
#json.dumps(a, cls=JsonEncoder)
print(json.dumps(a, cls=JsonEncoder))

