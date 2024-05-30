import json
import numpy as np

class NumpyJSONEncoder:
    def __init__(self, data):
        self.data = data

    @staticmethod
    def convert_numpy(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.bool_, bool)):
            return bool(obj)
        elif isinstance(obj, (np.str_, str)):
            return str(obj)
        else:
            return obj

    @classmethod
    def convert_dict(cls, d):
        if isinstance(d, dict):
            return {cls.convert_numpy(k): cls.convert_dict(v) for k, v in d.items()}
        elif isinstance(d, list):
            return [cls.convert_dict(i) for i in d]
        else:
            return cls.convert_numpy(d)

    def to_json(self):
        converted_data = self.convert_dict(self.data)
        return converted_data


