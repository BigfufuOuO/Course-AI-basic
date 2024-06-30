import json
import numpy as np
import matplotlib.pyplot as plt

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
    
class PicPoints:
    def __init__(self):
        self.cluster = [0]*6 + [1]*8 + [2]*9 + [3]*7 + [4]*6 + [5]*4 + [6]*6
    
    def plot(self, points, words):
        # points: [n_points, 2]
        # words: [n_points, ]
        plt.figure(figsize=(12, 12))
        plt.scatter(points[:, 0], points[:, 1], c=self.cluster)
        for i, word in enumerate(words):
            plt.annotate(word, (points[i, 0], points[i, 1]))
        plt.savefig("points.png")
        pass
    
    def to_mtrx(self, K):
        # K: [n_samples, n_samples]
        with open("kernel.txt", "w") as f:
            for cls in np.unique(self.cluster):
                cls_idx = np.where(self.cluster == cls)[0]
                cls_K = K[cls_idx][:, cls_idx]
                f.write(f"cluster {cls}:\n")
                f.write(f"{cls_K}\n")
        
        
    


