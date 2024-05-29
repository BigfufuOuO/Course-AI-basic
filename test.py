import numpy as np

import pandas as pd

# 创建 DataFrame
df = pd.DataFrame({'A': [1, 1], 'B': [2, 2]})

# 统计唯一行的个数
unique_counts = df.duplicated(keep=False).all()

print(df.shape)
print(unique_counts)
