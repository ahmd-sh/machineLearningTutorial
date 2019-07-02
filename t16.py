import numpy as np
import matplotlib.pyplot as plt
import warnings
from collections import Counter
from matplotlib import style
style.use('fivethirtyeight')

dataset = {
    'k': [[1,2],[2,3],[3,1]],
    'r': [[6,5],[7,7],[8,6]]
}
new_features = [3,5]

# [[plt.scatter(ii[0], ii[1], color=i, s=100) for ii in dataset[i]] for i in dataset]
# plt.scatter(new_features[0], new_features[1])
# plt.show()

def k_nearest_neighbors(data, predict, k=3):
    if len(dataset) >= k:
        warnings.warn("Dumbo!")
    
    #knnalgos
    return vote_result