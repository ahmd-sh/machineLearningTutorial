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


def k_nearest_neighbors(data, predict, k=3):
    if len(dataset) >= k:
        warnings.warn("Dumbo!")
    
    distances = []
    for group in data:
        for feature in data[group]:
            euclidean_distance = np.linalg.norm(np.array(feature) - np.array(predict))
            distances.append([euclidean_distance, group])
    
    votes = [i[1] for i in sorted(distances)[:k]]
    votes_result = Counter(votes).most_common(1)[0][0]
    return votes_result

result = k_nearest_neighbors(dataset, new_features, k=3)
print(result)

[[plt.scatter(ii[0], ii[1], color=i, s=100) for ii in dataset[i]] for i in dataset]
plt.scatter(new_features[0], new_features[1], color=result)
plt.show()