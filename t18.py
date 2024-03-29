import numpy as np
import warnings
from collections import Counter
import pandas as pd
import random

accuracies = []

for i in range(25):
    def k_nearest_neighbors(data, predict, k=5):
        if len(data) >= k:
            warnings.warn("Dumbo!")
        
        distances = []
        for group in data:
            for feature in data[group]:
                euclidean_distance = np.linalg.norm(np.array(feature) - np.array(predict))
                distances.append([euclidean_distance, group])
        
        votes = [i[1] for i in sorted(distances)[:k]]
        votes_result = Counter(votes).most_common(1)[0][0]
        confidence = Counter(votes).most_common(1)[0][1] / float(k)
        return votes_result

    df = pd.read_csv("breast-cancer-wisconsin.data")
    df.replace('?', -99999, inplace=True)
    df.drop(['id'], 1, inplace=True)
    full_data = df.astype(float).values.tolist()
    random.shuffle(full_data)

    test_size = 0.2
    train_set = {2:[], 4:[]}
    test_set = {2:[], 4:[]}
    train_data = full_data[:-int(test_size*len(full_data))]
    test_data = full_data[-int(test_size*len(full_data)):]

    for i in train_data:
        train_set[i[-1]].append(i[:-1])

    for i in test_data:
        test_set[i[-1]].append(i[:-1])

    correct = 0
    total = 0

    for group in test_set:
        for data in test_set[group]:
            vote, confidence = k_nearest_neighbors(train_set, data, k=5)
            if vote == group:
                correct += 1
            total += 1
    accuracy = correct/total
    accuracies.append(accuracy)

print(sum(accuracies)/len(accuracies))