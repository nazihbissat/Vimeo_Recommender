# !/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May  8 16:18:41 2018

@author: christianduffydeabreu
"""

import operator

import pandas as pd

from distance_transformation import *


# Removing those without Columns
def remove_empty_categories(data):
    data = data.reindex(index=data.index[::-1])
    rows_to_delete = []
    for i in range(len(data)):
        if data['main categories'].values[i] == []:
            rows_to_delete.append(i)
    data.drop(data.index[rows_to_delete], inplace=True)
    data.reset_index(inplace=True)
    return (data)


# Importing Data
all_data = ft.load_whole_file("similar-staff-picks-challenge-clips.csv",
                              "similar-staff-picks-challenge-clip-categories.csv",
                              "similar-staff-picks-challenge-categories.csv")
all_data = remove_empty_categories(all_data)
all_data = ot.strdate_to_int(all_data)

df = all_data[['created', 'filesize', 'total_comments', 'total_plays', 'total_likes']]
images = df.values.tolist()


# %%
def feature_similarity(clip_id, p=3, data=all_data, images=images, mode='euclidean', k=11):
    d = []
    test_image_index = np.where(data['id'] == clip_id)[0][0]
    test_image = images[test_image_index]
    test_cat = data.iloc[test_image_index, 13][0]
    # Comparing To Every Other Image:
    for idx, item in enumerate(images):
        if idx == test_image_index:
            d.append(0)
            continue
        temp = ot.calculate_distance(test_image, item, mode, p)
        d.append(temp)
    ddf = pd.DataFrame(d, columns=['Distance'])
    scores_df = data
    scores_df['Distance'] = d
    top_k = scores_df.sort_values(by=['Distance'])[1:k]

    return (top_k, test_cat)


def calculate_purity(top):
    cat_dict = {}
    total = 0;
    for count, cat in enumerate(top['main categories']):
        total += 1
        if cat[0] not in cat_dict.keys():
            cat_dict[cat[0]] = 1
        else:
            cat_dict[cat[0]] += 1
    max_cat = max(cat_dict.items(), key=operator.itemgetter(1))[0]
    accuracy = float(cat_dict[max_cat] / total)
    return (accuracy)


def calculate_accuracy(top, test_cat, k):
    cat_count = 0
    total = k - 1;
    if len(test_cat[0]) > 1:
        for cat in test_cat[0]:
            for im in top['main categories']:
                if im[0] == cat:
                    cat_count += 1
    else:
        for im in top['main categories']:
            if im[0] == test_cat:
                cat_count += 1
    accuracy = float(cat_count / total)
    return (accuracy)


# %%

# rand_int = random.sample(range(1, len(all_data)), len(all_data))

pur_final = []
acc_final = [];
k = 11  # spits top ten similar
modes = ['cosine', 'minkowsky', 'euclidean', 'manhattan', 'chebyshev']
for mode1 in modes:
    acc = []
    pur = []
    for i in range(len(all_data)):
        clip_id = all_data['id'][i]
        [top_clips, test_cat] = feature_similarity(clip_id=clip_id, mode=mode1, k=k)
        acc.append(calculate_accuracy(top_clips, test_cat, k))
        pur.append(calculate_purity(top_clips))
    acc_final.append(np.mean(acc))
    pur_final.append(np.mean(pur))
    print(mode1 + ' done')
## Plotting Purity and Accuracy Results
fig, ax = plt.subplots(1, 1)
plt.plot(range(5), pur_final)
plt.plot(range(5), acc_final)
plt.title('Extra Features Category Similarity')
plt.xlabel('Distance Metric')
plt.ylabel('Accuracy')
ax.set_xticks(range(5))
ax.set_xticklabels(modes, minor=False, rotation=0)
plt.legend(['Purity', 'Test Category Match'])
