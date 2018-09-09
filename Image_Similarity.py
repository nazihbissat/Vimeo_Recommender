# !/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May  6 20:16:06 2018

@author: christianduffydeabreu
"""
from sklearn.feature_extraction.text import TfidfVectorizer
import feature_extraction as ft
import other_features as ot
import numpy as np
import math
import matplotlib.pyplot as plt
import pickle
import os
import matplotlib.image as mpimg
import pandas as pd
from distance_transformation import *
import operator
from image_analysis import rgb2gray

# Importing Data
all_data = ft.load_whole_file("similar-staff-picks-challenge-clips.csv",
                              "similar-staff-picks-challenge-clip-categories.csv",
                              "similar-staff-picks-challenge-categories.csv")
data = extract_data_with_thumbid("similar-staff-picks-challenge-clips.csv")


# %%
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


all_data = remove_empty_categories(all_data)
data = remove_empty_categories(data)

images = vectorize_images('vect_images', data)
dir_path = os.path.dirname(os.path.realpath('__file__'))
webppath = dir_path + "/WebP_Files/"
jpgpath = dir_path + "/samples/"


def image_list(color):
    image_list = []
    ''' Saving Images to List '''
    for image in os.listdir(jpgpath):
        if '.DS_Store' in image:  # This may sometimes be found in a folder preventing uploads
            continue
        img = mpimg.imread(image)
        if color == "gray":
            image_list.append(rgb2gray(img))
        elif color == "rgb":
            image_list.append(img)
    return (image_list)


os.chdir(jpgpath)
rgb_images = image_list('rgb')
os.chdir(dir_path)


def image_similarity(clip_id, p=3, all_images=rgb_images, data=data, images=images, mode='euclidean', k=10, show=False):
    d = []
    test_image_index = np.where(data['id'] == clip_id)[0][0]
    test_image = images[test_image_index]
    test_cat = data.iloc[test_image_index, 12][0]
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

    # Show for plotting
    if show == True:
        indices = top_k.index.get_values()
        for index in indices:
            plt.figure()
            plt.imshow(rgb_images[index])
        plt.figure()
        plt.imshow(rgb_images[np.where(data['id'] == clip_id)[0][0]])
        plt.title('Original Image')
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
        for cat in test_cat:
            for im in top['main categories']:
                if im[0] == cat:
                    cat_count += 1
    else:
        for im in top['main categories']:
            if im[0] == test_cat:
                cat_count += 1
    accuracy = float(cat_count / total)
    return (accuracy)


# %% There is a more efficient way to go about this but for now it taks around 15 minutes to run each type of accuracy, so essentially an hour to run this code
import random

rand_int = random.sample(range(1, len(data)), 200)

pur_final = []
acc_final = [];
k = 11  # spits top ten similar
modes = ['cosine', 'minkowsky', 'euclidean', 'manhattan', 'chebyshev']
for mode1 in modes:
    acc = []
    pur = []
    for i in rand_int:
        print(i)
        clip_id = data['id'][i]
        [top_clips, test_cat] = image_similarity(clip_id=clip_id, mode=mode1, k=k, show=False)
        acc.append(calculate_accuracy(top_clips, test_cat, k))
        pur.append(calculate_purity(top_clips))
    acc_final.append(np.mean(acc))
    pur_final.append(np.mean(pur))
    print(mode1 + ' done')

# %%
fig, ax = plt.subplots(1, 1)
plt.plot(range(5), pur_final)
plt.plot(range(5), acc_final)
plt.title('Image Category Similarity')
plt.xlabel('Distance Metric')
plt.ylabel('Accuracy')
ax.set_xticks(range(5))
ax.set_xticklabels(modes, minor=False, rotation=0)
plt.legend(['Purity', 'Test Category Match'])
