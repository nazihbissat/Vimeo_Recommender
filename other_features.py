import feature_extraction
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import datetime as dt
from sklearn.decomposition import PCA


def calculate_distance(a, b, mode, p=3):
    """ Calculates distances between two vectors"""
    if mode not in ['cosine', 'minkowsky', 'euclidean', 'manhattan', 'chebyshev']:
        raise ValueError('Distance Mode not available')

    if mode == 'cosine':
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        num = sum([a1 * b1 for a1, b1 in zip(a, b)])
        den = norm_a * norm_b
        cos_sim = num / den
        return cos_sim
    elif mode == 'minkowsky':
        if p <= 0:
            raise ValueError('p must be a positive, non-zero, real-valued number')
        dist = sum([(abs(a1 - b1)) ** p for a1, b1 in zip(a, b)])
        mink_dist = dist ** (1 / p)
        return mink_dist
    elif mode == 'euclidean':
        dist = sum([(a1 - b1) ** 2 for a1, b1 in zip(a, b)])
        euc_dist = dist ** (1 / 2)
        return euc_dist
    elif mode == 'manhattan':
        man_dist = sum([abs(a1 - b1) for a1, b1 in zip(a, b)])
        return man_dist
    elif mode == 'chebyshev':
        cheb_dist = max([abs(a1 - b1) for a1, b1 in zip(a, b)])
        return cheb_dist


def rank_misc_distance(clip_id, clip_df, mode, pca=False, p=3):
    """ Returns DataFrame with most similar clips"""
    """ mode in ["cosine", "minkowsky", "euclidean", "manhattan", "chebyshev"] """
    """ if mode == "chebyshev", provide p """

    # select other features
    clip_misc_df = clip_df[["created", "filesize", "duration", "total_comments",
                            "total_plays", "total_likes"]]
    # convert created to integer
    clip_misc_df = strdate_to_int(clip_misc_df)

    # Standardize Columns
    sc = StandardScaler()
    clip_misc_df = sc.fit_transform(clip_misc_df)

    # PCA
    if pca:
        pca_dcp = PCA(n_components=3)
        clip_misc_df = pca_dcp.fit_transform(clip_misc_df)

    # set target clip
    target_clip = clip_misc_df[clip_df['id'] == clip_id][0]

    # calculate distances between target clip and other clips based on misc features
    clip_df['dist_to_target'] = [calculate_distance(target_clip,
                                                    clip_misc_df[i, :], mode, p) for i in range(len(clip_misc_df))]

    clip_df = clip_df[clip_df['id'] != clip_id]
    clip_df.sort_values("dist_to_target", inplace=True)
    max_d = max(clip_df["dist_to_target"])
    min_d = min(clip_df["dist_to_target"])
    delta = max_d - min_d
    clip_df["dist_to_target"] = clip_df["dist_to_target"].apply(lambda x: (x - min_d) / delta)
    return clip_df


def strdate_to_int(df):
    """ Convert "created" Pandas Dataframe column from datetime to integer"""

    df["created"] = df["created"].apply(lambda x: x.split("T")[0])
    df["created"] = df["created"].apply(lambda x: dt.datetime.strptime(x, '%Y-%m-%d'))
    oldest_post = min(df["created"])
    df["created"] = df["created"].apply(lambda x: (x - oldest_post).days)
    return df
