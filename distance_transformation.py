from sklearn.feature_extraction.text import TfidfVectorizer
import feature_extraction as ft
import other_features as ot
import numpy as np
import math
from sklearn.decomposition import TruncatedSVD
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from text_analysis import transform_caption
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
import pickle


class FeatureSpace(object):
    def __init__(self, data_file, pickled_images):
        print("Extracting Data")
        self.data = extract_data_with_thumbid(data_file)
        print("Extracting Vectorized Images")
        self.images = vectorize_images(pickled_images, self.data)

    def generate_text_vector(self):
        """ Transform Text to (n_clips x n_components) Matrix"""
        print("Generating Text Vector")
        print("        Generating TF IDF Caption Vector")
        tfidf_captions = tf_idf_captions(self.data)
        print("        Performing SVD on Sparse TF IDF Matrix")
        self.svd_dp = TruncatedSVD(n_components=100)
        self.text_vect = self.svd_dp.fit_transform(tfidf_captions)
        return self.text_vect

    def generate_misc_vector(self, features):
        """ Create Matrix of Other Features"""
        print("Generating Other Features Vector")
        misc_vect = [np.array([row[feature] for feature in features]) for idx, row in self.data.iterrows()]
        self.misc_vect = np.array(misc_vect)
        return self.misc_vect

    def generate_img_vector(self, n_pca=0.6):
        """ Transform Images to (n_clips x n_tsne) Matrix"""
        print("Generating Image Vector")
        images = self.images
        # Perform PCA prior to t-SNE
        print("       Performing PCA on CNN output")
        sc = StandardScaler()
        images = sc.fit_transform(self.images)
        if n_pca < 1:
            pca_dp = PCA(n_components=n_pca, svd_solver='full')
        else:
            pca_dp = PCA(n_components=n_pca)
        self.pca = pca_dp
        images = pca_dp.fit_transform(images)
        self.img_vect = images
        return self.img_vect

    def generate_final_vector(self, misc_features, tsne_perplexity, tsne_iter, n_pca_img=0.6):
        """ Generates Text, Image, Misc Vectors and Perform Final t-SNE"""
        print("Generating Transformed Feature Space: ETA <1 min")
        img_vec = self.generate_img_vector(n_pca_img)
        txt_vec = self.generate_text_vector()
        misc_vec = self.generate_misc_vector(features=misc_features)
        self.test = zip(txt_vec, misc_vec, img_vec)
        final_vect = [np.concatenate([txt, misc, img]) for txt, misc, img in zip(txt_vec, misc_vec, img_vec)]
        # t-SNE
        print("Performing Final t-SNE")
        tsne_clt = TSNE(n_components=2, verbose=1, perplexity=tsne_perplexity, n_iter=tsne_iter)
        self.final_vect = tsne_clt.fit_transform(final_vect)
        self.tsne = tsne_clt
        return self.final_vect

    def plot_fspace_2d(self, fspace):
        """ Plot any Intermediate or Final Feature Space"""
        """ fspace in ['text', 'misc', 'img', 'final']"""
        if fspace == 'text': plot_vect = self.text_vect
        if fspace == 'misc': plot_vect = self.misc_vect
        if fspace == 'img': plot_vect = self.img_vect
        if fspace == 'final': plot_vect = self.final_vect
        plt.clf()
        plt.scatter(plot_vect[:, 0], plot_vect[:, 1])
        plt.title('Final Feature Space (t-SNE Two Principal Components)')
        plt.xlabel('First t-SNE Component')
        plt.ylabel('Second t-SNE Component')
        plt.show()

    def get_similar(self, clip_id, mode='euclidean', p=3, with_cat=False):
        clip_index = self.data.index[self.data['id'] == clip_id]
        target_clip = self.final_vect[clip_index][0]
        distances = [(idx, ot.calculate_distance(target_clip, clip, mode, p), self.data['main categories'][idx]) for
                     idx, clip in enumerate(self.final_vect)]
        distances = sorted(distances, key=lambda x: x[1])
        self.distances = distances
        if with_cat:
            top_10_clips = []
            for item in distances:
                if len(top_10_clips) == 10:
                    break
                else:
                    if item[2] == []:
                        continue
                    else:
                        top_10_clips += [data['id'][item[0]]]
        else:
            top_10 = distances[:10]
            top_10_indexes = [item[0] for item in top_10]
            top_10_clips = [data['id'][index] for index in top_10_indexes]
        return top_10_clips


def extract_data_with_thumbid(path):
    """ Extract data as formatted dataframe"""
    data = ft.load_whole_file(path, "similar-staff-picks-challenge-clip-categories.csv",
                              "similar-staff-picks-challenge-categories.csv")
    data.drop(['Unnamed: 0'], axis=1, inplace=True)
    data = ot.strdate_to_int(data)
    data["thumb_id"] = data["thumbnail"].apply(lambda x: x.split("_")[0].split("/")[-1])
    return data


def tf_idf_captions(df):
    """ Return Sparse Matrix with TF IDF for each clip caption"""
    df = transform_caption(df)
    captions = df["caption"].tolist()
    captions = [x if isinstance(x, str) else "" for x in captions]
    tfidf_vect = TfidfVectorizer(stop_words='english')
    tfidf_captions = tfidf_vect.fit_transform(captions)
    return tfidf_captions


def vectorize_images(pickle_file, df):
    """ Return Ordered (consistent with df) Matrix of Vectorized Images"""
    file = open(pickle_file, 'rb')
    images = pickle.load(file)
    vect_img_mtx = []
    for i in range(len(df)):
        thumb = str(df["thumb_id"][i])
        for item in images:
            if item[0] == thumb:
                vect_img_mtx += [list(map(float, item[1]))]
    return vect_img_mtx


fs = FeatureSpace("similar-staff-picks-challenge-clips.csv", 'vect_images')
fs.data["main categories"] = fs.data["main categories"].apply(lambda x: x[0] if len(x) != 0 else [])

# """""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

# Testing Accuracy and Purity
test = False

# """""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

if test:
    purity = 0
    accuracy = 0
    num_clips = 0
    for item in fs.data['id']:
        true_category = fs.data[fs.data['id'] == item]['main categories'].values[0]
        if true_category == []:
            continue
        else:
            top_10 = fs.get_similar(item, with_cat=True)
            categories = []
            for similar in top_10:
                categories += [fs.data[fs.data['id'] == similar]['main categories'].values[0]]
            unique_cat = list(np.unique(np.array(categories)))
            counts = []
            for cat in unique_cat:
                counts += [categories.count(cat)]
            purity += max(counts)
            accuracy += categories.count(true_category)
            num_clips += 1
    purity = purity / (10 * num_clips)
    accuracy = accuracy / (10 * num_clips)
