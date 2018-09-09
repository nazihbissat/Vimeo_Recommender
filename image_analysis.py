'''
What you need in your directory:
    similar-similar-staff-picks-challenge-clips.csv

Running Code Will:
    create two subdirectories: WebP_Files nad JPG_Files containing all images
    from urls scraped.
'''
# %%Converting pics to gracyscale

from PIL import Image
from scipy.misc import imread
from scipy.linalg import norm
from scipy import sum, average
from sklearn.cluster import KMeans

import matplotlib.image as mpimg
import numpy as np
import matplotlib.pyplot as plt
import skimage.feature as feature
import csv
import pandas as pd
import os
import sys
import numpy as np
import feature_extraction

# Extracting Dataset
pdf = feature_extraction.PandaFrames("similar-staff-picks-challenge-clips.csv",
                                     "similar-staff-picks-challenge-clip-categories.csv",
                                     "similar-staff-picks-challenge-categories.csv")
data1 = pdf.get_train_file()
data2 = pdf.get_test_file()
all_data = pd.concat([data1, data2])
clips = all_data[['id', 'thumbnail']]
dir_path = os.path.dirname(os.path.realpath('__file__'))
webppath = dir_path + "/WebP_Files/"
jpgpath = dir_path + "/JPG_Files/"
# %% Scraping List of Urls for Images and Saving Them to WebP Folder
try:
    from urlparse import urljoin
except ImportError:
    from urllib.parse import urljoin
import requests
from bs4 import BeautifulSoup


class Scraper:
    def __init__(self):
        self.visited = set()
        self.session = requests.Session()
        self.session.headers = {
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_12_5) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/59.0.3071.109 Safari/537.36"}
        requests.packages.urllib3.disable_warnings()  # turn off SSL warnings

    def visit_url(self, url, level):
        print(url)
        if url in self.visited:
            return

        self.visited.add(url)

        content = self.session.get(url, verify=False).content
        soup = BeautifulSoup(content, "lxml")

        for img in soup.select("img[src]"):
            image_url = img["src"]
            if not image_url.startswith(("data:image", "javascript")):
                self.downlzoad_image(urljoin(url, image_url))

        if level > 0:
            for link in soup.select("a[href]"):
                self.visit_url(urljoin(url, link["href"]), level - 1)

    def download_image(self, image_url):
        local_filename = image_url.split('/')[-1].split("?")[0]
        r = self.session.get(image_url, stream=True, verify=False)
        webppath = dir_path + "/WebP_Files/"
        if not os.path.exists(webppath):
            os.makedirs(webppath)
        with open(webppath + local_filename, 'wb') as f:
            for chunk in r.iter_content(chunk_size=1024):
                f.write(chunk)


if __name__ == '__main__':
    dir_path = os.path.dirname(os.path.realpath('__file__'))
    for idx, image in enumerate(clips['thumbnail']):
        scraper = Scraper()
        scraper.visit_url(image, 1)
        scraper.download_image(image)


# %% Importing images as jpg
def import_jpeg():
    dir_path = os.path.dirname(os.path.realpath('__file__'))
    if not os.path.exists(jpgpath):
        os.makedirs(jpgpath)
    os.chdir(webppath)
    for thumbnail in os.listdir(webppath):
        if '.DS_Store' in thumbnail:
            continue
        """" This line gives OSError: cannot identify image file '100041054_780x439.webp ? Image.open cannot find image with id 100041054"""
        ''' This is probably because of the paths. If you were able to download the images to a WebP Folder, then this should work. I'll try this again'''
        im = Image.open(thumbnail).convert("RGB")
        im.save(jpgpath + thumbnail[:-5] + ".jpg", "jpeg")
    os.chdir(dir_path)


# Grayscale Conversion
def rgb2gray(rgb):
    return np.dot(rgb[..., :3], [0.299, 0.587, 0.114])


# Importing List of Images
def image_list(color):
    image_list = []
    for image in os.listdir(jpgpath):
        if '.DS_Store' in image:  # This may sometimes be found in a folder preventing uploads
            continue
        img = mpimg.imread(image)
        if color == "gray":
            image_list.append(rgb2gray(img))
        elif color == "rgb":
            image_list.append(img)
    return (image_list)


def hog_image(color):
    img_converted = []
    if color == "gray":
        for i, image in enumerate(gray_images):
            img_converted.append(feature.hog(image))
    if color == "rgb":
        for k, image in enumerate(rgb_images):
            img_converted.append(feature.hog(image))
    return (img_converted)


# Extracting List of Images
os.chdir(jpgpath)
import_jpeg()
gray_images = image_list("gray")
rgb_images = image_list("rgb")


# %%
###############################################################################


#                            Interacting With Images                          #


###############################################################################
# %% Comparitive Functions
def normalize(arr):
    rng = arr.max() - arr.min()
    amin = arr.min()
    return (arr - amin) * 255 / rng


def compare_images(img1, img2, distance):
    # normalize to compensate for exposure difference, this may be unnecessary
    # consider disabling it
    #    img1 = normalize(img1)
    #    img2 = normalize(img2)
    # calculate the difference and its norms
    diff = img1 - img2  # elementwise for scipy arrays
    if distance == "m":
        d = sum(abs(diff))  # Manhattan norm
    if distance == "z":
        d = norm(diff.ravel(), 0)  # Zero norm
    return (d)


# clip_id as int, image(gray or rgb), k for top returned, and 'z' or 'm' for distance type
def compare_all(clip_id, images, k, distance, show):
    d = []
    test_image_index = np.where(clips['id'] == clip_id)[0][0]
    test_image = images[test_image_index]
    for idx, item in enumerate(images):
        if idx == test_image_index:
            d.append(0)
            continue
        if np.shape(item) == (439, 780):
            item = item[1:]
        temp = compare_images(test_image, item, distance)
        d.append(temp)
    ddf = pd.DataFrame(d, columns=['Norm Distance'])
    scores_df = clips
    scores_df['Norm Distance'] = d
    top_k = scores_df.sort_values(by=['Norm Distance'])[1:k]

    if show == True:
        indices = top_k.index.get_values()
        for index in indices:
            plt.figure()
            plt.imshow(images[index])
    return (scores_df, top_k)


# Example Usage:
clip_id = 249450406
images = gray_images
k = 10
distance = 'm'
show = True
compare_all(clip_id, images, k, distance, show)

# %%
