import os
import re
import string
from collections import *
from itertools import *

import gensim
import matplotlib.pyplot as plt
import numpy as np
from nltk import pos_tag
from nltk.corpus import stopwords
from nltk.corpus import wordnet as wn
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

import feature_extraction

# # Uncomment the following 4 lines when running code for the first time
# nltk.download()
# nltk.download('wordnet')
# nltk.download('stopwords')
# nltk.download('averaged_perceptron_tagger')

# Initialize WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
# Set of stopwords
stopWords = set(stopwords.words('english'))
# Regex to remove punctuation
regex = re.compile('[%s]' % re.escape(string.punctuation))


# Function to tokenize a caption
def tokenize(caption):
    words = []

    caption_array = regex.sub('', caption)
    caption_array = ' '.join(word for word in caption_array.split() if word not in stopWords)
    data = caption_array.split(" ")

    for word in data:
        # Make word lower-case and append lemmatized word
        words.append(lemmatizer.lemmatize(word.lower()))

    return words


# Function to add the 'tokenized caption' column, containing the tokenized caption, to a dataframe
def transform_caption(dataframe):
    dataframe['tokenized caption'] = ""
    for i in range(len(dataframe)):
        caption = dataframe.at[i, 'caption']
        if caption is np.NaN:
            dataframe.at[i, 'tokenized caption'] = []
        else:
            dataframe.at[i, 'tokenized caption'] = tokenize(str(dataframe.at[i, 'caption']))
    return dataframe


def create_gen_docs(dataframe):
    gen_docs = []
    caption_id_dict = {}
    for i in range(len(dataframe)):
        caption = dataframe.at[i, 'caption']
        tokens = [w.lower() for w in word_tokenize(str(caption)) if w not in stopWords]
        gen_docs.append(tokens)
        caption_id_dict[tuple(tokens)] = dataframe.at[i, 'id']
    return [gen_docs, caption_id_dict]


# Function to convert the Penn part of speech tag to the WordNet synset part of speech tag
def synset_pos_tag(tag):
    if tag.startswith('V'):
        return 'v'
    elif tag.startswith('N'):
        return 'n'
    elif tag.startswith('R'):
        return 'r'
    elif tag.startswith('J'):
        return 'a'
    else:
        return None


# Function to retrieve the WordNet synset given a word
def word_to_synset(word, pos):
    tag = synset_pos_tag(pos)
    if tag is None:
        return None
    else:
        sets = wn.synsets(word, tag)
        if sets == []:
            return None
        elif sets is None:
            return None
        else:
            return sets[0]


# Function to calculate the path similarity given two synsets
def path_sim(s1, s2):
    # Need two synsets to have same part of speech to calculate path similarity
    if (s1.pos() == s2.pos()):
        return s1.path_similarity(s2)
    else:
        return 0


# Function to calculate the Wu-Palmer similarity given two synsets
def wup_sim(s1, s2):
    # Need two synsets to have same part of speech to calculate Wu-Palmer similarity
    if (s1.pos() == s2.pos()):
        return s1.wup_similarity(s2)
    else:
        return 0


# Function to calculate the Leacock-Chodorow similarity given two synsets
def lch_sim(s1, s2):
    # Need two synsets to have same part of speech to calculate Leacock-Chodorow similarity
    if (s1.pos() == s2.pos()):
        return s1.lch_similarity(s2)
    else:
        return 0


# Try using all synsets for each word instead of just one (using wn.synsets('word'))
# The argument 'f_sim' should be one of [path_sim, lch_sim, wup_sim]
def sentence_similarity(sentence1, sentence2, f_sim):
    postag1 = pos_tag(sentence1)
    postag2 = pos_tag(sentence2)

    synsets1_init = [word_to_synset(*word) for word in postag1]
    synsets2_init = [word_to_synset(*word) for word in postag2]

    synsets1 = [word for word in synsets1_init if word is not None]
    synsets2 = [word for word in synsets2_init if word is not None]

    similarity_main = 0
    num_words_main = 0
    similarity2 = 0
    num_words2 = 0

    if len(synsets1) == 0 or len(synsets2) == 0:
        return 0

    elif len(synsets1) < len(synsets2):
        for set1 in synsets1:
            sim_array = []
            for set2 in synsets2:
                sim12 = f_sim(set1, set2)
                if sim12 is not None:
                    sim_array.append(sim12)
            top_similarity = np.max(sim_array)
            if top_similarity is not None:
                similarity_main += top_similarity
            num_words_main += 1

        return similarity_main / num_words_main

    elif len(synsets1) > len(synsets2):
        for set2 in synsets2:
            sim_array = []
            for set1 in synsets1:
                sim21 = f_sim(set2, set1)
                if sim21 is not None:
                    sim_array.append(sim21)
            top_similarity = np.max(sim_array)
            if top_similarity is not None:
                similarity_main += top_similarity
            num_words_main += 1

        return similarity_main / num_words_main

    else:
        for set1 in synsets1:
            sim_array = []
            for set2 in synsets2:
                sim12 = f_sim(set1, set2)
                if sim12 is not None:
                    sim_array.append(sim12)
            top_similarity = np.max(sim_array)
            if top_similarity is not None:
                similarity_main += top_similarity
            num_words_main += 1
        for set2 in synsets2:
            sim_array = []
            for set1 in synsets1:
                sim21 = f_sim(set2, set1)
                if sim21 is not None:
                    sim_array.append(sim21)
            top_similarity2 = np.max(sim_array)
            if top_similarity2 is not None:
                similarity2 += top_similarity2
            num_words2 += 1

        return (similarity_main / num_words_main + similarity2 / num_words2) / 2


# The argument 'f_sim' should be one of [path_sim, lch_sim, wup_sim]
def caption_similarity(df, clip_id, f_sim):
    # df = transform_caption(df)
    df['caption similarity'] = 0.0
    clip_index = df[df['id'] == clip_id].index[0]
    target_clip = df[df['id'] == clip_id]
    df = df.drop([clip_index]).reset_index().drop(['index'], axis=1)
    df['caption similarity'] = [sentence_similarity(target_clip.at[clip_index, 'tokenized caption'],
                                                    df.at[i, 'tokenized caption'], f_sim) for i in range(len(df))]
    df = df.sort_values(by=['caption similarity'], ascending=False).reset_index().drop(['index'], axis=1)
    return df


# Function to find accuracy with categories as labels and threshold for 1 vs 0. Iterates over every id in the df.
# The argument 'f_sim' should be one of [path_sim, lch_sim, wup_sim]
def calculate_accuracy(df, thresh, top_k, f_sim):
    accuracies = 0
    for id in df['id']:
        sim_df = caption_similarity(df, id, f_sim)
        correct = 0
        for i in range(top_k):
            if sim_df.at[i, 'category'] == df.at[id, 'category']:
                correct += 1
        if correct >= thresh:
            accuracies += 1
    return accuracies / len(df)


# Removing those without main categories
def remove_empty_categories(data):
    data = data.reindex(index=data.index[::-1])
    rows_to_delete = []
    for i in range(len(data)):
        if data['main categories'].values[i] == []:
            rows_to_delete.append(i)
    data.drop(data.index[rows_to_delete], inplace=True)
    data.reset_index(inplace=True)
    return (data)


# This class creates a tf-IDF model based on textual features and calculates purity and accuracy scores.
# It also provides accuracy scores for different thresholds. (threshold = number of documents out of top 10 that we consider similar)
class TextualAccuracy(object):
    def __init__(self, filepath, clip_category_file_path, category_map_file_path, threshold):
        dataset = feature_extraction.load_whole_file(filepath,
                                                     clip_category_file_path,
                                                     category_map_file_path)

        # updated_dataset = feature_extraction.get_updated_captions_for_whole_file(dataset)
        self.train_text = remove_empty_categories(dataset)
        docs = create_gen_docs(self.train_text)
        self.training_documents = docs[0]
        self.tokens_id_dict = docs[1]

        # creating gensim TF-IDF, Similarity Models
        self.dictionary = gensim.corpora.Dictionary(self.training_documents)
        self.corpus = [self.dictionary.doc2bow(gen_doc) for gen_doc in self.training_documents]
        self.tf_idf = gensim.models.TfidfModel(self.corpus)
        self.similarity_model = gensim.similarities.Similarity(os.path.dirname(os.path.abspath(__file__)),
                                                               self.tf_idf[self.corpus],
                                                               num_features=len(self.dictionary))
        self.threshold = threshold

    def calculate_tf_idf_accuracy(self):

        correct_classifications = 0

        purity_scores = []
        accuracy_scores = []
        for id in self.train_text['id']:  # For each video in our dataset

            index = self.train_text.loc[self.train_text['id'] == id].index[0]  # get its index
            caption = self.train_text.loc[index, 'caption']  # find the associated caption
            original_category = self.train_text.loc[index, 'main categories']  # Get categories of that video
            # print('Category of this video was {}'.format(original_category))

            query_doc = [w.lower() for w in word_tokenize(str(caption)) if
                         w not in stopWords]  # tokenize caption of that video

            query_doc_bow = self.dictionary.doc2bow(query_doc)

            query_doc_tf_idf = self.tf_idf[query_doc_bow]  # create tf_idf_model of that query (video)

            similarity_scores = (
                self.similarity_model[query_doc_tf_idf])  # Find similarity scores with all other videos
            top_10 = (similarity_scores.argsort()[-10:][::-1])  # Get top 10 similarity scores
            similar_category_list = []

            # This loop tracks indexes of these top scores and finds the main categories of the videos associated with top-10 similarity scores
            for s in top_10:
                i = (self.tokens_id_dict[tuple(self.training_documents[int(s)])])
                index = self.train_text.loc[self.train_text['id'] == i].index[0]
                similar_category_list.append((self.train_text.loc[index, 'main categories']))

            categories = list(chain(*similar_category_list))  # now we have all categories of these videos in a list

            # Calculate how many of these categories are the same with original category
            similar_category_counter = 0
            counter = Counter(categories)

            # calculate purity score
            max_category = max(counter, key=counter.get)
            purity_scores.append(float(counter[max_category] / len(categories)))

            for cat in original_category:
                similar_category_counter = similar_category_counter + counter[cat]

            # calculate accuracy score
            accuracy_scores.append(float(similar_category_counter / len(categories)))
            # If the number is above a certain threshold (i.e. 3 out of 10), count it as correct classification
            if (similar_category_counter > self.threshold):
                correct_classifications = correct_classifications + 1

        accuracy = float(correct_classifications / len(self.train_text))
        print("The purity of TF-IDF Model was: {}".format(np.asarray(purity_scores).mean()))
        print("The accuracy of TF-IDF Model was: {}".format(np.asarray(accuracy_scores).mean()))

        # print("The accuracy of TF-IDF Model for a threshold of {} documents out of 10 was {}".format(self.threshold,
        #                                                                                            accuracy))
        return accuracy

#To Run the TF-IDF Model Purity and Accuracy Scores, run the following string:

"""
TextModel = TextualAccuracy('similar-staff-picks-challenge-clips.csv',
                           'similar-staff-picks-challenge-clip-categories.csv',
                            'similar-staff-picks-challenge-categories.csv', 3)    
acc = TextModel.calculate_tf_idf_accuracy()

"""



#Word-net Model Example usage

"""

#Word-net Model Example usage 
for id in train_text['id']:
    similar_captions = caption_similarity(train, id, path_sim)

    print('Currently investigating Video ID: {}'.format(id))
    index = train_text.loc[train_text['id'] == id].index[0]
    print('Category of this video was {}'.format(train_text.loc[index, 'main categories']))
    print('Similar videos:')
    print((similar_captions[['main categories']].head(10)))
    # print('The main categories of most similar videos were {}'.format(similar_captions['main categories']))

"""
