import pickle

import unicodedata
import numpy as np

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Embedding, LSTM, Dropout

import src.app.utils.nltkmodule
import nltk
from nltk.corpus import stopwords

import gensim


class FakeNewsRnnHandler:
    def __init__(self):
        self.stopwords = set(stopwords.words('spanish'))
        with open("../../pickles/tokenizer.pickle", "rb") as handler:
            self.tokenizer = pickle.load(handler)
        self.model = load_model("../../pickles/myModel.h5")

    def predict(self, article):
        new_text = self.process_string(article)
        new_text = [new_text]
        new_text = self.tokenizer.texts_to_sequences(new_text)
        new_text = pad_sequences(new_text, maxlen=900)
        return self.model.predict(new_text)[0].tolist()[0]

    @staticmethod
    def strip_accents(s):
        if s is np.nan or s is None:
            return ''
        return ''.join(c for c in unicodedata.normalize('NFD', s)\
                       if unicodedata.category(c) != 'Mn')

    @staticmethod
    def remove_non_alphanum(s):
        return ''.join(c for c in s if c.isalnum() or c.isspace())

    @staticmethod
    def process_string(s):
        s = FakeNewsRnnHandler.strip_accents(s)
        s = FakeNewsRnnHandler.remove_non_alphanum(s)
        s = s.lower()
        return s
