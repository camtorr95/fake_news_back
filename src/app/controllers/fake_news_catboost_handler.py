import pickle

import unicodedata

import src.app.utils.nltkmodule
import nltk
import re
import numpy as np
import pandas as pd
import os
from pathlib import Path

from googletrans import Translator
from textblob import TextBlob
from collections import Counter
from nltk import word_tokenize, bigrams, trigrams
from nltk.util import ngrams
from nltk.corpus import stopwords


class FakeNewsCatboostHandler:
    def __init__(self):
        self.topics = [
            "Educacion",
            "Sociedad",
            "Ciencia",
            "Seguridad",
            "Salud",
            "Economia",
            "Deportes",
            "Política",
            "Entretenimiento",
            "Covid-19",
            "Internacional",
            "Deporte",
            "Ambiental",
        ]
        self.stopwords = set(stopwords.words('spanish'))
        self.translator = Translator()
        with open("../../pickles/catboost.pickle", "rb") as handler:
            self.model = pickle.load(handler)
        with open("../../pickles/preprocessor.pickle", "rb") as handler:
            self.preprocessor = pickle.load(handler)

    @staticmethod
    def __get_proporcion_mayusculas(text):
        result = re.findall(r'[A-Z]+', text)
        return len(''.join(result)) / len(text)

    @staticmethod
    def __get_numero_number(text):
        return len(re.findall(r'number+', text.lower()))

    @staticmethod
    def __get_proporcion_number(text):
        result = FakeNewsCatboostHandler.__get_numero_number(text)
        return result / len(text.split())

    @staticmethod
    def __get_non_alphanumeric_count(text):
        result = re.findall(r'[^a-zA-Z0-9 ]', text)
        return len(result)

    def __get_stopword_count(self, text):
        count = 0
        _split = text.lower().split()
        stop_words_set = self.stopwords
        for word in _split:
            if word in stop_words_set:
                count += 1
        return count / len(_split)

    @staticmethod
    def __get_sentences_count(text):
        return len(nltk.sent_tokenize(text))

    @staticmethod
    def __get_palabras_unicas(text):
        return len(set(text.lower().split()))

    @staticmethod
    def __get_average_sentiment(text):
        sentences = text.split('.')
        textblob_sentiment = []
        for s in sentences:
            res = TextBlob(s)
            a = res.sentiment.polarity
            b = res.sentiment.subjectivity
            textblob_sentiment.append([a, b])

        average_polarity = np.mean([item[0] for item in textblob_sentiment])
        average_subjectivity = np.mean([item[1] for item in textblob_sentiment])
        return average_polarity, average_subjectivity

    @staticmethod
    def __strip_accents(s):
        if s is np.nan or s is None:
            return ''
        return ''.join(c for c in unicodedata.normalize('NFD', s)\
                       if unicodedata.category(c) != 'Mn')

    @staticmethod
    def __remove_non_alphanum(s):
        return ''.join(c for c in s if c.isalnum() or c.isspace())

    def __process_string(self, s):
        s = self.__strip_accents(s)
        s = self.__remove_non_alphanum(s)
        s = s.lower()
        return s

    def build_ngram(self, text, n=2, top=10):
        ntext = self.__process_string(text)
        all_str = ' '.join([word for word in ntext.split() if word not in self.stopwords])
        top_ngrams = (pd.Series(ngrams(all_str.split(), n)).value_counts())[:top]
        top_ngrams = list(top_ngrams.to_dict().items())
        processed_ngrams = []
        for ngram in top_ngrams:
            _ngram = ', '.join(ngram[0])
            _ngram = f'({_ngram})'
            processed_ngrams.append((_ngram, ngram[1]))
        return processed_ngrams

    def get_features(self, topic, headline, article):
        data = pd.DataFrame({
            'Topic': [topic],
            'headline': [headline],
            'article': [article]
        })
        data['headline_palabras'] = data['headline'].apply(lambda x: len(x.split()))
        data['headline_palabras_avg_len'] = data['headline'].apply(lambda x: len(x) / len(x.split()))
        data['headline_mayusculas'] = data['headline'].apply(self.__get_proporcion_mayusculas)
        data['headline_numbers'] = data['headline'].apply(self.__get_proporcion_number)
        data['headline_especiales'] = data['headline'].apply(self.__get_non_alphanumeric_count)
        data['headline_stopwords'] = data['headline'].apply(self.__get_stopword_count)
        data['headline_unicas'] = data['headline'].apply(
            lambda x: self.__get_palabras_unicas(x) / len(x.split()))

        data['text_palabras'] = data['article'].apply(lambda x: len(x.split()))
        data['text_palabras_avg_len'] = data['article'].apply(lambda x: len(x) / len(x.split()))
        data['text_mayusculas'] = data['article'].apply(self.__get_proporcion_mayusculas)
        data['text_numbers'] = data['article'].apply(self.__get_proporcion_number)
        data['text_especiales'] = data['article'].apply(self.__get_non_alphanumeric_count)
        data['text_stopwords'] = data['article'].apply(self.__get_stopword_count)
        data['text_unicas'] = data['article'].apply(lambda x: self.__get_palabras_unicas(x) / len(x.split()))
        data['text_oraciones'] = data['article'].apply(self.__get_sentences_count)
        data['text_oraciones_avg_len'] = data['text_oraciones'] / data['text_palabras']

        data['eng_headline'] = data['headline'].apply(lambda x: self.translator.translate(x, src='es').text)
        data['eng_text'] = data['article'].apply(lambda x: self.translator.translate(x, src='es').text)

        data['headline_sentiment'] = data['eng_headline'].apply(self.__get_average_sentiment)
        data['headline_avg_polarity'] = data['headline_sentiment'].apply(lambda sent: sent[0])
        data['headline_avg_subjetivity'] = data['headline_sentiment'].apply(lambda sent: sent[1])

        data['text_sentiment'] = data['eng_text'].apply(self.__get_average_sentiment)
        data['text_avg_polarity'] = data['text_sentiment'].apply(lambda sent: sent[0])
        data['text_avg_subjetivity'] = data['text_sentiment'].apply(lambda sent: sent[1])

        return data

    def get_probability(self, data):
        x_columns = [
            'Topic',
            'headline_palabras',
            'headline_palabras_avg_len',
            'headline_mayusculas',
            'headline_numbers',
            'headline_especiales',
            'headline_stopwords',
            'headline_unicas',
            'headline_avg_subjetivity',
            'text_palabras',
            'text_palabras_avg_len',
            'text_mayusculas',
            'text_numbers',
            'text_especiales',
            'text_stopwords',
            'text_unicas',
            'text_oraciones',
            'text_oraciones_avg_len',
        ]

        processed_data = pd.DataFrame(self.preprocessor.transform(data[x_columns]))
        processed_data.columns = [f'topic_{i}' for i, value in enumerate(self.topics) if i > 0] + \
                                 [col for col in x_columns if col != 'Topic']
        # This returns a tuple of probabilities for each class. We only need one ([0]) that corresponds to
        # the 1 class (Fake).
        return self.model.predict_proba(processed_data)[0].tolist()[1]
        # return f"({', '.join(map(str, self.model.predict_proba(processed_data)[0].tolist()))})"

    def get_feature_values(self, data):
        _features = [
            'headline_palabras',
            'headline_palabras_avg_len',
            'headline_mayusculas',
            'headline_numbers',
            'headline_especiales',
            'headline_stopwords',
            'headline_unicas',
            'headline_avg_subjetivity',
            'text_palabras',
            'text_palabras_avg_len',
            'text_mayusculas',
            'text_numbers',
            'text_especiales',
            'text_stopwords',
            'text_unicas',
            'text_oraciones',
            'text_oraciones_avg_len',
        ]
        features = data[_features]
        features_dict = features.to_dict()
        for key in features_dict:
            value = features_dict[key]
            features_dict[key] = value[0]

        return features_dict
