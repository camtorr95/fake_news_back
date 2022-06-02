import pickle

import src.app.utils.nltkmodule
import nltk
import re
import numpy as np
import pandas as pd


from googletrans import Translator
from textblob import TextBlob
from nltk import word_tokenize, bigrams, trigrams
from nltk.util import ngrams
from nltk.corpus import stopwords


translator = Translator()


class FakeNewsCatboostHandler:
    def __init__(self):
        handler = open("preprocessor.pickle", "rb")
        self.preprocessor = pickle.load(handler)
        handler = open("catboost.pickle", "rb")
        self.model = pickle.load(handler)

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

    @staticmethod
    def __get_stopword_count(text):
        count = 0
        _split = text.lower().split()
        stop_words_set = set(stopwords.words('spanish'))
        for word in _split:
            if word in stop_words_set:
                count += 1
        return count / len(_split)

    @staticmethod
    def __get_sentences_count(text):
        return len(nltk.sent_tokenize(text))

    @staticmethod
    def __get_palabras_unicas(self, text):
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

    def get_features(self, topic, headline, article):
        data = pd.DataFrame({
            'topic': [topic],
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

        data['eng_headline'] = data['headline'].apply(lambda x: translator.translate(x, src='es').text)
        data['eng_text'] = data['article'].apply(lambda x: translator.translate(x, src='es').text)

        data['headline_sentiment'] = data['eng_headline'].apply(self.__get_average_sentiment)
        data['headline_avg_polarity'] = data['headline_sentiment'].apply(lambda sent: sent[0])
        data['headline_avg_subjetivity'] = data['headline_sentiment'].apply(lambda sent: sent[1])

        data['text_sentiment'] = data['eng_text'].apply(self.__get_average_sentiment)
        data['text_avg_polarity'] = data['text_sentiment'].apply(lambda sent: sent[0])
        data['text_avg_subjetivity'] = data['text_sentiment'].apply(lambda sent: sent[1])

        return data
