from flask import Flask
from flask.globals import request

from src.app.controllers.fake_news_catboost_handler import FakeNewsCatboostHandler

app = Flask(__name__)

catboost_handler = FakeNewsCatboostHandler()


@app.route('/fake_news/predict', methods=['POST'])
def predict():
    """
    Handles the prediction and different statistics for a
    news article. the body must contain the following:
    {
        topic: <topic>,
        headline: <headline>,
        text: <text>
    }
    :return: Values associated to the prediction.
    """
    body = request.get_json(force=False)
    topic = body['topic']
    headline = body['headline']
    article = body['article']
    data = catboost_handler.get_features(topic, headline, article)
    return {
        'probability': {
            'catboost': catboost_handler.get_probability(data)
        },
        'sentiment': {
            'headline': {
                'polarity': data['headline_avg_polarity'][0],
                'subjetivity': data['headline_avg_subjetivity'][0]
            },
            'text': {
                'polarity': data['text_avg_polarity'][0],
                'subjetivity': data['text_avg_subjetivity'][0]
            }
        },
        'ngrams': {
            'text_bigrams': catboost_handler.build_ngram(article),
            'text_trigrams': catboost_handler.build_ngram(article, n=3)
        },
        'variables': catboost_handler.get_feature_stats(data)
    }


if __name__ == '__main__':
    # run() method of Flask class runs the application
    # on the local development server.
    app.run()
