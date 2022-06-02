from flask import Flask
from flask.globals import request

app = Flask(__name__)


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


if __name__ == '__main__':
    # run() method of Flask class runs the application
    # on the local development server.
    app.run()
