from flask import Flask, request, jsonify
from flask_cors import CORS  # Import CORS
import pandas as pd
from datetime import datetime
from joblib import load

app = Flask(__name__)

# Load the trained model and vectorizer
model = load('knn_sentiment_model.joblib')
vectorizer = load('tfidf_vectorizer.joblib')

# Define a path to the CSV file
csv_file = '../intern/sentiment_records.csv'

@app.route('/write_feed_back', methods=['GET'])
def analyze_comment():
    comment = request.args.get('comment', '')
    text_tfidf = vectorizer.transform([comment])
    sentiment = model.predict(text_tfidf)[0]
    now = datetime.now()
    date_time = now.strftime("%Y-%m-%d %H:%M:%S")

    df = pd.DataFrame([[comment, sentiment, date_time]], columns=['Comment', 'Sentiment', 'DateTime'])
    df.to_csv(csv_file, mode='a', header=not pd.read_csv(csv_file).shape[0], index=False)

    return jsonify({'comment': comment, 'sentiment': sentiment, 'dateTime': date_time})
@app.route('/countall')
def sentiment_counts():
    try:
        df = pd.read_csv(csv_file)
        sentiment_counts = df['Sentiment'].value_counts().to_dict()
        return jsonify(sentiment_counts)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/getfeeds', methods=['GET'])
def comments():
    try:
        df = pd.read_csv(csv_file)
        return jsonify(df.to_dict(orient='records'))
    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True)
