from flask import Flask, request, jsonify
from flask_cors import CORS  # If you need CORS support, ensure it's installed and uncomment usage.
import pandas as pd
from datetime import datetime
from joblib import load
import os

app = Flask(__name__)
CORS(app)  # Enable CORS if needed

# Load the trained random forest model and vectorizer
model = load('random_forest_sentiment_model.joblib')
vectorizer = load('tfidf_vectorizer_f.joblib')

# Define a path to the CSV file
csv_file = 'sentiment_records.csv'

# Ensure the CSV file exists with proper headers if needed
if not os.path.exists(csv_file):
    pd.DataFrame(columns=['Comment', 'Sentiment', 'DateTime']).to_csv(csv_file, index=False)

@app.route('/analyze', methods=['GET'])
def analyze_comment():
    comment = request.args.get('comment', '')
    text_tfidf = vectorizer.transform([comment])
    sentiment = model.predict(text_tfidf)[0]
    now = datetime.now()
    date_time = now.strftime("%Y-%m-%d %H:%M:%S")

    # Load the existing CSV
    df_existing = pd.read_csv(csv_file)
    # Append the new data
    new_row = pd.DataFrame([[comment, sentiment, date_time]], columns=['Comment', 'Sentiment', 'DateTime'])
    df_updated = pd.concat([df_existing, new_row], ignore_index=True)
    # Save back to CSV
    df_updated.to_csv(csv_file, index=False)

    return jsonify({'comment': comment, 'sentiment': sentiment, 'dateTime': date_time})

@app.route('/sentiment-counts', methods=['GET'])
def sentiment_counts():
    try:
        df = pd.read_csv(csv_file)
        sentiment_counts = df['Sentiment'].value_counts().to_dict()
        return jsonify(sentiment_counts)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/comments', methods=['GET'])
def comments():
    try:
        df = pd.read_csv(csv_file)
        return jsonify(df.to_dict(orient='records'))
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
