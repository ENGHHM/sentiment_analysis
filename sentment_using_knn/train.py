import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report
from joblib import dump

def load_data(file_path, encoding='latin1'):
    return pd.read_csv(file_path, encoding=encoding)

# Load the training and testing data with 'latin1' encoding
train_data = load_data('../intern/train.csv')
test_data = load_data('../intern/test.csv')

# Handle missing values directly in the DataFrame
train_data.fillna({'text': '', 'sentiment': 'neutral'}, inplace=True)
test_data.fillna({'text': '', 'sentiment': 'neutral'}, inplace=True)

# Convert sentiment to string if not already
train_data['sentiment'] = train_data['sentiment'].astype(str)
test_data['sentiment'] = test_data['sentiment'].astype(str)

# Extract features and target labels
X_train = train_data['text']
y_train = train_data['sentiment']
X_test = test_data['text']
y_test = test_data['sentiment']

# Text vectorization with TF-IDF
vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# Create k-NN classifier with n_neighbors set to 5
knn = KNeighborsClassifier(n_neighbors=5)

# Train the model
knn.fit(X_train_tfidf, y_train)

# Predicting the test set results
y_pred = knn.predict(X_test_tfidf)

# Accuracy and classification report
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

print("Accuracy:", accuracy)
print("Classification Report:\n", report)

# Save the trained model and the vectorizer
dump(knn, 'knn_sentiment_model.joblib')
dump(vectorizer, 'tfidf_vectorizer.joblib')
