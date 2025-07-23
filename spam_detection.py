import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

# Load and prepare the data
data = pd.read_csv('spam.csv', encoding='latin-1')
data = data[['v1', 'v2']]
data.columns = ['label', 'message']
data['label'] = data['label'].map({'ham': 0, 'spam': 1})

# Split the data
X_train, X_test, y_train, y_test = train_test_split(data['message'], data['label'], test_size=0.2, random_state=1)

# Vectorize the messages
vectorizer = CountVectorizer()
X_train_counts = vectorizer.fit_transform(X_train)
X_test_counts = vectorizer.transform(X_test)

# Train the model
model = MultinomialNB()
model.fit(X_train_counts, y_train)

# Make predictions
predictions = model.predict(X_test_counts)
print('Accuracy:', accuracy_score(y_test, predictions))

# Function to predict if a message is spam or not
def predict_spam(message):
    message_count = vectorizer.transform([message])
    prediction = model.predict(message_count)
    return "Spam" if prediction[0] == 1 else "Ham"

# Test the function
test_message = "Congratulations! You've won a free ticket to Bahamas. Call now!"
result = predict_spam(test_message)
print(f"The message is: {result}")
