from flask import Flask, render_template, request
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

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

# Flask app setup
app = Flask(__name__)

# Home route (displays the form)
@app.route('/')
def home():
    return render_template('index.html')

# Predict route (handles form submission and displays result)
@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        message = request.form['message']  # Get message from form
        message_count = vectorizer.transform([message])  # Transform message to vectorized format
        prediction = model.predict(message_count)  # Predict if it's spam or ham
        result = "Spam" if prediction[0] == 1 else "Ham"  # Get result
        return render_template('result.html', prediction=result)  # Render result template with prediction

if __name__ == '__main__':
    app.run(debug=True)
