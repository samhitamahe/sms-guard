from flask import Flask, render_template, request
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import make_scorer, precision_score, recall_score, f1_score

# Load and prepare the data
data = pd.read_csv('spam.csv', encoding='latin-1')
data = data[['v1', 'v2']]
data.columns = ['label', 'message']
data['label'] = data['label'].map({'ham': 0, 'spam': 1})

# Vectorize the messages
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(data['message'])
y = data['label']

# Define the model
model = MultinomialNB()

# Set up cross-validation scoring metrics
scoring = {
    'accuracy': 'accuracy',
    'precision': make_scorer(precision_score),
    'recall': make_scorer(recall_score),
    'f1': make_scorer(f1_score)
}

# Perform cross-validation
cv_results = cross_val_score(model, X, y, cv=5, scoring='accuracy')
print(f"Cross-Validated Accuracy: {cv_results.mean() * 100:.2f}% (± {cv_results.std() * 100:.2f}%)")

# For multiple metrics
from sklearn.model_selection import cross_validate
multi_cv_results = cross_validate(model, X, y, cv=5, scoring=scoring)

# Print the average cross-validation results for each metric
print(f"Average Cross-Validated Precision: {multi_cv_results['test_precision'].mean() * 100:.2f}%")
print(f"Average Cross-Validated Recall: {multi_cv_results['test_recall'].mean() * 100:.2f}%")
print(f"Average Cross-Validated F1 Score: {multi_cv_results['test_f1'].mean() * 100:.2f}%")

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
