import pickle

# Load the saved model
with open('spam_detection_model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

# Load the saved vectorizer
with open('vectorizer.pkl', 'rb') as vectorizer_file:
    vectorizer = pickle.load(vectorizer_file)

# Check if loading was successful
print("Model and vectorizer loaded successfully.")
# Example message for testing
test_message = "Congratulations! You've won a free ticket to Bahamas!"

# Vectorize the message
test_message_count = vectorizer.transform([test_message])

# Predict
prediction = model.predict(test_message_count)
result = "Spam" if prediction[0] == 1 else "Ham"

print(f"Test message: '{test_message}' is classified as: {result}")
