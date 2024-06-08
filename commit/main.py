import json
import re
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Load intents.json
with open('intents.json') as file:
    data = json.load(file)

# Extract questions and their corresponding labels (minor or major)
questions = []
labels = []
feedbacks = []
for intent in data['intents']:
    for pattern in intent['patterns']:
        questions.append(pattern)
        # Assuming labels are encoded based on the 'tag' field
        label = intent['tag']
        if label == 'major':
            labels.append(1)  # Major mistake
        else:
            labels.append(0)  # Minor mistake
        feedbacks.append(intent['responses'][0])

# Check the distribution of labels
print(pd.Series(labels).value_counts())

# Convert to a DataFrame for easier manipulation
df = pd.DataFrame({
    'question': questions,
    'label': labels,
    'feedback': feedbacks
})

# Step 2: Data Preprocessing
df['question'] = df['question'].str.lower()
df['question'] = df['question'].str.replace(r'\W', ' ', regex=True)

# Step 3: Feature Extraction
tfidf = TfidfVectorizer()
X = tfidf.fit_transform(df['question'])
y = df['label']

# Step 4: Model Training
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LogisticRegression()
model.fit(X_train, y_train)

# Step 5: Evaluation
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print(f'Accuracy: {accuracy:.2f}')
print(f'Precision: {precision:.2f}')
print(f'Recall: {recall:.2f}')
print(f'F1 Score: {f1:.2f}')

# Step 6: User Input Handling and Feedback Generation
def classify_user_input(user_input):
    user_input_processed = user_input.lower()
    user_input_processed = re.sub(r'\W', ' ', user_input_processed)
    user_input_vector = tfidf.transform([user_input_processed])
    prediction = model.predict(user_input_vector)[0]
    feedback = df[df['label'] == prediction]['feedback'].values[0]
    if prediction == 1:
        return f"This is a major mistake. Feedback: {feedback}"
    else:
        return f"This is a minor mistake. Feedback: {feedback}"

# Example usage
user_input = "Oxygen isn't flammable."
result = classify_user_input(user_input)
print(result)