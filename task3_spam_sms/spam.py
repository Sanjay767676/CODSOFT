import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
df = pd.read_csv('spam.csv', encoding='latin-1')
df = df[['v1', 'v2']].rename(columns={'v1': 'label', 'v2': 'message'})
df['label'] = df['label'].map({'ham': 0, 'spam': 1})
def preprocess_text(text):
    text = text.lower()  
    text = ''.join([c for c in text if c.isalpha() or c == ' '])  
    return text.strip()

df['clean_message'] = df['message'].apply(preprocess_text)
X = df['clean_message']
y = df['label']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

tfidf = TfidfVectorizer(max_features=5000, stop_words='english')
X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf = tfidf.transform(X_test)

def train_evaluate_model(model, X_train, y_train, X_test, y_test):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    print(f"\n{model.__class__.__name__} Results:")
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print(f"Precision: {precision_score(y_test, y_pred):.4f}")
    print(f"Recall: {recall_score(y_test, y_pred):.4f}")
    print(f"F1-Score: {f1_score(y_test, y_pred):.4f}")
    
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot()
    plt.title(f'{model.__class__.__name__} Confusion Matrix')
    plt.show()
    
    return model
models = [
    MultinomialNB(),
    LogisticRegression(max_iter=1000),
    SVC(kernel='linear', probability=True)
]
trained_models = []
for model in models:
    trained_model = train_evaluate_model(model, X_train_tfidf, y_train, X_test_tfidf, y_test)
    trained_models.append(trained_model)
def plot_word_cloud(text, title):
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud)
    plt.axis('off')
    plt.title(title)
    plt.show()
spam_words = ' '.join(df[df['label'] == 1]['clean_message'])
ham_words = ' '.join(df[df['label'] == 0]['clean_message'])

plot_word_cloud(spam_words, 'Most Common Words in Spam Messages')
plot_word_cloud(ham_words, 'Most Common Words in Ham Messages')
def predict_spam(model, vectorizer, new_message):
    cleaned_message = preprocess_text(new_message)
    message_vector = vectorizer.transform([cleaned_message])
    prediction = model.predict(message_vector)
    probability = model.predict_proba(message_vector)[0][1]
    
    print(f"\nMessage: {new_message}")
    print(f"Cleaned version: {cleaned_message}")
    print(f"Prediction: {'SPAM' if prediction[0] == 1 else 'HAM'}")
    print(f"Spam probability: {probability:.4f}")
best_model = trained_models[1]
predict_spam(best_model, tfidf, "WINNER!! You've been selected for a free $1000 Amazon gift card! Click here to claim now!")
predict_spam(best_model, tfidf, "Hey, are we still meeting for lunch tomorrow? Let me know the time.")