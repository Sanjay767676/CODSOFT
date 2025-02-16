import re
import csv
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from collections import defaultdict
def refine_text(text):
    text = re.sub(r'[^a-zA-Z\s]', '', text.lower())
    text = re.sub(r'\b(?:movie|film|story|character)\b', '', text)  
    return re.sub(r'\s+', ' ', text).strip()
train_data = []
train_labels = []
with open('train_data.txt', 'r', encoding='utf-8', errors='replace') as f:
    for line in f:
        parts = line.strip().split(':::')
        if len(parts) >= 4 and len(parts[3]) > 50: 
            train_data.append(refine_text(parts[3]))
            train_labels.append(parts[2].strip().lower())
test_entries = []
test_raw_text = []
test_ids = []
with open('test_data.txt', 'r', encoding='utf-8', errors='replace') as f:
    for line in f:
        parts = line.strip().split(':::')
        if len(parts) >= 3:
            test_raw_text.append(parts[2].strip())
            test_entries.append(refine_text(parts[2]))
            test_ids.append(parts[0].strip())
solution_map = {}
with open('test_data_solution.txt', 'r', encoding='utf-8', errors='replace') as f:
    for line in f:
        parts = line.strip().split(':::')
        if len(parts) >= 4:
            solution_map[parts[0].strip()] = parts[2].strip().lower()
vectorizer = TfidfVectorizer(
    ngram_range=(1, 3),
    max_df=0.6,
    min_df=2,
    stop_words='english',
    sublinear_tf=True
)
X_train = vectorizer.fit_transform(train_data)
X_test = vectorizer.transform(test_entries)
model = SGDClassifier(
    loss='log_loss',
    penalty='elasticnet',
    class_weight='balanced',
    max_iter=3000,
    early_stopping=True,
    n_iter_no_change=15
)
model.fit(X_train, train_labels)
confidence_threshold = 0.25
probabilities = model.predict_proba(X_test)
predictions = []
confidence_scores = []

for prob in probabilities:
    max_prob = np.max(prob)
    if max_prob < confidence_threshold:
        predictions.append("uncertain")
    else:
        predictions.append(model.classes_[np.argmax(prob)])
    confidence_scores.append(max_prob)
final_predictions = []
for text, pred in zip(test_raw_text, predictions):
    if pred == 'drama':
        if any(kw in text.lower() for kw in ['murder', 'crime', 'suspense']):
            final_predictions.append('thriller')
        else:
            final_predictions.append(pred)
    else:
        final_predictions.append(pred)
with open('genre_analysis_report.csv', 'w', newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    writer.writerow(['ID', 'RawText', 'Predicted', 'Actual', 'Confidence'])
    
    for tid, raw, pred, actual, conf in zip(
        test_ids,
        test_raw_text,
        final_predictions,
        [solution_map.get(tid, 'missing') for tid in test_ids],
        confidence_scores
    ):
        writer.writerow([tid, raw, pred, actual, f"{conf:.2%}"])
correct = 0
total = 0
mismatches = defaultdict(list)

for tid, pred, actual in zip(test_ids, final_predictions, 
                            [solution_map.get(tid, '') for tid in test_ids]):
    if actual != 'missing':
        total += 1
        if pred == actual:
            correct += 1
        else:
            mismatches[f"{actual}->{pred}"].append(tid)

print(f"\nFinal Accuracy: {correct/total:.2%}")
print("Common Mismatches:")
for conflict, ids in mismatches.items():
    print(f"{conflict}: {len(ids)} cases")

print("\nDetailed report saved to genre_analysis_report.csv")
