# Imports
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

# Load the data
data = pd.read_csv('/Users/lucasginevro/Downloads/reviews.csv', sep='\t')  # still use \t if it's tab-separated

# Map RatingValue to Sentiment
def map_sentiment(rating):
    if rating in [1, 2]:
        return 0  # negative
    elif rating == 3:
        return 1  # neutral
    else:  # 4 or 5
        return 2  # positive

data['Sentiment'] = data['RatingValue'].apply(map_sentiment)

# Smart balancing — limit only the positive class
neg_neut = data[data['Sentiment'].isin([0, 1])]
pos = data[data['Sentiment'] == 2]

# Get average size of negative and neutral classes
avg_size = int(neg_neut['Sentiment'].value_counts().mean())

# Downsample positive class to the average size
pos_downsampled = pos.sample(n=avg_size, random_state=42)

# Combine balanced dataset
balanced_data = pd.concat([neg_neut, pos_downsampled]).reset_index(drop=True)

# Train-validation split
train_df, valid_df = train_test_split(
    balanced_data,
    test_size=0.2,
    stratify=balanced_data['Sentiment'],
    random_state=42
)

# Save the train and validation sets
train_df.to_csv('train.csv', index=False)
valid_df.to_csv('valid.csv', index=False)


# Quick checks
print("Total samples:", len(balanced_data))
print("Train shape:", train_df.shape)
print("Validation shape:", valid_df.shape)

print("\nClass balance in train:")
print(train_df['Sentiment'].value_counts())

print("\nClass balance in validation:")
print(valid_df['Sentiment'].value_counts())


# EDA Check Through

print("Missing values in train:\n", train_df.isnull().sum())
print("Missing values in valid:\n", valid_df.isnull().sum())

print("\nTrain class distribution:")
print(train_df['Sentiment'].value_counts())

print("\nValidation class distribution:")
print(valid_df['Sentiment'].value_counts())

# Text Cleaning
import re

def clean_text(text):
    text = text.lower()
    text = re.sub(r'\n', ' ', text)
    text = re.sub(r'[^\w\s]', '', text)
    return text

train_df['CleanedReview'] = train_df['Review'].apply(clean_text)
valid_df['CleanedReview'] = valid_df['Review'].apply(clean_text)


# Building Sentiment Classifier
# Vectorize the Review text
vectorizer = CountVectorizer(stop_words='english')  # ignore common stopwords
X_train = vectorizer.fit_transform(train_df['Review'])
X_valid = vectorizer.transform(valid_df['Review'])

y_train = train_df['Sentiment']
y_valid = valid_df['Sentiment']

# Train the Logistic Regression model
model = LogisticRegression(max_iter=1000, random_state=42)
model.fit(X_train, y_train)

# Predict on validation set
y_pred = model.predict(X_valid)

# Evaluate performance
accuracy = accuracy_score(y_valid, y_pred)
f1 = f1_score(y_valid, y_pred, average='weighted')
conf_matrix = confusion_matrix(y_valid, y_pred)

print("accuracy:", accuracy)
print("\nF1_score:", f1)
print("\nConfusion_matrix:")
print("           negative  neutral  positive")
for row_label, row in zip(['negative', 'neutral', 'positive'], conf_matrix):
    print(f"{row_label:>10}  {row}")


# Making Adjustments
# Adjusting to having a balanced dataset

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

# Train the classifier (with class_weight='balanced')
model = LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42)
model.fit(X_train, y_train)

# Predict on validation set
y_pred = model.predict(X_valid)

# Evaluate
accuracy = accuracy_score(y_valid, y_pred)
f1 = f1_score(y_valid, y_pred, average='weighted')
conf_matrix = confusion_matrix(y_valid, y_pred)

# Print performance
print("accuracy:", accuracy)
print("\nF1_score:", f1)
print("\nConfusion_matrix:")
print("           negative  neutral  positive")
for label, row in zip(['negative', 'neutral', 'positive'], conf_matrix):
    print(f"{label:>10}  {row}")


# Evaluating Lemmatization
# Note that this was not used because it scored lower than the model above (54% accuracy/F1 score)
# import nltk
# from nltk.stem import WordNetLemmatizer
# nltk.download('wordnet')
# nltk.download('omw-1.4')

# lemmatizer = WordNetLemmatizer()

# def clean_and_lemmatize(text):
    # text = text.lower()
    # text = re.sub(r'[^\w\s]', '', text)
    # tokens = text.split()
    # lemmatized = [lemmatizer.lemmatize(word) for word in tokens]
    # return ' '.join(lemmatized)

# train_df['CleanedReview'] = train_df['Review'].apply(clean_and_lemmatize)
# valid_df['CleanedReview'] = valid_df['Review'].apply(clean_and_lemmatize)

# Re-vectorizing
# from sklearn.feature_extraction.text import TfidfVectorizer

# vectorizer = TfidfVectorizer(stop_words='english', ngram_range=(1, 1))  # stick to unigrams for now
# X_train = vectorizer.fit_transform(train_df['CleanedReview'])
# X_valid = vectorizer.transform(valid_df['CleanedReview'])

# y_train = train_df['Sentiment']
# y_valid = valid_df['Sentiment']

# from sklearn.linear_model import LogisticRegression
# from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

# model = LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42)
# model.fit(X_train, y_train)

# y_pred = model.predict(X_valid)

# accuracy = accuracy_score(y_valid, y_pred)
# f1 = f1_score(y_valid, y_pred, average='weighted')
# conf_matrix = confusion_matrix(y_valid, y_pred)

# print("accuracy:", accuracy)
# print("\nF1_score:", f1)
# print("\nConfusion_matrix:")
# print("           negative  neutral  positive")
# for label, row in zip(['negative', 'neutral', 'positive'], conf_matrix):
    # print(f"{label:>10}  {row}")


# Evaluating Bigrams
# Note: Similar to Lemmatization above, not used because it scored lower results (same accuracy, slightly lower F1 score)
# vectorizer = TfidfVectorizer(stop_words='english', ngram_range=(1, 2))

# X_train = vectorizer.fit_transform(train_df['CleanedReview'])
# X_valid = vectorizer.transform(valid_df['CleanedReview'])

# model = LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42)
# model.fit(X_train, y_train)

# y_pred = model.predict(X_valid)

# accuracy = accuracy_score(y_valid, y_pred)
# f1 = f1_score(y_valid, y_pred, average='weighted')
# conf_matrix = confusion_matrix(y_valid, y_pred)

# print("accuracy:", accuracy)
# print("\nF1_score:", f1)
# print("\nConfusion_matrix:")
# print("           negative  neutral  positive")
# for label, row in zip(['negative', 'neutral', 'positive'], conf_matrix):
    # print(f"{label:>10}  {row}")


# --- Predict on the final test set (test.csv) if available ---
try:
    test_df = pd.read_csv('test.csv', sep='\t')
    
    # Map RatingValue to Sentiment (same as before)
    test_df['Sentiment'] = test_df['RatingValue'].apply(map_sentiment)

    # Clean the review text
    test_df['CleanedReview'] = test_df['Review'].apply(clean_text)

    # Vectorize the test reviews using the same vectorizer
    X_test = vectorizer.transform(test_df['CleanedReview'])
    y_test = test_df['Sentiment']

    # Predict and evaluate
    y_pred_test = model.predict(X_test)

    test_accuracy = accuracy_score(y_test, y_pred_test)
    test_f1 = f1_score(y_test, y_pred_test, average='weighted')
    test_conf_matrix = confusion_matrix(y_test, y_pred_test)

    print("\n--- TEST SET PERFORMANCE ---")
    print("accuracy:", test_accuracy)
    print("\nF1_score:", test_f1)
    print("\nConfusion_matrix:")
    print("           negative  neutral  positive")
    for label, row in zip(['negative', 'neutral', 'positive'], test_conf_matrix):
        print(f"{label:>10}  {row}")

except FileNotFoundError:
    print("\n'test.csv' not found — skipping test set evaluation.")

