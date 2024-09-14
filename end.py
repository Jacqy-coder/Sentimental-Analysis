# Import necessary libraries
import streamlit as st
import pandas as pd
import spacy
from textblob import TextBlob
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import re
from collections import Counter

# Load spaCy model
nlp = spacy.load("en_core_web_sm")

# Title of the app
st.title("IT ENDS WITH US, the movie, sentiment Analysis")

# Read the CSV file (replace with your file path)
file_path = 'end.csv'
df = pd.read_csv(file_path)

# Display the DataFrame
st.write("Preview of the dataset:")
st.dataframe(df.head())

# Handle missing data (optional)
st.write("Missing values in the dataset:")
st.write(df.isnull().sum())
df.fillna(df.mode().iloc[0], inplace=True)

st.write(df.isnull().sum())

# Descriptive statistics (optional)
st.write("Descriptive statistics of the dataset:")
st.write(df.describe())

# Rating distribution
rating_counts = df['Rating'].value_counts().sort_index()

# Bar plot for rating distribution
st.subheader("Distribution of Ratings")
fig, ax = plt.subplots(figsize=(8, 6))
ax.bar(rating_counts.index, rating_counts.values, color=['#ff9999','#66b3ff','#99ff99','#ffcc99','#c2c2f0'])
ax.set_xlabel('Rating (0-5 Scale)')
ax.set_ylabel('Number of Occurrences')
ax.set_title('Distribution of Ratings (0-5 Scale)')
ax.set_xticks(rating_counts.index)
plt.tight_layout()  # Ensure proper layout
st.pyplot(fig)

# Function to clean review text using spaCy
def clean_review(text):
    doc = nlp(text)
    tokens = [token.text for token in doc if not token.is_stop and not token.is_punct]
    return ' '.join(tokens)

# Clean reviews and perform sentiment analysis
df['Cleaned_Review'] = df['Review'].apply(clean_review)
df['Sentiments'] = df['Cleaned_Review'].apply(lambda review: TextBlob(review).sentiment)
df['Compound Score'] = df['Sentiments'].apply(lambda sentiment: sentiment.polarity)

# Modify sentiment labeling to include 'neutral'
def label_sentiment(score):
    if score > 0.2:  # Positive sentiment threshold
        return 'positive'
    elif score < -0.2:  # Negative sentiment threshold
        return 'negative'
    else:  # Neutral sentiment
        return 'neutral'

df['Sentiment Label'] = df['Compound Score'].apply(label_sentiment)

# Display sentiment analysis results
st.write("Sentiment analysis results with neutral sentiment:")
st.dataframe(df[['Review', 'Cleaned_Review', 'Sentiments', 'Compound Score', 'Sentiment Label']].head())

# Sentiment label distribution
st.subheader("Sentiment Label Distribution")
sentiment_label_dist = df['Sentiment Label'].value_counts(normalize=True) * 100
st.write(sentiment_label_dist)

# Word Cloud
st.subheader("Word Cloud of Cleaned Reviews")
all_reviews = ' '.join(df['Cleaned_Review'])
wordcloud = WordCloud(width=800, height=400, background_color='white').generate(all_reviews)

fig, ax = plt.subplots(figsize=(10, 5))
ax.imshow(wordcloud, interpolation='bilinear')
ax.axis("off")
plt.tight_layout()  # Ensure layout is tight
st.pyplot(fig)

# Most common words
st.subheader("Most Common Words in Reviews")
words = [word for word in all_reviews.split() if word.isalpha()]
word_freq = Counter(words)
most_common_words = word_freq.most_common(10)

# Convert the list of most common words to a DataFrame
most_common_df = pd.DataFrame(most_common_words, columns=['Word', 'Frequency'])

# Display the DataFrame
st.write(most_common_df)

# Overall Sentiment
average_sentiment = df['Compound Score'].mean()
st.subheader("Overall Sentiment Analysis")
st.write(f"Average Sentiment Score: {average_sentiment:.2f}")

# Use consistent thresholds for overall sentiment
if average_sentiment > 0.2:
    st.write("Overall sentiment towards the 'End With Us' Movie is positive.")
elif average_sentiment < -0.2:
    st.write("Overall sentiment towards the 'End With Us' Movie is negative.")
else:
    st.write("Overall sentiment towards the 'End With Us' Movie is neutral.")