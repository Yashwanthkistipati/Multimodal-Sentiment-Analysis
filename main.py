import pandas as pd
import streamlit as st
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
import assemblyai as aai
import streamlit as st
from transformers import pipeline

st.set_page_config(layout="wide")
st.title("Enhancing Customer Expereince🤗😏😑 With Multimodal Sentiment Analysis")
## -------------------------------------------------------
# Tab 1
tab1, tab2, tab3, tab4 = st.tabs(["Home", "📅 Data", "Text Model","Audio Model"])

# Display content in Tab 1
tab1.markdown("""
### An Overview
Multimodal sentiment analysis is a future-oriented research area that aims to capture the sentiment of audience from multiple modal, such as audio, text. Through integration of data from different sources the multimodal sentiment analysis seeks to give a more depictive perception regarding what is expressed with various forms of media. By looking into the situation of this project, multimodal sentiment analysis refers to the development of the models that can assess the sentiment from the both text and the sound files. That project will involve feed-forward networks, recurrent neural networks (RNNs) and deep neural networks like convolutional neural networks (CNNs). Thus, the predictive models will be able to accurately classify the sentiment from the different data sources. Taking as an example, for a case of textual data, the project could be to preprocess language, distill features and train models to classify the sentiment expressed in text into positive, negative or neutral sentences. In addition, the same project may be engaged in preprocessing of the audio signals, extraction of significant features, like spectrograms or MFCC (Mel-frequency cepstral coefficients), and training algorithms for the sentiment classification in sound recordings. This project seeks to create an algorithm that combines information from both textual and auditory modalities to develop a comprehensive understanding of emotions, which could be quite useful in the areas of sentiment analysis in customers feedback, speech emotion recognition, or mental health monitoring. The presence of datasets that contain relevant examples of text and audio day enables creation of models that are more viable for multimodal sentiment analysis, thus these models would be utilized in decision-making processes and user experience across different sectors.

You can find the source code in the [GitHub Repository](https://github.com/Yashwanthkistipati/Multimodal-Sentiment-Analysis)
""")

### ------------------------------------------------------------
# Tab 2

# Display content in Tab 2
tab2.markdown("""
# About Dataset
""")

tab2.markdown("""
# Text Dataset
The primary idea behind text sentiment analysis, which is a fundamental technology, is making efficient models via AI based techniques such as machine learning and network to get the sentiment conveyed within different text elements. Precisely how the detection from textual information of the early emotions is the top priority in multiple application areas, such as customer feedback analysis, sentiment analysis of social media posts, and mental health estimation. The project aims to tag semantic undertones in text repositories, also known as labeled text repositories, adding labels like "positive," "negative," or "neutral." These datasets are widely used to train sentiment analysis projects. The project concentrates on preprocessing the textual data, text feature extraction, which is better known as text features, and attaching the scores with the text. The development of natural language understanding (NLU) mechanisms is now supported by text tagged datasets as well as evolving of deep learning frameworks and NLP libraries. This makes the process of multi-language sentiment analysis a real and applicable one. Thus, these models play a highly important role in making assertive decisions and producing a superior feeling over lots of diverse domains and application.
""")

# Load data
df1 = pd.read_csv("Copy of amazonreviews.csv")
# Create an expander within Tab 2
see_data = tab2.expander('You can click here to see the Dataset 👉')
# With the expander context, display dataframe
with see_data:
    st.dataframe(data=df1.reset_index(drop=True))


tab2.markdown("""
# Audio Dataset
Audio sentiment analysis, as a core technology, focuses on creating practical models with the help of AI methods
 such as machine learning and neural network technology, for classifying sentiments expressed within audio tracks. 
 Likewise, early detection of mood from sound-based data is crucial in many services,
  for example, customer feedback analysis, emotion recognition through speech, and mental health monitoring. 
  The key exploit of the project is labeling audio datasets which are known as labeled audio datasets and these kinds of datasets contain 
  sentiment labels which are generally labeled as “positive”, “negative”, or “neutral”.
   The project focusing mainly on preprocessing the audio data, extracting features which are audio features and sentiment labels which are generally labeled as “positive”, “negative”, or “neutral”, and then training different models such as recur With the access of labeled audio datasets, in addition to further developments in deep learning frameworks and audio processing libraries, the development of reliable audio sentiment analysis models becomes feasible other than just being a futuristic approach. 
   Hence, it contributes to the decision making process and enhances user experiences across various areas of activities 
""")

# Load data
df2 = pd.read_csv("TRAIN.csv")
# Create an expander within Tab 2
see_data = tab2.expander('You can click here to see the Dataset 👉')
# With the expander context, display dataframe
with see_data:
    st.dataframe(data=df2.reset_index(drop=True))

## ------------------------------------------------------

# Tab3



# Load the pre-trained sentiment analysis model
sentiment_analysis = pipeline("sentiment-analysis")

# Streamlit app
def main():
    tab3.title("Text Sentiment Analysis ")
    tab3.write("Enter text below to analyze its sentiment:")

    # Text input
    text_input = tab3.text_area("Input Text:", "")

    if tab3.button("Analyze"):
        if text_input:
            # Predict sentiment
            result = sentiment_analysis(text_input)
            label = result[0]['label']
            score = result[0]['score']
            tab3.write(f"Predicted Sentiment: {label}, Score: {score}")

if __name__ == "__main__":
    main()

## ------------------------------------------------------

# Tab4



# Only necessary for the first time running the program
nltk.download("vader_lexicon")

def analyze_sentiment(text):
    sid = SentimentIntensityAnalyzer()
    polarity_scores = sid.polarity_scores(text)

    if polarity_scores['pos'] >= polarity_scores['neg'] and polarity_scores['compound'] >= 0.05:
        return ("Positive", polarity_scores['pos'], polarity_scores['neg'], polarity_scores['compound'])
    elif polarity_scores['neg'] >= polarity_scores['pos'] and polarity_scores['compound'] <= -0.05:
        return ("Negative", polarity_scores['pos'], polarity_scores['neg'], polarity_scores['compound'])
    else:
        return ("Neutral", polarity_scores['pos'], polarity_scores['neg'], polarity_scores['compound'])

# Replace with your API key
aai.settings.api_key = "2ce777c3f2484a5babaf9ba85e9d3e34"

tab4.title("Audio Sentiment Analysis")

# File upload
uploaded_file = tab4.file_uploader("Choose a file", type=["wav", "mp3"])

if uploaded_file is not None:
    # Transcribe the audio file
    transcriber = aai.Transcriber()
    transcript = transcriber.transcribe(uploaded_file)

    if transcript.status == aai.TranscriptStatus.error:
        st.error(transcript.error)
    else:
        tab4.write("Transcript:")
        tab4.write(transcript.text)

        # Analyze the sentiment
        sentiment, pos, neg, neutral = analyze_sentiment(transcript.text)
        tab4.write(f"Sentiment: {sentiment}")
        tab4.write(f"Positive: {pos:.2f}")
        tab4.write(f"Negative: {neg:.2f}")
        tab4.write(f"Neutral: {neutral:.2f}")
