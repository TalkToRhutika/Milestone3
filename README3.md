# Sentiment Analysis with DistilBERT and Streamlit

This is a simple web application built using Streamlit to perform sentiment analysis on text inputs using the DistilBERT model pretrained on the given dataset.

### Overview

The application allows users to input one or more sentences and predicts whether each sentence expresses a positive or negative sentiment. It utilizes the Hugging Face Transformers library for natural language processing tasks and Streamlit for creating an interactive web interface.

### Features

- **Input:** Users can input multiple sentences, each on a new line, for sentiment analysis.
- **Prediction:** For each input sentence, the application predicts whether the sentiment is 'POSITIVE' or 'NEGATIVE'.
- **Confidence Score:** Provides a confidence score indicating the model's certainty about its prediction.
- **Interactive Interface:** Simple and intuitive web interface powered by Streamlit.

### Libraries Used

   - streamlit
   - torch
   - transformers
   - random

### Application
1. **Running the Application:**
   - Navigate to the directory containing `finetuneapp.py`.
   - Run the Streamlit application:

     ```
     streamlit run finetuneapp.py
     ```

   - It will redirect to you browser

2. **Usage:**
   - Enter one or more sentences in the text area provided.
   - Click on the "Predict" button to see the sentiment analysis results for each sentence.

### Some screenshots

