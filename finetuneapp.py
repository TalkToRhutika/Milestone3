import streamlit as st
import torch
from torch.utils.data import Dataset
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification, Trainer, TrainingArguments
import random

#customdataset
class CustomDataset(Dataset):
    def __init__(self, texts, labels, tokenizer):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        encoding = self.tokenizer(text, truncation=True, padding='max_length', max_length=128, return_tensors='pt')
        inputs = {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'labels': torch.tensor(label, dtype=torch.long)
        }
        return inputs

#function to generate eval_labels
def generate_eval_labels(texts):
    eval_labels = [random.randint(0, 1) for _ in texts]  # Generate random 0 or 1 labels
    return eval_labels

#load pretrained tokenizer
tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased-finetuned-sst-2-english")

#load pretrained model
model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased-finetuned-sst-2-english", num_labels=2)

#training arguments
training_args = TrainingArguments(
    output_dir='./results',          # output directory
    eval_strategy='epoch',           # evaluate every epoch
    logging_dir='./logs',            # directory for storing logs
    per_device_train_batch_size=8,   # batch size for training
    num_train_epochs=3,              # number of training epochs
    logging_steps=10,
    save_steps=100,
)

#prediction
def predict_sentiments(texts):
    results = []
    for text in texts:
        inputs = tokenizer(text, return_tensors='pt')
        with torch.no_grad():
            outputs = model(**inputs)
        logits = outputs.logits
        predicted_label = 'POSITIVE' if torch.argmax(logits) == 1 else 'NEGATIVE'
        score = torch.softmax(logits, dim=1)[0][torch.argmax(logits)].item()
        results.append({'label': predicted_label, 'score': score})
    return results

#streamlit app
def main():
    st.title("Sentiment Analysis with DistilBERT")
    #st.write("Enter one or more sentences to predict their sentiment.")

    #user input
    text_input = st.text_area("Input Text", "Type here... (one sentence per line)")

    if st.button("Predict"):
        #split input
        sentences = text_input.splitlines()
        results = predict_sentiments(sentences)

        #results
        st.write("\n\n")
        st.markdown("### Results:")
        for result in results:
            st.write(f"{{'label': '{result['label']}', 'score': {result['score']:.4f}}}")
            st.write("\n")

if __name__ == '__main__':
    main()
