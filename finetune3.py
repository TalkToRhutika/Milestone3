import os
import pandas as pd
import streamlit as st
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset
from sklearn.model_selection import train_test_split
import requests
from io import BytesIO

# Load the dataset
@st.cache_data
def load_data():
    url = "https://huggingface.co/datasets/HUPD/hupd/resolve/main/hupd_metadata_2022-02-22.feather"
    response = requests.get(url)
    data = BytesIO(response.content)
    df = pd.read_feather(data)
    return df

# Tokenizer and model loading
def load_tokenizer_and_model(model_name, num_labels):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)
    return tokenizer, model

# Tokenize and prepare the dataset
def prepare_data(df, tokenizer):
    df['filing_date'] = pd.to_datetime(df['filing_date'])
    jan_2016_df = df[df['filing_date'].dt.to_period('M') == '2016-01']
    
    # Get only 5 unique labels
    unique_labels = jan_2016_df['patent_number'].astype('category').cat.categories[:5]
    jan_2016_df = jan_2016_df[jan_2016_df['patent_number'].isin(unique_labels)]
    
    # Re-map labels to integers starting from 0
    label_mapping = {label: idx for idx, label in enumerate(unique_labels)}
    jan_2016_df['label'] = jan_2016_df['patent_number'].map(label_mapping)

    texts = jan_2016_df['invention_title'].tolist()
    labels = jan_2016_df['label'].tolist()
    num_labels = len(unique_labels)

    # Define tokenization function
    def tokenize_function(texts):
        return tokenizer(texts, padding=True, truncation=True, return_tensors="pt", max_length=512)

    # Tokenize texts
    tokenized_data = tokenize_function(texts)

    # Create dataset
    dataset_dict = {
        'input_ids': [x.tolist() for x in tokenized_data['input_ids']],
        'attention_mask': [x.tolist() for x in tokenized_data['attention_mask']],
        'labels': labels
    }
    
    dataset = Dataset.from_dict(dataset_dict)
    
    return dataset, num_labels

# Define Streamlit app
def main():
    st.title("Patent Classification with Fine-Tuned BERT")
    
    # Initialize model directory path
    model_dir = './finetuned_model'
    
    # Load data
    df = load_data()
    
    # Show data
    st.subheader("Data from January 2016")
    st.write(df.head())
    
    # Prepare data
    model_name = "bert-base-uncased"
    tokenizer, model = load_tokenizer_and_model(model_name, num_labels=5)
    dataset, num_labels = prepare_data(df, tokenizer)
    
    # Update the model with the correct number of labels based on the data
    if num_labels != 5:
        model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)
    
    # Split the dataset
    train_data, eval_data = train_test_split(list(zip(dataset['input_ids'], dataset['attention_mask'], dataset['labels'])), test_size=0.2, random_state=42)
    
    def create_dataset(data):
        return Dataset.from_dict({
            'input_ids': [item[0] for item in data],
            'attention_mask': [item[1] for item in data],
            'labels': [item[2] for item in data]
        })

    train_dataset = create_dataset(train_data)
    eval_dataset = create_dataset(eval_data)
    
    # Show training data
    st.subheader("Training Data")
    train_df = pd.DataFrame({
        'input_ids': [ids[:10] for ids in train_dataset['input_ids'][:5]],  
        'attention_mask': [mask[:10] for mask in train_dataset['attention_mask'][:5]],
        'labels': train_dataset['labels'][:5]
    })
    st.write(train_df)

    # Fine-tune model
    training_args = TrainingArguments(
        output_dir=model_dir,
        evaluation_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        num_train_epochs=3,
        weight_decay=0.01,
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer
    )
    
    st.subheader("Training the Model")
    if st.button('Train Model'):
        with st.spinner('Training in progress...'):
            trainer.train()
            model.save_pretrained(model_dir)
            tokenizer.save_pretrained(model_dir)
            st.success("Model training complete and saved.")
    
    # Display pretrained model data
    st.subheader("Pretrained Model")
    if st.button('Show Pretrained Model'):
        if os.path.exists(model_dir):
            files = [f for f in os.listdir(model_dir) if f.endswith('.json')]
            st.write("Contents of `.json` files in `./finetuned_model` directory:")
            for file in files:
                file_path = os.path.join(model_dir, file)
                st.write(f"**{file}:**")
                with open(file_path, 'r', encoding='utf-8') as f:
                    st.write(f.read())
        else:
            st.write("Directory `./finetuned_model` does not exist.")

if __name__ == "__main__":
    main()
