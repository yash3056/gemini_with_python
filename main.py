import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
import torch

# Load the dataset
file_path = 'constraint_Hindi_Train - Sheet1.csv'
data = pd.read_csv(file_path)

# Clean the text data
def clean_text(text):
    text = text.str.replace(r'http\S+|www.\S+', '', case=False)
    text = text.str.replace(r'[^a-zA-Z0-9\s]', '', case=False)
    return text

data['Post'] = clean_text(data['Post'])

# Encode the labels
data['Labels'] = data['Labels Set'].apply(lambda x: x.split(','))
mlb = MultiLabelBinarizer()
labels = mlb.fit_transform(data['Labels'])

# Split the dataset into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(data['Post'], labels, test_size=0.2, random_state=42)

# Load the tokenizer and model
tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')
model = BertForSequenceClassification.from_pretrained('bert-base-multilingual-cased', num_labels=5)

# Tokenize the text data
def tokenize_function(texts):
    return tokenizer(texts, padding="max_length", truncation=True, max_length=128)

train_encodings = tokenize_function(X_train.tolist())
val_encodings = tokenize_function(X_val.tolist())

# Convert encodings to PyTorch tensors
class HateSpeechDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

train_dataset = HateSpeechDataset(train_encodings, y_train)
val_dataset = HateSpeechDataset(val_encodings, y_val)

# Define the training arguments
training_args = TrainingArguments(
    output_dir='/home/aza/workspace/hatespeech/output',          # output directory
    num_train_epochs=3,              # number of training epochs
    per_device_train_batch_size=16,  # batch size for training
    per_device_eval_batch_size=64,   # batch size for evaluation
    warmup_steps=500,                # number of warmup steps for learning rate scheduler
    weight_decay=0.01,               # strength of weight decay
    logging_dir='/home/aza/workspace/hatespeech/logs',            # directory for storing logs
    logging_steps=10,
)

# Create Trainer instance
trainer = Trainer(
    model=model,                         # the instantiated 🤗 Transformers model to be trained
    args=training_args,                  # training arguments, defined above
    train_dataset=train_dataset,         # training dataset
    eval_dataset=val_dataset             # evaluation dataset
)

# Training the model
trainer.train()
