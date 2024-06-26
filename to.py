import pandas as pd
from sklearn.model_selection import train_test_split
import torch
from transformers import BertTokenizer, BertForSequenceClassification
from transformers import Trainer, TrainingArguments
from transformers import DataCollatorWithPadding
from torch.utils.data import DataLoader, Dataset
import numpy as np

# Load the dataset
df = pd.read_csv("/u/irist_guest/Desktop/pdfs/wy.csv")

# Split the dataset into training, validation, and test sets
train_df, temp_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df['Instructor'])
valid_df, test_df = train_test_split(temp_df, test_size=0.2, random_state=42, stratify=temp_df['Instructor'])

# Reset index for all datasets
train_df.reset_index(drop=True, inplace=True)
valid_df.reset_index(drop=True, inplace=True)
test_df.reset_index(drop=True, inplace=True)

# Define a custom dataset class
class CustomDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=False,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

# Prepare the tokenizer and datasets
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
max_len = 128

train_dataset = CustomDataset(train_df['text'].tolist(), train_df['Instructor'].tolist(), tokenizer, max_len)
valid_dataset = CustomDataset(valid_df['text'].tolist(), valid_df['Instructor'].tolist(), tokenizer, max_len)
test_dataset = CustomDataset(test_df['text'].tolist(), test_df['Instructor'].tolist(), tokenizer, max_len)

# Prepare data loaders
data_collator = DataCollatorWithPadding(tokenizer)
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, collate_fn=data_collator)
valid_loader = DataLoader(valid_dataset, batch_size=16, shuffle=False, collate_fn=data_collator)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False, collate_fn=data_collator)

# Initialize the model
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=df['Instructor'].nunique())

# Define training arguments
training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=3,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    evaluation_strategy='epoch',
    save_strategy='epoch',
    logging_dir='./logs',
    logging_steps=10,
)

# Initialize the Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=valid_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator
)

# Train the model
trainer.train()

# Evaluate the model
eval_results = trainer.evaluate(eval_dataset=test_dataset)
print(f"Test results: {eval_results}")

# Save the model
model.save_pretrained('./saved_model')
tokenizer.save_pretrained('./saved_model')
