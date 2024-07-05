import numpy as np
import pandas as pd
import re
import torch
import random
import torch.nn as nn
import transformers
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from transformers import AutoModel, BertTokenizerFast
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler, Dataset
import torch.optim.lr_scheduler as lr_scheduler
import json
import gradio
import matplotlib
matplotlib.use('agg')  # Use a non-interactive backend
import matplotlib.pyplot as plt

# Load the CSV data
df = pd.read_csv('/u/irist_guest/Desktop/al/p.csv')

# Filter the dataframe to only include source_name and content columns
df = df[['Label', 'Text']]

# Show the first 5 rows
df.head()
device = torch.device('cuda' if torch.cuda.is_available() else "cpu")
df["Text"] = df['Text'].apply(lambda x: '' if pd.isna(x) else str(x))

def clean_text(text):
    text = text.replace("\n", "").strip()
    return text

df["Text"] = df["Text"].apply(clean_text)

train_df, test_df = train_test_split(df, test_size=0.2, stratify=df["Label"], random_state=42)

# Encode labels to integers
label_encoder = LabelEncoder()
df["Label"] = label_encoder.fit_transform(df["Label"])
source_to_idx = {label: idx for idx, label in enumerate(label_encoder.classes_)}

class SyllabiDataset(Dataset):
    def __init__(self, data, tokenizer, max_len, source_to_idx):
        self.data = data
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.source_to_idx = source_to_idx

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        row = self.data.iloc[index]
        textt = row["Text"]
        labell = row["Label"]
        
        inputs = self.tokenizer.encode_plus(textt, None, add_special_tokens=True, max_length=self.max_len, padding="max_length", return_token_type_ids=True, truncation=True)
        return {
            "ids": torch.tensor(inputs["input_ids"], dtype=torch.long),
            "mask": torch.tensor(inputs["attention_mask"], dtype=torch.long), 
            "targets": torch.tensor(self.source_to_idx[labell], dtype=torch.long)
        }

tokenizer = transformers.DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
MAX_LEN = 256
BATCH_SIZE = 17

train_set = SyllabiDataset(train_df, tokenizer, MAX_LEN, source_to_idx)
test_set = SyllabiDataset(test_df, tokenizer, MAX_LEN, source_to_idx)

train_loader = DataLoader(train_set, sampler=RandomSampler(train_set), batch_size=BATCH_SIZE)
test_loader = DataLoader(test_set, sampler=SequentialSampler(test_set), batch_size=BATCH_SIZE)

class Label(torch.nn.Module):
    def __init__(self):
        super(Label, self).__init__()
        self.distilbert = transformers.DistilBertModel.from_pretrained('distilbert-base-uncased')
        self.dropout = torch.nn.Dropout(0.3)
        self.output = torch.nn.Linear(768, len(df['Label'].unique()))

    def forward(self, ids, mask):
        output = self.distilbert(ids, attention_mask=mask)
        output = self.dropout(output[0][:, 0, :])  # Use the [CLS] token representation
        output = self.output(output)
        return output

model = Label()
model.to(device)

EPOCHS = 10
LEARNING_RATE = 1e-7
optimizer = torch.optim.Adam(params=model.parameters(), lr=LEARNING_RATE)
loss_function = torch.nn.CrossEntropyLoss()

for epoch in range(EPOCHS):
    model.train()
    train_bar = tqdm(train_loader, desc=f"Epoch {epoch+1} - Training")
    for batch in train_bar:
        ids = batch['ids'].to(device)
        mask = batch['mask'].to(device)
        targets = batch['targets'].to(device)

        optimizer.zero_grad()
        outputs = model(ids, mask)
        loss = loss_function(outputs, targets)
        loss.backward()
        optimizer.step()

        train_bar.set_postfix(loss=loss.item())

    model.eval()
    total_loss = 0
    total_correct = 0
    total_samples = 0

    test_bar = tqdm(test_loader, desc=f"Epoch {epoch+1} - Testing")
    for batch in test_bar:
        ids = batch['ids'].to(device)
        mask = batch['mask'].to(device)
        targets = batch['targets'].to(device)

        with torch.no_grad():
            outputs = model(ids, mask)
            loss = loss_function(outputs, targets)
            total_loss += loss.item()

            _, predicted = torch.max(outputs, 1)
            total_correct += (predicted == targets).sum().item()
            total_samples += targets.size(0)

            test_bar.set_postfix(loss=loss.item())

    avg_loss = total_loss / len(test_loader)
    accuracy = total_correct / total_samples

    print(f"Epoch: {epoch + 1}, Loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f}")

# import numpy as np
# import pandas as pd
# import re
# import torch
# import random
# import torch.nn as nn
# import transformers
# import matplotlib.pyplot as plt
# from sklearn.preprocessing import LabelEncoder
# from sklearn.model_selection import train_test_split
# from tqdm import tqdm
# from transformers import AutoModel, BertTokenizerFast
# from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler, Dataset
# import torch.optim.lr_scheduler as lr_scheduler
# import json
# import gradio
# import matplotlib
# matplotlib.use('agg')  # Use a non-interactive backend
# import matplotlib.pyplot as plt
# # Load the CSV data
# df = pd.read_csv('/u/irist_guest/Desktop/al/p.csv')
# # Filter the dataframe to only include source_name and content columns
# df = df[['Label', 'Text']]

# # Show the first 5 rows
# df.head()
# device = torch.device('cuda' if torch.cuda.is_available() else "cpu")
# df["Text"] = df['Text'].apply(lambda x: '' if pd.isna(x) else str(x))

# def clean_text(text):
#     text = text.replace("\n", "").strip()
#     return text

# df["Text"] = df["Text"].apply(clean_text)

# train_df, test_df = train_test_split(df, test_size = 0.2, stratify=df["Label"], random_state = 42)

# class SyllabiDataset(Dataset):
#     def __init__(self, data, tokenizer, max_len, source_to_idx):
#         self.data = data
#         self.tokenizer = tokenizer
#         self.max_len = max_len
#         self.source_to_idx = source_to_idx

#     def __len__(self):
#         return len(self.data)
    
#     def __getitem__(self, index):
#         row = self.data.iloc[index]
#         textt = row["Text"]
#         labell = row["Label"]
        
#         inputs = self.tokenizer.encode_plus(textt, None, add_special_tokens = True, max_length = self.max_len, padding = "max_length", return_token_type_ids= True, truncation = True)
#         return {
#             "ids": torch.tensor(inputs["input_ids"], dtype = torch.long),
#             "mask": torch.tensor(inputs["attention_mask"], dtype= torch.lang), 
#             "targets": torch.tensor(self.source_to_idx[labell], dtype = torch.lang)
#         }

# tokenizer = transformers.DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
# MAX_LEN = 256
# BATCH_SIZE = 17


# train_set = SyllabiDataset(train_df, tokenizer, MAX_LEN, source_to_idx)
# test_set = SyllabiDataset(train_df, tokenizer, MAX_LEN, source_to_idx)
# class Label(torch.nn.Module):
#     def __init__(self):
#         super(Label, self).__init__()
#         self.distilbert = transformers.DistilBertModel.from_pretrained('distilbert-base-uncased')
#         self.dropout = torch.nn.Dropout(0.3)
#         self.output = torch.nn.Linear(768, len(df['Label'].unique()))

#     def forward(self, ids, mask):
#         output = self.distilbert(ids, attention_mask=mask)
#         output = self.dropout(output[0][:, 0, :])  # Use the [CLS] token representation
#         output = self.output(output)
#         return output
# model = Label()
# model.to(device)

# EPOCHS = 100
# LEARNING_RATE = 1e-7
# optimizer = torch.optim.Adam(params=model.parameters(), lr=LEARNING_RATE)
# loss_function = torch.nn.CrossEntropyLoss()

# for epoch in range(EPOCHS):
#     model.train()
#     train_bar = tqdm(train_loader, desc=f"Epoch {epoch+1} - Training")
#     for batch in train_bar:
#         ids = batch['ids'].to(device)
#         mask = batch['mask'].to(device)
#         targets = batch['targets'].to(device)

#         optimizer.zero_grad()
#         outputs = model(ids, mask)
#         loss = loss_function(outputs, targets)
#         loss.backward()
#         optimizer.step()

#         train_bar.set_postfix(loss=loss.item())

#     model.eval()
#     total_loss = 0
#     total_correct = 0
#     total_samples = 0

#     test_bar = tqdm(test_loader, desc=f"Epoch {epoch+1} - Testing")
#     for batch in test_bar:
#         ids = batch['ids'].to(device)
#         mask = batch['mask'].to(device)
#         targets = batch['targets'].to(device)

#         with torch.no_grad():
#             outputs = model(ids, mask)
#             loss = loss_function(outputs, targets)
#             total_loss += loss.item()

#             _, predicted = torch.max(outputs, 1)
#             total_correct += (predicted == targets).sum().item()
#             total_samples += targets.size(0)

#             test_bar.set_postfix(loss=loss.item())

#     avg_loss = total_loss / len(test_loader)
#     accuracy = total_correct / total_samples

#     print(f"Epoch: {epoch + 1}, Loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f}")