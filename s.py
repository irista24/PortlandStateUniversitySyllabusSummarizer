import numpy as np
import pandas as pd
import re
import torch
import random
import torch.nn as nn
import transformers
import matplotlib.pyplot as plt
from sklearn.utils import compute_class_weight
from torch.optim import AdamW
from sklearn.preprocessing import LabelEncoder as le
from transformers import AutoModel, BertTokenizerFast
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
import torch.optim.lr_scheduler as lr_scheduler
import json
import gradio as gr
import matplotlib
matplotlib.use('agg')  # Use a non-interactive backend
import matplotlib.pyplot as plt

# Load the CSV file
df_faq = pd.read_csv('/u/irist_guest/Desktop/al/p.csv')

# Load intents JSON file
data = json.load(open("/u/irist_guest/Desktop/pdfs/x.json", "r"))

# specify GPU
device = torch.device('cuda' if torch.cuda.is_available() else "cpu")

# Training
df = pd.read_csv('/u/irist_guest/Desktop/al/p.csv')
df['Label'] = le().fit_transform(df['Label'])
train_text, train_labels = df["Text"], df['Label']

# BERT
tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")
bert = AutoModel.from_pretrained("bert-base-uncased")

max_seq_len = 50
tokens_train = tokenizer(train_text.tolist(), max_length=max_seq_len, pad_to_max_length=True, truncation=True, return_token_type_ids=False)
train_seq = torch.tensor(tokens_train['input_ids'])
train_mask = torch.tensor(tokens_train['attention_mask'])
train_y = torch.tensor(train_labels.tolist())

batch_size = 16
train_data = TensorDataset(train_seq, train_mask, train_y)
train_sampler = RandomSampler(train_data)
train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size)

class BERT_Arch(nn.Module):
    def __init__(self, bert):
        super(BERT_Arch, self).__init__()
        self.bert = bert
        self.dropout = nn.Dropout(0.2)
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(768, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 17)
        self.softmax = nn.LogSoftmax(dim=1)
    
    def forward(self, sent_id, mask):
        cls_hs = self.bert(sent_id, attention_mask=mask)[0][:,0]
        x = self.fc1(cls_hs)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc3(x)
        x = self.softmax(x)
        return x

for param in bert.parameters():
    param.requires_grad = False

model = BERT_Arch(bert).to(device)
optimizer = AdamW(model.parameters(), lr=1e-3)
class_weights = compute_class_weight(class_weight = "balanced",
                                        classes = np.unique(train_labels),
                                        y = train_labels )
weights = torch.tensor(class_weights, dtype=torch.float).to(device)
cross_entropy = nn.NLLLoss(weight=weights)

train_losses = []
epochs = 25
lr_sch = lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.1)

def train():
    model.train()
    total_loss = 0
    total_preds = []
    for step, batch in enumerate(train_dataloader):
        if step % 50 == 0 and not step == 0:
            print('  Batch {:>17,}  of  {:>17,}.'.format(step, len(train_dataloader)))
        batch = [r.to(device) for r in batch]
        sent_id, mask, labels = batch
        preds = model(sent_id, mask)
        loss = cross_entropy(preds, labels)
        total_loss += loss.item()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        optimizer.zero_grad()
        preds = preds.detach().cpu().numpy()
        total_preds.append(preds)
    avg_loss = total_loss / len(train_dataloader)
    total_preds = np.concatenate(total_preds, axis=0)
    return avg_loss, total_preds

for epoch in range(epochs):
    print('\n Epoch {:} / {:}'.format(epoch + 1, epochs))
    train_loss, _ = train()
    train_losses.append(train_loss)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
print(f'\nTraining Loss: {train_loss:.3f}')

def extract_keywords(question):
    words = re.findall(r'\w+', question.lower())
    return words
def find_best_match(keywords, df):
    best_match = None
    highest_score = 0
    
    for index, row in df.iterrows():
        question_keywords = set(extract_keywords(row['Label']))
        score = len(set(keywords) & question_keywords)  # Count common keywords
        
        if score > highest_score:
            highest_score = score
            best_match = row['Text']
    
    return best_match if best_match else "I'm sorry, I don't have an answer for that question."


def get_prediction(str):
    str = re.sub(r'[^a-zA-Z ]+', '', str)
    test_text = [str]
    model.eval()
    tokens_test_data = tokenizer(test_text, max_length=max_seq_len, pad_to_max_length=True, truncation=True, return_token_type_ids=False)
    test_seq = torch.tensor(tokens_test_data['input_ids'])
    test_mask = torch.tensor(tokens_test_data['attention_mask'])
    with torch.no_grad():
        preds = model(test_seq.to(device), test_mask.to(device))
    preds = preds.detach().cpu().numpy()
    preds = np.argmax(preds, axis=1)
    return le.inverse_transform(preds)[0]

def get_response(message):
    keywords = extract_keywords(message)
    best_match = find_best_match(keywords, df_faq)
    if not best_match:
        intent = get_prediction(message)
        for i in data['intents']:
            if i["tag"] == intent:
                best_match = random.choice(i["responses"])
                break
    return best_match

def chat(message):
    response = get_response(message)
    return response

iface = gr.Interface(fn=chat, inputs="text", outputs="text", title="Chatbot")
iface.launch()
