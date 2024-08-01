import pandas as pd
import torch
import torch.nn as nn
import transformers
from transformers import AutoModel, BertTokenizerFast
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, Dataset
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib
matplotlib.use('agg') 
import matplotlib.pyplot as plt
import gradio as gr
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from collections import defaultdict
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# Load the CSV data
df = pd.read_csv('/u/irist_guest/Desktop/x/cw.csv')


# Filter the dataframe to only include source_name and content columns
df = df[['Label', 'Text', 'Summary']]


# Check class distribution
class_counts = df["Label"].value_counts()
print(class_counts)


# Filter out labels with fewer than 2 samples
min_samples_per_class = 2
valid_labels = class_counts[class_counts >= min_samples_per_class].index
df = df[df["Label"].isin(valid_labels)]


# Proceed with data splitting
train_df, temp_df = train_test_split(df, test_size=0.2, stratify=df["Label"], random_state=42)
val_df, test_df = train_test_split(temp_df, test_size=0.5, stratify=temp_df["Label"], random_state=42)


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
BATCH_SIZE = 19


train_set = SyllabiDataset(train_df, tokenizer, MAX_LEN, source_to_idx)
val_set = SyllabiDataset(val_df, tokenizer, MAX_LEN, source_to_idx)
test_set = SyllabiDataset(test_df, tokenizer, MAX_LEN, source_to_idx)


train_loader = DataLoader(train_set, sampler=RandomSampler(train_set), batch_size=BATCH_SIZE)
val_loader = DataLoader(val_set, sampler=SequentialSampler(val_set), batch_size=BATCH_SIZE)
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


EPOCHS = 100
LEARNING_RATE = 1.9e-5
optimizer = torch.optim.Adam(params=model.parameters(), lr=LEARNING_RATE)
loss_function = torch.nn.CrossEntropyLoss()


train_losses = []
val_losses = []


# Early stopping parameters
early_stopping_patience = 10
best_val_loss = float('inf')
patience_counter = 0


for epoch in range(EPOCHS):
    model.train()
    train_bar = tqdm(train_loader, desc=f"Epoch {epoch+1} - Training")
    total_train_loss = 0
    for batch in train_bar:
        ids = batch['ids'].to(device)
        mask = batch['mask'].to(device)
        targets = batch['targets'].to(device)


        optimizer.zero_grad()
        outputs = model(ids, mask)
        loss = loss_function(outputs, targets)
        loss.backward()
        optimizer.step()


        total_train_loss += loss.item()
        train_bar.set_postfix(loss=loss.item())


    avg_train_loss = total_train_loss / len(train_loader)
    train_losses.append(avg_train_loss)


    model.eval()
    total_val_loss = 0
    total_correct = 0
    total_samples = 0


    val_bar = tqdm(val_loader, desc=f"Epoch {epoch+1} - Validation")
    for batch in val_bar:
        ids = batch['ids'].to(device)
        mask = batch['mask'].to(device)
        targets = batch['targets'].to(device)


        with torch.no_grad():
            outputs = model(ids, mask)
            loss = loss_function(outputs, targets)
            total_val_loss += loss.item()


            _, predicted = torch.max(outputs, 1)
            total_correct += (predicted == targets).sum().item()
            total_samples += targets.size(0)


            val_bar.set_postfix(loss=loss.item())


    avg_val_loss = total_val_loss / len(val_loader)
    val_losses.append(avg_val_loss)
    accuracy = total_correct / total_samples


    print(f"Epoch: {epoch + 1}, Validation Loss: {avg_val_loss:.4f}, Validation Accuracy: {accuracy:.4f}")


    # Check for early stopping
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        patience_counter = 0
        torch.save(model.state_dict(), '/u/irist_guest/Desktop/x/modelll.pth')
    else:
        patience_counter += 1


    if patience_counter >= early_stopping_patience:
        print("Early stopping triggered")
        break


# Save the losses to a CSV file
loss_data = {
    'epoch': list(range(1, len(train_losses) + 1)),
    'train_loss': train_losses,
    'val_loss': val_losses,
}
loss_df = pd.DataFrame(loss_data)
loss_df.to_csv('model_losses.csv', index=False)


model.eval()
total_test_loss = 0
total_correct = 0
total_samples = 0


test_bar = tqdm(test_loader, desc="Testing")
for batch in test_bar:
    ids = batch['ids'].to(device)
    mask = batch['mask'].to(device)
    targets = batch['targets'].to(device)


    with torch.no_grad():
        outputs = model(ids, mask)
        loss = loss_function(outputs, targets)
        total_test_loss += loss.item()


        _, predicted = torch.max(outputs, 1)
        total_correct += (predicted == targets).sum().item()
        total_samples += targets.size(0)


        test_bar.set_postfix(loss=loss.item())


avg_test_loss = total_test_loss / len(test_loader)
accuracy = total_correct / total_samples


print(f"Test Loss: {avg_test_loss:.4f}, Test Accuracy: {accuracy:.4f}")


# Load the losses from CSV
loss_df = pd.read_csv('model_losses.csv')
train_losses = loss_df['train_loss']
val_losses = loss_df['val_loss']


# Plot loss curves
plt.figure(figsize=(10, 5))
plt.plot(loss_df['epoch'], train_losses, label='Training Loss')
plt.plot(loss_df['epoch'], val_losses, label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.title('Training and Validation Loss Curves')
plt.savefig('/u/irist_guest/Desktop/x/loss_curvesss.png')


# Create TF-IDF Vectorizer
tfidf_vectorizer = TfidfVectorizer()
tfidf_matrix = tfidf_vectorizer.fit_transform(df["Text"])


# Function to generate sentences based on keywords
def generate_sentences(keyword):
    keyword = keyword.lower()
    filtered_df = df[df['Text'].str.contains(keyword, case=False, na=False)]
    sentences = filtered_df['Text'].tolist()
    return sentences


# Function to summarize content for specific keywords
# Function to summarize content by keyword, considering summaries as well
def summarize_content(keyword):
    # Filter the dataframe based on the keyword
    relevant_rows = df[df['Label'].str.contains(keyword, case=False, na=False)]
  
    # Collect all summaries related to the keyword
    all_summaries = relevant_rows['Summary'].tolist()
  
    # If no summaries found, use the text column as a fallback
    if not all_summaries:
        all_texts = relevant_rows['Text'].tolist()
        all_summaries = [text for text in all_texts if text]
  
    # Join all relevant summaries or texts
    summarized_text = ' '.join(all_summaries)
  
    return summarized_text


# Function to answer questions
def answer_question(question):
    question_vec = tfidf_vectorizer.transform([question])
    cosine_similarities = cosine_similarity(question_vec, tfidf_matrix)
    similar_idx = cosine_similarities.argsort()[0][-1]
    return df.iloc[similar_idx]['Text']


# Gradio interface update
def gradio_interface(question):
    if "summary" in question.lower():
        keyword = question.split('summary')[0].strip()
        answer = summarize_content(keyword)
    else:
        answer = answer_question(question)
    return answer


iface = gr.Interface(fn=gradio_interface, inputs="text", outputs="text", title="Keyword-based Q&A")



