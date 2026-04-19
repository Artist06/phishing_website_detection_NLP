import pandas as pd
import numpy as np
import pickle
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler

INPUT_FILE = './input_data/input_data.pkl'
MODEL_OUTPUT_FILE = './model/content_ann_model.pkl'
PREDICTIONS_CSV = './model/content_ann_predictions.csv'
N_EPOCHS = 20
BATCH_SIZE = 64
LEARNING_RATE = 0.001
TITLE_VEC_SIZE = 10
CONTENT_VEC_SIZE = 100

class ContentDataset(Dataset):
    def __init__(self, title_vec, content_vec, labels):
        self.title_vec = torch.tensor(title_vec, dtype=torch.float32)
        self.content_vec = torch.tensor(content_vec, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.float32).unsqueeze(1)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.title_vec[idx], self.content_vec[idx], self.labels[idx]

class ContentANN(nn.Module):
    """
    Two-headed model: Head 1 Title Vector (10 features), Head 2 Content Vector (100 features)
    """
    def __init__(self, title_in, content_in, title_out=8, content_out=32):
        super(ContentANN, self).__init__()
        self.title_head = nn.Sequential(
            nn.Linear(title_in, title_out),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        self.content_head = nn.Sequential(
            nn.Linear(content_in, 64),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(64, content_out),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        # Combine
        self.combiner = nn.Sequential(
            nn.Linear(title_out + content_out, 16),
            nn.ReLU(),
            nn.Linear(16, 1) 
        )

    def forward(self, x_title, x_content):
        t = self.title_head(x_title)
        c = self.content_head(x_content)
        combined = torch.cat((t, c), dim=1)
        output = self.combiner(combined)
        return output

def train_epoch(model, dataloader, optimizer, loss_fn, device):
    model.train()
    for x_title, x_content, y_batch in dataloader:
        x_title, x_content, y_batch = x_title.to(device), x_content.to(device), y_batch.to(device)
        
        y_pred_logits = model(x_title, x_content)
        loss = loss_fn(y_pred_logits, y_batch)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

def evaluate_model(model, dataloader, device):
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for x_title, x_content, y_batch in dataloader:
            x_title, x_content, y_batch = x_title.to(device), x_content.to(device), y_batch.to(device)
            
            y_pred_logits = model(x_title, x_content)
            y_pred_probs = torch.sigmoid(y_pred_logits)
            y_pred_class = (y_pred_probs > 0.5).float()
            
            all_preds.extend(y_pred_class.cpu().numpy())
            all_labels.extend(y_batch.cpu().numpy())

    # Flatten lists
    all_preds = [item for sublist in all_preds for item in sublist]
    all_labels = [item for sublist in all_labels for item in sublist]        
    accuracy = accuracy_score(all_labels, all_preds)
    cm = confusion_matrix(all_labels, all_preds)
    report = classification_report(all_labels, all_preds, target_names=['Safe (0)', 'Phishing (1)'], zero_division=0)

    return accuracy, cm, report, all_preds

if __name__ == "__main__":
    try:
        with open(INPUT_FILE, 'rb') as data_file:
            input_data = pickle.load(data_file)
    except Exception as e:
        print(f"Error loading pickle file: {e}")
        exit()
    data = pd.DataFrame(input_data, columns=['index', 'url', 'page_title_vector', 'page_content_vector', 'label'])
    data.drop(columns=['index'], inplace=True)
    
    train_df, test_df = train_test_split(
        data,
        test_size=0.2,
        random_state=42,
        stratify=data['label']
    )
    title_train = np.stack(train_df['page_title_vector'].values)
    content_train = np.stack(train_df['page_content_vector'].values)
    y_train = train_df['label'].values
    
    title_test = np.stack(test_df['page_title_vector'].values)
    content_test = np.stack(test_df['page_content_vector'].values)
    y_test = test_df['label'].values

    title_scaler = StandardScaler()
    title_train = title_scaler.fit_transform(title_train)
    title_test = title_scaler.transform(title_test)
    
    content_scaler = StandardScaler()
    content_train = content_scaler.fit_transform(content_train)
    content_test = content_scaler.transform(content_test)
    
    print(f"Data scaled and split. Training on {len(y_train)}, testing on {len(y_test)}.")
    train_dataset = ContentDataset(title_train, content_train, y_train)
    test_dataset = ContentDataset(title_test, content_test, y_test)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    model = ContentANN(TITLE_VEC_SIZE, CONTENT_VEC_SIZE).to(device)
    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    print(f"Starting training for {N_EPOCHS} epochs")
    for epoch in range(N_EPOCHS):
        train_epoch(model, train_loader, optimizer, loss_fn, device)
        if (epoch + 1) % 2 == 0:
            acc, cm, report, _ = evaluate_model(model, test_loader, device)
            print(f"\nEpoch {epoch + 1}/{N_EPOCHS}")
            print(f"Test Accuracy: {acc * 100:.2f}%")

    #Final Evaluation
    print("FINAL MODEL EVALUATION (PyTorch ANN)")
    acc, cm, report, final_preds = evaluate_model(model, test_loader, device)
    
    print(f"Final Test Accuracy: {acc * 100:.2f}%")
    print("\nFinal Confusion Matrix:")
    print(cm)
    if cm.shape == (2,2):
        print(f"False Negatives: {cm[1][0]}")
    print("\nFinal Classification Report:")
    print(report)
    
    print(f"Saving trained model to '{MODEL_OUTPUT_FILE}'...")
    torch.save(model.state_dict(), MODEL_OUTPUT_FILE)
    print(f"Saving predictions to '{PREDICTIONS_CSV}'...")
    df_predictions = pd.DataFrame({
        'url': test_df['url'],
        'actual_label': test_df['label'],
        'predicted_label': final_preds
    })
    df_predictions.to_csv(PREDICTIONS_CSV, index=False)
    print("Predictions saved successfully.")