import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler

INPUT_FILE = './Datasets/hyperlink_features_with_pct.csv'
N_EPOCHS = 20 
BATCH_SIZE = 64
LEARNING_RATE = 0.001


class HyperlinkDataset(Dataset):
    def __init__(self, X_data, y_data):
        self.X_data = torch.tensor(X_data, dtype=torch.float32)
        self.y_data = torch.tensor(y_data, dtype=torch.float32).unsqueeze(1)
        
    def __len__(self):
        return len(self.y_data)

    def __getitem__(self, idx):
        return self.X_data[idx], self.y_data[idx]

class SimpleANN(nn.Module):
    """  Architecture: Input -> 64 -> 32 -> 1 (Output) """
    def __init__(self, num_features):
        super(SimpleANN, self).__init__()
        
        self.layer_1 = nn.Linear(num_features, 64)
        self.relu_1 = nn.ReLU()
        
        self.layer_2 = nn.Linear(64, 32)
        self.relu_2 = nn.ReLU()
        self.dropout = nn.Dropout(0.3)
        
        self.output_layer = nn.Linear(32, 1)

    def forward(self, x):
        x = self.relu_1(self.layer_1(x))
        x = self.relu_2(self.layer_2(x))
        x = self.dropout(x)
        x = self.output_layer(x)
        return x


def train_epoch(model, dataloader, optimizer, loss_fn, device):
    model.train()
    for X_batch, y_batch in dataloader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        
        y_pred_logits = model(X_batch)
        loss = loss_fn(y_pred_logits, y_batch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

def evaluate_model(model, dataloader, device):
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for X_batch, y_batch in dataloader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            
            y_pred_logits = model(X_batch)

            y_pred_probs = torch.sigmoid(y_pred_logits)
            y_pred_class = (y_pred_probs > 0.5).float()
            
            all_preds.extend(y_pred_class.cpu().numpy())
            all_labels.extend(y_batch.cpu().numpy())
    accuracy = accuracy_score(all_labels, all_preds)
    cm = confusion_matrix(all_labels, all_preds)
    report = classification_report(all_labels, all_preds, target_names=['Safe (0)', 'Phishing (1)'], zero_division=0)
    
    return accuracy, cm, report

if __name__ == "__main__":
    try:
        data = pd.read_csv(INPUT_FILE)
    except FileNotFoundError:
        print(f"Error: The file '{INPUT_FILE}' was not found.")
        exit()
        
    data.dropna(inplace=True)
    
    y = data['label']
    X = data.drop(['label', 'url'], axis=1)
    
    num_features = X.shape[1]
    print(f"Found {num_features} features.")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    train_dataset = HyperlinkDataset(X_train_scaled, y_train.values)
    test_dataset = HyperlinkDataset(X_test_scaled, y_test.values)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    print(f"Data ready. Training on {len(train_dataset)} samples, testing on {len(test_dataset)} samples.")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    model = SimpleANN(num_features=num_features).to(device)

    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    print(f"Starting training for {N_EPOCHS} epochs")
    for epoch in range(N_EPOCHS):
        train_epoch(model, train_loader, optimizer, loss_fn, device)

        acc, cm, report = evaluate_model(model, test_loader, device)
        print(f"\nEpoch {epoch + 1}/{N_EPOCHS}")
        print(f"Test Accuracy: {acc * 100:.2f}%")
        print(f"False Negatives: {cm[1][0] if cm.shape == (2,2) else 'N/A'}")

    print("FINAL MODEL EVALUATION (PyTorch ANN)")
    acc, cm, report = evaluate_model(model, test_loader, device)
    
    print(f"Final Test Accuracy: {acc * 100:.2f}%")
    print("\nFinal Confusion Matrix:")
    print(cm)
    print("\nFinal Classification Report:")
    print(report)
