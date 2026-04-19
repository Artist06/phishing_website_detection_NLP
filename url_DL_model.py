import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler

CSV_FILE = './Datasets/url_features_extracted.csv'
MAX_LEN = 200 
N_HEURISTIC_FEATURES = 21
EMBEDDING_DIM = 32
N_EPOCHS = 10
BATCH_SIZE = 64
LEARNING_RATE = 0.001
PADDING_IDX = 0

class PhishingDataset(Dataset):
    def __init__(self, dataframe, char_cols, heuristic_cols, scaler=None):
        char_data = dataframe[char_cols].fillna(PADDING_IDX).values
        char_data = np.asarray(char_data, dtype=np.int64)
        if (char_data < 0).any():
            print("[Dataset] Warning: negative indices found in char data; replacing with PADDING_IDX (0).")
            char_data[char_data < 0] = PADDING_IDX
        self.char_seq = torch.tensor(char_data, dtype=torch.long)

        h_features = dataframe[heuristic_cols].values
        if scaler is not None:
            h_features = scaler.transform(h_features)
        self.heuristic = torch.tensor(h_features, dtype=torch.float32)

        self.labels = torch.tensor(dataframe['label'].values, dtype=torch.float32).unsqueeze(1)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.char_seq[idx], self.heuristic[idx], self.labels[idx]

class HybridModel(nn.Module):
    """CNN : Char seq , FC layers : heuristic features"""
    def __init__(self, vocab_size, embed_dim, seq_len, heuristic_in_features,
                 conv_out_features=64, heuristic_out_features=16, padding_idx=0):
        super(HybridModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=padding_idx)

        self.conv1 = nn.Conv1d(in_channels=embed_dim, out_channels=64, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool1d(2)

        self.conv2 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool1d(2)

        self.flatten = nn.Flatten()
        flat_size = 128 * (seq_len // 4)
        self.cnn_fc_out = nn.Linear(flat_size, conv_out_features)
        self.cnn_relu_out = nn.ReLU()

        self.heuristic_fc = nn.Linear(heuristic_in_features, heuristic_out_features)
        self.heuristic_relu = nn.ReLU()

        self.combiner_fc1 = nn.Linear(conv_out_features + heuristic_out_features, 64)
        self.combiner_relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        self.output_fc = nn.Linear(64, 1)

    def forward(self, x_char_seq, x_heuristic):
        device = self.embedding.weight.device
        if x_char_seq.dtype != torch.long:
            x_char_seq = x_char_seq.long()
        x_char_seq = x_char_seq.to(device)
        x_heuristic = x_heuristic.to(device)

        min_idx = int(x_char_seq.min().item())
        max_idx = int(x_char_seq.max().item())
        if min_idx < 0 or max_idx >= self.embedding.num_embeddings:
            raise ValueError(
                f"[Embedding index error] char indices out of range: min={min_idx}, max={max_idx}. "
                f"Embedding expects indices in [0 .. {self.embedding.num_embeddings - 1}]."
            )

        # Embedding -> Conv pipeline
        x_embed = self.embedding(x_char_seq)  # [batch, seq_len, embed_dim]
        x_cnn = x_embed.permute(0, 2, 1)      # [batch, embed_dim, seq_len]
        x_cnn = self.pool1(self.relu1(self.conv1(x_cnn)))
        x_cnn = self.pool2(self.relu2(self.conv2(x_cnn)))
        x_cnn = self.flatten(x_cnn)
        x_cnn = self.cnn_relu_out(self.cnn_fc_out(x_cnn))
        # Heuristic head
        x_heu = self.heuristic_relu(self.heuristic_fc(x_heuristic))
        # Combine
        x_combined = torch.cat((x_cnn, x_heu), dim=1)
        x_combined = self.combiner_relu(self.combiner_fc1(x_combined))
        x_combined = self.dropout(x_combined)
        output = self.output_fc(x_combined)
        return output

def train_epoch(model, dataloader, optimizer, loss_fn, device):
    model.train()
    running_loss = 0.0
    for x_char, x_heu, y_batch in dataloader:
        x_char, x_heu, y_batch = x_char.to(device), x_heu.to(device), y_batch.to(device)
        y_pred_logits = model(x_char, x_heu)
        loss = loss_fn(y_pred_logits, y_batch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * x_char.size(0)
    return running_loss / len(dataloader.dataset)

def evaluate_model(model, dataloader, loss_fn, device):
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for x_char, x_heu, y_batch in dataloader:
            x_char, x_heu, y_batch = x_char.to(device), x_heu.to(device), y_batch.to(device)
            y_pred_logits = model(x_char, x_heu)
            y_pred_probs = torch.sigmoid(y_pred_logits)
            y_pred_class = (y_pred_probs > 0.5).float()
            all_preds.extend(y_pred_class.cpu().numpy())
            all_labels.extend(y_batch.cpu().numpy())

    accuracy = accuracy_score(all_labels, all_preds)
    cm = confusion_matrix(all_labels, all_preds)
    report = classification_report(all_labels, all_preds, target_names=['Safe (0)', 'Phishing (1)'])
    return accuracy, cm, report

if __name__ == "__main__":
    print(f"Loading data from '{CSV_FILE}'")
    try:
        data = pd.read_csv(CSV_FILE)
    except FileNotFoundError:
        print(f"Error: The file '{CSV_FILE}' was not found.")
        raise SystemExit(1)

    data.dropna(inplace=True)
    char_cols = [f'c_{i+1}' for i in range(MAX_LEN)]
    for c in char_cols:
        if c not in data.columns:
            print(f"[Warning] Column {c} not found in CSV — filling with PADDING_IDX (0).")
            data[c] = PADDING_IDX
    all_cols = data.columns.tolist()
    heuristic_cols = [col for col in all_cols if col not in char_cols and col not in ['url', 'label']]
    if len(heuristic_cols) != N_HEURISTIC_FEATURES:
        print(f"[Warning] Found {len(heuristic_cols)} heuristic features (expected {N_HEURISTIC_FEATURES}). Using found count.")
        N_HEURISTIC_FEATURES = len(heuristic_cols)

    char_df = data[char_cols].apply(pd.to_numeric, errors='coerce').fillna(PADDING_IDX).astype(int)
    data[char_cols] = char_df 
    char_min = int(char_df.values.min())
    char_max = int(char_df.values.max())
    print(f"[Diagnostics] Raw char index range in CSV (min max): {char_min} {char_max}")

    if char_min < 0:
        print("[Diagnostics] Negative indices detected, mapping negatives to PADDING_IDX (0).")
        data[char_cols] = data[char_cols].clip(lower=PADDING_IDX)
    required_vocab_size = int(data[char_cols].values.max()) + 1
    if required_vocab_size <= PADDING_IDX:
        required_vocab_size = PADDING_IDX + 2 
    print(f"[Diagnostics] Setting vocab_size dynamically to: {required_vocab_size}")

    train_df, test_df = train_test_split(
        data,
        test_size=0.2,
        random_state=42,
        stratify=data['label']
    )

    scaler = StandardScaler()
    scaler.fit(train_df[heuristic_cols])

    train_dataset = PhishingDataset(train_df, char_cols, heuristic_cols, scaler)
    test_dataset = PhishingDataset(test_df, char_cols, heuristic_cols, scaler)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    print(f"Data ready. Training on {len(train_dataset)} samples, testing on {len(test_dataset)} samples.")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    model = HybridModel(
        vocab_size=required_vocab_size,
        embed_dim=EMBEDDING_DIM,
        seq_len=MAX_LEN,
        heuristic_in_features=N_HEURISTIC_FEATURES,
        padding_idx=PADDING_IDX
    ).to(device)

    assert train_dataset.char_seq.dtype == torch.long, "char_seq must be torch.long"
    if train_dataset.char_seq.min().item() < 0:
        raise ValueError("Negative char index found after preprocessing.")
    if train_dataset.char_seq.max().item() >= model.embedding.num_embeddings:
        raise ValueError(
            f"Maximum char index {int(train_dataset.char_seq.max().item())} >= embedding vocab {model.embedding.num_embeddings}. "
            "This should not happen after dynamic vocab sizing."
        )
    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # Training loop
    for epoch in range(N_EPOCHS):
        train_loss = train_epoch(model, train_loader, optimizer, loss_fn, device)
        acc, cm, report = evaluate_model(model, test_loader, loss_fn, device)
        print(f"\nEpoch {epoch + 1}/{N_EPOCHS}")
        print(f"Train Loss: {train_loss:.6f}")
        print(f"Test Accuracy: {acc * 100:.2f}%")
        print("Test Confusion Matrix:\n", cm)
        if cm.shape == (2,2):
            print(f"False Negatives: {cm[1][0]}")

    # Final evaluation
    print("FINAL MODEL EVALUATION")
    acc, cm, report = evaluate_model(model, test_loader, loss_fn, device)
    print(f"Final Test Accuracy: {acc * 100:.2f}%")
    print("\nFinal Confusion Matrix:")
    print(cm)
    print("\nFinal Classification Report:")
    print(report)
