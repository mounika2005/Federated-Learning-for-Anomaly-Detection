import os
import torch
import torch.optim as optim
import numpy as np
from src.utils.data_loader import TrafficLoader
from src.utils.preprocessing import Preprocessor, get_dataloaders
from src.client.model import Autoencoder, get_loss_criterion

def train_model(epochs=10, batch_size=32):
    # 1. Generate & Preprocess Data
    simulator = TrafficLoader(n_samples=2000)
    df = simulator.generate_data()
    
    preprocessor = Preprocessor()
    # Train only on NORMAL data (label 0)
    normal_df = df[df['label'] == 0]
    X_train, y_train = preprocessor.fit_transform(normal_df)
    
    # Validation set including anomalies to test threshold later
    val_df = df # Full dataset
    X_val, y_val = preprocessor.transform(val_df)
    
    train_loader, _ = get_dataloaders(X_train, y_train, batch_size=batch_size)
    
    # 2. Init Model
    input_dim = X_train.shape[1]
    model = Autoencoder(input_dim)
    criterion = get_loss_criterion()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # 3. Training Loop
    print("Starting local training on normal traffic...")
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for data, _ in train_loader:
            optimizer.zero_grad()
            outputs = model(data)
            loss = criterion(outputs, data)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * data.size(0)
            
        train_loss /= len(train_loader.dataset)
        if (epoch+1) % 2 == 0:
            print(f"Epoch {epoch+1}/{epochs}, Loss: {train_loss:.4f}")
            
    # 4. Save Model & Scaler to repository-relative defaults or env overrides
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    default_model_path = os.path.join(project_root, 'src', 'client', 'autoencoder.pth')
    default_scaler_path = os.path.join(project_root, 'src', 'client', 'scaler.pkl')

    model_path = os.environ.get('FLARE_MODEL_PATH', default_model_path)
    scaler_path = os.environ.get('FLARE_SCALER_PATH', default_scaler_path)

    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    preprocessor.save(scaler_path)
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to: {model_path}")
    print(f"Scaler saved to: {scaler_path}")
    
    return model, preprocessor, input_dim

if __name__ == "__main__":
    train_model()
