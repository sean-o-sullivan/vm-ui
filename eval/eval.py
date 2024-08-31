import torch
import torch.nn as nn
import torch.nn.functional as F
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class TransformerEncoder(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=2, num_heads=4, dropout=0.1):
        super(TransformerEncoder, self).__init__()
        self.input_proj = nn.Linear(input_size, hidden_size)
        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_size, nhead=num_heads, dim_feedforward=hidden_size*4, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.output_proj = nn.Linear(hidden_size, hidden_size // 2)
        self.norm = nn.LayerNorm(hidden_size // 2)

    def forward(self, x):
        x = self.input_proj(x).unsqueeze(0)  # Add sequence dimension
        x = self.transformer_encoder(x)
        x = x.squeeze(0)  # Remove sequence dimension
        x = self.output_proj(x)
        x = self.norm(x)
        return F.normalize(x, p=2, dim=1)  # L2 normalization

class SiameseTransformerNetwork(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(SiameseTransformerNetwork, self).__init__()
        self.encoder = TransformerEncoder(input_size, hidden_size)

    def forward(self, x1, x2):
        out1 = self.encoder(x1)
        out2 = self.encoder(x2)
        return out1, out2

def load_model(model_path, input_size, hidden_size):
    siamese_net = SiameseTransformerNetwork(input_size, hidden_size).to(device)
    checkpoint = torch.load(model_path, map_location=device)
    siamese_net.load_state_dict(checkpoint['model_state_dict'])
    siamese_net.eval()
    # threshold = checkpoint['threshold']
    threshold = 0.5
    return siamese_net, threshold

# Global variables to store the loaded model and threshold
global_model = None
global_threshold = None

def predict_author(focus_context_embedding, focus_check_embedding):
    global global_model, global_threshold
    
    # Load the model if it hasn't been loaded yet
    if global_model is None:
        input_size = 112
        hidden_size = 256
        current_dir = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(current_dir, "BnG_10_best_transformer_siamese_model.pth")
        global_model, global_threshold = load_model(model_path, input_size, hidden_size)
    
    # Convert embeddings to tensors
    context_tensor = torch.tensor(focus_context_embedding, dtype=torch.float32).unsqueeze(0).to(device)
    check_tensor = torch.tensor(focus_check_embedding, dtype=torch.float32).unsqueeze(0).to(device)
    
    # Predict authorship
    with torch.no_grad():
        context_out, check_out = global_model(context_tensor, check_tensor)
        distance = F.pairwise_distance(context_out, check_out).item()
        is_same_author = distance < global_threshold
        confidence = 1 - (distance / global_threshold) if is_same_author else (distance / global_threshold) - 1
        confidence = max(0, min(confidence, 1))  # Clamp confidence between 0 and 1
    
    return is_same_author, distance, confidence