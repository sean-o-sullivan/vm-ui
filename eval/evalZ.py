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
        x = self.input_proj(x).unsqueeze(0) 
        
        x = self.transformer_encoder(x)
        x = x.squeeze(0)  
        x = self.output_proj(x)
        x = self.norm(x)
        return x, F.normalize(x, p=2, dim=1)  # Return both raw and normalized embeddings

class SiameseTransformerNetwork(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(SiameseTransformerNetwork, self).__init__()
        self.encoder = TransformerEncoder(input_size, hidden_size)

    def forward(self, x1, x2):
        raw_out1, norm_out1 = self.encoder(x1)
        raw_out2, norm_out2 = self.encoder(x2)
        return raw_out1, raw_out2, norm_out1, norm_out2

def load_model(model_path, input_size, hidden_size):
    siamese_net = SiameseTransformerNetwork(input_size, hidden_size).to(device)
    checkpoint = torch.load(model_path, map_location=device)
    siamese_net.load_state_dict(checkpoint['model_state_dict'])
    siamese_net.eval()
    threshold = 0.5
    return siamese_net, threshold

global_model = None
global_threshold = None

def predict_author(focus_context_embedding, focus_check_embedding):
    global global_model, global_threshold
    
    if global_model is None:
        input_size = 112
        hidden_size = 256
        current_dir = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(current_dir, "BnG_10_best_transformer_siamese_model.pth")
        global_model, global_threshold = load_model(model_path, input_size, hidden_size)

    context_tensor = torch.tensor(focus_context_embedding, dtype=torch.float32).unsqueeze(0).to(device)
    check_tensor = torch.tensor(focus_check_embedding, dtype=torch.float32).unsqueeze(0).to(device)

    with torch.no_grad():
        _, _, norm_context, norm_check = global_model(context_tensor, check_tensor)
        distance = F.pairwise_distance(norm_context, norm_check).item()
        is_same_author = distance < global_threshold
        confidence = 1 - (distance / global_threshold) if is_same_author else (distance / global_threshold) - 1
        confidence = max(0, min(confidence, 1))  # Clamp confidence between 0 and 1

        #now we start in the making of the other standardised samples

    return is_same_author, distance, confidence, norm_context.cpu().numpy().tolist(),  norm_check.cpu().numpy().tolist()