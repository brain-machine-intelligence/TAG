import torch
import torch.nn as nn


# Ensure you are using MPS if available
try:
    device = torch.device('mps') if torch.backends.mps.is_available() else torch.device('cpu')
except AttributeError:
    device = torch.device('cpu')

# Constants
num_envs = 1000  # Number of environments for training
test_envs = 200  # Number of environments for the test set
grid_size = 10
input_dim = 32  # Final total feature size after transforming vertex and adding block feature
test_interval = 5  # Test every 5 epochs

# Feature Sizes
vertex_dim = 4  # Original size of vertex feature (will be transformed to 8)
border_dim = 4  # Binary feature (either 0 or 1)
place_dim = 100  # Positive float values
grid_dim = 9  # Continuous values between -1 and 1
projected_place_dim = 10  # We project place feature to 10 dimensions to balance with other features
block_dim = 1  # 1 if block, 0 otherwise (binary feature)


# Vertex transformation function (4D -> 8D based on the rules provided)
def transform_corner(vertex):
    # Vertex shape: (num_envs, grid_size, grid_size, vertex_dim)
    batch_size, grid_h, grid_w, _ = vertex.shape
    
    # Create an empty tensor for the transformed vertex (new dimension is 8)
    transformed_vertex = torch.zeros(batch_size, grid_h, grid_w, 8).to(vertex.device)
    
    for i in range(4):
        # If the element is -1, set 1 at index 4 + element_index
        transformed_vertex[:, :, :, 4 + i] = (vertex[:, :, :, i] == -1).float()

        # If the element is 1, set 1 at index element_index
        transformed_vertex[:, :, :, i] = (vertex[:, :, :, i] == 1).float()

    return transformed_vertex


# Define positional encoding function with concatenation
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, height=10, width=10, scale=8.0):
        super(PositionalEncoding, self).__init__()
        self.d_model = d_model

        # Positional encoding matrix for 2D with adjusted scale for 10x10 grid
        pos_encoding = torch.zeros(height, width, d_model)
        for pos_x in range(height):
            for pos_y in range(width):
                for i in range(0, d_model, 2):  # We handle pairs (sin/cos) in the loop
                    div_term = torch.exp(torch.tensor(i, dtype=torch.float32) * -torch.log(torch.tensor(scale)) / d_model)
                    pos_encoding[pos_x, pos_y, i] = torch.sin(pos_x * div_term)
                    if i + 1 < d_model:
                        pos_encoding[pos_x, pos_y, i + 1] = torch.cos(pos_y * div_term)

        # Register buffer to store positional encodings (so they are not treated as parameters)
        self.register_buffer('pos_encoding', pos_encoding.unsqueeze(0))  # Add batch dimension

    def forward(self, x):
        # Concatenate positional encoding to the input (increase embedding size)
        # x has shape (batch_size, height, width, d_model)
        batch_size, height, width, _ = x.shape
        pos_encoding = self.pos_encoding.repeat(batch_size, 1, 1, 1)  # Repeat positional encoding for the batch size
        return torch.cat((x, pos_encoding), dim=-1)  # Concatenate along the last dimension (feature dimension)

# Transformer model with projection for place feature
class TransformerClassifier(nn.Module):
    def __init__(self, input_dim, output_dim, num_layers=6, nhead=8, dim_feedforward=512, dropout=0.1):
        super(TransformerClassifier, self).__init__()

        # Linear projection for 'place' feature to reduce its dimensionality
        self.place_projection = nn.Linear(place_dim, projected_place_dim)

        # Positional encoding
        self.pos_encoder = PositionalEncoding(input_dim, grid_size, grid_size)

        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=input_dim * 2, nhead=nhead, dim_feedforward=dim_feedforward, dropout=dropout, batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Classification head
        self.classifier = nn.Linear(input_dim * 2, output_dim)
        self.output_dim = output_dim

    def forward(self, vertex, border, place, grid, block, return_attn=False):
        # Project 'place' feature to the desired dimension
        place = self.place_projection(place)

        # Concatenate all features along the last dimension, including the block feature
        src = torch.cat([vertex, border, place, grid, block], dim=-1)

        # Add positional encoding
        src = self.pos_encoder(src)

        # Reshape for transformer: (batch_size, sequence_length, embedding_size)
        src = src.flatten(1, 2)  # (num_envs*grid_size*grid_size, 1, input_dim)

        encoded_output = self.transformer_encoder(src)

        # Classification
        output = self.classifier(encoded_output)

        # Transformer encoding with attention weight extraction
        attn_weights_list = []
        if return_attn:
            encoded_output_attn = src
            for layer in self.transformer_encoder.layers:
                # Forward pass through multi-head attention with attention weights
                attn_output, attn_weights = layer.self_attn(
                    encoded_output_attn,
                    encoded_output_attn,
                    encoded_output_attn,
                    need_weights=True,
                )

                # Apply residual connection + layer normalization (part of TransformerEncoderLayer)
                encoded_output_attn = layer.norm1(attn_output + encoded_output_attn)

                # Feed-forward network with dropout and another layer normalization (completing TransformerEncoderLayer operations)
                ff_output = layer.linear2(
                    layer.dropout(layer.activation(layer.linear1(encoded_output_attn)))
                )
                encoded_output_attn = layer.norm2(ff_output + encoded_output_attn)

                # Collect attention weights for this layer
                attn_weights_list.append(attn_weights)
            output_attn = self.classifier(encoded_output_attn)
            assert output_attn.detach().cpu().numpy().all() == output.detach().cpu().numpy().all()

        # Reshape back to original grid shape
        output = output.view(-1, grid_size, grid_size, self.output_dim)

        if return_attn:
            return output, attn_weights_list  # Return both output and attention weights
        else:
            return output  # Only return output


if __name__ == '__main__':
    model = TransformerClassifier(input_dim=input_dim, output_dim=4, num_layers=6, nhead=8)
    model.load_state_dict(
                torch.load("data/outer_loop/model_weights_dict_v_6_8_0.001_10.pth")
            )
    model.to(torch.device('cpu'))
    torch.save(model.state_dict(), "data/outer_loop/model_weights_dict_v_6_8_0.001_10_cpu.pth")
