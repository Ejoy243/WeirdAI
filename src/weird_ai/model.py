import torch
import torch.nn as nn
from weird_ai.transformer import TransformerBlock
from weird_ai.layer_norm import LayerNorm

class WeirdAIModel(nn.Module):
    def __init__(self, vocab_size, num_layers = 4):
        super().__init__()

        # TODO:
        # Create embeddings
        # Create attention layers
        # Create output head

        self.tok_emb = nn.Embedding(vocab_size, 128)
        self.pos_emb = nn.Embedding(128, 128)
        self.drop = nn.Dropout(0.1)

        self.blocks = nn.Sequential(*[
            TransformerBlock(128, 128, 4, 0.1)
            for _ in range(num_layers)
        ])

        self.final_norm = LayerNorm(128)
        self.out_head = nn.Linear(128, vocab_size, bias=False)



    def forward(self, x):
        # TODO:
        # Implement forward pass

        batch_size, seq_len = x.shape

        tok = self.tok_emb(x)
        pos = self.pos_emb(torch.arange(seq_len, device=x.device))
        x = self.drop(tok + pos)

        x = self.blocks(x)
        x = self.final_norm(x)
        return self.out_head(x)
        
