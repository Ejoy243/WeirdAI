from torch import nn as nn
from weird_ai import attention
from weird_ai import feed_forward
from weird_ai import layer_norm
class TransformerBlock(nn.Module):
   

    def __init__(self, emb_dim, context_length, num_heads, dropout=0.0, qkv_bias=False):
        super().__init__()

        self.num_heads = num_heads

        self.norm1 = layer_norm.LayerNorm(emb_dim)
        self.norm2 = layer_norm.LayerNorm(emb_dim)
        self.att = attention.SelfAttention(embedding_dim = emb_dim, output_dim=emb_dim, qkv_bias=qkv_bias,)
        self.ff = feed_forward.FeedForward(emb_dim)

    def forward( self, x):
        shortcut = x
        x = self.norm1(x)
        x, _ = self.att(x)
        x = x + shortcut

        shortcut = x
        x = self.norm2(x)
        x = self.ff(x)
        x = x + shortcut
        return x


    
