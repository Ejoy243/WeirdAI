import torch
from torch.utils.data import DataLoader, Subset

from weird_ai.model import WeirdAIModel
from weird_ai.tokenizer import SimpleCharacterTokenizer
from weird_ai.config import PROJECT_ROOT, SAMPLE_LYRICS_FILE
from weird_ai.dataset import LyricsDataset
from weird_ai.trainer import train_model_simple, save_checkpoint

CONTEXT_LENGTH = 128   
CHECKPOINT_PATH = PROJECT_ROOT / "models" / "lesson-05-pretrained" / "checkpoint.pt"

def main():
    print("Starting training...")

    text = SAMPLE_LYRICS_FILE.read_text(encoding="utf-8")
    tokenizer = SimpleCharacterTokenizer(text)

    vocab_size = len(tokenizer.stoi)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = WeirdAIModel(vocab_size=vocab_size)

    split_idx = int(0.90 * len(text))
    train_data = LyricsDataset(tokenizer.encode(text[:split_idx]), CONTEXT_LENGTH)
    val_data = LyricsDataset(tokenizer.encode(text[split_idx:]), CONTEXT_LENGTH)

    train_data = Subset(train_data, range(0, len(train_data), CONTEXT_LENGTH))
    val_data = Subset(val_data, range(0, len(val_data), CONTEXT_LENGTH))

    torch.manual_seed(123)
    train_loader = DataLoader(train_data, batch_size=16, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_data, batch_size=16, shuffle=False)

    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0004, weight_decay=0.1)

    num_epochs = 3
    train_losses, val_losses, tokens_seen = train_model_simple(
        model, train_loader, val_loader, optimizer, device,
        num_epochs=num_epochs, eval_freq=100, eval_iter=5,
        start_context="love is", tokenizer=tokenizer, context_size=CONTEXT_LENGTH,
    )

    CHECKPOINT_PATH.parent.mkdir(parents=True, exist_ok=True)
    save_checkpoint(model, optimizer, num_epochs, train_losses, val_losses, tokens_seen, CHECKPOINT_PATH)

   
    print("Training complete.")

if __name__ == "__main__":
    main()