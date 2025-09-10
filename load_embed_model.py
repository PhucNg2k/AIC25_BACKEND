import torch
import torch.nn.functional as F
from torch import Tensor
from transformers import AutoTokenizer, AutoModel
from model_loading import device

def average_pool(last_hidden_states: Tensor, attention_mask: Tensor) -> Tensor:
    last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
    return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]

def get_asr_embedding(text: str):
    """Get E5 embedding for a passage chunk using HF Transformers."""
    input_text = f"passage: {text}"  # E5 requires prefix
    batch = tokenizer(
        input_text,
        max_length=512,
        padding=True,
        truncation=True,
        return_tensors='pt'
    )
    batch = {k: v.to(device) for k, v in batch.items()}
    with torch.no_grad():
        outputs = asr_model(**batch)
        embeddings = average_pool(outputs.last_hidden_state, batch['attention_mask'])
        embeddings = F.normalize(embeddings, p=2, dim=1)
    return embeddings[0].cpu().numpy().tolist()


# Initialize E5 tokenizer and model (official usage)
print("Loading ASR model...")
tokenizer = AutoTokenizer.from_pretrained('intfloat/multilingual-e5-base')
asr_model = AutoModel.from_pretrained('intfloat/multilingual-e5-base')

asr_model.to(device)
asr_model.eval()
print("ASR MODEL: ", asr_model.device)