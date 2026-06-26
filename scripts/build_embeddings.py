"""
Build document embeddings for the ctmatch IR corpus and save as .npy.

Usage:
    # Standard sentence-transformers model
    python build_embeddings.py --model NeuML/pubmedbert-base-embeddings

    # MedCPT (separate article/query encoders, not sentence-transformers compatible)
    python build_embeddings.py --medcpt

    # Custom output path and push to HF Hub
    python build_embeddings.py --medcpt --output doc_embeddings_medcpt.npy --push

Recommended models:
    ncbi/MedCPT-Article-Encoder  (query: ncbi/MedCPT-Query-Encoder)  -- biomedical specialist
    NeuML/pubmedbert-base-embeddings                                   -- good biomedical baseline
    BAAI/bge-base-en-v1.5                                              -- strong general model

Note on MedCPT: docs are encoded with MedCPT-Article-Encoder; at query time the
pipeline must use MedCPT-Query-Encoder (set embedding_model_checkpoint accordingly).
"""

import argparse
import numpy as np
from pathlib import Path
from tqdm import tqdm
from datasets import load_dataset


CTMATCH_IR_DATASET_ROOT = "semaj83/ctmatch_ir"
DOC_TEXTS_PATH = "doc_texts.txt"


def load_doc_texts(local_path=None):
    if local_path and Path(local_path).exists():
        with open(local_path) as f:
            return [line.strip() for line in f if line.strip()]
    print("Loading doc texts from HuggingFace Hub...")
    ds = load_dataset(CTMATCH_IR_DATASET_ROOT, data_files=DOC_TEXTS_PATH, split='train')
    return [row['text'] for row in ds]


def encode_sentence_transformers(model_name, texts, batch_size, normalize):
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(model_name)
    dim = model.get_sentence_embedding_dimension()
    print(f"Encoding {len(texts)} docs with {model_name} (dim={dim})...")
    return model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        normalize_embeddings=normalize,
        convert_to_numpy=True,
    )


def encode_medcpt(texts, batch_size):
    import torch
    from transformers import AutoTokenizer, AutoModel

    model_name = "ncbi/MedCPT-Article-Encoder"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device).eval()
    print(f"Encoding {len(texts)} docs with MedCPT article encoder (dim=768) on {device}...")

    all_embeddings = []
    for i in tqdm(range(0, len(texts), batch_size)):
        batch = texts[i:i + batch_size]
        inputs = tokenizer(batch, return_tensors="pt", truncation=True, padding=True, max_length=512).to(device)
        with torch.no_grad():
            emb = model(**inputs).last_hidden_state[:, 0, :]  # CLS token
            emb = torch.nn.functional.normalize(emb, dim=1)
        all_embeddings.append(emb.cpu().numpy())

    return np.vstack(all_embeddings)


def main():
    parser = argparse.ArgumentParser(description="Build ctmatch document embeddings as .npy")
    parser.add_argument('--model', default='NeuML/pubmedbert-base-embeddings',
                        help='HuggingFace sentence-transformers model name')
    parser.add_argument('--medcpt', action='store_true',
                        help='Use MedCPT article encoder (ignores --model)')
    parser.add_argument('--output', default=None,
                        help='Output .npy path (default: doc_embeddings_<model-slug>.npy)')
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--no-normalize', action='store_true',
                        help='Skip L2 normalization (not recommended)')
    parser.add_argument('--local-texts', default=None,
                        help='Local doc_texts.txt path (otherwise loads from HF Hub)')
    parser.add_argument('--push', action='store_true',
                        help='Push output .npy to HuggingFace Hub after encoding')
    args = parser.parse_args()

    texts = load_doc_texts(args.local_texts)
    print(f"Loaded {len(texts)} documents")

    if args.medcpt:
        embeddings = encode_medcpt(texts, args.batch_size)
        model_slug = 'medcpt'
    else:
        embeddings = encode_sentence_transformers(args.model, texts, args.batch_size, not args.no_normalize)
        model_slug = args.model.split('/')[-1]

    print(f"Embeddings shape: {embeddings.shape}, dtype: {embeddings.dtype}")

    out_path = args.output or f'doc_embeddings_{model_slug}.npy'
    np.save(out_path, embeddings.astype(np.float32))
    size_mb = Path(out_path).stat().st_size / 1e6
    print(f"Saved to {out_path} ({size_mb:.1f} MB)")

    if args.push:
        from huggingface_hub import HfApi
        api = HfApi()
        print(f"Pushing {out_path} to {CTMATCH_IR_DATASET_ROOT}...")
        api.upload_file(
            path_or_fileobj=out_path,
            path_in_repo=out_path,
            repo_id=CTMATCH_IR_DATASET_ROOT,
            repo_type='dataset',
        )
        print("Done.")


if __name__ == '__main__':
    main()
