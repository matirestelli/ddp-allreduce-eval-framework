"""
Tokenize a text corpus into nanoGPT-style uint16 .bin files.

Run ONCE on a Polaris LOGIN node (compute nodes have no internet):

    pip install --user tiktoken datasets numpy
    python prepare_tokens.py --dataset openwebtext --out ./data/openwebtext
    python prepare_tokens.py --dataset tinystories --out ./data/tinystories

Output (identical format to nanoGPT's data/openwebtext/prepare.py):
    <out>/train.bin   uint16 GPT-2 BPE token ids, documents separated by EOT
    <out>/val.bin
    <out>/meta.json   token counts, vocab size

Sizes (approx):
    openwebtext  ~9.0B train tokens, train.bin ~17 GB, HF download ~54 GB
                 -> paper / nanoGPT default; put both cache and output on eagle
    tinystories  ~0.47B tokens, train.bin ~0.9 GB, much faster to stage
                 -> fine for the communication study: the gradient size is
                    set by the model and the vocab, not by the corpus

Use --max-docs to make a quick subset for smoke tests.
"""

import argparse
import json
import os

import numpy as np

SOURCES = {
    # name: (hf_path, hf_config, text_field, has_val_split)
    "openwebtext": ("Skylion007/openwebtext", None, "text", False),
    "tinystories": ("roneneldan/TinyStories", None, "text", True),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=sorted(SOURCES), required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--num-proc", type=int, default=16)
    ap.add_argument("--max-docs", type=int, default=None)
    ap.add_argument("--val-fraction", type=float, default=0.0005)
    ap.add_argument("--seed", type=int, default=2357)
    args = ap.parse_args()

    import tiktoken
    from datasets import load_dataset

    hf_path, hf_cfg, field, has_val = SOURCES[args.dataset]
    enc = tiktoken.get_encoding("gpt2")
    eot = enc.eot_token  # 50256

    ds = load_dataset(hf_path, hf_cfg, num_proc=args.num_proc)
    if has_val:
        splits = {"train": ds["train"], "val": ds["validation"]}
    else:
        sp = ds["train"].train_test_split(test_size=args.val_fraction,
                                          seed=args.seed, shuffle=True)
        splits = {"train": sp["train"], "val": sp["test"]}

    if args.max_docs:
        splits = {k: v.select(range(min(len(v), args.max_docs if k == "train"
                                        else max(args.max_docs // 100, 100))))
                  for k, v in splits.items()}

    def tok(ex):
        ids = enc.encode_ordinary(ex[field])
        ids.append(eot)
        return {"ids": ids, "len": len(ids)}

    os.makedirs(args.out, exist_ok=True)
    meta = {"tokenizer": "gpt2", "vocab_size": enc.n_vocab, "eot": eot}

    for split, dset in splits.items():
        dset = dset.map(tok, remove_columns=dset.column_names,
                        desc=f"tokenizing {split}", num_proc=args.num_proc)
        total = int(np.sum(dset["len"], dtype=np.uint64))
        path = os.path.join(args.out, f"{split}.bin")
        arr = np.memmap(path, dtype=np.uint16, mode="w+", shape=(total,))
        n_shards = min(1024, max(1, len(dset)))
        idx = 0
        for s in range(n_shards):
            batch = dset.shard(num_shards=n_shards, index=s,
                               contiguous=True).with_format("numpy")
            chunk = np.concatenate(batch["ids"]).astype(np.uint16)
            arr[idx: idx + len(chunk)] = chunk
            idx += len(chunk)
        arr.flush()
        meta[f"{split}_tokens"] = total
        print(f"{split}: {total:,} tokens -> {path}")

    with open(os.path.join(args.out, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)


if __name__ == "__main__":
    main()
    

#python prepare_tokens.py --dataset tinystories --num-proc 8 --out /lus/eagle/projects/UIC-HPC/mrest/mpiAllReduceCompressed/data/tinystories