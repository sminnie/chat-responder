"""LoRA fine-tuning upgrade path (optional; requires the `train` extra).

Frames next-responder prediction as sequence classification over a FIXED role set
(the common case: a stable group). Fine-tunes a small encoder with LoRA, then serves
a softmax over roles that is restricted to the live candidates and renormalized.

Install:  pip install -e ".[train]"
Train:    python -m chatresponder.train_lora --export chat.txt --roles roles.json --out ./lora-out
"""

from __future__ import annotations

import argparse
from pathlib import Path

from .dataset import build_examples, chronological_split
from .parser import parse_whatsapp_export
from .profiles import load_roles

_BASE_MODEL = "distilbert-base-uncased"


def train(
    export_path: str,
    roles_path: str,
    out_dir: str,
    base_model: str = _BASE_MODEL,
    epochs: int = 3,
    window: int = 6,
) -> None:
    import numpy as np
    import torch
    from datasets import Dataset
    from peft import LoraConfig, TaskType, get_peft_model
    from transformers import (
        AutoModelForSequenceClassification,
        AutoTokenizer,
        Trainer,
        TrainingArguments,
    )

    roles = load_roles(roles_path)
    label_names = [r.name for r in roles]
    label_to_id = {name: i for i, name in enumerate(label_names)}

    messages = parse_whatsapp_export(Path(export_path).read_text(encoding="utf-8"))
    examples = [
        ex
        for ex in build_examples(messages, window=window)
        if ex.responder in label_to_id
    ]
    if not examples:
        raise SystemExit("No usable examples: check that roles match chat senders.")
    train_ex, eval_ex = chronological_split(examples)

    tokenizer = AutoTokenizer.from_pretrained(base_model)

    def to_dataset(items):
        return Dataset.from_dict(
            {
                "text": [ex.query_text() for ex in items],
                "label": [label_to_id[ex.responder] for ex in items],
            }
        )

    def tokenize(batch):
        return tokenizer(batch["text"], truncation=True, max_length=256)

    train_ds = to_dataset(train_ex).map(tokenize, batched=True)
    eval_ds = to_dataset(eval_ex).map(tokenize, batched=True) if eval_ex else None

    model = AutoModelForSequenceClassification.from_pretrained(
        base_model,
        num_labels=len(label_names),
        id2label=dict(enumerate(label_names)),
        label2id=label_to_id,
    )
    lora = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=["q_lin", "v_lin"],
    )
    model = get_peft_model(model, lora)

    def metrics(pred):
        logits, labels = pred
        preds = np.argmax(logits, axis=-1)
        return {"accuracy": float((preds == labels).mean())}

    args = TrainingArguments(
        output_dir=out_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=32,
        learning_rate=2e-4,
        eval_strategy="epoch" if eval_ds else "no",
        save_strategy="epoch",
        logging_steps=10,
        use_mps_device=torch.backends.mps.is_available(),
    )
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        compute_metrics=metrics if eval_ds else None,
    )
    trainer.train()
    model.save_pretrained(out_dir)
    tokenizer.save_pretrained(out_dir)
    (Path(out_dir) / "labels.txt").write_text("\n".join(label_names), encoding="utf-8")
    print(f"Saved LoRA adapter + tokenizer to {out_dir}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="chatresponder.train_lora")
    parser.add_argument("--export", required=True)
    parser.add_argument("--roles", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--base-model", default=_BASE_MODEL)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--window", type=int, default=6)
    args = parser.parse_args(argv)
    train(
        export_path=args.export,
        roles_path=args.roles,
        out_dir=args.out,
        base_model=args.base_model,
        epochs=args.epochs,
        window=args.window,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
