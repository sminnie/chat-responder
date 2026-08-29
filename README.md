# chat-responder (MVP)

Real-time group-chat responder scoring. Given a message, output a probability
distribution over **who should respond next** — e.g. `Bob 0.70, Carol 0.30`
(always sums to 1). Zero training required to start.

## How it works

1. **Parse** a WhatsApp export into structured messages ([parser.py](src/chatresponder/parser.py)).
2. **Encode** the recent conversation once with a small sentence-transformer, score
   it against cached per-role profile embeddings, and softmax over candidates
   ([scorer.py](src/chatresponder/scorer.py)). Group-size independent, low latency.
3. **Serve** live via a rolling window that excludes the sender
   ([engine.py](src/chatresponder/engine.py)), either headless or through a
   Slack-style browser UI ([ui.py](src/chatresponder/ui.py)).

Ground truth for evaluation is free: **who actually replied next** in the chat
history ([dataset.py](src/chatresponder/dataset.py)).

## Setup

```bash
cd chat-responder-mvp
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
```

## Run

```bash
# Slack/WhatsApp-style chat UI in the browser (stdlib server, no extra deps)
chat-responder --roles examples/roles.json ui

# Replay the sample chat live, showing suggested responders per message
chat-responder --roles examples/roles.json simulate examples/sample_chat.txt

# Evaluate against next-responder ground truth (top-1 accuracy, recall@2)
chat-responder --roles examples/roles.json eval examples/sample_chat.txt

# Interactive: type "Sender: text" lines on stdin
chat-responder --roles examples/roles.json predict
```

In the UI you pick who is speaking from the sidebar and send a message. A
**Responder bot** sits in the channel and replies inline — "Bob, this one looks
like yours (57% confident). Otherwise: Carol 43%." — with the full distribution
shown as chips. The sender auto-advances to the top pick so a conversation flows
naturally. Use `--seed examples/sample_chat.txt` to preload a transcript,
`--port` to change the port, and `--no-open` to skip launching a browser.

Define your group in a roles JSON (see [examples/roles.json](examples/roles.json)):
one entry per participant with a short profile describing their role.

## Test

```bash
pytest            # runs without any model download (uses a fake encoder)
```

## Upgrade path: LoRA fine-tuning

The cosine baseline needs no data. Once you have chat history, fine-tune a small
encoder with LoRA for better accuracy ([train_lora.py](src/chatresponder/train_lora.py)):

```bash
pip install -e ".[train]"
python -m chatresponder.train_lora --export chat.txt --roles examples/roles.json --out ./lora-out
```

## Deferred (post-MVP)

- `needs_response` and `response_type` (support / like / thinking / answer) heads
- ONNX int8 export + latency benchmark harness (target p95 ≤ 50 ms)
- Live platform connectors (Discord/Slack first; WhatsApp bridge opt-in)
