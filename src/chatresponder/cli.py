"""Command-line interface: simulate, evaluate, and one-shot predict."""

from __future__ import annotations

import argparse
import sys
import webbrowser
from pathlib import Path

from .dataset import build_examples, chronological_split
from .engine import RealtimeEngine
from .evaluate import evaluate
from .parser import parse_whatsapp_export
from .profiles import load_roles
from .scorer import ResponderScorer
from .simulator import format_line, replay
from .ui import ChatSession, create_server


def _build_scorer(roles_path: str, temperature: float) -> ResponderScorer:
    roles = load_roles(roles_path)
    return ResponderScorer(roles, temperature=temperature)


def _cmd_simulate(args: argparse.Namespace) -> int:
    scorer = _build_scorer(args.roles, args.temperature)
    engine = RealtimeEngine(scorer, window=args.window)
    text = Path(args.export).read_text(encoding="utf-8")
    for msg, ranked in replay(engine, parse_whatsapp_export(text), top_k=args.top_k):
        print(format_line(msg, ranked))
    return 0


def _cmd_eval(args: argparse.Namespace) -> int:
    scorer = _build_scorer(args.roles, args.temperature)
    text = Path(args.export).read_text(encoding="utf-8")
    examples = build_examples(parse_whatsapp_export(text), window=args.window)
    _, held_out = chronological_split(examples, eval_fraction=args.eval_fraction)
    print(evaluate(scorer, held_out).as_table())
    return 0


def _cmd_predict(args: argparse.Namespace) -> int:
    scorer = _build_scorer(args.roles, args.temperature)
    engine = RealtimeEngine(scorer, window=args.window)
    for line in sys.stdin:
        line = line.strip()
        if not line or ":" not in line:
            continue
        sender, text = line.split(":", 1)
        dist = engine.add_message(sender.strip(), text.strip())
        ranked = sorted(dist.items(), key=lambda kv: kv[1], reverse=True)
        print(", ".join(f"{n} {s:.2f}" for n, s in ranked))
    return 0


def _cmd_ui(args: argparse.Namespace) -> int:
    scorer = _build_scorer(args.roles, args.temperature)
    session = ChatSession(RealtimeEngine(scorer, window=args.window), top_k=args.top_k)
    if args.seed:
        for msg in parse_whatsapp_export(Path(args.seed).read_text(encoding="utf-8")):
            session.send(msg.sender, msg.text)
    server = create_server(session, args.host, args.port)
    url = f"http://{args.host}:{server.server_address[1]}"
    print(f"chat UI on {url}  (ctrl-c to stop)")
    if args.open:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="chat-responder")
    parser.add_argument("--roles", required=True, help="path to roles JSON")
    parser.add_argument("--window", type=int, default=6)
    parser.add_argument("--temperature", type=float, default=0.1)
    sub = parser.add_subparsers(dest="command", required=True)

    p_sim = sub.add_parser("simulate", help="replay a WhatsApp export live")
    p_sim.add_argument("export")
    p_sim.add_argument("--top-k", type=int, default=3)
    p_sim.set_defaults(func=_cmd_simulate)

    p_eval = sub.add_parser("eval", help="evaluate against next-responder ground truth")
    p_eval.add_argument("export")
    p_eval.add_argument("--eval-fraction", type=float, default=0.2)
    p_eval.set_defaults(func=_cmd_eval)

    p_pred = sub.add_parser("predict", help="read 'Sender: text' lines from stdin")
    p_pred.set_defaults(func=_cmd_predict)

    p_ui = sub.add_parser("ui", help="serve a Slack-style chat UI in the browser")
    p_ui.add_argument("--host", default="127.0.0.1")
    p_ui.add_argument("--port", type=int, default=8000)
    p_ui.add_argument("--top-k", type=int, default=3)
    p_ui.add_argument("--seed", help="WhatsApp export to preload into the transcript")
    p_ui.add_argument("--no-open", dest="open", action="store_false")
    p_ui.set_defaults(func=_cmd_ui, open=True)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
