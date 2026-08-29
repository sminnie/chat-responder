from chatresponder.dataset import build_examples, chronological_split
from chatresponder.parser import parse_whatsapp_export

SAMPLE = (
    "[8/29/26, 10:15:32 AM] Alice: one\n"
    "[8/29/26, 10:16:00 AM] Bob: two\n"
    "[8/29/26, 10:16:10 AM] Bob: still bob\n"
    "[8/29/26, 10:16:30 AM] Alice: three"
)


def test_build_examples_labels_next_responder():
    msgs = parse_whatsapp_export(SAMPLE)
    examples = build_examples(msgs, window=6)
    # Bob->Bob consecutive is skipped; Alice->Bob and Bob->Alice remain.
    assert [ex.responder for ex in examples] == ["Bob", "Alice"]


def test_query_text_includes_context():
    msgs = parse_whatsapp_export(SAMPLE)
    ex = build_examples(msgs, window=6)[0]
    assert "Alice: one" in ex.query_text()


def test_chronological_split():
    examples = list(range(10))
    train, held = chronological_split(examples, eval_fraction=0.2)
    assert train == list(range(8))
    assert held == [8, 9]
