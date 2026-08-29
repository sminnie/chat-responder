from datetime import datetime

from chatresponder.parser import parse_whatsapp_export


def test_parses_ios_format():
    text = "[8/29/26, 10:15:32 AM] Alice: Hello there"
    msgs = parse_whatsapp_export(text)
    assert len(msgs) == 1
    assert msgs[0].sender == "Alice"
    assert msgs[0].text == "Hello there"
    assert isinstance(msgs[0].timestamp, datetime)


def test_parses_android_format():
    text = "8/29/26, 10:15 - Bob: hey"
    msgs = parse_whatsapp_export(text)
    assert len(msgs) == 1
    assert msgs[0].sender == "Bob"
    assert msgs[0].text == "hey"


def test_multiline_continuation():
    text = "[8/29/26, 10:15:32 AM] Alice: line one\nline two\nline three"
    msgs = parse_whatsapp_export(text)
    assert len(msgs) == 1
    assert msgs[0].text == "line one\nline two\nline three"


def test_strips_invisible_marks():
    text = "\u200e[8/29/26, 10:15:32 AM] \u200eAlice: hi"
    msgs = parse_whatsapp_export(text)
    assert msgs[0].sender == "Alice"
    assert msgs[0].text == "hi"


def test_multiple_messages():
    text = (
        "[8/29/26, 10:15:32 AM] Alice: one\n"
        "[8/29/26, 10:16:00 AM] Bob: two\n"
        "[8/29/26, 10:16:30 AM] Alice: three"
    )
    msgs = parse_whatsapp_export(text)
    assert [m.sender for m in msgs] == ["Alice", "Bob", "Alice"]
