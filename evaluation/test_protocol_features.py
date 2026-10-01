import pandas as pd
import pytest

from experiment.prepare_protocol_features import FIELDS, parse_protocols, protocol_matrix


def test_protocol_parser_does_not_execute_expressions():
    assert parse_protocols("['tcp', 'udp', 'tcp']") == {"tcp", "udp"}
    with pytest.raises((ValueError, SyntaxError)):
        parse_protocols("__import__('os').system('echo unsafe')")


def test_unknown_protocol_is_flagged_without_expanding_training_vocabulary():
    vocab = {field: ["tcp"] for field in FIELDS}
    frame = pd.DataFrame({field: ["['tcp']", "['udp']"] for field in FIELDS})
    encoded = protocol_matrix(frame, vocab)
    assert encoded.shape == (2, 6)
    assert encoded[0].tolist() == [1, 0, 1, 0, 1, 0]
    assert encoded[1].tolist() == [0, 1, 0, 1, 0, 1]
    assert vocab[FIELDS[0]] == ["tcp"]
