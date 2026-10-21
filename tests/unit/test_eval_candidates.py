"""Tests for parsing Ollama-generated evaluation candidates."""

import pytest

from eval.generate_candidates import _extract_json_array


def test_extract_json_array_accepts_fenced_json():
    text = '```json\n[{"question":"Q","ground_truth_answer":"A","ground_truth_sources":[]}]\n```'

    assert _extract_json_array(text)[0]["question"] == "Q"


def test_extract_json_array_rejects_non_json():
    with pytest.raises(ValueError, match="JSON array"):
        _extract_json_array("I could not create JSON.")
