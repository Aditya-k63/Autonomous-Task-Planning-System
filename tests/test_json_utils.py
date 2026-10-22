"""Tests for robust JSON extraction from model output."""

from app.json_utils import extract_json_object, strip_code_fences, undouble_braces


def test_plain_json():
    assert extract_json_object('{"a": 1}') == {"a": 1}


def test_json_in_code_fence():
    raw = '```json\n{"a": 1}\n```'
    assert extract_json_object(raw) == {"a": 1}


def test_json_with_prose_around_it():
    raw = 'Sure! Here is the plan:\n{"tasks": []}\nHope that helps.'
    assert extract_json_object(raw) == {"tasks": []}


def test_doubled_braces():
    raw = '{{"pass": true, "score": 1.0}}'
    assert extract_json_object(raw) == {"pass": True, "score": 1.0}


def test_doubled_braces_inside_a_fence():
    raw = '```\n{{"pass": true}}\n```'
    assert extract_json_object(raw) == {"pass": True}


def test_nested_objects_survive():
    raw = '{"a": {"b": {"c": 1}}, "d": [1, 2]}'
    assert extract_json_object(raw) == {"a": {"b": {"c": 1}}, "d": [1, 2]}


def test_garbage_returns_none():
    assert extract_json_object("not json at all") is None
    assert extract_json_object("") is None


def test_helpers():
    assert strip_code_fences("```json\n{}\n```") == "{}"
    assert undouble_braces("{{\"a\": 1}}") == '{"a": 1}'
