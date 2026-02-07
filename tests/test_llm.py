"""Tests for lib/llm.py LLM integration functions."""

from unittest.mock import MagicMock, patch

import pytest

from lib.llm import (
    build_prompt,
    call_llm,
    call_llm_with_validation,
    parse_llm_json,
    validate_qcm,
    MISTRAL_BASE_URL,
)


class TestBuildPrompt:
    def test_basic_prompt(self):
        question = {
            "question_id": "1_1",
            "question": "Est-ce que le projet est écologique?",
            "options": ["Oui", "Non", "Partiellement"],
            "context": "Contexte environnemental",
        }
        fields = {"Nom": "Projet Test"}
        chunks = []

        prompt = build_prompt(question, fields, chunks)

        assert "1_1" in prompt
        assert "Est-ce que le projet est écologique?" in prompt
        assert "Contexte environnemental" in prompt
        assert "Projet Test" in prompt

    def test_options_formatted_as_letters(self):
        question = {
            "question": "Choix?",
            "options": ["Premier", "Deuxième", "Troisième"],
        }

        prompt = build_prompt(question, {}, [])

        assert "A. Premier" in prompt
        assert "B. Deuxième" in prompt
        assert "C. Troisième" in prompt

    def test_no_options(self):
        question = {"question": "Question ouverte?", "options": []}

        prompt = build_prompt(question, {}, [])

        assert "Aucune option fournie" in prompt

    def test_with_chunks(self):
        question = {"question": "Test?"}
        chunks = [
            {"score": 0.95, "start_char": 0, "text": "Chunk 1 content"},
            {"score": 0.85, "start_char": 100, "text": "Chunk 2 content"},
        ]

        prompt = build_prompt(question, {}, chunks)

        assert "score=0.950" in prompt
        assert "Chunk 1 content" in prompt
        assert "Chunk 2 content" in prompt

    def test_fields_included(self):
        question = {"question": "Test?"}
        fields = {
            "Nom": "Mon Projet",
            "Adresse": "123 rue Test",
            "Vide": "",  # Empty fields should be excluded
        }

        prompt = build_prompt(question, fields, [])

        assert "Nom: Mon Projet" in prompt
        assert "Adresse: 123 rue Test" in prompt
        assert "Vide" not in prompt

    def test_json_format_instruction(self):
        question = {"question": "Test?"}

        prompt = build_prompt(question, {}, [])

        assert "JSON" in prompt
        assert "text" in prompt
        assert "qcm" in prompt


class TestParseJson:
    def test_valid_json(self):
        text = '{"text": "La réponse", "qcm": "A"}'
        result = parse_llm_json(text)

        assert result["text"] == "La réponse"
        assert result["qcm"] == "A"

    def test_json_with_markdown_wrapper(self):
        text = '''```json
{"text": "Réponse", "qcm": "B"}
```'''
        result = parse_llm_json(text)

        assert result["text"] == "Réponse"
        assert result["qcm"] == "B"

    def test_json_with_surrounding_text(self):
        text = '''Voici ma réponse:
{"text": "Analyse complète", "qcm": "C"}
J'espère que cela vous aide.'''
        result = parse_llm_json(text)

        assert result["text"] == "Analyse complète"
        assert result["qcm"] == "C"

    def test_invalid_json(self):
        text = "Ceci n'est pas du JSON"
        result = parse_llm_json(text)

        assert result == {}

    def test_malformed_json(self):
        text = '{"text": "incomplete'
        result = parse_llm_json(text)

        assert result == {}

    def test_empty_string(self):
        result = parse_llm_json("")

        assert result == {}


class TestValidateQcm:
    def test_valid_single_letter(self):
        options = ["Option A", "Option B", "Option C"]

        assert validate_qcm("A", options) is True
        assert validate_qcm("B", options) is True
        assert validate_qcm("C", options) is True

    def test_lowercase_accepted(self):
        options = ["Option A", "Option B"]

        assert validate_qcm("a", options) is True
        assert validate_qcm("b", options) is True

    def test_invalid_letter(self):
        options = ["Option A", "Option B"]

        assert validate_qcm("C", options) is False
        assert validate_qcm("Z", options) is False

    def test_unknown_always_valid(self):
        options = ["Option A", "Option B"]

        assert validate_qcm("UNKNOWN", options) is True

    def test_empty_options_always_valid(self):
        assert validate_qcm("A", []) is True
        assert validate_qcm("anything", []) is True

    def test_invalid_values(self):
        options = ["Option A", "Option B"]

        assert validate_qcm("AB", options) is False
        assert validate_qcm("1", options) is False
        assert validate_qcm("Option A", options) is False


class TestCallLlm:
    @patch("lib.llm.requests.post")
    def test_successful_call(self, mock_post):
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "choices": [{"message": {"content": '{"text": "response", "qcm": "A"}'}}]
        }
        mock_post.return_value = mock_response

        result = call_llm(
            prompt="Test prompt",
            api_key="test-key",
            model="test-model",
            temperature=0.5,
            max_tokens=100,
        )

        assert result == '{"text": "response", "qcm": "A"}'
        mock_post.assert_called_once()

    @patch("lib.llm.requests.post")
    def test_uses_default_mistral_url(self, mock_post):
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "choices": [{"message": {"content": "response"}}]
        }
        mock_post.return_value = mock_response

        call_llm("prompt", "key", "model", 0.5, 100)

        call_args = mock_post.call_args
        assert MISTRAL_BASE_URL in call_args[0][0]

    @patch("lib.llm.requests.post")
    def test_custom_base_url(self, mock_post):
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "choices": [{"message": {"content": "response"}}]
        }
        mock_post.return_value = mock_response

        call_llm("prompt", "key", "model", 0.5, 100, base_url="http://localhost:8000/v1")

        call_args = mock_post.call_args
        assert "localhost:8000" in call_args[0][0]
        assert "/chat/completions" in call_args[0][0]

    @patch("lib.llm.requests.post")
    def test_payload_structure(self, mock_post):
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "choices": [{"message": {"content": "response"}}]
        }
        mock_post.return_value = mock_response

        call_llm(
            prompt="My prompt",
            api_key="my-key",
            model="my-model",
            temperature=0.7,
            max_tokens=256,
        )

        call_args = mock_post.call_args
        payload = call_args[1]["json"]

        assert payload["model"] == "my-model"
        assert payload["temperature"] == 0.7
        assert payload["max_tokens"] == 256
        assert payload["messages"][0]["role"] == "user"
        assert payload["messages"][0]["content"] == "My prompt"

    @patch("lib.llm.requests.post")
    def test_authorization_header(self, mock_post):
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "choices": [{"message": {"content": "response"}}]
        }
        mock_post.return_value = mock_response

        call_llm("prompt", "secret-key", "model", 0.5, 100)

        call_args = mock_post.call_args
        headers = call_args[1]["headers"]

        assert headers["Authorization"] == "Bearer secret-key"


class TestCallLlmWithValidation:
    @patch("lib.llm.call_llm")
    def test_valid_response_first_try(self, mock_call_llm):
        mock_call_llm.return_value = '{"text": "Response", "qcm": "A"}'
        options = ["Option 1", "Option 2"]

        response_text, parsed = call_llm_with_validation(
            prompt="Test",
            options=options,
            api_key="key",
            model="model",
            temperature=0.5,
            max_tokens=100,
        )

        assert parsed["qcm"] == "A"
        assert parsed["text"] == "Response"
        assert mock_call_llm.call_count == 1

    @patch("lib.llm.call_llm")
    def test_retries_on_invalid_qcm(self, mock_call_llm):
        # First two calls return invalid QCM, third is valid
        mock_call_llm.side_effect = [
            '{"text": "Bad", "qcm": "Z"}',
            '{"text": "Still bad", "qcm": "X"}',
            '{"text": "Good", "qcm": "A"}',
        ]
        options = ["Option 1", "Option 2"]

        response_text, parsed = call_llm_with_validation(
            prompt="Test",
            options=options,
            api_key="key",
            model="model",
            temperature=0.5,
            max_tokens=100,
            max_retries=2,
        )

        assert parsed["qcm"] == "A"
        assert mock_call_llm.call_count == 3

    @patch("lib.llm.call_llm")
    def test_returns_unknown_after_max_retries(self, mock_call_llm):
        # All calls return invalid QCM
        mock_call_llm.return_value = '{"text": "Response", "qcm": "Z"}'
        options = ["Option 1", "Option 2"]

        response_text, parsed = call_llm_with_validation(
            prompt="Test",
            options=options,
            api_key="key",
            model="model",
            temperature=0.5,
            max_tokens=100,
            max_retries=2,
        )

        assert parsed["qcm"] == "UNKNOWN"
        assert parsed["validation_failed"] is True
        assert mock_call_llm.call_count == 3  # 1 initial + 2 retries

    @patch("lib.llm.call_llm")
    def test_unknown_is_valid(self, mock_call_llm):
        mock_call_llm.return_value = '{"text": "No info", "qcm": "UNKNOWN"}'
        options = ["Option 1", "Option 2"]

        response_text, parsed = call_llm_with_validation(
            prompt="Test",
            options=options,
            api_key="key",
            model="model",
            temperature=0.5,
            max_tokens=100,
        )

        assert parsed["qcm"] == "UNKNOWN"
        assert "validation_failed" not in parsed
        assert mock_call_llm.call_count == 1

    @patch("lib.llm.call_llm")
    def test_passes_base_url(self, mock_call_llm):
        mock_call_llm.return_value = '{"text": "R", "qcm": "A"}'

        call_llm_with_validation(
            prompt="Test",
            options=["O1"],
            api_key="key",
            model="model",
            temperature=0.5,
            max_tokens=100,
            base_url="http://custom:8000/v1",
        )

        call_args = mock_call_llm.call_args
        assert call_args[0][5] == "http://custom:8000/v1"  # base_url is 6th positional arg
