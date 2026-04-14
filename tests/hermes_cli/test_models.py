"""Tests for the hermes_cli models module."""

from unittest.mock import patch

from hermes_cli.models import (
    fetch_openrouter_models,
    menu_labels,
    model_ids,
    detect_provider_for_model,
    openrouter_picker_groups,
    openrouter_picker_model_ids,
    filter_nous_free_models,
    _NOUS_ALLOWED_FREE_MODELS,
    is_nous_free_tier,
    partition_nous_models_by_tier,
    check_nous_free_tier,
    _FREE_TIER_CACHE_TTL,
)
import hermes_cli.models as _models_mod

LIVE_OPENROUTER_MODELS = [
    ("anthropic/claude-opus-4.6", ""),
    ("qwen/qwen3.6-plus", ""),
    ("nvidia/nemotron-3-super-120b-a12b:free", "free"),
]


class TestModelIds:
    def test_returns_non_empty_list(self):
        with patch(
            "hermes_cli.models.openrouter_picker_model_ids",
            return_value=[mid for mid, _ in LIVE_OPENROUTER_MODELS],
        ):
            ids = model_ids()
        assert isinstance(ids, list)
        assert len(ids) > 0

    def test_ids_match_picker_catalog(self):
        with patch(
            "hermes_cli.models.openrouter_picker_model_ids",
            return_value=[mid for mid, _ in LIVE_OPENROUTER_MODELS],
        ):
            ids = model_ids()
        expected = [mid for mid, _ in LIVE_OPENROUTER_MODELS]
        assert ids == expected

    def test_all_ids_contain_provider_slash(self):
        with patch(
            "hermes_cli.models.openrouter_picker_model_ids",
            return_value=[mid for mid, _ in LIVE_OPENROUTER_MODELS],
        ):
            for mid in model_ids():
                assert "/" in mid, f"Model ID '{mid}' missing provider/ prefix"

    def test_no_duplicate_ids(self):
        with patch(
            "hermes_cli.models.openrouter_picker_model_ids",
            return_value=[mid for mid, _ in LIVE_OPENROUTER_MODELS],
        ):
            ids = model_ids()
        assert len(ids) == len(set(ids)), "Duplicate model IDs found"


class TestMenuLabels:
    def test_same_length_as_model_ids(self):
        with patch("hermes_cli.models.fetch_openrouter_models", return_value=LIVE_OPENROUTER_MODELS), patch(
            "hermes_cli.models.openrouter_picker_model_ids",
            return_value=[mid for mid, _ in LIVE_OPENROUTER_MODELS],
        ):
            assert len(menu_labels()) == len(model_ids())

    def test_free_label_is_annotated(self):
        with patch("hermes_cli.models.fetch_openrouter_models", return_value=LIVE_OPENROUTER_MODELS):
            labels = menu_labels()
        assert "free" in labels[-1].lower()

    def test_each_label_contains_its_model_id(self):
        with patch("hermes_cli.models.fetch_openrouter_models", return_value=LIVE_OPENROUTER_MODELS), patch(
            "hermes_cli.models.openrouter_picker_model_ids",
            return_value=[mid for mid, _ in LIVE_OPENROUTER_MODELS],
        ):
            for label, mid in zip(menu_labels(), model_ids()):
                assert mid in label, f"Label '{label}' doesn't contain model ID '{mid}'"

    def test_non_free_labels_have_no_tag(self):
        with patch("hermes_cli.models.fetch_openrouter_models", return_value=LIVE_OPENROUTER_MODELS):
            labels = menu_labels()
        for label in labels[:-1]:
            assert "free" not in label.lower(), f"Unexpected 'free' in '{label}'"


class TestFetchOpenRouterModels:
    def test_live_fetch_recomputes_free_tags(self, monkeypatch):
        class _Resp:
            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb):
                return False

            def read(self):
                return b'{"data":[{"id":"anthropic/claude-opus-4.6","pricing":{"prompt":"0.000015","completion":"0.000075"},"supported_parameters":["tools","tool_choice"],"architecture":{"input_modalities":["text","image"],"output_modalities":["text"]}},{"id":"qwen/qwen3.6-plus","pricing":{"prompt":"0.000000325","completion":"0.00000195"},"supported_parameters":["tools","tool_choice"],"architecture":{"input_modalities":["text"],"output_modalities":["text"]}},{"id":"nvidia/nemotron-3-super-120b-a12b:free","pricing":{"prompt":"0","completion":"0"},"supported_parameters":["tools","tool_choice"],"architecture":{"input_modalities":["text"],"output_modalities":["text"]}}]}'

        monkeypatch.setattr(_models_mod, "_openrouter_catalog_cache", None)
        with patch("hermes_cli.models.urllib.request.urlopen", return_value=_Resp()):
            models = fetch_openrouter_models(force_refresh=True)

        assert models == [
            ("anthropic/claude-opus-4.6", ""),
            ("nvidia/nemotron-3-super-120b-a12b:free", "free"),
            ("qwen/qwen3.6-plus", ""),
        ]

    def test_without_live_metadata_returns_blank_tags(self, monkeypatch):
        monkeypatch.setattr(_models_mod, "_openrouter_catalog_cache", None)
        with patch(
            "agent.models_dev.list_agentic_models",
            return_value=["anthropic/claude-opus-4.6", "qwen/qwen3.6-plus"],
        ), patch(
            "hermes_cli.models.urllib.request.urlopen",
            side_effect=OSError("boom"),
        ):
            models = fetch_openrouter_models(force_refresh=True)

        assert models == [
            ("anthropic/claude-opus-4.6", ""),
            ("qwen/qwen3.6-plus", ""),
        ]


class TestOpenRouterPickerModelIds:
    def test_sorts_models_by_vendor_then_model(self, monkeypatch):
        class _Resp:
            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb):
                return False

            def read(self):
                return b'{"data":[{"id":"google/gemma-4-31b-it","supported_parameters":["tools","tool_choice"],"architecture":{"input_modalities":["text","image"],"output_modalities":["text"]}},{"id":"openai/gpt-5.4","supported_parameters":["tools","tool_choice"],"architecture":{"input_modalities":["text"],"output_modalities":["text"]}},{"id":"anthropic/claude-opus-4.6","supported_parameters":["tools","tool_choice"],"architecture":{"input_modalities":["text","image"],"output_modalities":["text"]}}]}'

        monkeypatch.setattr(_models_mod, "_openrouter_catalog_cache", None)
        with patch("hermes_cli.models.urllib.request.urlopen", return_value=_Resp()), patch(
            "agent.models_dev.list_agentic_models",
            return_value=[
                "google/gemma-4-31b-it",
                "openai/gpt-5.4",
                "anthropic/claude-opus-4.6",
            ],
        ):
            result = openrouter_picker_model_ids(force_refresh=True)

        assert result == [
            "anthropic/claude-opus-4.6",
            "google/gemma-4-31b-it",
            "openai/gpt-5.4",
        ]

    def test_excludes_synthetic_openrouter_ids(self, monkeypatch):
        class _Resp:
            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb):
                return False

            def read(self):
                return b'{"data":[{"id":"openrouter/free","supported_parameters":["tools","tool_choice"],"architecture":{"input_modalities":["text"],"output_modalities":["text"]}},{"id":"openrouter/auto","supported_parameters":["tools","tool_choice"],"architecture":{"input_modalities":["text"],"output_modalities":["text"]}},{"id":"anthropic/claude-opus-4.6","supported_parameters":["tools","tool_choice"],"architecture":{"input_modalities":["text","image"],"output_modalities":["text"]}}]}'

        monkeypatch.setattr(_models_mod, "_openrouter_catalog_cache", None)
        with patch("hermes_cli.models.urllib.request.urlopen", return_value=_Resp()), patch(
            "agent.models_dev.list_agentic_models",
            return_value=["openrouter/free", "openrouter/auto", "anthropic/claude-opus-4.6"],
        ):
            result = openrouter_picker_model_ids(force_refresh=True)

        assert result == ["anthropic/claude-opus-4.6"]


class TestOpenRouterPickerGroups:
    def test_groups_models_by_vendor(self, monkeypatch):
        monkeypatch.setattr(_models_mod, "_openrouter_groups_cache", None)
        with patch(
            "hermes_cli.models.openrouter_picker_model_ids",
            return_value=[
                "anthropic/claude-opus-4.6",
                "anthropic/claude-sonnet-4.6",
                "openai/gpt-5.4",
            ],
        ):
            groups = openrouter_picker_groups()

        assert groups == [
            ("anthropic", ("anthropic/claude-opus-4.6", "anthropic/claude-sonnet-4.6")),
            ("openai", ("openai/gpt-5.4",)),
        ]


class TestFindOpenrouterSlug:
    def test_exact_match(self):
        from hermes_cli.models import _find_openrouter_slug
        with patch(
            "hermes_cli.models.openrouter_picker_model_ids",
            return_value=[mid for mid, _ in LIVE_OPENROUTER_MODELS],
        ):
            assert _find_openrouter_slug("anthropic/claude-opus-4.6") == "anthropic/claude-opus-4.6"

    def test_bare_name_match(self):
        from hermes_cli.models import _find_openrouter_slug
        with patch(
            "hermes_cli.models.openrouter_picker_model_ids",
            return_value=[mid for mid, _ in LIVE_OPENROUTER_MODELS],
        ):
            result = _find_openrouter_slug("claude-opus-4.6")
        assert result == "anthropic/claude-opus-4.6"

    def test_case_insensitive(self):
        from hermes_cli.models import _find_openrouter_slug
        with patch(
            "hermes_cli.models.openrouter_picker_model_ids",
            return_value=[mid for mid, _ in LIVE_OPENROUTER_MODELS],
        ):
            result = _find_openrouter_slug("Anthropic/Claude-Opus-4.6")
        assert result is not None

    def test_unknown_returns_none(self):
        from hermes_cli.models import _find_openrouter_slug
        with patch(
            "hermes_cli.models.openrouter_picker_model_ids",
            return_value=[mid for mid, _ in LIVE_OPENROUTER_MODELS],
        ):
            assert _find_openrouter_slug("totally-fake-model-xyz") is None


class TestDetectProviderForModel:
    def test_anthropic_model_detected(self):
        with patch(
            "hermes_cli.models.openrouter_picker_model_ids",
            return_value=[mid for mid, _ in LIVE_OPENROUTER_MODELS],
        ):
            result = detect_provider_for_model("claude-opus-4-6", "openai-codex")
        assert result is not None
        assert result[0] == "anthropic"

    def test_deepseek_model_detected(self):
        result = detect_provider_for_model("deepseek-chat", "openai-codex")
        assert result is not None
        assert result[0] in ("deepseek", "openrouter")

    def test_current_provider_model_returns_none(self):
        assert detect_provider_for_model("gpt-5.3-codex", "openai-codex") is None

    def test_openrouter_slug_match(self):
        with patch(
            "hermes_cli.models.openrouter_picker_model_ids",
            return_value=[mid for mid, _ in LIVE_OPENROUTER_MODELS],
        ):
            result = detect_provider_for_model("anthropic/claude-opus-4.6", "openai-codex")
        assert result is not None
        assert result[0] == "openrouter"
        assert result[1] == "anthropic/claude-opus-4.6"

    def test_bare_name_gets_openrouter_slug(self, monkeypatch):
        for env_var in (
            "ANTHROPIC_API_KEY",
            "ANTHROPIC_TOKEN",
            "CLAUDE_CODE_TOKEN",
            "CLAUDE_CODE_OAUTH_TOKEN",
        ):
            monkeypatch.delenv(env_var, raising=False)
        with patch(
            "hermes_cli.models.openrouter_picker_model_ids",
            return_value=[mid for mid, _ in LIVE_OPENROUTER_MODELS],
        ):
            result = detect_provider_for_model("claude-opus-4.6", "openai-codex")
        assert result is not None
        assert result[1] == "anthropic/claude-opus-4.6"

    def test_unknown_model_returns_none(self):
        with patch(
            "hermes_cli.models.openrouter_picker_model_ids",
            return_value=[mid for mid, _ in LIVE_OPENROUTER_MODELS],
        ):
            assert detect_provider_for_model("nonexistent-model-xyz", "openai-codex") is None

    def test_aggregator_not_suggested(self):
        with patch(
            "hermes_cli.models.openrouter_picker_model_ids",
            return_value=[mid for mid, _ in LIVE_OPENROUTER_MODELS],
        ):
            result = detect_provider_for_model("claude-opus-4-6", "openai-codex")
        assert result is not None
        assert result[0] not in ("nous",)


class TestFilterNousFreeModels:
    _PAID = {"prompt": "0.000003", "completion": "0.000015"}
    _FREE = {"prompt": "0", "completion": "0"}

    def test_paid_models_kept(self):
        models = ["anthropic/claude-opus-4.6", "openai/gpt-5.4"]
        pricing = {m: self._PAID for m in models}
        assert filter_nous_free_models(models, pricing) == models

    def test_free_non_allowlist_models_removed(self):
        models = ["anthropic/claude-opus-4.6", "arcee-ai/trinity-large-preview:free"]
        pricing = {
            "anthropic/claude-opus-4.6": self._PAID,
            "arcee-ai/trinity-large-preview:free": self._FREE,
        }
        result = filter_nous_free_models(models, pricing)
        assert result == ["anthropic/claude-opus-4.6"]

    def test_allowlist_model_kept_when_free(self):
        models = ["anthropic/claude-opus-4.6", "xiaomi/mimo-v2-pro"]
        pricing = {
            "anthropic/claude-opus-4.6": self._PAID,
            "xiaomi/mimo-v2-pro": self._FREE,
        }
        result = filter_nous_free_models(models, pricing)
        assert result == ["anthropic/claude-opus-4.6", "xiaomi/mimo-v2-pro"]

    def test_allowlist_model_removed_when_paid(self):
        models = ["anthropic/claude-opus-4.6", "xiaomi/mimo-v2-pro"]
        pricing = {
            "anthropic/claude-opus-4.6": self._PAID,
            "xiaomi/mimo-v2-pro": self._PAID,
        }
        result = filter_nous_free_models(models, pricing)
        assert result == ["anthropic/claude-opus-4.6"]

    def test_no_pricing_returns_all(self):
        models = ["anthropic/claude-opus-4.6", "nvidia/nemotron-3-super-120b-a12b:free"]
        assert filter_nous_free_models(models, {}) == models

    def test_model_with_no_pricing_entry_treated_as_paid(self):
        models = ["anthropic/claude-opus-4.6", "openai/gpt-5.4"]
        pricing = {"anthropic/claude-opus-4.6": self._PAID}
        result = filter_nous_free_models(models, pricing)
        assert result == models

    def test_mixed_scenario(self):
        models = [
            "anthropic/claude-opus-4.6",
            "nvidia/nemotron-3-super-120b-a12b:free",
            "xiaomi/mimo-v2-pro",
            "xiaomi/mimo-v2-omni",
            "openai/gpt-5.4",
        ]
        pricing = {
            "anthropic/claude-opus-4.6": self._PAID,
            "nvidia/nemotron-3-super-120b-a12b:free": self._FREE,
            "xiaomi/mimo-v2-pro": self._FREE,
            "xiaomi/mimo-v2-omni": self._PAID,
            "openai/gpt-5.4": self._PAID,
        }
        result = filter_nous_free_models(models, pricing)
        assert result == [
            "anthropic/claude-opus-4.6",
            "xiaomi/mimo-v2-pro",
            "openai/gpt-5.4",
        ]

    def test_allowlist_contains_expected_models(self):
        assert "xiaomi/mimo-v2-pro" in _NOUS_ALLOWED_FREE_MODELS
        assert "xiaomi/mimo-v2-omni" in _NOUS_ALLOWED_FREE_MODELS


class TestIsNousFreeTier:
    def test_paid_plus_tier(self):
        assert is_nous_free_tier({"subscription": {"plan": "Plus", "tier": 2, "monthly_charge": 20}}) is False

    def test_free_tier_by_charge(self):
        assert is_nous_free_tier({"subscription": {"plan": "Free", "tier": 0, "monthly_charge": 0}}) is True

    def test_no_charge_field_not_free(self):
        assert is_nous_free_tier({"subscription": {"plan": "Free", "tier": 0}}) is False

    def test_plan_name_alone_not_free(self):
        assert is_nous_free_tier({"subscription": {"plan": "free"}}) is False

    def test_empty_subscription_not_free(self):
        assert is_nous_free_tier({"subscription": {}}) is False

    def test_no_subscription_not_free(self):
        assert is_nous_free_tier({}) is False

    def test_empty_response_not_free(self):
        assert is_nous_free_tier({}) is False


class TestPartitionNousModelsByTier:
    _PAID = {"prompt": "0.000003", "completion": "0.000015"}
    _FREE = {"prompt": "0", "completion": "0"}

    def test_paid_tier_all_selectable(self):
        models = ["anthropic/claude-opus-4.6", "xiaomi/mimo-v2-pro"]
        pricing = {"anthropic/claude-opus-4.6": self._PAID, "xiaomi/mimo-v2-pro": self._FREE}
        sel, unav = partition_nous_models_by_tier(models, pricing, free_tier=False)
        assert sel == models
        assert unav == []

    def test_free_tier_splits_correctly(self):
        models = ["anthropic/claude-opus-4.6", "xiaomi/mimo-v2-pro", "openai/gpt-5.4"]
        pricing = {
            "anthropic/claude-opus-4.6": self._PAID,
            "xiaomi/mimo-v2-pro": self._FREE,
            "openai/gpt-5.4": self._PAID,
        }
        sel, unav = partition_nous_models_by_tier(models, pricing, free_tier=True)
        assert sel == ["xiaomi/mimo-v2-pro"]
        assert unav == ["anthropic/claude-opus-4.6", "openai/gpt-5.4"]

    def test_no_pricing_returns_all(self):
        models = ["anthropic/claude-opus-4.6", "openai/gpt-5.4"]
        sel, unav = partition_nous_models_by_tier(models, {}, free_tier=True)
        assert sel == models
        assert unav == []

    def test_all_free_models(self):
        models = ["xiaomi/mimo-v2-pro", "xiaomi/mimo-v2-omni"]
        pricing = {m: self._FREE for m in models}
        sel, unav = partition_nous_models_by_tier(models, pricing, free_tier=True)
        assert sel == models
        assert unav == []

    def test_all_paid_models(self):
        models = ["anthropic/claude-opus-4.6", "openai/gpt-5.4"]
        pricing = {m: self._PAID for m in models}
        sel, unav = partition_nous_models_by_tier(models, pricing, free_tier=True)
        assert sel == []
        assert unav == models


class TestCheckNousFreeTierCache:
    def setup_method(self):
        _models_mod._free_tier_cache = None

    def teardown_method(self):
        _models_mod._free_tier_cache = None

    @patch("hermes_cli.models.fetch_nous_account_tier")
    @patch("hermes_cli.models.is_nous_free_tier", return_value=True)
    def test_result_is_cached(self, mock_is_free, mock_fetch):
        mock_fetch.return_value = {"subscription": {"monthly_charge": 0}}
        with patch("hermes_cli.auth.get_provider_auth_state", return_value={"access_token": "tok"}), \
             patch("hermes_cli.auth.resolve_nous_runtime_credentials"):
            result1 = check_nous_free_tier()
            result2 = check_nous_free_tier()

        assert result1 is True
        assert result2 is True
        assert mock_fetch.call_count == 1

    @patch("hermes_cli.models.fetch_nous_account_tier")
    @patch("hermes_cli.models.is_nous_free_tier", return_value=False)
    def test_cache_expires_after_ttl(self, mock_is_free, mock_fetch):
        mock_fetch.return_value = {"subscription": {"monthly_charge": 20}}
        with patch("hermes_cli.auth.get_provider_auth_state", return_value={"access_token": "tok"}), \
             patch("hermes_cli.auth.resolve_nous_runtime_credentials"):
            result1 = check_nous_free_tier()
            assert mock_fetch.call_count == 1

            cached_result, cached_at = _models_mod._free_tier_cache
            _models_mod._free_tier_cache = (cached_result, cached_at - _FREE_TIER_CACHE_TTL - 1)

            result2 = check_nous_free_tier()
            assert mock_fetch.call_count == 2

        assert result1 is False
        assert result2 is False

    def test_cache_ttl_is_short(self):
        assert _FREE_TIER_CACHE_TTL <= 300
