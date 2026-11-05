"""
The AI reading of the nova under the "AI texts" rule (D4): a cached text is tied to the data it was
written from, is labelled with its model and its date, and the prompt asks for nothing but a
reading of the data supplied.

No LLM is ever called: the interpreter of the endpoint is replaced by a counting stand-in. The
sub-group and claim names are synthetic.
"""

import json
import re
from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import pytest_asyncio

from app.models.models import CosmeticInterpretation, User
from app.services.ai_interpreter import LocalAIInterpreter
from app.services.cosmomics_nova import (
    NOVA_MODEL_TAG,
    ClaimSubgroupMap,
    SubgroupTerm,
    TermHit,
    build_nova,
)
from app.services.report_common import ai_label, nova_inputs_sha256

MODEL = "test-model-1"
URL = "/api/v1/datasets/{ds}/comparisons/A_vs_B/cosmomics-nova"


def synthetic_nova(alpha: float = 0.05, version: str = "1") -> dict:
    """One claim with a sub-group for it (A) and a sub-group against it (B)."""
    rows = [
        SubgroupTerm("Firming", "Group A", "GO:0000001", "Term A", "up"),
        SubgroupTerm("Firming", "Group B", "GO:0000002", "Term B", "up"),
    ]
    hits = [
        TermHit("GO:0000001", "Term A", 1e-6, "up"),
        TermHit("GO:0000002", "Term B", 1e-5, "down"),
    ]
    return build_nova(
        hits, ClaimSubgroupMap(rows, version=version), alpha=alpha, contrast="A_vs_B"
    )


def many_claims_nova(n: int) -> dict:
    return {
        "alpha": 0.05,
        "claims": [
            {
                "name": f"Claim {i}",
                "state": "for_only",
                "coherence": 1.0,
                "n_for": 1,
                "n_against": 0,
                "n_balanced": 0,
                "subgroups": [
                    {"name": "Group A", "obs": "up", "nl": 4.0, "delta": 4.0, "terms": []}
                ],
            }
            for i in range(n)
        ],
        "unlinked": [],
        "provenance": {},
    }


# ── The label and the fingerprint ───────────────────────────────────────────


def test_ai_label_names_model_and_date():
    day = datetime(2026, 11, 5, 23, 30, tzinfo=timezone.utc)
    assert ai_label(MODEL, day) == f"AI-generated text (model {MODEL}, generated 2026-11-05)"
    assert ai_label(MODEL, None) == f"AI-generated text (model {MODEL})"


def test_ai_label_dates_in_utc():
    from datetime import timedelta

    late = datetime(2026, 11, 5, 23, 30, tzinfo=timezone(timedelta(hours=-5)))  # 04:30 UTC, the 6th
    assert ai_label(MODEL, late).endswith("generated 2026-11-06)")
    assert ai_label(MODEL, datetime(2026, 11, 5, 8, 0)).endswith("generated 2026-11-05)")  # naive


def test_inputs_hash_covers_exactly_what_the_text_was_written_from():
    base = synthetic_nova()
    digest = nova_inputs_sha256(base, MODEL)
    assert re.fullmatch(r"[0-9a-f]{64}", digest)
    assert nova_inputs_sha256(synthetic_nova(), MODEL) == digest

    def changed(**edits):
        nova = synthetic_nova()
        for path, value in edits.items():
            target = nova["provenance"] if path.startswith("prov_") else nova
            target[path.removeprefix("prov_")] = value
        return nova_inputs_sha256(nova, MODEL)

    assert changed(alpha=0.01) != digest
    assert changed(prov_referential_version=2) != digest
    assert changed(prov_enrichment_dataset_ids=["another-dataset"]) != digest
    assert changed(unlinked=[{"name": "Group Z"}]) != digest
    other_claims = synthetic_nova()
    other_claims["claims"][0]["subgroups"][0]["nl"] += 1
    assert nova_inputs_sha256(other_claims, MODEL) != digest
    assert nova_inputs_sha256(base, "another-model") != digest

    # The clock, and what is not an input of the text, do not enter the fingerprint.
    later = synthetic_nova()
    later["provenance"]["generated_at"] = "2030-01-01T00:00:00+00:00"
    later["provenance"]["app_version"] = "9.9.9"
    later["coverage"]["n_terms"] += 10
    assert nova_inputs_sha256(later, MODEL) == digest


# ── The prompt ──────────────────────────────────────────────────────────────


def test_prompt_has_no_marketing_or_regulatory_section():
    interpreter = LocalAIInterpreter(base_url="http://127.0.0.1:9/v1")
    prompt = interpreter._build_cosmomics_prompt(synthetic_nova(), "A_vs_B")
    assert "marketing" not in prompt.lower()
    assert "Regulatory note" not in prompt
    assert "regulatory" not in prompt.lower()
    # A reading claim by claim, then a summary, that stay inside the data supplied.
    assert "## Claim-by-claim reading" in prompt and "## Summary" in prompt
    assert re.search(r"nothing beyond the data", prompt)
    # A claim with no evidence against it is flagged, never read as confirmed.
    assert re.search(r"no pathway (sub-group )?against", prompt)
    assert not re.search(r"\d+ of \d+ claims", prompt)  # nothing is cut: no "N of M"


def test_prompt_says_how_many_claims_it_carries_when_the_list_is_cut():
    interpreter = LocalAIInterpreter(base_url="http://127.0.0.1:9/v1")
    prompt = interpreter._build_cosmomics_prompt(many_claims_nova(12), "A_vs_B")
    assert "9 of 12 claims" in prompt
    assert "- Claim 8 (" in prompt and "- Claim 9 (" not in prompt
    assert not re.search(r"\d+ of \d+ claims", interpreter._build_cosmomics_prompt(
        many_claims_nova(9), "A_vs_B"
    ))


def test_prompt_says_what_the_p_values_are_in_its_own_words():
    interpreter = LocalAIInterpreter(base_url="http://127.0.0.1:9/v1")
    own = interpreter._build_cosmomics_prompt(synthetic_nova(), "A_vs_B")
    assert "Benjamini–Hochberg adjusted p-value" in own
    uploaded = synthetic_nova()
    uploaded["p_source"] = "adjusted p from uploaded tables (column ZQX_qvalue)"
    prompt = interpreter._build_cosmomics_prompt(uploaded, "A_vs_B")
    assert "read from uploaded tables" in prompt
    assert "Benjamini" not in prompt  # it does not assert what the payload does not say
    assert "ZQX_qvalue" not in prompt  # the column name is the user's text: it does not leave


def llm_client(text: str = "Reading A."):
    """A stand-in for the OpenAI-compatible client of the interpreter: nothing leaves the process."""
    response = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=text))],
        usage=SimpleNamespace(prompt_tokens=1, completion_tokens=1, total_tokens=2),
    )
    create = AsyncMock(return_value=response)
    return SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create))), create


async def test_the_comparison_name_never_reaches_the_llm():
    """The consent says what leaves: claim and sub-group names, p-values. A comparison name often
    holds the user's own condition or product names."""
    name = "ZQX-Vehicle_vs_ZQX-Product7"
    interpreter = LocalAIInterpreter(base_url="http://127.0.0.1:9/v1")
    interpreter.client, create = llm_client()
    text = await interpreter.interpret_cosmomics(synthetic_nova(), name)
    assert text == "Reading A."
    create.assert_awaited_once()
    messages = create.await_args.kwargs["messages"]
    assert [m["role"] for m in messages] == ["user"]
    # Every argument of the call, not only the prompt text.
    sent = json.dumps(create.await_args.kwargs, default=str)
    for part in (name, "ZQX", "Vehicle", "Product7"):
        assert part not in sent, part
    assert "this comparison" in messages[0]["content"]  # it refers to the comparison anonymously
    assert "A_vs_B" not in interpreter._build_cosmomics_prompt(synthetic_nova(), "A_vs_B")


def test_only_names_directions_counts_and_p_values_reach_the_prompt():
    """Nothing of the payload but claim names, sub-group names, directions, counts and p-values."""
    nova = synthetic_nova(alpha=0.0123)
    nova["contrast"], nova["condition"] = "ZQX-contrast", "ZQX-condition"
    nova["p_source"] = "adjusted p from uploaded tables (column ZQX_col)"
    nova["provenance"].update(
        dataset_id="ZQX-dataset",
        analysis_id="ZQX-analysis",
        enrichment_dataset_ids=["ZQX-enrichment"],
        padj_columns=["ZQX_col"],
        source_versions=["ZQX-v1"],
        app_version="ZQX-app",
    )
    nova["claims"][0]["coherence"] = 0.7777
    nova["claims"][0]["importance"] = 123.456
    interpreter = LocalAIInterpreter(base_url="http://127.0.0.1:9/v1")
    prompt = interpreter._build_cosmomics_prompt(nova, "ZQX-comparison")

    assert "ZQX" not in prompt
    # No score, no setting.
    for leaked in ("0.0123", "alpha", "78%", "0.78", "coherence", "123.4", "importance"):
        assert leaked not in prompt, leaked
    # What is meant to go is there: the claim, its sub-groups with direction and p, the counts.
    line = next(line for line in prompt.splitlines() if line.startswith("- Firming"))
    assert "1 sub-group for, 1 against" in line
    assert "supporting: Group A (up, p = 1.0e-06)" in line
    assert "contradicting: Group B (down, p = 1.0e-05)" in line


# ── The endpoints ───────────────────────────────────────────────────────────


def make_llm(text: str = "Reading A."):
    """A stand-in for `LocalAIInterpreter` that counts what the endpoint asks of it."""
    calls: list[str] = []

    class FakeInterpreter:
        model = MODEL

        def __init__(self):
            self.last_usage = {"total_tokens": 7}

        async def interpret_cosmomics(self, nova, comparison_name=None):
            calls.append(comparison_name)
            return text

    return FakeInterpreter, calls


@pytest.fixture
def stored():
    """The `cosmetic_interpretations` row the endpoint reads and writes (None: no cache)."""
    return {"row": None}


@pytest_asyncio.fixture
async def client_for(stored):
    from app.api.deps.supabase_deps import get_db
    from app.main import app
    from tests.test_project_access_control import (
        make_client,
        make_db,
        make_fixtures,
        make_user,
    )

    caller: User = make_user()
    caller.cosmetics_module_enabled = True
    dataset, project = make_fixtures(owner_id=caller.id)
    client = make_client(caller=caller, dataset=dataset, project=project, member=None)
    db = make_db(dataset=dataset, project=project, member=None)
    plain = db.execute.side_effect

    def execute(stmt, *args, **kwargs):
        res = plain(stmt, *args, **kwargs)
        try:
            entity = stmt.column_descriptions[0]["entity"]
        except Exception:  # noqa: BLE001
            entity = None
        if entity is CosmeticInterpretation:
            res.scalar_one_or_none.return_value = stored["row"]
        return res

    def add(obj):
        if isinstance(obj, CosmeticInterpretation):
            stored["row"] = obj

    db.execute = AsyncMock(side_effect=execute)
    db.add = MagicMock(side_effect=add)

    async def _db():
        yield db

    app.dependency_overrides[get_db] = _db
    async with client:
        yield client, dataset
    app.dependency_overrides.clear()


def nova_source(version: str = "1"):
    """`compute_nova` of the endpoint: the synthetic nova at whatever alpha is asked for."""

    async def compute(_db, _dataset, comparison_name, alpha):
        return synthetic_nova(alpha, version), []

    return patch("app.api.endpoints.cosmomics.compute_nova", new=AsyncMock(side_effect=compute))


async def post(client, dataset, **params):
    return await client.post(URL.format(ds=dataset.id) + "/interpret", params=params)


async def get(client, dataset, **params):
    return await client.get(URL.format(ds=dataset.id) + "/interpretation", params=params)


async def test_interpret_returns_model_and_generated_at_on_a_fresh_text(client_for, stored):
    client, dataset = client_for
    llm, calls = make_llm()
    before = datetime.now(timezone.utc)
    with nova_source(), patch("app.api.endpoints.cosmomics.LocalAIInterpreter", new=llm):
        resp = await post(client, dataset)
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["interpretation"] == "Reading A."
    assert (body["cached"], body["stale"]) == (False, False)
    assert body["model"] == MODEL
    assert body["generated_at"] is not None
    assert datetime.fromisoformat(body["generated_at"]) >= before
    assert body["inputs_sha256"] == nova_inputs_sha256(synthetic_nova(), MODEL)
    assert (body["claims_sent"], body["claims_total"]) == (1, 1)
    assert body["claims_count"] == 1  # kept for the clients that read it
    assert calls == ["A_vs_B"]

    row = stored["row"]
    assert row.inputs_sha256 == body["inputs_sha256"]
    assert row.model == f"{NOVA_MODEL_TAG}{MODEL}"
    assert row.updated_at is not None and row.created_at is not None


async def test_cached_text_is_served_while_its_data_stand(client_for):
    client, dataset = client_for
    llm, calls = make_llm()
    with nova_source(), patch("app.api.endpoints.cosmomics.LocalAIInterpreter", new=llm):
        first = (await post(client, dataset)).json()
        again = await post(client, dataset)
    assert again.status_code == 200, again.text
    body = again.json()
    assert (body["cached"], body["stale"]) == (True, False)
    assert body["generated_at"] == first["generated_at"]
    assert body["inputs_sha256"] == first["inputs_sha256"]
    assert len(calls) == 1


async def test_cached_text_of_another_nova_is_stale_and_not_served(client_for, stored):
    client, dataset = client_for
    llm, calls = make_llm()
    with nova_source(), patch("app.api.endpoints.cosmomics.LocalAIInterpreter", new=llm):
        await post(client, dataset)  # a text written at alpha = 0.05
        assert len(calls) == 1

        # The same data read at another alpha is another nova: the text is flagged ...
        seen = await get(client, dataset, alpha=0.01)
        assert seen.status_code == 200, seen.text
        assert seen.json()["stale"] is True
        assert seen.json()["interpretation"] == "Reading A."  # the panel can show it, flagged
        assert (await get(client, dataset)).json()["stale"] is False

        # ... and a request for that alpha does not serve it: it writes a new one.
        regenerated = await post(client, dataset, alpha=0.01)
        assert regenerated.status_code == 200, regenerated.text
        assert regenerated.json()["cached"] is False
        assert len(calls) == 2
        assert stored["row"].inputs_sha256 == nova_inputs_sha256(synthetic_nova(0.01), MODEL)

        # The one cached row now belongs to alpha = 0.01.
        assert (await get(client, dataset, alpha=0.01)).json()["stale"] is False
        assert (await get(client, dataset)).json()["stale"] is True


async def test_cached_text_is_stale_when_the_referential_or_the_data_changed(client_for):
    client, dataset = client_for
    llm, calls = make_llm()
    with patch("app.api.endpoints.cosmomics.LocalAIInterpreter", new=llm):
        with nova_source(version="1"):
            await post(client, dataset)
        with nova_source(version="2"):
            assert (await get(client, dataset)).json()["stale"] is True
            assert (await post(client, dataset)).json()["cached"] is False
    assert len(calls) == 2


async def test_cached_text_without_a_fingerprint_is_stale(client_for, stored):
    """A text cached before the fingerprint existed cannot be tied to its data."""
    client, dataset = client_for
    llm, calls = make_llm()
    with nova_source(), patch("app.api.endpoints.cosmomics.LocalAIInterpreter", new=llm):
        await post(client, dataset)
        stored["row"].inputs_sha256 = None
        assert (await get(client, dataset)).json()["stale"] is True
        assert (await post(client, dataset)).json()["cached"] is False
    assert len(calls) == 2


async def test_force_regenerate_writes_a_new_text(client_for):
    client, dataset = client_for
    llm, calls = make_llm()
    with nova_source(), patch("app.api.endpoints.cosmomics.LocalAIInterpreter", new=llm):
        await post(client, dataset)
        forced = await post(client, dataset, force_regenerate="true")
    assert forced.json()["cached"] is False
    assert len(calls) == 2


async def test_get_interpretation_never_calls_the_llm(client_for, stored):
    client, dataset = client_for
    llm = MagicMock()
    with nova_source(), patch("app.api.endpoints.cosmomics.LocalAIInterpreter", new=llm):
        nothing = await get(client, dataset)  # no text yet
        assert nothing.status_code == 404

        stored["row"] = CosmeticInterpretation(
            dataset_id=dataset.id,
            comparison_name="A_vs_B",
            interpretation="Reading A.",
            model=f"{NOVA_MODEL_TAG}{MODEL}",
            claims_count=1,
            inputs_sha256=nova_inputs_sha256(synthetic_nova(), MODEL),
            created_at=datetime(2026, 11, 5, 9, 0, tzinfo=timezone.utc),
            updated_at=datetime(2026, 11, 5, 9, 0, tzinfo=timezone.utc),
        )
        fresh = await get(client, dataset)
        stale = await get(client, dataset, alpha=0.01)
    assert fresh.status_code == stale.status_code == 200
    assert fresh.json()["stale"] is False and stale.json()["stale"] is True
    body = fresh.json()
    assert body["cached"] is True and body["model"] == MODEL
    assert datetime.fromisoformat(body["generated_at"]) == datetime(
        2026, 11, 5, 9, 0, tzinfo=timezone.utc
    )
    assert (body["claims_sent"], body["claims_total"]) == (1, 1)
    llm.assert_not_called()


async def test_a_text_that_is_not_a_nova_reading_is_not_served(client_for, stored):
    client, dataset = client_for
    with nova_source():
        stored["row"] = CosmeticInterpretation(
            dataset_id=dataset.id,
            comparison_name="A_vs_B",
            interpretation="Another kind of text.",
            model=MODEL,  # no nova tag
            claims_count=0,
        )
        resp = await get(client, dataset)
    assert resp.status_code == 404


@pytest.mark.parametrize("module,plan", [(False, "TEAM"), (True, "STARTER")])
async def test_get_interpretation_needs_the_skin_claims_add_on_and_ai_access(module, plan):
    from app.main import app
    from app.models.models import SubscriptionPlan
    from tests.test_project_access_control import make_client, make_fixtures, make_user

    try:
        caller = make_user()
        caller.cosmetics_module_enabled = module
        caller.subscription_plan = SubscriptionPlan[plan]
        dataset, project = make_fixtures(owner_id=caller.id)
        client = make_client(caller=caller, dataset=dataset, project=project, member=None)
        compute = AsyncMock(return_value=(synthetic_nova(), []))
        with patch("app.api.endpoints.cosmomics.compute_nova", new=compute):
            async with client as c:
                resp = await c.get(URL.format(ds=dataset.id) + "/interpretation")
        assert resp.status_code == 403
        compute.assert_not_awaited()
    finally:
        app.dependency_overrides.clear()


async def test_interpretation_alpha_is_range_checked(client_for):
    client, dataset = client_for
    resp = await get(client, dataset, alpha=0.5)
    assert resp.status_code == 422
