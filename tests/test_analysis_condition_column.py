"""
Colonne de condition choisie dans le wizard, du lancement jusqu'au pipeline R.

Le ContrastBuilder du wizard laisse choisir n'importe quelle colonne de la
feuille d'échantillons comme colonne de regroupement, mais ce choix n'arrivait
jamais au backend : `run_multimethod_pipeline.R` reprenait sa propre liste
d'alias (condition / group / treatment / genotype). Une colonne `groupe` ou
`genotype_detail` faisait donc échouer l'analyse, et une feuille qui contenait
aussi une colonne `condition` était comparée sur la mauvaise colonne.

Le choix voyage maintenant dans `params.condition_column` (colonne JSON, pas de
migration) et devient `--condition-col=<nom>` sur la ligne de commande R.
Absent, rien ne change : les analyses existantes gardent la détection par alias.
"""

import re
from pathlib import Path

import pytest
from pydantic import ValidationError

from app.api.endpoints.analyses import AnalysisParams
from app.worker.tasks import _build_pipeline_command
from tests.test_api_analyses_quota import (
    ENDPOINT,
    make_client,
    make_comparisons_dataset,
    make_user,
    payload_for,
)

R_SCRIPT = Path(__file__).resolve().parents[1] / "r_scripts" / "run_multimethod_pipeline.R"


def _cmd(params: dict) -> list[str]:
    return _build_pipeline_command(
        "/app/r_scripts/run_multimethod_pipeline.R",
        "/data/counts.tsv",
        "/data/samples.tsv",
        "/data/comparisons.tsv",
        "/data/out",
        params,
    )


# ── Schéma ────────────────────────────────────────────────────────────────────


def test_condition_column_defaults_to_none():
    assert AnalysisParams().condition_column is None
    assert AnalysisParams().model_dump()["condition_column"] is None


def test_condition_column_is_kept_and_trimmed():
    params = AnalysisParams(condition_column="  Genotype_Detail ")
    assert params.condition_column == "Genotype_Detail"
    assert params.model_dump()["condition_column"] == "Genotype_Detail"


def test_blank_condition_column_means_auto_detect():
    assert AnalysisParams(condition_column="   ").condition_column is None


@pytest.mark.parametrize("bad", ["geno\ttype", "geno\ntype", "x" * 256])
def test_condition_column_rejects_values_that_cannot_be_a_tsv_header(bad):
    with pytest.raises(ValidationError):
        AnalysisParams(condition_column=bad)


# ── Route de lancement ────────────────────────────────────────────────────────


def _added_analysis(db):
    added = [c.args[0] for c in db.add.call_args_list]
    return next(a for a in added if type(a).__name__ == "SelfServiceAnalysis")


async def test_launch_stores_the_chosen_condition_column():
    user = make_user()
    ds = make_comparisons_dataset(declared=1)
    client, db, project = make_client(as_user=user, comparisons_dataset=ds)
    payload = payload_for(project.id, ds.id)
    payload["params"] = {"condition_column": "groupe"}

    async with client:
        res = await client.post(ENDPOINT, json=payload)

    assert res.status_code == 201, res.text
    assert _added_analysis(db).params["condition_column"] == "groupe"
    assert res.json()["params"]["condition_column"] == "groupe"


async def test_launch_without_condition_column_still_works():
    """Les appelants d'API qui ignorent le champ ne sont pas cassés."""
    user = make_user()
    ds = make_comparisons_dataset(declared=1)
    client, db, project = make_client(as_user=user, comparisons_dataset=ds)

    async with client:
        res = await client.post(ENDPOINT, json=payload_for(project.id, ds.id))

    assert res.status_code == 201, res.text
    assert _added_analysis(db).params["condition_column"] is None


async def test_launch_rejects_a_condition_column_with_a_tab():
    user = make_user()
    ds = make_comparisons_dataset(declared=1)
    client, _db, project = make_client(as_user=user, comparisons_dataset=ds)
    payload = payload_for(project.id, ds.id)
    payload["params"] = {"condition_column": "a\tb"}

    async with client:
        res = await client.post(ENDPOINT, json=payload)

    assert res.status_code == 422


# ── Ligne de commande R ───────────────────────────────────────────────────────


def test_command_passes_the_condition_column():
    cmd = _cmd({"condition_column": "Genotype_Detail"})
    assert "--condition-col=Genotype_Detail" in cmd
    assert sum(a.startswith("--condition-col") for a in cmd) == 1


def test_command_keeps_a_column_starting_with_a_dash_as_one_value():
    """`--opt=valeur` : un nom commençant par « - » n'est pas lu comme une option."""
    cmd = _cmd({"condition_column": "-weird col"})
    assert "--condition-col=-weird col" in cmd
    assert "-weird col" not in cmd


@pytest.mark.parametrize("params", [{}, {"condition_column": None}, {"condition_column": ""}])
def test_command_without_condition_column_is_unchanged(params):
    """Analyses créées avant l'option : aucune option ajoutée, alias côté R."""
    cmd = _cmd(params)
    assert not any(a.startswith("--condition-col") for a in cmd)
    assert cmd == [
        "Rscript", "/app/r_scripts/run_multimethod_pipeline.R",
        "--counts", "/data/counts.tsv",
        "--samples", "/data/samples.tsv",
        "--comparisons", "/data/comparisons.tsv",
        "--outdir", "/data/out",
        "--design", "auto",
        "--fdr", "0.05",
        "--min-log2fc", "1.0",
        "--min-reads", "10",
        "--min-genes", "200",
        "--min-count", "5",
        "--min-reps", "2",
        "--threads", "4",
        "--species", "human",
        "--method", "all",
    ]


def test_command_reads_the_stored_params_of_a_launched_analysis():
    """Ce que la route stocke est exactement ce que le worker relit."""
    stored = AnalysisParams(condition_column="groupe", de_method="deseq2").model_dump()
    cmd = _cmd(stored)
    assert "--condition-col=groupe" in cmd
    assert cmd[cmd.index("--method") + 1] == "deseq2"


def test_r_script_declares_the_condition_col_option():
    """optparse refuse toute option inconnue : le worker et le script doivent
    évoluer ensemble, sinon chaque analyse du wizard planterait."""
    source = R_SCRIPT.read_text()
    assert re.search(r'make_option\(\s*"--condition-col"', source)


def test_r_script_aliases_include_groupe():
    """Le wizard auto-détecte `groupe` ; R doit l'accepter aussi."""
    source = R_SCRIPT.read_text()
    alias_line = next(
        line for line in source.splitlines() if '"condition", "group"' in line
    )
    assert '"groupe"' in alias_line
