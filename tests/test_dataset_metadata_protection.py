"""
`PATCH /datasets/{id}` fusionnait tel quel le dict fourni par le client dans
`dataset_metadata`. Deux conséquences.

La première est un contournement : `POST /analyses` refuse une analyse dont le
fichier de comparaisons déclare plus de lignes (`rows`) qu'il ne reste de
quota. Un contrôle dont la seule entrée est modifiable par le client n'en est
pas un — un `PATCH {"dataset_metadata": {"rows": 1}}` suffisait, et un admin de
projet passe `_check_project_admin`.

La seconde est indépendante du quota : ces clés (comptages DEG, chemins de
figures, provenance) sont lues par toute l'application, et un client pouvait
écraser des métadonnées calculées.
"""

import logging
from uuid import uuid4

from app.api.endpoints.datasets import (
    _CLIENT_PROTECTED_METADATA_KEYS,
    _merge_client_metadata,
)

DATASET_ID = uuid4()


def test_rows_cannot_be_overwritten_by_a_client():
    """La clé que lit la barrière de quota au lancement."""
    merged = _merge_client_metadata({"rows": 500}, {"rows": 1}, DATASET_ID)

    assert merged["rows"] == 500


def test_every_protected_key_keeps_its_server_value():
    existing = {key: "server" for key in _CLIENT_PROTECTED_METADATA_KEYS}
    incoming = {key: "client" for key in _CLIENT_PROTECTED_METADATA_KEYS}

    merged = _merge_client_metadata(existing, incoming, DATASET_ID)

    assert merged == existing


def test_descriptive_keys_still_go_through():
    """Le seul client connu (EditDatasetModal.tsx) n'envoie que ces deux clés ;
    un déni de liste doit les laisser passer, ainsi que les clés inconnues."""
    merged = _merge_client_metadata(
        {"rows": 12},
        {"is_normalized": True, "contains_all_genes": False, "note": "libre"},
        DATASET_ID,
    )

    assert merged == {
        "rows": 12,
        "is_normalized": True,
        "contains_all_genes": False,
        "note": "libre",
    }


def test_protected_keys_absent_from_the_server_value_are_not_created():
    merged = _merge_client_metadata({}, {"comparisons": {"forge": {}}}, DATASET_ID)

    assert "comparisons" not in merged


def test_dropped_keys_are_logged(caplog):
    with caplog.at_level(logging.WARNING):
        _merge_client_metadata(
            {"rows": 12}, {"rows": 1, "comparisons": {}, "note": "libre"}, DATASET_ID
        )

    assert "rows" in caplog.text
    assert "comparisons" in caplog.text
    assert "note" not in caplog.text


def test_no_warning_when_nothing_was_dropped(caplog):
    with caplog.at_level(logging.WARNING, logger="app.api.endpoints.datasets"):
        _merge_client_metadata({"rows": 12}, {"is_normalized": True}, DATASET_ID)

    assert [r for r in caplog.records if r.name == "app.api.endpoints.datasets"] == []


def test_none_metadata_is_treated_as_empty():
    merged = _merge_client_metadata(None, {"is_normalized": True}, DATASET_ID)

    assert merged == {"is_normalized": True}


async def test_the_denylist_covers_what_ingestion_writes():
    """Épingle le contrat avec le producteur de métadonnées : si
    `get_file_metadata` gagne une clé, ce test le signale au lieu de laisser
    apparaître silencieusement une clé calculée écrasable."""
    from app.services.data_processor import data_processor

    tsv = b"comparison\tcondition1\tcondition2\na_vs_b\ta\tb\n"
    metadata = await data_processor.get_file_metadata(tsv, ".tsv")

    assert set(metadata) <= _CLIENT_PROTECTED_METADATA_KEYS
