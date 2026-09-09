"""
`POST /datasets/{id}/rerun-enrichment` — la route doit atteindre le worker.

Le répertoire `app/worker/tasks/` masque le module `app/worker/tasks.py` adjacent :
Python résout `app.worker.tasks` sur le package, jamais sur le fichier. Le
package recharge donc explicitement le fichier via importlib et ré-expose ses
symboles un par un — et `_auto_run_enrichment` avait été oublié de cette liste.

Conséquence : `from app.worker.tasks import _auto_run_enrichment`, écrit dans le
corps de la route, levait `ImportError` à chaque appel. La route rendait 500 à
tout le monde, et le message d'erreur ne pointait pas vers la liste
d'exportations mais vers une ligne d'import d'apparence banale.

Le premier test épingle le ré-export lui-même : c'est le contrat que la prochaine
réorganisation du package doit préserver, et le seul endroit où sa rupture est
visible sans lancer la route entière.
"""

from unittest.mock import AsyncMock, patch
from uuid import uuid4

from tests.test_project_access_control import make_client, make_fixtures, make_user


def test_auto_run_enrichment_is_reexported_by_the_tasks_package():
    """`app.worker.tasks` doit exposer ce que la route importe.

    Assertion volontairement portée sur l'import lui-même plutôt que sur un
    appel : c'est l'import qui échouait, et une signature vérifiée ne dirait
    rien du symbole manquant.
    """
    from app.worker.tasks import _auto_run_enrichment

    assert callable(_auto_run_enrichment)


async def test_project_admin_reaches_the_worker():
    """Un membre ADMIN du projet traverse la barrière ET l'import.

    Avant, cette route répondait 500 quel que soit l'appelant : l'import était
    la première instruction du corps, donc il précédait même le contrôle
    d'accès.
    """
    owner = make_user()
    dataset, project = make_fixtures(owner_id=owner.id)
    dataset.dataset_metadata = {"min_log2fc": 1.5}
    client = make_client(caller=owner, dataset=dataset, project=project, member=None)

    runner = AsyncMock()
    with patch("app.worker.tasks._auto_run_enrichment", new=runner):
        async with client as c:
            resp = await c.post(f"/api/v1/datasets/{dataset.id}/rerun-enrichment")

    assert resp.status_code == 200, resp.text
    assert resp.json()["min_log2fc"] == 1.5
    runner.assert_awaited_once()
    # Le seuil vient de la métadonnée du dataset, pas de la valeur par défaut.
    assert runner.await_args.kwargs["min_log2fc"] == 1.5


async def test_outsider_never_reaches_the_worker():
    """Le refus doit précéder le déclenchement du calcul, pas le suivre.

    C'est la moitié de la barrière que le test d'accès générique ne peut pas
    voir : il constate le 403, pas le fait que rien n'a tourné derrière.
    """
    outsider = make_user()
    dataset, project = make_fixtures(owner_id=uuid4())
    client = make_client(caller=outsider, dataset=dataset, project=project, member=None)

    runner = AsyncMock()
    with patch("app.worker.tasks._auto_run_enrichment", new=runner):
        async with client as c:
            resp = await c.post(f"/api/v1/datasets/{dataset.id}/rerun-enrichment")

    assert resp.status_code == 403
    runner.assert_not_awaited()
