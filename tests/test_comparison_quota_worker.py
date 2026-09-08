"""
Le pipeline R ne touchait pas au compteur de quota : un compte STARTER pouvait
lancer autant d'analyses qu'il voulait par le chemin normal en gardant un
compteur à zéro.

**Une analyse coûte une unité.** C'est ce que déclare la grille tarifaire
(`app/config/pricing.json`, `billable_unit`) : « +1 par fichier accepté,
indépendamment du nombre de contrastes qu'il contient ». La première version
de ce décompte ajoutait une unité PAR COMPARAISON produite, ce qui faisait
qu'une analyse à 12 contrastes consommait 12 unités du plan — un STARTER à
30/mois épuisait son quota en 5 analyses. `test_counts_one_unit_whatever_the
_analysis_produced` épingle l'inverse et refuse le retour de ce comptage.

Ces tests portent sur l'aide `_count_pipeline_analysis` extraite de la boucle :
la boucle elle-même vit dans une fonction de 400 lignes qui lance des
sous-processus R, et n'est pas testable en isolation.

Deux invariants tirent dans des sens opposés : le compteur ne dépasse jamais le
quota, et une analyse déjà calculée n'est jamais annulée pour un dépassement.
D'où le plafonnement (`least`) plutôt qu'un incrément conditionnel. Le
troisième point est la fenêtre de verrou : un seul UPDATE, émis juste avant le
commit, parce que la boucle passe jusqu'à 1800 s par contraste dans
`subprocess.run`.
"""

import logging
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

from app.models.models import SubscriptionPlan, User, UserRole
from app.worker.tasks import _count_pipeline_analysis


def make_user(
    plan: SubscriptionPlan = SubscriptionPlan.STARTER,
    used: int = 0,
    role: UserRole = UserRole.USER,
) -> User:
    u = User()
    u.id = uuid4()
    u.email = "worker@example.com"
    u.role = role
    u.subscription_plan = plan
    u.analyses_used_this_month = used
    return u


def db_returning(new_count: int | None) -> AsyncMock:
    db = AsyncMock()
    db.commit = AsyncMock()
    db.rollback = AsyncMock()
    result = MagicMock()
    result.scalar = MagicMock(return_value=new_count)
    db.execute = AsyncMock(return_value=result)
    return db


def statement_of(db: AsyncMock):
    return db.execute.await_args.args[0]


async def test_counts_one_unit_whatever_the_analysis_produced():
    """Douze comparaisons produites, UNE unité décomptée.

    L'unité facturable est l'analyse, pas le contraste. Une seule requête au
    passage : la fenêtre de verrou sur la ligne `users` se réduit au commit.
    """
    user = make_user(used=3)
    db = db_returning(4)

    await _count_pipeline_analysis(user, db, 12)

    db.execute.assert_awaited_once()
    params = statement_of(db).compile().params
    values = list(params.values())
    assert 1 in values, f"l'increment n'est pas de 1 : {params}"
    assert (
        12 not in values
    ), f"le nombre de contrastes ne doit pas entrer dans le decompte : {params}"


async def test_the_counter_is_capped_at_the_quota():
    """`least(quota, used + 1)` : le compteur ne dépasse jamais le quota, sans
    quoi l'affichage et la facturation partiraient au-delà du plan.

    Le plafond mord encore, même à +1 : une analyse admise sur la réservation
    en vol peut terminer après qu'une autre a saturé le compteur.
    """
    user = make_user(used=30)
    db = db_returning(30)

    await _count_pipeline_analysis(user, db, 12)

    sql = str(statement_of(db)).lower()
    assert "least" in sql, sql


async def test_does_not_raise_when_the_quota_saturates(caplog):
    """Le calcul est terminé : on enregistre les datasets, on log, et on
    n'annule jamais une analyse déjà calculée pour un dépassement de quota."""
    user = make_user(used=30)
    db = db_returning(30)  # plafonné : deja au quota

    with caplog.at_level(logging.WARNING):
        await _count_pipeline_analysis(user, db, 12)

    assert "quota" in caplog.text.lower()
    db.rollback.assert_not_awaited()


async def test_no_warning_while_under_the_quota(caplog):
    user = make_user(used=3)
    db = db_returning(15)

    with caplog.at_level(logging.WARNING, logger="app.worker.tasks"):
        await _count_pipeline_analysis(user, db, 12)

    assert [r for r in caplog.records if r.name == "app.worker.tasks"] == []


async def test_no_op_when_user_is_none():
    """Utilisateur local introuvable : jamais de plantage d'analyse pour un
    problème de comptage."""
    db = db_returning(4)

    await _count_pipeline_analysis(None, db, 12)

    db.execute.assert_not_awaited()


async def test_no_op_when_no_comparison_was_produced():
    """Une analyse qui ne produit aucun DEG (mauvais mapping de conditions) ne
    doit rien décompter : on ne facture pas un résultat vide."""
    user = make_user(used=3)
    db = db_returning(4)

    await _count_pipeline_analysis(user, db, 0)

    db.execute.assert_not_awaited()


async def test_no_update_for_unlimited_users():
    user = make_user(plan=SubscriptionPlan.ON_PREMISE, used=999)
    db = db_returning(4)

    await _count_pipeline_analysis(user, db, 12)

    db.execute.assert_not_awaited()


async def test_no_update_for_privileged_roles():
    user = make_user(role=UserRole.ADMIN, used=999)
    db = db_returning(4)

    await _count_pipeline_analysis(user, db, 12)

    db.execute.assert_not_awaited()


async def test_leaves_transaction_control_to_the_caller():
    user = make_user(used=3)
    db = db_returning(15)

    await _count_pipeline_analysis(user, db, 12)

    db.commit.assert_not_awaited()
    db.rollback.assert_not_awaited()
