"""
Le pipeline R crée un dataset DEG par comparaison
(`app/worker/tasks.py`, boucle sur `comparisons/`) et ne touchait pas au
compteur de quota. Un compte STARTER pouvait donc produire un nombre illimité
de comparaisons par le chemin normal en gardant un compteur à zéro.

Ces tests portent sur l'aide `_count_pipeline_comparison` extraite de la
boucle : la boucle elle-même vit dans une fonction de 400 lignes qui lance des
sous-processus R, et n'est pas testable en isolation.
"""

import logging
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

from app.models.models import SubscriptionPlan, User, UserRole
from app.worker.tasks import _count_pipeline_comparison


def make_user(
    plan: SubscriptionPlan = SubscriptionPlan.STARTER,
    used: int = 0,
) -> User:
    u = User()
    u.id = uuid4()
    u.email = "worker@example.com"
    u.role = UserRole.USER
    u.subscription_plan = plan
    u.comparisons_used_this_month = used
    return u


def db_returning(new_count: int | None) -> AsyncMock:
    db = AsyncMock()
    db.commit = AsyncMock()
    db.rollback = AsyncMock()
    result = MagicMock()
    result.scalar = MagicMock(return_value=new_count)
    db.execute = AsyncMock(return_value=result)
    return db


async def test_counts_one_comparison_when_under_quota():
    user = make_user(used=3)
    db = db_returning(4)

    await _count_pipeline_comparison(user, db, "cond_A_vs_cond_B")

    db.execute.assert_awaited_once()


async def test_does_not_raise_when_quota_is_exhausted(caplog):
    """Le calcul est terminé : on enregistre le dataset et on log, on n'annule
    jamais une analyse déjà calculée pour un dépassement de quota."""
    user = make_user(used=30)
    db = db_returning(None)

    with caplog.at_level(logging.WARNING):
        await _count_pipeline_comparison(user, db, "cond_A_vs_cond_B")

    assert "quota" in caplog.text.lower()
    db.rollback.assert_not_awaited()


async def test_no_op_when_user_is_none():
    """Utilisateur local introuvable : jamais de plantage d'analyse pour un
    problème de comptage."""
    db = db_returning(4)

    await _count_pipeline_comparison(None, db, "cond_A_vs_cond_B")

    db.execute.assert_not_awaited()


async def test_leaves_transaction_control_to_the_caller():
    user = make_user(used=3)
    db = db_returning(4)

    await _count_pipeline_comparison(user, db, "cond_A_vs_cond_B")

    db.commit.assert_not_awaited()
    db.rollback.assert_not_awaited()
