"""
Tests de la comptabilisation du quota de comparaisons.

`increment_analysis_usage` fusionnait trois responsabilités : l'UPDATE
conditionnel atomique, la politique HTTP (`raise HTTPException(429)`) et le
contrôle de transaction (`rollback`/`commit`). Rien de tout cela n'est
utilisable depuis un worker Celery : une HTTPException levée là abandonnerait
une analyse dont le calcul coûteux a déjà tourné, et annulerait les datasets
DEG déjà enregistrés.

`try_increment_analysis_usage` isole l'UPDATE : pas de raise, pas de commit,
pas de rollback. La politique appartient à l'appelant.
"""

from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest
from fastapi import HTTPException

from app.api.deps.subscription import (
    increment_analysis_usage,
    try_increment_analysis_usage,
)
from app.models.models import SubscriptionPlan, User, UserRole


def make_user(
    plan: SubscriptionPlan = SubscriptionPlan.STARTER,
    role: UserRole = UserRole.USER,
    used: int = 0,
) -> User:
    """Utilisateur non persisté. `analyses_quota` est une property calculée
    depuis `subscription_plan` et `role` : STARTER=30, TEAM=150,
    ON_PREMISE=None, rôles admin=None."""
    u = User()
    u.id = uuid4()
    u.email = "quota@example.com"
    u.role = role
    u.subscription_plan = plan
    u.analyses_used_this_month = used
    return u


def db_returning(new_count: int | None) -> AsyncMock:
    """AsyncSession mockée dont l'UPDATE ... RETURNING rend `new_count`.
    `None` simule la ligne non mise à jour (quota atteint)."""
    db = AsyncMock()
    db.add = MagicMock()
    db.commit = AsyncMock()
    db.rollback = AsyncMock()
    db.refresh = AsyncMock()
    result = MagicMock()
    result.scalar = MagicMock(return_value=new_count)
    db.execute = AsyncMock(return_value=result)
    return db


# ── try_increment_analysis_usage ──────────────────────────────────────────


async def test_try_increment_returns_true_and_writes_when_under_quota():
    user = make_user(used=5)
    db = db_returning(6)

    assert await try_increment_analysis_usage(user, db) is True
    db.execute.assert_awaited_once()


async def test_try_increment_returns_false_when_quota_blocks():
    user = make_user(used=30)  # STARTER: quota 30, saturé
    db = db_returning(None)

    assert await try_increment_analysis_usage(user, db) is False


async def test_try_increment_skips_write_for_unlimited_plan():
    user = make_user(plan=SubscriptionPlan.ON_PREMISE, used=999)
    db = db_returning(None)

    assert await try_increment_analysis_usage(user, db) is True
    db.execute.assert_not_awaited()


async def test_try_increment_skips_write_for_admin_role():
    user = make_user(role=UserRole.ADMIN, used=999)
    db = db_returning(None)

    assert await try_increment_analysis_usage(user, db) is True
    db.execute.assert_not_awaited()


async def test_try_increment_never_controls_the_transaction():
    """Le contrat qui rend la fonction utilisable dans un worker Celery :
    la transaction appartient à l'appelant, dans les deux issues."""
    for new_count, used in ((6, 5), (None, 30)):
        user = make_user(used=used)
        db = db_returning(new_count)

        await try_increment_analysis_usage(user, db)

        db.commit.assert_not_awaited()
        db.rollback.assert_not_awaited()


async def test_try_increment_never_raises_when_quota_blocks():
    user = make_user(used=30)
    db = db_returning(None)

    # Ne doit pas lever : un raise ici abandonnerait une analyse déjà calculée.
    assert await try_increment_analysis_usage(user, db) is False


# ── increment_analysis_usage (politique HTTP, inchangée) ──────────────────


async def test_http_increment_raises_429_and_rolls_back_when_blocked():
    user = make_user(used=30)
    db = db_returning(None)

    with pytest.raises(HTTPException) as exc:
        await increment_analysis_usage(user, db)

    assert exc.value.status_code == 429
    db.rollback.assert_awaited_once()


async def test_http_increment_leaves_the_commit_to_its_caller():
    """Le décompte s'écrit dans la transaction de l'appelant, sans la clore.

    C'est ce qui rend le `rollback` de la variante bloquée utile : l'insert du
    dataset est encore en vol au moment où le quota refuse. Un commit ici le
    rendait durable, et le 429 arrivait sur un dataset déjà créé.
    """
    user = make_user(used=5)
    db = db_returning(6)

    await increment_analysis_usage(user, db)

    db.execute.assert_awaited_once()
    db.commit.assert_not_awaited()
    db.rollback.assert_not_awaited()


async def test_http_increment_is_noop_for_unlimited_plan():
    user = make_user(plan=SubscriptionPlan.ON_PREMISE, used=999)
    db = db_returning(None)

    await increment_analysis_usage(user, db)

    db.execute.assert_not_awaited()
    db.commit.assert_not_awaited()
