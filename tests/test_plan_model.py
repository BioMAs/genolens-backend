"""Tests for the new STARTER / TEAM / ON_PREMISE plan model."""
import pytest
from uuid import uuid4
from app.models.models import User, UserRole, SubscriptionPlan


def make_user(plan: SubscriptionPlan) -> User:
    u = User()
    u.id = uuid4()
    u.email = "test@example.com"
    u.role = UserRole.USER
    u.subscription_plan = plan
    u.ai_interpretations_used = 0
    u.ai_tokens_purchased = 0
    u.ai_tokens_used = 0
    u.comparisons_used_this_month = 0
    u.is_active = True
    return u


def test_plan_enum_values_exist():
    assert SubscriptionPlan.STARTER == "STARTER"
    assert SubscriptionPlan.TEAM == "TEAM"
    assert SubscriptionPlan.ON_PREMISE == "ON_PREMISE"

def test_basic_premium_advanced_do_not_exist():
    with pytest.raises(AttributeError):
        _ = SubscriptionPlan.BASIC
    with pytest.raises(AttributeError):
        _ = SubscriptionPlan.PREMIUM
    with pytest.raises(AttributeError):
        _ = SubscriptionPlan.ADVANCED

def test_starter_cannot_use_ai():
    u = make_user(SubscriptionPlan.STARTER)
    assert u.can_use_ai is False

def test_team_can_use_ai():
    u = make_user(SubscriptionPlan.TEAM)
    assert u.can_use_ai is True

def test_on_premise_can_use_ai():
    u = make_user(SubscriptionPlan.ON_PREMISE)
    assert u.can_use_ai is True

def test_admin_can_use_ai_regardless_of_plan():
    u = make_user(SubscriptionPlan.STARTER)
    u.role = UserRole.ADMIN
    assert u.can_use_ai is True

def test_starter_can_launch_analyses():
    u = make_user(SubscriptionPlan.STARTER)
    assert u.can_launch_analyses is True

def test_team_can_launch_analyses():
    u = make_user(SubscriptionPlan.TEAM)
    assert u.can_launch_analyses is True

def test_starter_quota_is_30():
    u = make_user(SubscriptionPlan.STARTER)
    assert u.comparisons_quota == 30

def test_team_quota_is_150():
    u = make_user(SubscriptionPlan.TEAM)
    assert u.comparisons_quota == 150

def test_on_premise_quota_is_none():
    u = make_user(SubscriptionPlan.ON_PREMISE)
    assert u.comparisons_quota is None

def test_admin_quota_is_none():
    u = make_user(SubscriptionPlan.STARTER)
    u.role = UserRole.ADMIN
    assert u.comparisons_quota is None

def test_comparisons_remaining_full():
    u = make_user(SubscriptionPlan.STARTER)
    u.comparisons_used_this_month = 0
    assert u.comparisons_remaining == 30

def test_comparisons_remaining_partial():
    u = make_user(SubscriptionPlan.STARTER)
    u.comparisons_used_this_month = 20
    assert u.comparisons_remaining == 10

def test_comparisons_remaining_exhausted():
    u = make_user(SubscriptionPlan.STARTER)
    u.comparisons_used_this_month = 30
    assert u.comparisons_remaining == 0

def test_comparisons_remaining_unlimited():
    u = make_user(SubscriptionPlan.ON_PREMISE)
    u.comparisons_used_this_month = 9999
    assert u.comparisons_remaining is None

def test_starter_cannot_use_multi_comparison():
    u = make_user(SubscriptionPlan.STARTER)
    assert u.can_use_multi_comparison is False

def test_team_can_use_multi_comparison():
    u = make_user(SubscriptionPlan.TEAM)
    assert u.can_use_multi_comparison is True

def test_starter_cannot_export_pdf():
    u = make_user(SubscriptionPlan.STARTER)
    assert u.can_export_advanced is False

def test_team_can_export_pdf():
    u = make_user(SubscriptionPlan.TEAM)
    assert u.can_export_advanced is True

def test_starter_max_projects():
    u = make_user(SubscriptionPlan.STARTER)
    assert u.max_projects == 15

def test_team_max_projects():
    u = make_user(SubscriptionPlan.TEAM)
    assert u.max_projects is None

def test_starter_max_datasets_per_project():
    u = make_user(SubscriptionPlan.STARTER)
    assert u.max_datasets_per_project == 5

def test_team_max_datasets_per_project():
    u = make_user(SubscriptionPlan.TEAM)
    assert u.max_datasets_per_project is None
