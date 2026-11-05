"""
The images install from requirements.txt alone, so what it pins is what runs in production.

kombu is a dependency of celery that the file did not pin: a fresh build resolved kombu 5.7.0, which
refuses redis-py 5.0.1, and the r-worker container did not start. It is pinned to the version the
recette ran with, next to celery.
"""

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _pins(name: str) -> dict[str, str]:
    pins = {}
    for line in (ROOT / name).read_text().splitlines():
        match = re.match(r"^([A-Za-z0-9_.\-]+)(?:\[[^\]]*\])?==([^\s#;]+)", line.strip())
        if match:
            pins[match.group(1).lower()] = match.group(2)
    return pins


def test_kombu_is_pinned_to_the_version_the_recette_ran_with():
    pins = _pins("requirements.txt")
    assert pins.get("kombu") == "5.6.2"
    # Next to celery, and the pins it has to agree with are the ones that were there.
    assert (pins["celery"], pins["redis"]) == ("5.3.6", "5.0.1")
    names = list(pins)  # the pinned packages in file order, comments left out
    assert abs(names.index("kombu") - names.index("celery")) == 1


def test_every_image_stage_installs_from_requirements_txt_alone():
    """No other requirements file reaches an image: pinning kombu in requirements.txt is enough."""
    installs = re.findall(r"pip install[^\n]*", (ROOT / "Dockerfile").read_text())
    assert installs, "the Dockerfile installs nothing with pip?"
    for line in installs:
        assert "-r requirements.txt" in line, line
    copied = re.findall(r"^COPY\s+(requirements\S*)", (ROOT / "Dockerfile").read_text(), re.M)
    assert copied and set(copied) == {"requirements.txt"}
    # The development file only adds to it (CI and the tests), it never replaces a pin.
    dev = (ROOT / "requirements-dev.txt").read_text()
    assert re.search(r"^-r requirements\.txt$", dev, re.M)
    assert "kombu" not in _pins("requirements-dev.txt")
