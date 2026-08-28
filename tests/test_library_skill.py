from __future__ import annotations

import re
import sys
from pathlib import Path

if sys.version_info >= (3, 11):
    import tomllib
else:  # pragma: no cover
    import tomli as tomllib

ROOT = Path(__file__).parents[1]
SKILL_ROOT = ROOT / ".library-skills" / "use-benchbro"


def test_library_skill_has_valid_structure_and_references() -> None:
    skill = (SKILL_ROOT / "SKILL.md").read_text(encoding="utf-8")
    assert skill.startswith("---\nname: use-benchbro\ndescription:")
    assert "\n---\n" in skill[4:]
    assert "TODO" not in skill

    references = re.findall(r"\]\((references/[^)]+)\)", skill)
    assert references == [
        "references/reliable-measurements.md",
        "references/advanced-patterns.md",
    ]
    assert all((SKILL_ROOT / reference).is_file() for reference in references)
    assert (SKILL_ROOT / "agents" / "openai.yaml").is_file()


def test_library_skill_is_mapped_into_distributions() -> None:
    pyproject = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    targets = pyproject["tool"]["hatch"]["build"]["targets"]

    assert targets["sdist"]["artifacts"] == [".library-skills"]
    assert targets["wheel"]["force-include"] == {
        ".library-skills": "benchbro/.agents/skills"
    }
