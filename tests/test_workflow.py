from pathlib import Path

import yaml


def test_workflow_has_three_inputs_and_never_interpolates_inputs_in_shell():
    # BaseLoader preserves GitHub's YAML 'on' key instead of interpreting it as a boolean.
    data = yaml.load(Path(".github/workflows/generate-video.yml").read_text(), Loader=yaml.BaseLoader)
    assert set(data["on"]["workflow_dispatch"]["inputs"]) == {"title", "description", "movies"}
    steps = data["jobs"]["produce"]["steps"]
    for step in steps:
        assert "${{ inputs." not in step.get("run", "")
    upload = next(step for step in steps if step.get("uses", "").startswith("actions/upload-artifact"))
    assert upload["with"]["path"] == "output/"
    assert "if" not in upload  # success only, never publish a failed partial package
    assert data["permissions"]["contents"] == "read"
