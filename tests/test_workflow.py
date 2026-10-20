from pathlib import Path

import yaml


def test_workflow_has_checked_4k_option_and_never_interpolates_inputs_in_shell():
    # BaseLoader preserves GitHub's YAML 'on' key instead of interpreting it as a boolean.
    data = yaml.load(Path(".github/workflows/generate-video.yml").read_text(), Loader=yaml.BaseLoader)
    inputs = data["on"]["workflow_dispatch"]["inputs"]
    assert set(inputs) == {"title", "description", "movies", "duration_minutes", "try_4k"}
    assert inputs["try_4k"]["type"] == "boolean"
    assert inputs["try_4k"]["default"] == "true"
    assert inputs["try_4k"]["required"] == "true"
    minutes = inputs["duration_minutes"]
    assert minutes["type"] == "choice"
    assert minutes["default"] == "Auto"
    assert minutes["required"] == "true"
    assert minutes["options"] == ["Auto", *map(str, range(1,31))]
    assert "100" not in minutes["options"]
    env = data["jobs"]["produce"]["env"]
    assert env["VIDEO_MINUTES"] == "${{ inputs.duration_minutes }}"
    assert env["TRY_4K"] == "${{ inputs.try_4k }}"
    # No truthy fallback may turn an explicitly unchecked value back on.
    assert "||" not in env["TRY_4K"]
    assert "VIDEO_RESOLUTION" not in env
    steps = data["jobs"]["produce"]["steps"]
    for step in steps:
        assert "${{ inputs." not in step.get("run", "")
    upload = next(step for step in steps if step.get("uses", "").startswith("actions/upload-artifact"))
    assert upload["with"]["path"] == "output/"
    assert "if" not in upload  # success only, never publish a failed partial package
    assert data["permissions"]["contents"] == "read"
