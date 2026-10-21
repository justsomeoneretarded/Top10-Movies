from pathlib import Path

import yaml


def test_workflow_has_checked_4k_option_and_never_interpolates_inputs_in_shell():
    # BaseLoader preserves GitHub's YAML 'on' key instead of interpreting it as a boolean.
    data = yaml.load(Path(".github/workflows/generate-video.yml").read_text(), Loader=yaml.BaseLoader)
    inputs = data["on"]["workflow_dispatch"]["inputs"]
    assert set(inputs) == {"title", "description", "movies", "content_type", "narrator", "duration_minutes", "try_4k"}
    assert inputs["content_type"]["type"] == "choice"
    assert inputs["content_type"]["default"] == "Movies"
    assert inputs["content_type"]["required"] == "true"
    assert inputs["content_type"]["options"] == ["Movies", "Shows"]
    assert inputs["narrator"]["type"] == "choice"
    assert inputs["narrator"]["default"] == "Warguy"
    assert inputs["narrator"]["required"] == "true"
    assert inputs["narrator"]["options"] == ["Warguy", "Kit Laser", "Female"]
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
    assert env["VIDEO_TYPE"] == "${{ inputs.content_type }}"
    assert env["NARRATOR"] == "${{ inputs.narrator }}"
    # No truthy fallback may turn an explicitly unchecked value back on.
    assert "||" not in env["TRY_4K"]
    assert "VIDEO_RESOLUTION" not in env
    steps = data["jobs"]["produce"]["steps"]
    for step in steps:
        assert "${{ inputs." not in step.get("run", "")
    upload = next(step for step in steps if step.get("uses", "").startswith("actions/upload-artifact"))
    assert upload["with"]["path"] == "output/"
    assert upload["with"]["retention-days"] == "2"
    assert "if" not in upload  # success only, never publish a failed partial package
    python_setup = next(step for step in steps if step.get("uses", "").startswith("actions/setup-python"))
    assert python_setup["with"]["cache"] == "pip"
    model_cache = next(step for step in steps if step.get("name") == "Restore model cache")
    assert model_cache["with"]["path"] == "work/models"
    assert model_cache["with"]["key"].startswith("models-cpu-v3-thumbnails-")
    checkpoints = next(step for step in steps if step.get("uses", "").startswith("actions/cache/save"))
    assert checkpoints["with"]["path"] == "work/jobs"
    assert checkpoints["with"]["key"] == "production-v2-${{ github.run_id }}-${{ github.run_attempt }}"
    assert data["permissions"]["contents"] == "read"


def test_speech_diagnostics_are_failure_only_and_optional_models_are_separately_cached():
    data = yaml.load(Path(".github/workflows/generate-video.yml").read_text(), Loader=yaml.BaseLoader)
    steps = data["jobs"]["produce"]["steps"]
    diagnostics = next(step for step in steps if step.get("name") == "Speech diagnostics if production fails")
    assert diagnostics["if"] == "failure()"
    assert diagnostics["with"]["retention-days"] == "2"
    assert "speech-verification.json" in diagnostics["with"]["path"]
    assert "last-unverified.wav" in diagnostics["with"]["path"]
    assert "models" not in diagnostics["with"]["path"]
    model_restore = next(step for step in steps if step.get("id") == "speech-verifier-cache")
    assert model_restore["with"]["key"].startswith("models-cpu-")
    assert model_restore["with"]["path"] == "work/speech-verifier-model"
    model_save = next(step for step in steps if step.get("name") == "Save optional speech recognizer when downloaded")
    assert "cache-hit != 'true'" in model_save["if"]
    assert "config.json" in model_save["if"]
    assert model_save["with"] == model_restore["with"]
