"""Fix 9: license, dependency conflict, Docker gaps, README wording, GPU label."""
import re
from pathlib import Path

from quantize.providers import gpu_session_active

ROOT = Path(__file__).resolve().parent.parent


def test_license_is_apache_2():
    text = (ROOT / "LICENSE").read_text()
    assert "Apache License" in text and "Version 2.0, January 2004" in text


def test_only_one_onnxruntime_build_installed():
    reqs = (ROOT / "requirements.txt").read_text()
    active = [ln.split("#")[0].strip() for ln in reqs.splitlines() if ln.split("#")[0].strip()]
    assert "onnxruntime==1.21.0" in active
    assert not any(ln.startswith("onnxruntime-gpu") for ln in active)


def test_readme_wording_and_numbers():
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    low = readme.lower()
    assert "production-ready" not in low and "enterprise-grade" not in low
    assert "120 MB" not in readme and "479 MB" not in readme
    assert "199.6" not in readme
    assert re.search(r"F1 / EM.*\*\*Unproven", readme)


def test_docker_mounts_models_and_single_worker():
    compose = (ROOT / "docker-compose.yml").read_text()
    assert "./models:/app/models" in compose and "./data:/app/data" in compose
    assert '"--workers", "1"' in (ROOT / "Dockerfile").read_text()


def test_cpu_fallback_not_labelled_gpu():
    assert gpu_session_active(["CUDAExecutionProvider", "CPUExecutionProvider"])
    assert not gpu_session_active(["CPUExecutionProvider"])
    assert not gpu_session_active([])
    src = (ROOT / "quantize" / "quantize_int8.py").read_text()
    assert "gpu_session_active(sess.get_providers())" in src
