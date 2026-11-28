#!/usr/bin/env python3
"""Fork-only lab watch: run every registered test and every GPU harness mode of a KytyPS5 build, compare with the
previous run (each new failure rerun once: a pass makes it flaky, not new), and write a short report (Nawid3333/KytyPS5, .github/workflows/lab-watch.yml).

    python lab/watch_run.py --build _Build/windows --out results.json [--previous prev.json] [--summary summary.md]
                            [--mode-timeout 300] [--ctest-timeout 300]

Vulkan comes from the environment (VK_DRIVER_FILES, e.g. Mesa lavapipe on a GitHub runner). A harness mode is one
process: exit 0 pass, 77 skip, other exit fail, no exit in time timeout; "unknown test selector" means the build
does not offer the mode. ctest entries come from ctest's JUnit file. A test is NEW-FAIL when it passed in the
previous results and fails now; FIXED the other way round. Standard library only.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
import xml.etree.ElementTree as ET
from pathlib import Path

RE_MODE = re.compile(r'strcmp\(argv\[1\], "(--[a-z0-9-]+)"\)')
SKIP_EXIT = 77
HARNESS = "shader_recompiler_compute_tests"
SOURCE = "tests/ShaderRecompilerComputeTests.cpp"
TIMING_LIST = Path(__file__).with_name("watch_timing_tests.txt")


def first_error(output: str) -> str:
    for line in output.splitlines():
        if re.search(r"failed at|LLVM ERROR|Error|error:|unknown test selector|Assertion|EXIT", line):
            return line.strip()[:200]
    lines = [line.strip() for line in output.splitlines() if line.strip()]
    return lines[-1][:200] if lines else ""


def run_modes(build: Path, source: Path, timeout: int) -> dict:
    exe = build / f"{HARNESS}.exe"
    if not exe.exists():
        exe = build / HARNESS
    modes = sorted({m for m in RE_MODE.findall(source.read_text(encoding="utf-8", errors="replace"))
                    if "death" not in m})
    results = {}
    for mode in modes:
        started = time.monotonic()
        try:
            proc = subprocess.run([str(exe), mode], capture_output=True, text=True, errors="replace",
                                  timeout=timeout, cwd=str(build))
            output = proc.stdout + proc.stderr
            if proc.returncode != 0 and "unknown test selector" in output:
                continue
            status = "pass" if proc.returncode == 0 else "skip" if proc.returncode == SKIP_EXIT else "fail"
            code = proc.returncode
        except subprocess.TimeoutExpired as error:
            output = error.stdout.decode("utf-8", "replace") if isinstance(error.stdout, bytes) else (error.stdout or "")
            status, code = "timeout", None
        results[f"mode:{mode}"] = {"status": status, "exit": code, "seconds": round(time.monotonic() - started, 1),
                                   "error": "" if status == "pass" else first_error(output)}
    return results


def run_ctest(build: Path, timeout: int) -> dict:
    junit = (build / "lab-watch-ctest.xml").resolve()
    proc = subprocess.run(["ctest", "--test-dir", str(build), "--timeout", str(timeout), "--output-junit",
                           str(junit), "--parallel", str(max(1, (os.cpu_count() or 2) // 2))],
                          capture_output=True, text=True, errors="replace")
    results = {}
    if not junit.exists():
        tail = " | ".join((proc.stdout + proc.stderr).strip().splitlines()[-6:])
        return {"ctest:(no junit)": {"status": "fail", "exit": proc.returncode,
                                     "error": f"ctest wrote no JUnit file: {tail}"[:400]}}
    for case in ET.parse(junit).getroot().iter("testcase"):
        name = case.get("name", "?")
        if case.find("failure") is not None or case.find("error") is not None:
            node = case.find("failure") if case.find("failure") is not None else case.find("error")
            status, error = "fail", (node.get("message") or "")[:200]
        elif case.find("skipped") is not None:
            status, error = "skip", ""
        else:
            status, error = "pass", ""
        out = case.find("system-out")
        if status == "fail" and out is not None and out.text:
            error = first_error(out.text) or error
        results[f"ctest:{name}"] = {"status": status, "seconds": float(case.get("time") or 0), "error": error}
    return results


def rerun_passes(build: Path, name: str, timeout: int) -> bool:
    kind, _, test = name.partition(":")
    if kind == "mode":
        exe = build / f"{HARNESS}.exe"
        command = [str(exe if exe.exists() else build / HARNESS), test]
    else:
        command = ["ctest", "--test-dir", str(build), "--timeout", str(timeout), "-R", f"^{re.escape(test)}$"]
    try:
        return subprocess.run(command, capture_output=True, timeout=timeout, cwd=str(build)).returncode == 0
    except subprocess.TimeoutExpired:
        return False


def compare(now: dict, before: dict | None) -> dict:
    if not before:
        return {"new_fail": [], "flaky": [], "fixed": [], "new_tests": [], "gone": []}
    old = before.get("tests", {})
    cur = now["tests"]
    bad = ("fail", "timeout")
    return {
        "new_fail": sorted(k for k, v in cur.items() if v["status"] in bad and old.get(k, {}).get("status") == "pass"),
        "flaky": [],
        "fixed": sorted(k for k, v in cur.items() if v["status"] == "pass" and old.get(k, {}).get("status") in bad),
        "new_tests": sorted(k for k in cur if k not in old),
        "gone": sorted(k for k in old if k not in cur),
    }


def summary_text(now: dict, diff: dict, before: dict | None) -> str:
    counts = {}
    for value in now["tests"].values():
        counts[value["status"]] = counts.get(value["status"], 0) + 1
    lines = [f"**lab-watch** upstream code at `{now['sha'][:9]}`"
             + (f" (previous `{before['sha'][:9]}`)" if before else " (first run, no previous results)"),
             f"harness patch: {now['patch']}; driver: {now['driver']}"
             + ("; NOT COMPARED: rebase lab/harness-skips.patch on this upstream" if now["patch"] == "FAILED" else ""),
             "counts: " + ", ".join(f"{k} {v}" for k, v in sorted(counts.items()))]
    if diff["new_fail"]:
        lines.append(f"**NEW FAILURES ({len(diff['new_fail'])})**:")
        lines += [f"- `{k}`: {now['tests'][k]['error']}" for k in diff["new_fail"]]
    if diff.get("timing"):
        lines.append("timing tests failing (not alerted, lab/watch_timing_tests.txt): "
                     + ", ".join(f"`{k}`" for k in diff["timing"]))
    if diff["flaky"]:
        lines.append("flaky (failed, then passed on a rerun): " + ", ".join(f"`{k}`" for k in diff["flaky"]))
    if diff["fixed"]:
        lines.append("fixed: " + ", ".join(f"`{k}`" for k in diff["fixed"]))
    if diff["new_tests"] and before:
        lines.append(f"new tests: {len(diff['new_tests'])}")
    if diff["gone"]:
        lines.append(f"gone: {len(diff['gone'])}")
    if os.environ.get("GITHUB_SERVER_URL") and os.environ.get("GITHUB_RUN_ID"):
        lines.append(f"run: {os.environ['GITHUB_SERVER_URL']}/{os.environ.get('GITHUB_REPOSITORY')}/actions/runs/"
                     f"{os.environ['GITHUB_RUN_ID']}")
    return "\n".join(lines) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--build", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--previous")
    ap.add_argument("--summary")
    ap.add_argument("--sha", default=os.environ.get("GITHUB_SHA", "local"))
    ap.add_argument("--patch", default="unknown")
    ap.add_argument("--mode-timeout", type=int, default=300)
    ap.add_argument("--ctest-timeout", type=int, default=300)
    ap.add_argument("--no-ctest", action="store_true")
    args = ap.parse_args()
    build = Path(args.build)
    tests = {} if args.no_ctest else run_ctest(build, args.ctest_timeout)
    tests.update(run_modes(build, Path(SOURCE), args.mode_timeout))
    now = {"sha": args.sha, "patch": args.patch, "driver": os.environ.get("VK_DRIVER_FILES", "system"),
           "tests": tests}
    before = None
    if args.previous and Path(args.previous).exists() and args.patch != "FAILED":
        before = json.loads(Path(args.previous).read_text(encoding="utf-8"))
    diff = compare(now, before)
    timing = {line.strip() for line in TIMING_LIST.read_text(encoding="utf-8").splitlines()
              if line.strip() and not line.startswith("#")} if TIMING_LIST.exists() else set()
    diff["timing"] = sorted(k for k in diff["new_fail"] if k in timing)
    diff["new_fail"] = [k for k in diff["new_fail"] if k not in timing]
    for name in list(diff["new_fail"]):
        timeout = args.mode_timeout if name.startswith("mode:") else args.ctest_timeout
        if rerun_passes(build, name, timeout):
            diff["new_fail"].remove(name)
            diff["flaky"].append(name)
    now["diff"] = diff
    modes = [v for k, v in tests.items() if k.startswith("mode:")]
    broken = sum("vkCreateInstance" in v.get("error", "") for v in modes)
    now["environment"] = "broken" if modes and broken * 2 > len(modes) else "ok"
    Path(args.out).write_text(json.dumps(now, indent=1, sort_keys=True), encoding="utf-8")
    text = summary_text(now, diff, before)
    if now["environment"] == "broken":
        text = (f"**ENVIRONMENT BROKEN**: {broken} of {len(modes)} harness modes could not create a Vulkan "
                f"instance; these results are not a test verdict.\n\n" + text)
    if args.summary:
        Path(args.summary).write_text(text, encoding="utf-8")
    print(text)
    return 2 if now["environment"] == "broken" else 0


if __name__ == "__main__":
    sys.exit(main())
