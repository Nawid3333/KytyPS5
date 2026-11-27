#!/usr/bin/env python3
"""Fork-only lab watch: print the CMake targets that the registered ctest entries run (space separated), so the
watch builds exactly what ctest needs. macOS-only targets are left out.

    python lab/test_targets.py [CMakeLists.txt]
"""
import re
import sys
from pathlib import Path

MAC_ONLY = {"macos_sse4a_tests"}

text = Path(sys.argv[1] if len(sys.argv) > 1 else "CMakeLists.txt").read_text(encoding="utf-8", errors="replace")
targets = sorted(set(re.findall(r"\$<TARGET_FILE:([A-Za-z0-9_]+)>", text)) - MAC_ONLY)
print(" ".join(targets))
