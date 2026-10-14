#!/usr/bin/env python3
"""
Generic ZMQ policy server for the Daksha UI.

Serves any trained policy (ACT, Diffusion, SmolVLA, Pi-0.5, ...) over ZMQ: the
policy type is read from the checkpoint's own config (`--policy-path`), not
from a CLI flag, so a single server binary covers every policy the UI trains.

This is a thin wrapper around smolvla/serve_smolvla_zmq.py, whose
implementation is already policy-agnostic (it loads the policy class via
`PreTrainedConfig.from_pretrained` + `get_policy_class`). Kept as a separate
entry point at the daksha root so app.py doesn't need a per-policy script.
"""

import runpy
import sys
from pathlib import Path

if __name__ == "__main__":
    smolvla_server = Path(__file__).resolve().parent / "smolvla" / "serve_smolvla_zmq.py"
    sys.argv[0] = str(smolvla_server)
    runpy.run_path(str(smolvla_server), run_name="__main__")
