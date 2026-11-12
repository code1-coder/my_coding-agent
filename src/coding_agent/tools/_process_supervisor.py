"""Linux Shell supervisor, executed in a dedicated process (never in the agent).

Subreaper adoption keeps double-fork/setsid descendants waitable even after
their parents exit. Only waitpid's ECHILD confirms that the tree is empty.
The private socket carries stop requests and completion evidence separately
from command stdout/stderr.
"""

from __future__ import annotations

import ctypes
import json
import os
from pathlib import Path
import select
import signal
import socket
import subprocess
import sys
import time


def supervise(command: str, control: socket.socket) -> dict:
    libc = ctypes.CDLL(None, use_errno=True)
    # PR_SET_CHILD_SUBREAPER: set before launching any command.
    if libc.prctl(36, 1, 0, 0, 0) != 0:
        raise OSError(ctypes.get_errno(), "Cannot enable child subreaper")
    children_file = Path(f"/proc/self/task/{os.getpid()}/children")
    children_file.read_text()  # Fail before launch if kernel evidence is unavailable.
    os.set_inheritable(control.fileno(), False)
    shell = subprocess.Popen(command, shell=True, stdin=subprocess.DEVNULL)
    shell_status = None
    stopping = False
    deadline = None
    while True:
        # Reap *all* adopted children, including descendants outside the Shell's
        # original session/group. A closed output pipe is never completion proof.
        while True:
            try:
                pid, status = os.waitpid(-1, os.WNOHANG)
            except ChildProcessError:
                shell.returncode = shell_status
                return {"exit_code": shell_status, "tree_stopped": True}
            if pid == 0:
                break
            if pid == shell.pid:
                shell_status = os.waitstatus_to_exitcode(status)

        if not stopping and select.select([control], [], [], 0)[0]:
            control.recv(16)  # EOF (agent gone) also requests cleanup.
            stopping = True
            deadline = time.monotonic() + 5
        if stopping:
            # Kill direct children, then repeat as their children are adopted.
            # No wait/reap occurs between listing and signalling: even an exited
            # child retains its PID until we reap it, preventing PID reuse here.
            for child in children_file.read_text().split():
                try:
                    os.kill(int(child), signal.SIGKILL)
                except ProcessLookupError:
                    pass
            if time.monotonic() >= deadline:
                raise RuntimeError("Command descendants did not stop within 5 seconds")
        time.sleep(0.01)


def main() -> None:
    with socket.socket(fileno=int(sys.argv[1])) as control:
        try:
            status = supervise(sys.argv[2], control)
        except Exception as exc:
            status = {"error": str(exc), "tree_stopped": False}
        try:
            control.sendall(json.dumps(status).encode() + b"\n")
        except (BrokenPipeError, ConnectionResetError):
            pass


if __name__ == "__main__":
    main()
