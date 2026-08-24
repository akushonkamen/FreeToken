from __future__ import annotations

import sys
from collections.abc import Sequence
from typing import TextIO


def _print_help(file: TextIO) -> None:
    print(
        """usage: prom <command> [args]

Commands:
  serve       Start the Prometheus API server
  shell       Chat with a Prometheus server in the terminal
  ctl         Query and manage a running Prometheus server
  daemon      Run the Prometheus supervisor (persistent engine service)
  launch      Configure and launch an agent against a Prometheus server
  checkpoint  Convert an HF safetensors checkpoint to FTW
  bench       Run a micro-benchmark (e.g. "bench bw" = CPU vs PCIe bandwidth)

Use "prom <command> --help" for command-specific options.
Use "prom --version" to print the Prometheus version.""",
        file=file,
    )


def _run_serve(argv: list[str]) -> int:
    from prometheus.server import launch_server

    launch_server(argv=argv, prog="prom serve")
    return 0


def _run_shell(argv: list[str]) -> int:
    from prometheus.shell import main

    return main(argv, prog="prom shell")


def _run_launch(argv: list[str]) -> int:
    from prometheus.launch import main

    return main(argv, prog="prom launch")


def _run_checkpoint(argv: list[str]) -> int:
    from prometheus.checkpoint.__main__ import main

    return main(argv, prog="prom checkpoint")


def _run_ctl(argv: list[str]) -> int:
    from prometheus.control_cli import main

    return main(argv, prog="prom ctl")


def _run_daemon(argv: list[str]) -> int:
    from prometheus.daemon import main  # torch-free supervisor

    return main(argv, prog="prom daemon")


def _print_bench_help(file: TextIO) -> None:
    print(
        """usage: prom bench <subcommand> [args]

Subcommands:
  bw   Benchmark CPU vs PCIe bandwidth and pick the MoE backend (hybrid/offload)

Use "prom bench <subcommand> --help" for subcommand-specific options.""",
        file=file,
    )


def _run_bench(argv: list[str]) -> int:
    if not argv:
        _print_bench_help(sys.stderr)
        return 2
    sub = argv[0]
    if sub in {"-h", "--help"}:
        _print_bench_help(sys.stdout)
        return 0
    if sub == "bw":
        from prometheus.moe.benchbw import main

        return main(argv[1:], prog="prom bench bw")
    print(f"unknown prom bench subcommand: {sub}", file=sys.stderr)
    _print_bench_help(sys.stderr)
    return 2


COMMANDS = {
    "serve": "_run_serve",
    "shell": "_run_shell",
    "ctl": "_run_ctl",
    "daemon": "_run_daemon",
    "launch": "_run_launch",
    "checkpoint": "_run_checkpoint",
    "bench": "_run_bench",
}


def main(argv: Sequence[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if not args:
        _print_help(sys.stderr)
        return 2
    if args[0] in {"-h", "--help"}:
        _print_help(sys.stdout)
        return 0
    if args[0] in {"-V", "--version"}:
        from prometheus.version import __version__

        print(f"prometheus version {__version__}")
        return 0

    command = args[0]
    runner_name = COMMANDS.get(command)
    if runner_name is None:
        print(f"unknown prom command: {command}", file=sys.stderr)
        _print_help(sys.stderr)
        return 2

    runner = globals()[runner_name]
    return runner(args[1:])


if __name__ == "__main__":
    raise SystemExit(main())
