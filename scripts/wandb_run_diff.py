#!/usr/bin/env python3
"""
Minimal, order-insensitive diff of two W&B run configs.
Prints only what changed, in YAML, git-style unified format.

pip install wandb dictdiffer pyyaml
"""

import wandb
import yaml
from dictdiffer import diff as ddiff
import argparse


def clean_cfg(run):  # remove wandb internals
    return {k: v for k, v in run.config.items() if not k.startswith("_")}


def fetch(run_id, entity, project):
    return clean_cfg(wandb.Api().run(f"{entity}/{project}/{run_id}"))


CLR = {"add": "\033[32m", "remove": "\033[31m", "change": "\033[33m", "end": "\033[0m"}


def _yaml_lines(val):
    """Return YAML lines of *val* – always block style, sorted keys."""
    return (
        yaml.safe_dump(
            val,
            sort_keys=True,
            default_flow_style=False,
            indent=2,
        )
        .rstrip("\n")
        .splitlines()
    )


def _emit_lines(lines, color, prefix="| "):
    for ln in lines:
        print(f"{color}{prefix}{ln}{CLR['end']}")


def pretty_print(diffs):
    """
    Print dictdiffer diff items.
    Adds/removes show nested structures indented underneath.
    """
    for action, node, change in diffs:
        path = node
        if action == "change":
            old, new = change
            print(f"{CLR['change']}~ {path}: {old!r} → {new!r}{CLR['end']}")
            continue

        sign, color = ("+", CLR["add"]) if action == "add" else ("-", CLR["remove"])

        for k, v in change:
            full = f"{path}.{k}" if path else str(k)

            # primitive?
            if isinstance(v, (str, int, float, bool, type(None))):
                print(f"{color}{sign} {full}: {v!r}{CLR['end']}")
            else:
                # container → show a header then block-dump YAML underneath
                print(f"{color}{sign} {full}:{CLR['end']}")
                _emit_lines(_yaml_lines(v), color)


def main():
    parser = argparse.ArgumentParser(description="Diff two W&B run configs.")
    parser.add_argument("runA", help="Run ID or URL of the first run")
    parser.add_argument("runB", help="Run ID or URL of the second run")
    parser.add_argument("--entity", default="ucsd-wang-lab-lm", help="W&B entity")
    args = parser.parse_args()

    entity = args.entity

    def extract_run_id(run_or_url):
        # If input is a URL possibly with query parameters, extract run id.
        base = run_or_url.split("?", 1)[0]
        # W&B run URLs typically have the format:
        # https://wandb.ai/{entity}/{project}/runs/{run_id}
        # or sometimes
        # https://wandb.ai/{entity}/{project}/run/{run_id}
        parts = base.rstrip("/").split("/")
        # The run id is the last part after 'runs' or 'run' segment.
        if len(parts) >= 1:
            if parts[-2] in ("runs", "run"):
                return parts[-1], parts[-3]
        raise Exception("Can't find run ID")

    runA, projectA = extract_run_id(args.runA)
    runB, projectB = extract_run_id(args.runB)

    a_cfg, b_cfg = fetch(runA, entity, projectA), fetch(runB, entity, projectB)
    pretty_print(ddiff(a_cfg, b_cfg))


if __name__ == "__main__":
    main()
