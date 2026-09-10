#!/usr/bin/env python3
"""Reject stale or incomplete source recipes before external submission."""
import argparse
import base64
import re
import subprocess
import tomllib
from pathlib import Path

RECIPES = {
    "nix": "nix/default.nix",
    "conda": "conda/meta.yaml",
    "macports": "macports/Portfile",
}


def recipe_revision(kind, text):
    patterns = {
        "nix": r'rev = "([0-9a-f]{40})";',
        "conda": r'/archive/([0-9a-f]{40})\.tar\.gz',
        "macports": r'github\.setup\s+PhilipJohnBasile vecstore ([0-9a-f]{40})\b',
    }
    match = re.search(patterns[kind], text)
    if not match:
        raise ValueError(f"{kind}: an immutable source commit is required")
    return match[1]


def validate_recipe(kind, text, source_revision, source_version, requested_version):
    if "REPLACE_WITH" in text:
        raise ValueError(f"{kind}: placeholder checksums remain")
    if recipe_revision(kind, text) != source_revision:
        raise ValueError(f"{kind}: recipe source differs from the requested release ref")
    if source_version != requested_version:
        raise ValueError("Requested version differs from the source Cargo.toml version")
    patterns = {
        "nix": r'version = "([^"\n]+)";',
        "conda": r'set version = "([^"\n]+)"',
        "macports": r'(?m)^version\s+(\S+)',
    }
    match = re.search(patterns[kind], text)
    if not match or not (match[1] == source_version or (
        kind == "nix" and match[1].startswith(source_version + "-unstable-")
    )):
        raise ValueError(f"{kind}: recipe version differs from the source package version")
    if kind == "nix":
        for field in ("hash", "cargoHash"):
            match = re.search(r'\b' + field + r' = "sha256-([^"\n]+)";', text)
            try:
                valid = match and len(base64.b64decode(match[1], validate=True)) == 32
            except ValueError:
                valid = False
            if not valid:
                raise ValueError(f"nix: {field} must be a SHA-256 SRI digest, not a hex tarball hash")
    elif kind == "conda":
        if not re.search(r'(?m)^  sha256: [0-9a-f]{64}$', text):
            raise ValueError("conda: a source SHA-256 checksum is required")
    else:
        if not (re.search(r'\brmd160\s+[0-9a-f]{40}\b', text)
                and re.search(r'\bsha256\s+[0-9a-f]{64}\b', text)
                and re.search(r'\bsize\s+[1-9][0-9]*\b', text)):
            raise ValueError("macports: complete source checksums are required")
        if not re.search(r'(?m)^cargo\.crates\s+\\$', text):
            raise ValueError("macports: locked Cargo dependencies are required")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kind", choices=RECIPES, required=True)
    parser.add_argument("--version", required=True)
    parser.add_argument("--ref", required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    try:
        revision = subprocess.check_output(
            ["git", "rev-parse", "--verify", "--end-of-options", f"{args.ref}^{{commit}}"],
            cwd=root, text=True, stderr=subprocess.PIPE,
        ).strip()
        manifest = tomllib.loads(subprocess.check_output(
            ["git", "show", f"{revision}:Cargo.toml"], cwd=root, text=True,
        ))
        text = (root / RECIPES[args.kind]).read_text()
        validate_recipe(args.kind, text, revision, manifest["package"]["version"], args.version)
        if args.kind == "macports":
            lock = tomllib.loads(subprocess.check_output(
                ["git", "show", f"{revision}:Cargo.lock"], cwd=root, text=True,
            ))
            expected = {(p["name"], p["version"], p["checksum"]) for p in lock["package"] if p.get("source", "").startswith("registry+")}
            actual = set(re.findall(r'(?m)^    (\S+) (\S+) ([0-9a-f]{64})(?: \\)?$', text))
            if actual != expected:
                raise ValueError("macports: Cargo crate list differs from the pinned source lockfile")
    except (ValueError, subprocess.CalledProcessError, OSError) as exc:
        parser.exit(1, f"Packaging validation failed: {exc}\n")
    print(f"{args.kind}: source {revision}, package {args.version}; metadata checks passed")


if __name__ == "__main__":
    main()
