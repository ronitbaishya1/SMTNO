#!/usr/bin/env python3
"""Compare local SMTNO research source files against a checked-out GitHub repository."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
from collections import defaultdict
from pathlib import Path

SOURCE_SUFFIXES = {
    '.py', '.edp', '.ipynb', '.sh', '.jl', '.m', '.f90', '.f', '.cpp', '.cc',
    '.c', '.h', '.hpp', '.cu', '.yaml', '.yml', '.toml', '.md', '.tex', '.json'
}
SOURCE_FILENAMES = {'requirements.txt', 'environment.yml', 'environment.yaml', 'Dockerfile', 'Makefile'}
SKIP_DIRS = {
    '.git', '.venv', 'venv', 'env', '__pycache__', '.ipynb_checkpoints',
    '.pytest_cache', '.mypy_cache', 'node_modules', '.idea', '.vscode',
    'results', 'result', 'data', 'datasets', 'checkpoints', 'figures',
    'plots', 'outputs', 'output', 'runs', 'logs', 'generated', 'cache',
    'wandb', 'tensorboard', 'build', 'dist', 'site-packages'
}


def wanted(p: Path) -> bool:
    return p.suffix.lower() in SOURCE_SUFFIXES or p.name in SOURCE_FILENAMES


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def notebook_code_digest(path: Path) -> str | None:
    """Notebook source-equivalence ignores outputs and execution state."""
    if path.suffix.lower() != '.ipynb':
        return None
    try:
        obj = json.loads(path.read_text(encoding='utf-8'))
        cells = []
        for cell in obj.get('cells', []):
            if cell.get('cell_type') not in ('code', 'markdown'):
                continue
            source = cell.get('source', '')
            if isinstance(source, list):
                source = ''.join(source)
            cells.append((cell.get('cell_type'), source.rstrip()))
        return hashlib.sha256(json.dumps(cells, ensure_ascii=False, sort_keys=True).encode()).hexdigest()
    except (ValueError, OSError, UnicodeDecodeError):
        return None


def scan_local(root: Path, repo: Path, own: Path) -> list[Path]:
    paths = []
    for p in root.rglob('*'):
        if not p.is_file() or not wanted(p):
            continue
        if p.resolve() == own.resolve():
            continue  # Do not report this auditing script as missing.
        if repo == p or repo in p.parents:
            continue
        relative = p.relative_to(root)
        if any(part in SKIP_DIRS for part in relative.parts[:-1]):
            continue
        paths.append(p)
    return sorted(paths)


def tracked_repo_files(repo: Path) -> list[Path]:
    result = subprocess.run(
        ['git', '-C', str(repo), 'ls-files', '-z'],
        capture_output=True, check=True,
    )
    names = [Path(x.decode('utf-8', errors='replace')) for x in result.stdout.split(b'\0') if x]
    return sorted(repo / n for n in names if wanted(n) and (repo / n).is_file())


def save_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    with path.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description='Audit Mac SMTNO source files against GitHub checkout.')
    parser.add_argument('--root', default='/Users/ronitbaishya/Desktop/IIT Guwahati/SMTNO')
    parser.add_argument('--repo', default=None, help='Cloned GitHub checkout; defaults to ROOT/github_source')
    parser.add_argument('--outdir', default=None, help='Defaults to ROOT/results/github_audit')
    args = parser.parse_args()
    root = Path(args.root).expanduser().resolve()
    repo = Path(args.repo).expanduser().resolve() if args.repo else root / 'github_source'
    outdir = Path(args.outdir).expanduser().resolve() if args.outdir else root / 'results' / 'github_audit'
    if not root.is_dir():
        raise SystemExit(f'Mac project root does not exist: {root}')
    if not (repo / '.git').exists():
        raise SystemExit(f'GitHub checkout not found: {repo}\nClone it into github_source first.')
    outdir.mkdir(parents=True, exist_ok=True)

    local_files = scan_local(root, repo, Path(__file__).resolve())
    git_files = tracked_repo_files(repo)
    by_name = defaultdict(list)
    for p in git_files:
        by_name[p.name].append(p)

    local_rows, duplicates = [], []
    for p in local_files:
        candidates = by_name.get(p.name, [])
        digest = sha(p)
        matches = [g for g in candidates if sha(g) == digest]
        # Executed Colab notebooks can differ only in their output/metadata.
        if not matches and p.suffix.lower() == '.ipynb':
            normalized = notebook_code_digest(p)
            matches = [g for g in candidates if normalized and notebook_code_digest(g) == normalized]
        status = 'MATCH' if matches else ('DIFFERENT_CONTENT' if candidates else 'MISSING_IN_GITHUB')
        local_rows.append({
            'status': status,
            'mac_path': str(p.relative_to(root)),
            'github_candidate_paths': ' | '.join(str(g.relative_to(repo)) for g in candidates),
            'sha256_first12': digest[:12],
        })
    groups = defaultdict(list)
    for p in local_files:
        groups[p.name].append(p)
    for name, paths in sorted(groups.items()):
        if len(paths) > 1:
            unique_hashes = len({sha(p) for p in paths})
            duplicates.append({
                'filename': name,
                'versions': len(paths),
                'different_versions': 'YES' if unique_hashes > 1 else 'NO',
                'mac_paths': ' | '.join(str(p.relative_to(root)) for p in paths),
            })

    missing = [r for r in local_rows if r['status'] == 'MISSING_IN_GITHUB']
    changed = [r for r in local_rows if r['status'] == 'DIFFERENT_CONTENT']
    matches = [r for r in local_rows if r['status'] == 'MATCH']
    local_names = {p.name for p in local_files}
    github_only = [
        {'github_path': str(p.relative_to(repo))}
        for p in git_files if p.name not in local_names
    ]
    fields = ['status', 'mac_path', 'github_candidate_paths', 'sha256_first12']
    save_csv(outdir / 'all_local_source_files.csv', local_rows, fields)
    save_csv(outdir / 'missing_in_github.csv', missing, fields)
    save_csv(outdir / 'different_versions.csv', changed, fields)
    save_csv(outdir / 'duplicate_local_names.csv', duplicates, ['filename','versions','different_versions','mac_paths'])
    save_csv(outdir / 'github_only.csv', github_only, ['github_path'])
    summary = {
        'mac_source_files': len(local_files),
        'github_tracked_source_files': len(git_files),
        'matching_local_files': len(matches),
        'missing_from_github': len(missing),
        'different_content': len(changed),
        'duplicate_filenames_in_mac': len(duplicates),
        'github_only_filenames': len(github_only),
        'note': 'Comparison matches by filename and content because Mac and GitHub folders have different layouts. Generated data/results are ignored.'
    }
    (outdir / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n', encoding='utf-8')
    print('\nSMTNO source audit')
    print('-' * 40)
    for k,v in summary.items():
        if k != 'note':
            print(f'{k.replace("_", " ").capitalize()}: {v}')
    print(f'\nReports saved to: {outdir}')
    if missing:
        print('\nMISSING IN GITHUB (first 30):')
        for r in missing[:30]:
            print('  ', r['mac_path'])
    if changed:
        print('\nDIFFERENT VERSIONS (first 30):')
        for r in changed[:30]:
            print('  ', r['mac_path'], '->', r['github_candidate_paths'])
    if duplicates:
        print('\nDUPLICATE LOCAL FILENAMES (first 20):')
        for r in duplicates[:20]:
            print('  ', r['filename'], f"({r['versions']} copies)")
    if not missing and not changed:
        print('\nAll located Mac source files have matching content on GitHub.')


if __name__ == '__main__':
    main()
