#!/usr/bin/env python3
"""Compare an existing Mac SMTNO project with actual GitHub origin/main files.

Checks all project files and folders. Generated datasets, results, plots, weights,
logs and archives are inventoried but are classified as intentionally excluded
unless --include-generated is passed. Does not change or upload any files.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import os
import subprocess
from collections import Counter, defaultdict
from pathlib import Path

ROOT_DEFAULT = Path('/Users/ronitbaishya/Desktop/IIT Guwahati/SMTNO')
GENERATED_DIRS = {
    'results','result','data','datasets','outputs','output','figures',
    'checkpoints','logs','runs','wandb','tensorboard','__pycache__',
    '.ipynb_checkpoints','.venv','venv','node_modules','.pytest_cache',
    '.mypy_cache','.ruff_cache','build','dist'
}
GENERATED_SUFFIXES = {
    '.npz','.npy','.pt','.pth','.ckpt','.h5','.hdf5','.csv','.parquet',
    '.png','.jpg','.jpeg','.gif','.webp','.svg','.mp4','.mov','.avi',
    '.log','.zip','.tar','.gz','.7z','.pdf'
}
IGNORE_NAMES = {'.DS_Store','Thumbs.db'}


def git(repo: Path, *args: str) -> bytes:
    p = subprocess.run(['git','-C',str(repo),*args],capture_output=True)
    if p.returncode:
        raise RuntimeError(p.stderr.decode(errors='replace').strip())
    return p.stdout


def sha_file(path: Path) -> str:
    h=hashlib.sha256()
    with path.open('rb') as fh:
        for block in iter(lambda: fh.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def sha_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def map_paths(path: str) -> list[str]:
    # Possible repository locations after moving source into src/.
    out=[path]
    translations = [
        ('FEM/advection_diffusion_2d/', 'src/advection_diffusion/'),
        ('FEM/', 'src/advection_diffusion/'),
        ('Colab Notebooks/', 'colab/'),
        ('Colab/', 'colab/'),
        ('burgers/', 'src/burgers_pilots/'),
    ]
    for a,b in translations:
        if path.startswith(a):out.append(b+path[len(a):])
    bn=Path(path).name
    if bn.lower().startswith('step2'):
        out.append('src/burgers_pilots/'+bn)
    if bn.lower().startswith('step3'):
        out.append('src/advection_diffusion/'+bn)
    return list(dict.fromkeys(out))


def excluded(rel: str) -> bool:
    p=Path(rel)
    return any(q in GENERATED_DIRS for q in p.parts[:-1]) or p.suffix.lower() in GENERATED_SUFFIXES


def write_csv(path: Path, rows: list[dict], columns: list[str]):
    with path.open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=columns,extrasaction='ignore')
        w.writeheader(); w.writerows(rows)


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--root',type=Path,default=ROOT_DEFAULT)
    ap.add_argument('--repo',type=Path,default=None)
    ap.add_argument('--outdir',type=Path,default=None)
    ap.add_argument('--include-generated',action='store_true',help='Treat data/results/plots/checkpoints as missing if not on GitHub')
    a=ap.parse_args()
    root=a.root.expanduser().resolve()
    repo=(a.repo or root/'github_source').expanduser().resolve()
    out=(a.outdir or root/'results'/'github_full_audit').expanduser().resolve()
    if not root.is_dir():raise SystemExit('Mac project folder not found: '+str(root))
    if not (repo/'.git').exists():raise SystemExit('GitHub checkout not found: '+str(repo)+'\nClone it with: git clone https://github.com/ronitbaishya1/SMTNO.git "'+str(repo)+'"')
    # Compare against remote-tracking origin/main, not against uncommitted local checkout edits.
    try:
        git(repo,'fetch','--quiet','origin','main')
        commit=git(repo,'rev-parse','origin/main').decode().strip()
        remote_paths=[p.decode('utf-8') for p in git(repo,'ls-tree','-r','-z','--name-only','origin/main').split(b'\0') if p]
    except RuntimeError as e:raise SystemExit('Could not read GitHub origin/main: '+str(e))
    remote_content={p:sha_bytes(git(repo,'show','origin/main:'+p)) for p in remote_paths}
    by_name=defaultdict(list)
    by_hash=defaultdict(list)
    for p,h in remote_content.items():
        by_name[Path(p).name].append(p)
        by_hash[h].append(p)
    local_files=[]; local_dirs=[]
    out_rel=None
    if out.is_relative_to(root):out_rel=out.relative_to(root).parts
    repo_rel=repo.relative_to(root).parts if repo.is_relative_to(root) else None
    for cur,dirnames,filenames in os.walk(root):
        base=Path(cur); rel=base.relative_to(root)
        kept=[]
        for d in dirnames:
            entry=rel.parts+(d,)
            if d == '.git' or (repo_rel and entry[:len(repo_rel)]==repo_rel) or (out_rel and entry[:len(out_rel)]==out_rel):continue
            kept.append(d)
            local_dirs.append('/'.join(entry))
        dirnames[:]=kept
        for fn in filenames:
            if fn in IGNORE_NAMES:continue
            full=base/fn
            r=full.relative_to(root).as_posix()
            local_files.append((r,full))
    rows=[]; used_remote=set()
    for rel,full in sorted(local_files):
        is_generated=excluded(rel)
        if is_generated and not a.include_generated:
            rows.append(dict(local=rel,github='',status='INTENTIONALLY_EXCLUDED',size_bytes=full.stat().st_size,details='Generated data, plots, binary artifact or archived output'))
            continue
        sha=sha_file(full)
        candidates=[p for p in map_paths(rel) if p in remote_content]
        candidates.extend(p for p in by_name[full.name] if p not in candidates)
        matched=next((p for p in candidates if remote_content[p]==sha),None)
        if matched:
            status='MATCH' if matched in map_paths(rel) else 'MATCH_MOVED'
            used_remote.add(matched)
            rows.append(dict(local=rel,github=matched,status=status,size_bytes=full.stat().st_size,details='Identical SHA-256 contents'))
            continue
        identical=by_hash.get(sha,[])
        if identical:
            matched=identical[0];used_remote.add(matched)
            rows.append(dict(local=rel,github=matched,status='MATCH_RENAMED',size_bytes=full.stat().st_size,details='Identical content, different filename/path'))
        elif candidates:
            rows.append(dict(local=rel,github=' | '.join(candidates[:4]),status='DIFFERENT_CONTENT',size_bytes=full.stat().st_size,details='Name/path exists but file contents differ'))
        else:
            rows.append(dict(local=rel,github='',status='MISSING_ON_GITHUB',size_bytes=full.stat().st_size,details='No same-name or same-content file in origin/main'))
    # A Git repository has no empty tracked folders; paths are considered present
    # when they are ancestors of tracked files, or a mapped source directory.
    ghdirs=set()
    for p in remote_paths:
        parts=Path(p).parts
        for n in range(1,len(parts)):
            ghdirs.add('/'.join(parts[:n]))
    folder_rows=[]
    for d in sorted(local_dirs):
        if any(p in GENERATED_DIRS for p in Path(d).parts):
            status='INTENTIONALLY_EXCLUDED'
            target=''
        else:
            suggestions=map_paths(d+'/')
            targets=[s.rstrip('/') for s in suggestions if s.rstrip('/') in ghdirs]
            # A folder can still be represented if its source files are elsewhere.
            matching_children=[x for x in rows if x['local'].startswith(d+'/') and x['status'].startswith('MATCH')]
            if targets or matching_children:
                target=targets[0] if targets else '(files distributed across GitHub)'
                status='FOUND_ON_GITHUB'
            else:
                target=''
                status='MISSING_FOLDER_OR_EMPTY'  # Git doesn't track empty folders
        folder_rows.append(dict(local_folder=d,github_folder=target,status=status))
    remote_only=[dict(github=p) for p in remote_paths if p not in used_remote]
    out.mkdir(parents=True,exist_ok=True)
    write_csv(out/'all_mac_files.csv',rows,['local','github','status','size_bytes','details'])
    write_csv(out/'missing_or_changed.csv',[r for r in rows if r['status'] in {'MISSING_ON_GITHUB','DIFFERENT_CONTENT'}],['local','github','status','size_bytes','details'])
    write_csv(out/'folder_comparison.csv',folder_rows,['local_folder','github_folder','status'])
    write_csv(out/'github_not_matched.csv',remote_only,['github'])
    counts=Counter(r['status'] for r in rows)
    folder_counts=Counter(r['status'] for r in folder_rows)
    summary=dict(github_repository='ronitbaishya1/SMTNO',github_commit=commit,mac_root=str(root),files_on_github=len(remote_paths),local_files_scanned=len(rows),local_folders_scanned=len(folder_rows),file_status_counts=dict(counts),folder_status_counts=dict(folder_counts),report_directory=str(out))
    (out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n',encoding='utf-8')
    print('\nSMTNO vs GitHub origin/main:',commit[:10])
    print('GitHub tracked files:',len(remote_paths),'| Mac files scanned:',len(rows),'| Mac folders:',len(folder_rows))
    for key in ['MATCH','MATCH_MOVED','MATCH_RENAMED','DIFFERENT_CONTENT','MISSING_ON_GITHUB','INTENTIONALLY_EXCLUDED']:
        print(f'  {key:25} {counts[key]:>6}')
    print('Missing/changed folders:',folder_counts['MISSING_FOLDER_OR_EMPTY'])
    print('\nFirst 20 missing or different source files:')
    discrepancies=[x for x in rows if x['status'] in {'MISSING_ON_GITHUB','DIFFERENT_CONTENT'}]
    for r in discrepancies[:20]:
        print(' ',r['status'],r['local'], ('→ '+r['github'] if r['github'] else ''))
    print('\nReports:',out)
    print('This script did not upload, delete or change any Mac/GitHub files.')

if __name__=='__main__':main()
