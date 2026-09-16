# Repository guide

Keep the visible repository focused on the study, its reproducible methods, and its research outputs. Keep active private work in `internal/` and obsolete work in `archive/`.

## Where things belong

| Location | Purpose |
| --- | --- |
| `bdb_study/` | Cross-year Big Data Bowl study implementation |
| `rushing_study/` | Original confirmatory rushing study and shared storage |
| `configs/` | Study specifications, protocol amendments, and dependency records |
| `paper/` | Current methods brief, bibliography, and retained figures |
| `presentations/cassis2026/` | Conference presentation sources and assets |
| `abstract/` | Conference submission sources and assets |
| `results/` | Current checksummed aggregate results and their provenance |
| `experiments/` | Additional research experiments |
| `scripts/` | Standalone preprocessing, plotting, presentation, and cluster commands |
| `docs/` | Usage and maintenance guides |
| `internal/` | Active private notes, planning, and maintenance records; completely ignored |
| `archive/` | Obsolete work preserved locally; completely ignored by Git |
| `data/`, `.venv/` | Local datasets and Python environment; ignored |

Reusable implementation lives in the `bdb_study/` and `rushing_study/` packages.
Standalone commands live under `scripts/`, with `scripts/plots/`,
`scripts/presentations/`, and `scripts/betty/` grouping their purposes.
Shared neural and interval helpers belong to `rushing_study/`, rather than
being imported from a pilot runner. New tools should follow the same pattern.

The migration preserved completed result payloads and their recorded hashes.
Current source identities have changed; see [the migration notes](code-layout-migration.md).
Keep source, configuration, and result changes separate from historical evidence;
never rewrite an old receipt to make a new implementation pass validation.

Archive obsolete files without deleting them, preserving enough of their original folder structure to find them later. Do not put anything needed by a fresh clone only in `archive/` or `internal/`: ignored content will not be on GitHub. `internal/` is for current private work; move it to `archive/` when it is retired. Required execution evidence stays with the versioned study, even if its name sounds internal.

## Ignore files close to where they live

Use the root `.gitignore` for repository-wide exclusions such as `archive/`, `internal/`, datasets, environments, and Python caches. Use a folder's own `.gitignore` for that folder's generated or internal files. This keeps each rule easy to find and explain.

For example, a `paper/.gitignore` could contain:

```gitignore
# Local render and scratch directories, relative to paper/.
/build/
/tmp/
/output/

# LaTeX build products anywhere beneath paper/.
*.aux
*.log
*.synctex.gz

# Compiled PDFs directly in paper/; figures/ is unaffected.
/*.pdf

# Exclude generated exports while keeping their explanatory README.
/exports/*
!/exports/README.md
```

Avoid blanket rules such as `*.json`, `*.csv`, or `*.tex`: they would hide study configurations, useful aggregate results, and manuscript sources along with disposable files. Prefer specific directories or names, such as `/scratch/` or `*_predictions.csv`, after checking what they match.

The useful pattern rules are:

- A leading `/` anchors the pattern to the directory containing that `.gitignore`, not necessarily the repository root. `/output/` in `paper/.gitignore` targets `paper/output/`.
- A trailing `/` matches directories. A simple name such as `scratch/` can match at any depth beneath that `.gitignore`.
- `*` matches within one path component; `**` can span directories.
- `!` makes an exception to an earlier match. Git cannot re-include a file while its parent directory is still ignored. The example uses `/exports/*` so the parent remains visible; `/exports/` would prevent the README exception from working unless the parent were also unignored.

Commit shared `.gitignore` files with the project when ready. For an exclusion that is only your personal preference, add the pattern to `.git/info/exclude` instead; that file stays local to this checkout.

## A quick maintenance routine

Run these from the repository root. They inspect files without staging or committing anything:

```bash
# Ordinary changes and new files; Git may collapse whole directories.
git status --short

# Also show ignored files/directories, marked with !!.
git status --short --ignored

# List each new file that is eligible for a future commit.
git ls-files --others --exclude-standard

# Explain the matching ignore rule, including its file and line number.
git check-ignore -v --no-index paper/output/example.png

# Check whether a path is already tracked or staged.
git ls-files -- paper/output/example.png
```

For a new exclusion: add the narrowest rule in the nearest appropriate `.gitignore`, check a representative excluded file with `git check-ignore`, then check a neighboring source file to ensure it remains eligible. No output from `git check-ignore` normally means no rule matched; a matching `!` rule means the path was explicitly re-included. `--no-index` also explains rules for already tracked files, whose tracked status would otherwise mask the check.

Ignoring a file does not untrack an existing Git entry. If a generated file is already tracked, the following removes it from Git's index while leaving the local file on disk:

```bash
git rm --cached -- path/to/generated-file
```

That command stages a removal; review it and commit it later when intended. For a directory, use `git rm -r --cached -- path/to/generated-directory`. Neither is needed for files that have never been staged or committed.

Before the first commit, inspect the full candidate list from `git ls-files --others --exclude-standard`, review file contents and sizes, and confirm `archive/`, `internal/`, datasets, environments, and generated outputs are absent. Once you choose to stage files later, review `git diff --cached --stat` and `git diff --cached` before committing.

Ignoring is separate from deleting, encrypting, or removing prior Git history. It helps keep local files out of ordinary Git additions; it is not secret protection, and a forced add can bypass it.
