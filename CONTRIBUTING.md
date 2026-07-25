# Contributing

Thanks for your interest in AffKernel. Bug reports, reproduction reports and
focused pull requests are all welcome.

## Before you start

This is research code released alongside a paper. The priority is that the
published numbers stay reproducible, so changes that alter model behaviour or
the evaluation protocol need a stronger justification than changes that improve
clarity, portability or documentation.

## Environment

```bash
conda create -n affkernel python=3.8 -y
conda activate affkernel
pip install -r requirements.txt
pip install -r requirements-dev.txt
```

**The runtime is Python 3.8.** This is not negotiable in the current pinned
stack (torch 2.0.1 + cu117). Concretely:

- No `X | Y` union syntax at runtime, no `zip(..., strict=...)`, no `match`
  statements.
- PEP 585/604 annotations (`list[int]`, `str | None`) are only allowed in files
  that already carry `from __future__ import annotations`.
- **Never run `ruff check --fix` or `ruff format` across the whole tree.**
  Targeted fixes on files you actually touched are fine.

## Tests

```bash
python -m unittest discover -s tests -v
```

The suite needs neither a GPU nor network access. Tests that require IIT-AFF or
UMD on disk skip automatically when the data is absent, and they must stay that
way: a fresh clone has to produce a green run. If you add a data-dependent test,
guard it with `@unittest.skipUnless` following the pattern in
`tests/test_umd_dataset.py`.

Watch out for duplicate method names within a `TestCase` class. Python silently
keeps only the last definition, so a shadowed test never runs.

## Linting

```bash
ruff check src tools tests
```

Line length is 100. Ruff is pinned to `target-version = "py38"` for the reason
given above; please do not raise it.

## Pull requests

- Keep the change focused. One concern per PR.
- If you change anything that could move a published metric, say so explicitly
  in the description and include before/after numbers with the seed and config
  used.
- New dependencies need a justification; the install is deliberately small.
- Do not commit datasets, checkpoints, or generated artifacts. `.gitignore`
  covers the usual paths, but please check `git status` before committing.
- Never commit absolute paths from your own machine, personal directory names,
  or credentials.

## Reporting a reproduction failure

The most useful reports include:

1. The exact command, including `--seed`.
2. The config file used.
3. GPU model, driver, CUDA version, and the output of `pip freeze`.
4. The number you got and the number you expected.

Note that IIT-AFF has no validation split, so this codebase deliberately does
not select checkpoints on the test set. If your protocol picks the best epoch
by test score, your numbers will not match and that is expected. See
[`docs/reproduction.md`](docs/reproduction.md), section 8.

Also check the `beta` convention before reporting a mismatch: `beta^2 = 0.3` and
`beta^2 = 1` differ by about one point on this model, and mixing them is the
most common source of confusion in this literature.

## Attribution

Substantial parts of `src/` derive from RT-DETR and DETR, both Apache-2.0. If
you port additional code from an upstream project, add it to
[`THIRD_PARTY_LICENSES.md`](THIRD_PARTY_LICENSES.md) in the same PR, keep the
original copyright header intact, and note that the file was modified.

If you believe something in this repository is insufficiently attributed, please
open an issue. Attribution problems are treated as bugs.
