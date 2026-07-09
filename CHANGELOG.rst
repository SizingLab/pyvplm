=========
Changelog
=========

Version 0.2.0
=============

Modernization release: professional packaging, CI and test infrastructure.

Packaging
---------
- Migrated from Poetry to `uv <https://docs.astral.sh/uv/>`_ with a PEP 621
  ``pyproject.toml`` (hatchling build backend) and a committed ``uv.lock``.
- Moved the package to a ``src/`` layout.
- Supported Python versions: 3.10 - 3.13 (previously 3.8 - 3.9).
- Replaced the unmaintained ``pyDOE2`` dependency with ``pyDOE3``; removed
  unused dependencies (``latex``, ``xlwt``, ``xlrd``, ``openpyxl``).
- Compatibility with current scientific stack: numpy >= 2, pandas >= 2/3,
  scikit-learn >= 1.2, matplotlib >= 3.7, ipywidgets >= 8.

Testing & CI
------------
- New ``tests/`` suite (92 unit + non-regression tests) covering core classes,
  dimensional analysis, DOE generation, file I/O and the regression engine
  (a known variable power-law must be recovered exactly).
- GitHub Actions CI: ruff lint/format check + test matrix on Linux/Windows/
  macOS for Python 3.10-3.13; notebooks smoke-run as a non-blocking job.
- Release workflow now builds and publishes with ``uv``.
- Pre-commit migrated from black/flake8 to ruff (+ standard hygiene hooks).

Fixes
-----
- ``adapt_parameter_set`` was broken (undefined names after an old rename);
  it now works and no longer relies on ``exec`` into function locals
  (Python 3.13 compatible).
- ``perform_regression`` returned the plot error-label instead of the model
  expression; it now returns the actual expression.
- ``ConstraintSet.declare_doe_constraint`` was unusable (referenced the class
  instead of the instance, misused ``exec``); reimplemented.
- ``pi_dependency`` crashed with ``UnboundLocalError`` when ``x_list``/
  ``y_list`` were not provided.
- Temporary files (saved figures, latex formulas) now go to a proper
  cross-platform temp directory instead of a Windows-only hardcoded path.
- Fixed greek-letter capitalisation in plot labels (``.upper`` vs ``.upper()``).
- Removed debug prints and the warning emitted on private attribute access.
- ``perform_regression`` crashed on read-only input arrays (e.g.
  ``DataFrame.values`` under pandas >= 3) in linear space.

Notebooks & widgets
-------------------
- All four example notebooks now execute end-to-end against the modernized
  library and are checked in CI (blocking job).
- Motor example migrated from ``pyDOE2`` to ``pyDOE3`` and from Excel
  (``.xls``/``.xlsx``, removed dependencies) to CSV data files.
- The interactive ipywidgets paths (``pi_sensitivity``/``pi_dependency`` with
  widgets enabled) are covered by headless smoke tests, including the
  'Apply' button callbacks, validating ipywidgets >= 8 compatibility.

Version 0.1.0-beta
===================

- First beta release
