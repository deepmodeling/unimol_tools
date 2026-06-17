# Test Layout

The test suite is split by purpose:

- `unit/`: fast, deterministic tests that run locally without downloading data or model weights. This is the default pytest target.
- `cases/`: end-to-end usage cases for training, prediction, representation extraction, and dataset/model downloads. These are not run by default.

Run the default unit tests:

```bash
python -m pytest
```

Run the case tests without network access. Network-dependent tests will be collected but skipped:

```bash
python -m pytest tests/cases
```

Run network-dependent cases explicitly:

```bash
python -m pytest tests/cases --run-network
```

Markers:

- `network`: requires external network access.
- `case`: user-facing end-to-end usage case.
- `integration`: exercises multiple subsystems together.
- `slow`: expected to take longer than unit tests.
