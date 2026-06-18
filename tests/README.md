# Test Layout

The test suite is split by purpose:

- `unit/`: fast, deterministic tests that run locally without downloading data or model weights. This is the default pytest target.
- `pipeline/`: offline smoke tests for the main `MolTrain.fit()` -> `MolPredict.predict()` workflow. These use fake data, tiny test dictionaries, random initialization, and one-epoch training to verify that the training and prediction pipeline is wired correctly without downloading pretrained weights.
- `cases/`: end-to-end usage cases for training, prediction, representation extraction, and dataset/model downloads. These are not run by default.

Run the default unit tests:

```bash
python -m pytest
```

Run the offline pipeline smoke tests:

```bash
python -m pytest tests/pipeline
```

Run both default CI test groups locally:

```bash
python -m pytest tests/unit tests/pipeline
```

Run the case tests without network access. Network-dependent tests will be collected but skipped:

```bash
python -m pytest tests/cases
```

Run network-dependent cases explicitly:

```bash
python -m pytest tests/cases --run-network
```

## Pipeline Smoke Tests

Pipeline tests are intended to catch broken wiring across data loading, feature generation, model initialization, training, checkpoint saving, model reload, and prediction. They should stay small and deterministic:

- Use local fake data instead of downloaded datasets.
- Use `load_pretrained=False` so model weights are randomly initialized.
- Provide a tiny local dictionary through `pretrained_dict_path` for UniMol v1 and conformer feature generation.
- Use small batches, one epoch, CPU execution, and temporary output directories from pytest's `tmp_path`.
- Assert shapes, finite predictions, and expected artifacts such as `model_0.pth`; do not assert model quality.

The current pipeline smoke coverage should include both `unimolv1` and `unimolv2`, and both `regression` and `classification` tasks. Pretrained-weight behavior and dataset download behavior belong in `cases/` tests marked as `network` and/or `slow`, not in the default CI smoke path.

Markers:

- `network`: requires external network access.
- `case`: user-facing end-to-end usage case.
- `integration`: exercises multiple subsystems together.
- `slow`: expected to take longer than unit tests.
