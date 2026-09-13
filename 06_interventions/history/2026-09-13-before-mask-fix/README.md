# Historical intervention benchmark

These are the unchanged producer and receipt from the first three-seed run.
They describe Cadence commit
[`0ec1a7ff233375ff304a66f64f291dd9c9b1c60f`](https://github.com/muellerberndt/cadence/tree/0ec1a7ff233375ff304a66f64f291dd9c9b1c60f),
before the correction to masks supplied separately for each batch row. The
benchmark passed individual-query masks, so it did not exercise that bug.
The current example's receipt comes from a fresh run after the correction.

Preserved byte hashes:

- `receipt.json`: `22f9c34e7cf7c91cab8c094a4c72eb97e802feafac27bc4a1e2e1978108a50d2`
- `train.py`: `60f4b08a9f679d4ab43cfd02c9d9e9b6b2f85f4bc264ca2cbbc0d7a35af09013`

The receipt's canonical content digest is
`641e09c0673f783c70e341d52f9c243ffe2e5b9451bc2ccba5a44ef429e7f110`.
It binds the archived producer and all 17 Python modules of that Cadence revision.
This archive is historical evidence, not a result of the current library.

For source verification, install that exact Cadence revision in an isolated
environment and run the archived producer with `--verify receipt.json` from this
directory. A rerun should use `--output /tmp/interventions-historical.json` to
preserve the archived receipt. Strict arithmetic checks in this older producer
may require the recorded NumPy/platform environment; the current checker permits
small libm roundoff in residual metrics.
