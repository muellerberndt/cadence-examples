# Historical memory benchmark

These are the unchanged producer and receipt from the first three-seed run.
Its Cadence sources are preserved in
[`0ec1a7ff233375ff304a66f64f291dd9c9b1c60f`](https://github.com/muellerberndt/cadence/tree/0ec1a7ff233375ff304a66f64f291dd9c9b1c60f).
The current example was rerun after source documentation was clarified. The
memory update formula did not change; the receipt's source hashes did.

Preserved byte hashes:

- `receipt.json`: `b4c8c7067dacbace90ba817ba9d91bae3c46330e797eb1a441d7a754b8e2091f`
- `train.py`: `a6d0f5327d9a301f3dd037d3223b65b8821c7e1e68e61d8fdcb31094b463b038`

The receipt's canonical content digest is
`8d7585e167d3e98ce20db459ae80654071ac8275574c8482f61de97495cab1ff`.
It binds the archived producer and the historical `stream.py` and `receipts.py`.
This archive is historical evidence, not a result of the current library.

For source verification, install that exact Cadence revision in an isolated
environment and run the archived producer with `--verify receipt.json` from this
directory. A rerun should use `--output /tmp/memory-historical.json` to preserve
the archived receipt.
