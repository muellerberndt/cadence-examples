# Frozen assisted-learning solver

`brain.js` is the exact source used by the two historical 60,000-neuron
assisted-learning receipts. SHA256:
`1b4d546e3d933b82ec4f35346433473ea1428dc2ac4ae677e1a30bc8808e7948`.

The test helper checks every source pin and creates an isolated temporary
source tree, linking the unchanged payloads instead of duplicating them.
The current demo continues to use `web/brain.js`. Historical hashes and
receipt bytes are never rewritten to follow a new implementation.
