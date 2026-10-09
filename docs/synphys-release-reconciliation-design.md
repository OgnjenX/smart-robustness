# Selected-release reconciliation gate

This is an engineering design, not permission to read medium-release data
rows, reconstruct recording baselines, fit parameters, or run cells/networks.
Acquisition registration 1130 permits schema inspection only. Preserve the
frozen classic SMART baseline and every earlier failed coverage contract.

The helpers in `scripts/synphys_release_reconciliation.py` have no execution
entry point. They are tested against synthetic values and an in-memory SQLite
fixture, not the downloaded physiological records.

## Required execution boundary

After successful acquisition, independently stream the entire object's hash,
verify its exact byte count and SQLite signature, reconstruct its complete
schema, and verify the recorded acquisition/source dependencies. Do not use
a whole-file allocation to hash the 11 GB object. A separate registration
must pin those verified artifacts and the collector before any row queries.
An incomplete or failed acquisition must not trigger a replacement download
or a full-release fallback.

That registration must define the exact selected IDs and allowed columns for
both databases. Reconcile the previously selected 92 synapses, their complete
selected average/rested summary records, existing conductance baseline rows,
shared latency, original pulse-ID blobs, and connected pair/cell/experiment/
slice/measured-layer identities. Compare all rows in the registered selected
relationships, including nulls and absent/multiple records. Do not select only
complete records or infer equivalence from the common r2.1 release name.
Derive membership from sealed source inventories, never medium outcomes.
Keep the original global experiment partition and all insufficient strata.

## Exact comparison contract

- Primary IDs must be positive integers, unique in each projected table.
- Every row must have exactly the registered columns; extras are rejected.
- Row ordering is irrelevant. Missing/extra IDs are retained as differences.
- Typed values must match without tolerances, rounding, unit conversions,
  string normalization, or bool/integer coercion.
- Floating-point signed zero is preserved. NaN matches only NaN, never null;
  this rule does not impute values that SQLite exposes as null.
- Pulse-ID blobs must match byte-for-byte before separately checking their
  bounded numeric decoding under the existing no-pickle contract.
- Archive every changed value and missing row, not only a pass/fail flag.
- SQLite is read-only, immutable and query-only before installing an exact
  column authorizer. Empty-column callbacks are permitted only on registered
  tables to support INTEGER PRIMARY KEY reads. Functions, named waveform
  columns, writes, and unregistered tables remain denied.

Any difference closes the equivalence gate pending an independently assessed
source explanation. It does not permit repairing IDs, accepting an approximate
match, changing partitions, or selecting favorable records. Only an exact,
independently verified reconciliation can authorize a separately registered
pulse-to-recording baseline reconstruction. Those joins, physiological
scoring, fit objectives and model execution remain outside this design.

## Engineering verification

The initial focused suite passed 41 tests covering reconciliation, bounded
acquisition and pulse storage. Repository Ruff and diff whitespace checks
passed. This is not a full-regression claim or evidence that the actual two
releases are equal. A synthetic authorizer fixture initially failed at its
transaction close because the guard correctly denied commit; committing its
setup before installing the read-only guard resolved that fixture error
without relaxing the guard.
