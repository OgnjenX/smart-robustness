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
  tables to support INTEGER PRIMARY KEY reads. Only `substr` is permitted as
  a function: the pulse-ID projection retrieves at most 1 MiB plus one byte
  inside SQLite and rejects an overlong blob, rather than allocating its full
  extent in Python. Accepted blobs are unchanged, not truncated. Other
  functions, named waveform columns, writes, and unregistered tables remain
  denied.

The selected reader and source-inventory selection are now implemented as
callables without an execution entry point. Selection validates original
fit/synapse/pair/cell/experiment/slice lineage and the original global partition
algorithm using only the sealed small-release inventories. The same ID sets
and relationship keys are used for both releases, including all selected
one-to-many measured-location, rested-fit and conductance rows. Per-parent
rows are capped at 1,000, each table at 10,000, and aggregate scalar/blob
payload at 32 MiB per release. An exceeded limit is an engineering failure,
not partial equivalence or authority to raise the limit after outcomes.

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

After adding source-selection and two-database projection tests, 91 focused
tests passed across reconciliation, acquisition, pulse storage, numeric
targets, identity inventories, rested targets and composite latency. These
checks use only synthetic reconciliation databases; repository Ruff and diff
whitespace checks also passed. They do not establish real-release equality.

## Independent acquisition verifier

`scripts/verify_synphys_medium_acquisition.py` now supplies a separate streamed
hash/signature and complete schema reconstruction, without importing any
acquisition helper. It checks registration, collector, schema-inspector,
header-manifest and CA hashes; source scope; terminal transport status; exact
length; and source stability throughout verification. Its SQLite guard allows
only schema reads and `table_info`, not physiological data rows.

Run it only after the terminal acquisition manifest exists. It writes the
exclusive `results/synphys-medium-verification-1132` evidence directory and
returns a nonzero exit for a retained verification failure. A premature
observation creates no verification directory, and an existing directory is
never overwritten or rerun. This verifier does not download anything, repair
an incomplete object, authorize reconciliation row reads, or establish the
biological usefulness of the source. Its own tests use synthetic SQLite
objects and registration fixtures, including changed schema/hash/permissions,
quoted table identifiers and explicit denial of data reads.

With the independent verifier included, all 157 Synphys-focused tests pass;
repository Ruff and diff whitespace checks pass. The real acquisition has not
been verified by this newly implemented script yet. This is still engineering
evidence, not real-source equivalence, physiological validation or a full
repository regression result.

## Registered execution entry point

`scripts/run_synphys_release_reconciliation.py --registration <sealed-file>`
now supplies the execution wrapper. No real execution registration exists yet;
the wrapper has been exercised only on synthetic releases representing the
98 selected fit IDs and 92 selected synapse IDs.

A real registration must explicitly authorize selected-summary reconciliation
and deny baseline joins, waveform reads, additional downloads, fitting, and
cell/network execution. It must retain the exact projection, counts, bounds
and sealed source paths; hash-pin the small release and existing identity,
QC and numeric inventories; and pin the newly verified medium release,
acquisition manifest, independent verifier manifest and published acquisition
assessment. It must also pin the wrapper, projection helper and independent
verifier implementations.

The wrapper claims an exclusive result directory before hashing inputs,
checks the actual medium length and every source hash, requires the exact
independent acquisition checks and assessment, and then derives all selection
IDs from the old inventories. Every source is checked for mutation across
projection. All exact differences and source projections are archived;
engineering failures are retained and return failure status. Even exact
equivalence is labelled independent-assessment-pending, not biological
promotion or permission for baseline joins.

No actual release data were accessed while developing or testing this wrapper.
All 174 Synphys-focused tests pass with the wrapper included; repository Ruff
and diff whitespace checks pass. These include end-to-end synthetic equality,
retained summary differences, duplicate invocation, source tampering, denied
permission expansion, missing independent/published gates, and post-projection
input mutation. This remains a focused engineering check, not full regression
or actual-source equivalence.
