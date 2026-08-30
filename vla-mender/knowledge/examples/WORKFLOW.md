# VLA-Mender example catalog workflow

`knowledge/examples` is an index over two independent reference catalogs:

- `aspire/`: the public NVIDIA ASPIRE Task Gallery snapshot. It is static,
  pre-task public evidence and is refreshed only by its upstream sync script.
- `repair/`: local, evaluator-backed repair winners. These references may be
  outcome-derived and therefore keep their campaign, protocol, coverage, and
  limitation metadata.

Both catalogs contain reference programs, not executable truth. The current
task language, pinned API, public observations, Agent Constitution, and repair
state machine always take precedence.

## Agent-owned selection order

1. Require an exact suite/task/language and compatible failure-mode entry gate
   before treating a local `repair` example as task-specific prior evidence.
2. Retrieve `aspire` examples by task meaning. Prefer an exact public task match
   when one exists; otherwise borrow only observation-relative control ideas.
3. A local repair match and a public ASPIRE match may both be inspected. Keep
   their evidence domains distinct while reasoning about applicability.
4. Do not merge success rates across seen repair resets, debug subsets,
   randomized rollouts, and official rollouts.

## Repair-campaign use

Search the catalogs directly with filesystem tools. Start from this workflow,
the root `manifest.json`, `aspire/manifest.json`, and `repair/manifest.json`,
then read every program and evidence card that is plausibly related to the task,
failure phase, mechanism, perception, or control need. There is no fixed top-k.

Before repair code or rollouts, list the relevant files/symbols in the
prompt-required transient Markdown checklist, including why each item may help
and what must be adapted. If neither catalog contains a match, list the searched
scope and `none found`. The runtime does not ingest, freeze, hash, or validate
this checklist, and examples do not create candidate provenance fields.

Use a selected source only as a possible starting point. Static examples cannot
replace the verified `parent_ref` required for a non-initial candidate and never
count as repaired-seed coverage.

## Catalog maintenance

- Refresh/check only the public snapshot with `aspire/sync_from_nvidia.py`.
- Add local winners only through `repair/manifest.json`, with immutable program
  SHA-256, exact task identity, source artifact, evaluation protocol, coverage,
  and caveat fields.
- `manifest.json` at this directory is only the catalog index. Do not flatten
  ASPIRE and local repair entries into one provenance domain.
