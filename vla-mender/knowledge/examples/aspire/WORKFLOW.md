# NVIDIA Task-Gallery Strategy Reference Workflow

Use this workflow when an ASPIRE actor receives one or more `retrieved_strategy` files from the
public NVIDIA Task Gallery snapshot.

## 1. Treat gallery programs as references, not executable truth

The gallery files are successful upstream programs for their published task and runtime. They are
read-only few-shot strategy references. They are not admitted skills, do not add runtime APIs, and
must never override the current task language, API reference, policy, or observation.

Before using a reference:

1. Read the current runtime task language and pinned API contract.
2. Inspect the current initial image or causal failure evidence.
3. Read only the highest-ranked reference that answers a concrete planning question.
4. Optionally compare the next ranked reference when it offers a materially different observable
   strategy. Do not load the whole gallery.

## 2. Retrieve by task meaning

Automatic retrieval ranks references within the current benchmark domain using the runtime task
language and task name. Exact or near-exact task matches are preferred. A run can also pin reference
IDs with `strategy_references.include_ids` in its YAML config.

Use references to answer questions such as:

- which semantic roles and actionable parts need localization;
- whether a task should be decomposed into drawer/contact, grasp, transport, and placement stages;
- which stages should re-observe and verify visible progress;
- which observation-relative geometry or retry structure transferred between similar tasks.

## 3. Adapt before writing code

Never copy a gallery program verbatim without reviewing every call and constant.

- Replace unavailable or stale calls with functions documented by the current reduced API.
- Recompute object, handle, destination, and waypoint geometry from the current observation.
- Remove task IDs, seeds, benchmark paths, simulator state, hidden predicates, and filesystem or
  network access from generated robot code.
- Replace scene-specific world-coordinate filters with semantic masks and observable relational
  checks whenever possible.
- Call `commit_target_mask` for every final task-relevant instance selected through SAM3.
- Preserve only control structure supported by current evidence; do not inherit unrelated fallback
  branches merely because they appear in a successful upstream program.

The current API reference and Agent Constitution always win when a gallery script conflicts with
them.

## 4. Use references without leaking evaluation evidence

Task Gallery programs are public, static pre-task references. They may be selected before baseline
generation and during debug repair, but selection/final rollout outcomes must never influence which
reference is chosen. Record the selected reference IDs and hashes in the actor artifacts so the run
is auditable.

## Catalog maintenance

Refresh and verify the checked-in Task Gallery snapshot with:

```bash
python vla-mender/knowledge/examples/aspire/sync_from_nvidia.py
python vla-mender/knowledge/examples/aspire/sync_from_nvidia.py --check
```

`manifest.json` records every available successful program, its upstream URL, task metadata,
extracted call names, byte count, and SHA-256 digest.
