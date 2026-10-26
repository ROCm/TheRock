-- TheRock local artifact cache, revision 6: component and bundle keys with explicit membership.
-- Fresh SQLite schema, not a migration or a server/API contract.
-- Requires SQLite JSON functions (json_valid, json_type, json_each).
-- CMake computes fingerprints and incorporates source/config/dependency inputs.
-- This database never interprets components, dependency edges, or stage names.
-- No names are seeded: register EXPECTED output identities from the configured
-- build/exporter contract, including outputs whose .fprint is missing/invalid.
PRAGMA foreign_keys = ON; -- required for each connection, before transactions
BEGIN;

-- Cache identity comes from CMake/exporter metadata, never a checkout path.
-- Use the full unambiguous artifact identity, including configured target/split
-- identity where relevant. Do not derive identities by an unverified filename
-- truncation rule. Epoch versions the cache-key rules, not the source commit
-- or build location. Use separate databases for isolated experiments.
-- One row per emitted component identity or artifact/bundle/target identity.
-- Bundle member sets are scoped on artifact_keys.components_json.
-- Packaging components and bundle/target suffixes remain part of this identity.
-- Different identities may have identical raw .fprint values.
CREATE TABLE artifacts (
    artifact_id INTEGER PRIMARY KEY,
    artifact_identity TEXT NOT NULL UNIQUE CHECK (length(artifact_identity) > 0)
);

-- Only valid, actually generated fingerprints become cache keys. For bundles,
-- every component must have this same raw CMake fingerprint.
CREATE TABLE artifact_keys (
    artifact_key_id INTEGER PRIMARY KEY,
    artifact_id INTEGER NOT NULL REFERENCES artifacts(artifact_id),
    epoch INTEGER NOT NULL CHECK (typeof(epoch) = 'integer' AND epoch > 0),
    fingerprint TEXT NOT NULL CHECK (
        length(fingerprint) = 64 AND fingerprint NOT GLOB '*[^0-9a-f]*'
    ),
    -- Canonical sorted JSON. [] means an individual component. A bundle keeps
    -- the raw CMake fingerprint and additionally keys on its exact member set.
    components_json TEXT NOT NULL DEFAULT '[]' CHECK (
        json_valid(components_json) AND json_type(components_json) = 'array'
    ),
    UNIQUE (epoch, artifact_id, fingerprint, components_json),
    UNIQUE (artifact_key_id, artifact_id)
);

-- Optional experiment/provenance records, not part of the lookup key.
-- One experiment is one configured checkout/overlay and build configuration.
-- NULL timings mean unmeasured; 0 is a measured no-op, not a default estimate.
-- Use monotonic wall-clock timers. lookup_ms includes connection + batch query
-- + result processing. preflight_ms measures the whole start-to-build-decision
-- path; restore_ms is measured separately. Record initial/prepared source state
-- so a prepared checkout result does not conceal source acquisition costs.
CREATE TABLE prototype_runs (
    prototype_run_id INTEGER PRIMARY KEY,
    label TEXT NOT NULL,
    repository_commit TEXT NOT NULL,
    configure_state TEXT NOT NULL DEFAULT 'not_run'
        CHECK (configure_state IN ('not_run','succeeded','failed')),
    source_revisions_json TEXT NOT NULL DEFAULT '{}'
        CHECK (json_valid(source_revisions_json) AND json_type(source_revisions_json) = 'object'),
    overlay_identity TEXT,
    source_state TEXT NOT NULL CHECK (source_state IN ('initial','prepared')),
    parameters_json TEXT NOT NULL DEFAULT '{}'
        CHECK (json_valid(parameters_json) AND json_type(parameters_json) = 'object'),
    source_prepare_ms REAL CHECK (source_prepare_ms >= 0),
    cmake_configure_ms REAL CHECK (cmake_configure_ms >= 0),
    fingerprint_collect_ms REAL CHECK (fingerprint_collect_ms >= 0),
    lookup_ms REAL CHECK (lookup_ms >= 0),
    plan_ms REAL CHECK (plan_ms >= 0),
    preflight_ms REAL CHECK (preflight_ms >= 0),
    restore_ms REAL CHECK (restore_ms >= 0),
    created_at INTEGER NOT NULL DEFAULT (CAST(strftime('%s','now') AS INTEGER))
);

-- Several productions may satisfy the same key. Paths are local PoC paths.
-- expected_files_json is a complete, nonempty array of relative output paths,
-- supplied from the successful artifact/exporter contract, not from a scan of
-- whatever files happen to exist. If this inventory is unknown, do not publish.
-- Fingerprint-only experiments create keys/observations but no ready entry.
CREATE TABLE cache_entries (
    cache_entry_id INTEGER PRIMARY KEY,
    artifact_key_id INTEGER NOT NULL REFERENCES artifact_keys(artifact_key_id),
    producer_run_id INTEGER NOT NULL REFERENCES prototype_runs(prototype_run_id),
    local_directory TEXT NOT NULL CHECK (length(local_directory) > 0),
    expected_files_json TEXT NOT NULL CHECK (
        json_valid(expected_files_json) AND json_type(expected_files_json) = 'array'
    ),
    state TEXT NOT NULL DEFAULT 'pending' CHECK (state IN ('pending','ready','invalid')),
    validation_state TEXT NOT NULL DEFAULT 'unvalidated'
        CHECK (validation_state IN ('unvalidated','build_passed','tests_passed')),
    validated_at INTEGER,
    expires_at INTEGER,
    created_at INTEGER NOT NULL DEFAULT (CAST(strftime('%s','now') AS INTEGER)),
    UNIQUE (cache_entry_id, artifact_key_id),
    CHECK (state <> 'ready' OR validated_at IS NOT NULL)
);
CREATE INDEX cache_entries_lookup
    ON cache_entries(artifact_key_id, state, created_at DESC);

-- Bytes live under local_directory; relative_path is also the manifest key.
-- Input fingerprints are not output-content integrity digests.
CREATE TABLE artifact_files (
    cache_entry_id INTEGER NOT NULL REFERENCES cache_entries(cache_entry_id),
    relative_path TEXT NOT NULL CHECK (length(relative_path) > 0),
    content_sha256 TEXT NOT NULL CHECK (
        length(content_sha256) = 64 AND content_sha256 NOT GLOB '*[^0-9a-f]*'
    ),
    size_bytes INTEGER NOT NULL CHECK (size_bytes >= 0),
    availability TEXT NOT NULL DEFAULT 'available'
        CHECK (availability IN ('available','missing','corrupt')),
    PRIMARY KEY (cache_entry_id, relative_path)
);

-- One observation per EXPECTED output per configured experiment. Keep a row
-- even when no usable .fprint exists; scanning only existing files loses these.
-- fingerprint_state describes generation; lookup_result describes cache use.
-- Invalid/missing fingerprints NEVER get placeholder artifact_keys rows.
-- selected_entry_id is a historical lookup result, not a permanent live lease.
CREATE TABLE artifact_observations (
    prototype_run_id INTEGER NOT NULL REFERENCES prototype_runs(prototype_run_id),
    artifact_id INTEGER NOT NULL REFERENCES artifacts(artifact_id),
    artifact_key_id INTEGER,
    fingerprint_path TEXT,
    fingerprint_state TEXT NOT NULL
        CHECK (fingerprint_state IN ('valid','missing','invalid')),
    detail TEXT,
    lookup_result TEXT NOT NULL DEFAULT 'not_checked'
        CHECK (lookup_result IN ('not_checked','hit','miss','uncacheable')),
    selected_entry_id INTEGER,
    PRIMARY KEY (prototype_run_id, artifact_id),
    FOREIGN KEY (artifact_key_id, artifact_id)
        REFERENCES artifact_keys(artifact_key_id, artifact_id),
    FOREIGN KEY (selected_entry_id, artifact_key_id)
        REFERENCES cache_entries(cache_entry_id, artifact_key_id),
    CHECK (
        (fingerprint_state = 'valid' AND artifact_key_id IS NOT NULL
            AND lookup_result IN ('not_checked','hit','miss'))
        OR
        (fingerprint_state IN ('missing','invalid') AND artifact_key_id IS NULL
            AND lookup_result = 'uncacheable'
            AND detail IS NOT NULL AND length(detail) > 0)
    ),
    CHECK ((lookup_result = 'hit' AND selected_entry_id IS NOT NULL)
        OR (lookup_result <> 'hit' AND selected_entry_id IS NULL))
);
CREATE INDEX artifact_observations_key ON artifact_observations(artifact_key_id);

CREATE TRIGGER artifacts_no_update
BEFORE UPDATE ON artifacts
BEGIN
    SELECT RAISE(ABORT, 'artifact output identity is immutable');
END;

-- A failed configure may leave old files on disk: never ingest them as valid.
CREATE TRIGGER artifact_observations_configure_succeeded_insert
BEFORE INSERT ON artifact_observations
WHEN NEW.fingerprint_state = 'valid' AND COALESCE((
    SELECT configure_state FROM prototype_runs
    WHERE prototype_run_id = NEW.prototype_run_id
), '') <> 'succeeded'
BEGIN
    SELECT RAISE(ABORT, 'valid observation requires successful configure');
END;

CREATE TRIGGER artifact_observations_configure_succeeded_update
BEFORE UPDATE OF prototype_run_id, artifact_key_id, fingerprint_state ON artifact_observations
WHEN NEW.fingerprint_state = 'valid' AND COALESCE((
    SELECT configure_state FROM prototype_runs
    WHERE prototype_run_id = NEW.prototype_run_id
), '') <> 'succeeded'
BEGIN
    SELECT RAISE(ABORT, 'valid observation requires successful configure');
END;

CREATE TRIGGER artifact_keys_no_update
BEFORE UPDATE ON artifact_keys
BEGIN
    SELECT RAISE(ABORT, 'artifact key identity is immutable');
END;

CREATE TRIGGER cache_entries_start_pending
BEFORE INSERT ON cache_entries
WHEN NEW.state <> 'pending'
BEGIN
    SELECT RAISE(ABORT, 'new cache entry must start pending');
END;

-- Freeze candidate identity/inventory; a new production gets a new row.
CREATE TRIGGER cache_entries_identity_immutable
BEFORE UPDATE OF artifact_key_id, producer_run_id, local_directory,
    expected_files_json, created_at ON cache_entries
BEGIN
    SELECT RAISE(ABORT, 'cache candidate identity and inventory are immutable');
END;

CREATE TRIGGER cache_entries_manifest_valid
BEFORE INSERT ON cache_entries
BEGIN
    SELECT CASE WHEN json_array_length(NEW.expected_files_json) = 0
        THEN RAISE(ABORT, 'expected output inventory must not be empty') END;
    SELECT CASE WHEN EXISTS (
        SELECT 1 FROM json_each(NEW.expected_files_json)
        WHERE type <> 'text' OR length(value) = 0
    ) THEN RAISE(ABORT, 'expected outputs must be nonempty path strings') END;
    SELECT CASE WHEN (
        SELECT count(*) FROM json_each(NEW.expected_files_json)
    ) <> (
        SELECT count(DISTINCT value) FROM json_each(NEW.expected_files_json)
    ) THEN RAISE(ABORT, 'duplicate expected output path') END;
END;

CREATE TRIGGER artifact_files_expected_only
BEFORE INSERT ON artifact_files
BEGIN
    SELECT CASE WHEN (SELECT state FROM cache_entries
                     WHERE cache_entry_id = NEW.cache_entry_id) <> 'pending'
        THEN RAISE(ABORT, 'register files before publication') END;
    SELECT CASE WHEN NOT EXISTS (
        SELECT 1 FROM cache_entries e, json_each(e.expected_files_json) r
        WHERE e.cache_entry_id = NEW.cache_entry_id
          AND r.value = NEW.relative_path
    ) THEN RAISE(ABORT, 'file is outside expected output inventory') END;
END;

CREATE TRIGGER artifact_files_identity_immutable
BEFORE UPDATE OF cache_entry_id, relative_path, content_sha256, size_bytes ON artifact_files
BEGIN
    SELECT RAISE(ABORT, 'file identity is immutable; availability may change');
END;

CREATE TRIGGER cache_entries_publish_complete
BEFORE UPDATE OF state, validation_state, validated_at ON cache_entries
WHEN NEW.state = 'ready'
BEGIN
    SELECT CASE WHEN COALESCE((
        SELECT configure_state FROM prototype_runs
        WHERE prototype_run_id = NEW.producer_run_id
    ), '') <> 'succeeded'
        THEN RAISE(ABORT, 'ready requires successfully configured producer') END;
    SELECT CASE WHEN NEW.validation_state NOT IN ('build_passed','tests_passed')
        OR NEW.validated_at IS NULL
        THEN RAISE(ABORT, 'ready requires verified build outputs') END;
    SELECT CASE WHEN EXISTS (
        SELECT 1 FROM json_each(NEW.expected_files_json) r
        LEFT JOIN artifact_files f
          ON f.cache_entry_id = NEW.cache_entry_id AND f.relative_path = r.value
        WHERE f.cache_entry_id IS NULL OR f.availability <> 'available'
    ) THEN RAISE(ABORT, 'ready requires every expected output') END;
END;

CREATE TRIGGER cache_entries_invalid_final
BEFORE UPDATE OF state ON cache_entries
WHEN OLD.state = 'invalid' AND NEW.state <> 'invalid'
BEGIN
    SELECT RAISE(ABORT, 'invalid candidates remain retired');
END;

-- Eligibility is re-evaluated, so missing/corrupt/deleted rows invalidate only
-- their own candidate. A newer incomplete candidate cannot hide an older hit.
-- This view checks metadata, not physical disk contents. The local client MUST
-- verify actual files/digests during restore, then mark failures missing/corrupt.
CREATE VIEW reusable_cache_entries AS
SELECT e.* FROM cache_entries e
JOIN prototype_runs p ON p.prototype_run_id = e.producer_run_id
WHERE p.configure_state = 'succeeded' AND e.state = 'ready'
  AND e.validation_state IN ('build_passed','tests_passed')
  AND e.validated_at IS NOT NULL
  AND (e.expires_at IS NULL OR e.expires_at > CAST(strftime('%s','now') AS INTEGER))
  AND NOT EXISTS (
      SELECT 1 FROM json_each(e.expected_files_json) r
      LEFT JOIN artifact_files f
        ON f.cache_entry_id = e.cache_entry_id AND f.relative_path = r.value
      WHERE f.cache_entry_id IS NULL OR f.availability <> 'available'
  );

PRAGMA user_version = 6;
COMMIT;

-- BATCH LOOKUP: parameterized, one result per EXPECTED output, including those
-- with missing/invalid fingerprints (NULL fingerprint + explicit state).
-- The collector validates 64-hex before passing fingerprint_state='valid'.
-- Use a VALUES row per output, not one query per ROCm component.
-- Bundle requests additionally match canonical components_json. See CacheDB's
-- LOOKUP_SQL for the current batched query; the following component-only example
-- fixes components_json='[]'.
-- WITH requested(request_id, epoch, artifact_identity, fingerprint, fingerprint_state) AS (
--     VALUES (?, ?, ?, ?, ?), (?, ?, ?, ?, ?)
-- ), ranked AS (
--     SELECT r.request_id, e.cache_entry_id,
--            ROW_NUMBER() OVER (
--                PARTITION BY r.request_id
--                ORDER BY e.created_at DESC, e.cache_entry_id DESC
--            ) AS choice
--     FROM requested r
--     JOIN artifacts a ON a.artifact_identity = r.artifact_identity
--     JOIN artifact_keys k
--       ON k.artifact_id = a.artifact_id
--      AND k.epoch = r.epoch
--      AND k.fingerprint = r.fingerprint
--      AND k.components_json = '[]'
--     JOIN reusable_cache_entries e USING (artifact_key_id)
--     WHERE r.fingerprint_state = 'valid'
-- )
-- SELECT r.request_id, r.artifact_identity,
--        CASE WHEN r.fingerprint_state <> 'valid' OR r.fingerprint IS NULL
--                 THEN 'uncacheable'
--             WHEN c.cache_entry_id IS NULL THEN 'miss' ELSE 'hit' END AS result,
--        c.cache_entry_id
-- FROM requested r
-- LEFT JOIN ranked c ON c.request_id = r.request_id AND c.choice = 1
-- ORDER BY r.request_id;
--
-- OPERATION / LOCAL EXPERIMENT LOOP
-- 1. Configure representative commits with the SAME local build environment.
--    Enumerate expected artifact output identities BEFORE scanning .fprint files.
--    After successful configure, collect REAL generated keys and observations.
--    Missing/invalid fingerprints create uncacheable observations with reasons.
--    A failed configure is not a source of valid fingerprints; ignore stale files.
--    Use per-experiment build directories/known-fresh outputs and record source
--    revisions, overlays, flags, tools, and environment in the run.
-- 2. Reconfigure unchanged inputs; fingerprints must remain stable. Relocate
--    the checkout/build directory to test independence from incidental paths.
--    Simulate COMMITTED source/config changes and check the actual deltas.
--    Separately test dirty working trees: the implementation in PR #2432 can
--    invalidate Git-backed fingerprints instead of hashing uncommitted edits.
--    Audit the current revision for broader monorepo/subtree invalidation.
-- 3. Build selected artifacts and copy their complete exports into an immutable
--    local cache directory. Register a pending candidate, expected inventory,
--    and files. Verify bytes, sizes, digests, and successful build externally.
--    In one short transaction, mark the candidate validated and ready.
-- 4. Batch-query current keys. Restore hits using the existing artifact import/
--    prebuilt mechanism; source-build misses and uncacheable units. The cache
--    does not reconstruct the DAG or infer which subprojects changed.
--    If a restore fails verification, reject only that candidate and try another
--    production or build that unit. Measure actual end-to-end behavior.
-- 5. Keep fingerprint stability comparisons separate from usable output hits.
--    Record real timings; do not populate fake benchmark numbers in this SQL.
--
-- BOUNDARIES / SAFETY OF THE LOCAL PoC
-- - Current .fprint input coverage (container/toolchain/flags/etc.) is an AUDIT
--   target, not assumed complete. Restrict initial reuse to one known local
--   environment until sensitivity checks establish compatibility. Fix missing
--   inputs in fingerprint generation and bump CACHE_EPOCH when prior productions must be invalidated.
-- - parameters_json is diagnostic metadata; it does not secretly add another
--   cache-key comparison or replace correct fingerprint generation.
-- - Output identity distinguishes same-hash packaging components and bundles.
--   Reuse a complete emitted output only where the existing importer/build
--   planner supports it; output-level hits do not automatically skip a stage.
-- - Cache at the current configured artifact granularity. A BLAS bundle is one
--   reuse unit unless CMake/exporter defines finer units with their own keys
--   and complete output inventories. Do not invent per-library reuse in SQL.
-- - Existing artifact/stage planning chooses what can be built/imported. If a
--   stage executor cannot build selected missing artifacts, record this as a
--   PoC gap; per-artifact lookup alone does not implement selective execution.
-- - Off bypasses cache use; dry-run reports decisions while still source-building.
-- - Safe local path resolution/extraction, expected-inventory correctness,
--   physical file verification, and publication authorization are client duties.
-- - No remote access, workflow edits, or component-name rules are required here.
