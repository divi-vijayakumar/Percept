# percept-contracts

The **frozen membrane** of the Percept SDK, extracted into a standalone, **zero-dependency**
Python package so consumers (clients, the IRA, dashboards, the JS edge twin) can import the
contract directly instead of re-declaring it.

It holds the vendor-neutral DTOs (`Box`, `Observation`, `Verdict`, `Goal`, `Concern`,
`Consequence`, `Evidence`, `Decision`, `WatchSpec`, `Tier0Signal`, `Incident`, …) and the seam
`Protocol`s (`Vision`, `Conductor`, `STT`, `TTS`, `Identity`, `Persistence`, …). These shapes are
**immutable within a phase**: implement *behind* them, add *beside* them.

```python
from percept_contracts import Verdict, Concern, Incident, CONTRACT_VERSION, negotiate

negotiate(CONTRACT_VERSION, "1.4.0")   # ok — same major
```

`percept.adapters` re-exports everything here, so every existing
`from percept.adapters import X` keeps working unchanged.

## Contract version

`CONTRACT_VERSION` is the semver of the whole frozen surface. `negotiate(local, remote)` returns
`True` when the MAJOR versions match and raises `ContractVersionMismatch` otherwise.
(`WatchSpec.version` is a separate wire integer kept for byte-for-byte payload compat.)

## JSON Schema

`python3 -m percept_contracts.jsonschema` regenerates the committed JSON Schema under `schema/`
(a combined `percept-contracts.schema.json` plus one file per DTO), stdlib only.

## Incident

`Incident` is the durable record of one concern firing, carrying an optional `IncidentSnippet`
(the captured relevant frame / clip / audio). It is referenced later by `feedback(incident_id,
label)`, health dashboards, and privacy redaction.
