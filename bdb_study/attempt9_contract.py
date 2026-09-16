"""Trust-root identity for attempt 9's post-probe operational-only changes.

The MIG45 probe manifest binds the complete code tree it executed.  Exactly
five admission/dispatch modules changed afterward to introduce runtime-plan
v4; this independent module pins their resulting bytes without creating a
self-hash cycle inside ``runtime_phases.py``.
"""

from __future__ import annotations


ATTEMPT9_OPERATIONAL_BINDING_SCHEMA_VERSION = (
    "bdb-betty-attempt9-operational-source-binding-v1"
)

# Populated only after the five v4 admission surfaces are final.  This module
# is itself bound by the attempt9 controller implementation receipt.
ATTEMPT9_OPERATIONAL_SOURCE_BINDINGS = {
    "bdb_study/cli.py": {
        "sha256": "d3a7d532e1b07283a33a54bc8d74299de5f1b7cae3885143c7ce27e4cd77445c",
        "size_bytes": 70661,
    },
    "bdb_study/execution.py": {
        "sha256": "2eaef79395fdb64ce76be1088fca59a42f82df2a1b627eaa84d9214a1e636bc1",
        "size_bytes": 66481,
    },
    "bdb_study/manifest.py": {
        "sha256": "908faffbabbb7464e9dce8fee093bc413a7dbf877162e295d961ec110b3020a0",
        "size_bytes": 76331,
    },
    "bdb_study/preflight.py": {
        "sha256": "fef84fba9874dc56397ac813ce96031ed9b376330c965423e09e13266d36f6a3",
        "size_bytes": 96629,
    },
    "bdb_study/runtime_phases.py": {
        "sha256": "dba82b802776a3f06d59a486de6598eee75702ff2f665e16ecd163d06a38735f",
        "size_bytes": 43453,
    },
}
