# Inspecting recorded formation runs

Open a calculated `.sidereel` file with `sidereel path/to/run.sidereel` or the
player's Open playback file button. Live view and its control board remain the
default. All plots use the xy projection, including recordings with z coordinates.
The player reads stored samples; it never advances the calculation.

Use Recorded state to move in either direction. The sample number and phase
identify separate states at an identical event time. Choose an event to pause at
its recorded time and inspect its details. Use the state slider to inspect both
sides of a contact. Calculation status, simulated time, assumptions, and provenance
are visible below the plot; the details area scrolls. A collision is an event, and
does not change a complete recording into an incomplete one.

Choose a body to see its parcel contributions in solar masses. White rings mark
the earlier bodies carrying a survivor's parcels. For a removed body, Body history
lists its recorded existence and the following absent state. Its removal event
and recorded status remain inspectable. Old files without parcel data explicitly
show “not recorded”; no ancestry amounts are inferred.

Compare selects the existing left/right plots. Playback and scrubbing stay inside
the interval both recordings cover and pause at its end. Each side uses the latest
stored sample at or before the common playhead time, without interpolating states.
The details include both provenances, disclosed velocity changes, checkpoint ids,
stored starting states, and the state before and after the branch change. Different checkpoint ids are explicitly identified.

For already calculated prototype branches, headless code can package the two
histories into the same playback format:

```python
import json
from pathlib import Path
from simulations.star_system.observatory import formation_playback

# left and right are already calculated FormationExperiment instances.
Path('branches.sidereel').write_text(json.dumps(formation_playback(left, right)))
```

The ordinary experiment builder also preserves formation inspection data. Optional
run fields `assumptions`, `provenance`, `formation_events`, and `removed_bodies`
and `branch_point` retain the prototype's disclosed metadata. Each formation sample adds a
`contributions` mapping from body id to parcel id/mass records. These additions
preserve compatibility with existing playback documents; `events` keeps its
existing watch-filtered meaning while `formation_events` retains all formation
outcomes, including events outside the watch filter. No physics changes are needed.
