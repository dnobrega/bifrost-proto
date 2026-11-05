# Building instrument-response resources

These scripts convert the legacy AIA/XRT IDL `.sav` tables and the Solar
Orbiter/EUI HRI 174 FITS response into the runtime format used by
`bifrost_proto`:

```text
instrument_responses/<name>/
    table.npy
    logt.npy
    logne.npy
    metadata.json
```

The runtime never needs IDL, SolarSoft, CHIANTI, Astropy, or SunPy. SciPy is
used only to read legacy `.sav` files during conversion; Astropy is required
only when converting the EUI FITS response.

## Convert the complete AIA/XRT set

With the historical source filenames in one directory:

```bash
python tools/build_instrument_responses/convert_all.py \
    /path/to/response_functions
```

By default the output is written to
`bifrost_proto/resources/instrument_responses/` in the current source tree.
Use `--output-root` to place the resources elsewhere.

The batch converter performs a preflight check before writing anything. It
expects the seven `aia_*_response_ne_tg.sav` files and the nine historical XRT
files (`xrt_al-med...`, `xrt_al-poly...`, etc.) and maps them to the canonical
runtime names (`aia193`, `xrt_alpoly`, ...).

## Include EUI/HRI 174

The legacy EUI recipe additionally divided the FITS response by the exact
`proton_dens(T)` vector returned by SolarSoft. Preserve that normalization by
saving that vector once (for example as `eui174_nhone.sav`) and run:

```bash
python tools/build_instrument_responses/convert_all.py \
    /path/to/response_functions \
    --nh-over-ne /path/to/response_functions/eui174_nhone.sav
```

Use `--nh-var` if the variable inside the `.sav` is not named `nhone`.

## Single-table conversion

`convert_aia.py`, `convert_xrt.py`, and `convert_eui.py` remain available for
converting one response at a time. All converters preserve the response axes
from the source files/header; no temperature or density grid origin is
hardcoded.
