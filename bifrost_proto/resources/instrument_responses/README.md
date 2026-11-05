# Instrument response resources

Each response lives in its own directory and uses the same canonical layout:

- `table.npy`: `G(ne,T)` with shape `(n_logne, n_logt)`
- `logne.npy`: strictly increasing `log10(ne / cm^-3)`
- `logt.npy`: strictly increasing `log10(T / K)`
- `metadata.json`: provenance, abundance, ionization equilibrium, units and fill value

The runtime quantity is `ne * nH * G(ne,T)` and is returned as
`DN cm^-1 s^-1 pix^-1` in solar/CGS units.

The package intentionally contains no dependency on IDL, SolarSoft, CHIANTI or
Astropy at runtime. Use `tools/build_instrument_responses/` to convert legacy
AIA/XRT `.sav` tables and the EUI/HRI FITS response into this format.
