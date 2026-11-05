# Third-party notices

Bifrost Proto is distributed under the **BSD 3-Clause License**; see [`LICENSE`](LICENSE). This license covers the original Bifrost Proto code and does not override any rights or terms applicable to third-party software, calibration data, or other source materials.

## Solar instrument colormaps (SunPy)

The AIA, EUI, and XRT colormaps included with Bifrost Proto reproduce the corresponding color-table conventions from [SunPy](https://sunpy.org/). SunPy's copyright notice, license terms, and provenance information are retained in [`bifrost_proto/resources/colormaps/SUNPY_LICENSE.txt`](bifrost_proto/resources/colormaps/SUNPY_LICENSE.txt).

The required color tables are bundled with Bifrost Proto; **SunPy is not a runtime dependency**.

## Solar instrument response tables (SolarSoft / SSWIDL)

The numerical response tables for **SDO/AIA**, **Hinode/XRT**, and **Solar Orbiter/EUI** were generated using scientific routines and calibration resources available through **SolarSoft (SSWIDL)**. The calculated results were saved as intermediate IDL data products (including `.sav` files) and converted to NumPy `.npy` arrays for use by Bifrost Proto.

Bifrost Proto does **not** redistribute SolarSoft IDL source code or those intermediate `.sav` files. The packaged numerical tables are located under [`bifrost_proto/resources/instrument_responses/`](bifrost_proto/resources/instrument_responses/).

Each response directory includes a `metadata.json` file recording available provenance information, such as the originating response file, checksums, abundance assumptions, ionization equilibrium, and normalization. The underlying responses may incorporate instrument-team calibrations and atomic data from other scientific resources, including CHIANTI, where applicable.

The original instrument teams, calibration providers, SolarSoft contributors, and atomic-data authors should receive appropriate acknowledgment and scientific citations when their work is used. **These provenance statements do not, by themselves, establish or grant redistribution rights for third-party calibration or atomic data.** Any applicable upstream terms remain in effect, independently of Bifrost Proto's BSD 3-Clause License.
