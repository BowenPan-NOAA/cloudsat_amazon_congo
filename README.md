# save-AryehPedestalvars_EnvParas_V8_xarray.py

## Purpose
This script processes and updates cloud object datasets with environmental parameters, using **xarray** for improved performance and streamlined data handling.

Main objectives:
- Switch data reading method to **xarray** (to improve speed)
- Match **Juliet's dataset** with **CLDCLASS**
- Split scenes into multiple **cloud objects**
- Save outputs to **monthly CSV files**
- Only save **surface flag**
- Add **environmental parameters** to CSV
- Add **ECMWF-AUX** environmental variables to CSV

---

## Output

### Surface Flag Output (`.txt`)
Contains:
- `JulietDate`
- `Granule`
- `SN`
- `Pedestal/Anvil width`
- `Cutoff height`

### Environmental Parameters Output (`.csv`)
Contains:
- `JulietDate`, `Granule`, `SN`, `COmeanLon`, `COmeanLat`
- `mCAPE`, `mCIN`, `mLCL`, `mSOIL`, `mSFLX`, `mLFLX`, `mMSLP`, `mSST`
- `momega`, `mlowVWS`, `mmidVWS`, `mupVWS`, `mttlVWS`
- `mdirlowVWS`, `mdirmidVWS`, `mdirupVWS`, `mdirttlVWS`
- `mlowSH`, `mmidSH`, `mupSH`, `mttlSH`
- `COmeanAOD`, `COmeanRAIN`, `COmeanCTT`, `COminCTT`
- `COmeanIWP`, `COmeanSWP`

---

## Update History

- **Apr 5, 2023** — Use Juliet's updated dataset
- **Feb 10, 2023** — Update to xarray read-in; update 2007 ERA5 reading; handle 2006 ERA5 format differences
- **Jan 30, 2023** — Add IWV & saturation IWV; include vertical profiles of T, Td, U, V
- **Nov 8, 2022** — Switch to xarray read-in
- **Nov 3, 2022** — Fix SST, LCL, CIN, SOIL reading bug (masked area issue)
- **Sep 23, 2022** — Update ERA5 read-in (format change from Jan 2007)
- **Aug 31, 2022** — Update pedestal definition with tropopause & topography
- **May 31, 2022** — Remove convection fraction threshold
- **May 26, 2022** — Remove short CloudSat-only region (`fdx-idx=1`); adjust longitude convention to match CLDCLASS
- **May 25, 2022** — Add soil moisture saturation fraction; adjust ECMWF-AUX calculations
- **May 19, 2022** — Change ERA5 environment (VWS) source to ECMWF-AUX
- **Apr 27, 2022** — Add surface `.txt` to CSV output
- **Apr 24, 2022** — Add environmental `.txt` output
- **Apr 6, 2022** — Initial file creation
- **Jan 26, 2022** — Add deep convective core dimension; consider adding vapor MR, surface moisture, LWP, VWS direction
