# insar-rs

A Rust engine for InSAR time-series analysis (Sentinel-1): ground deformation
by Small-Baseline Subset (SBAS) and Persistent Scatterer selection, with a
native non-closure phase-bias correction and numerical validation against
MintPy.

> Status: v0.3.0. See `CHANGELOG.md` for what changed, `PLAN.md` for the
> architecture, and `docs/` for validation and benchmarks.

## Capabilities (insar-core, 14 modules)

| Module | What it does |
|---|---|
| `types` | Contracts: stacks (interferogram/amplitude/unwrapped), pairs, series, metadata |
| `io` | Own stack format (`stack.json` + GeoTIFF, optional coherence), native reader/writer through SurtGIS, **no GDAL** |
| `io::isce` | **Native ISCE reader** (VRT + raw): `.unw` (NoData masked), `.int` (CFloat32), `.cor`, `.unw.conncomp`, `los.rdr` (per-pixel geometry), topsStack baselines |
| `io::licsar` | **LiCSAR/COMET reader** (GEOC wrapped phase and coherence) with AOI crop and date range; discovers incomplete networks |
| `ps` | Amplitude dispersion and PS candidate selection |
| `network` | Small-baseline network (double threshold), design matrix, connectivity |
| `unwrap` | Quality-guided 2D unwrapping with coherence and a `min_quality` threshold; **optional SNAPHU backend** (`unwrap::snaphu`, shell-out) |
| `unwrap_error` | 2π jump correction by **phase closure** (with an effectiveness check) and the `nonzero_closure_count` QC |
| `phase_bias` | **Non-closure phase-bias correction** (Maghsoudi et al. 2022) estimated from wrapped closures, for arbitrary span and incomplete networks; robust closure masking; applied after unwrapping by default |
| `inversion` | SBAS OLS / **coherence-weighted WLS** / **robust L1 (IRLS)**; **DEM error** (∝ B⊥); velocity with formal SE and **bootstrap**; **temporal model** (polynomial + seasonal + steps); temporal coherence; referencing |
| `atmosphere` | Spatio-temporal APS filter (temporal high-pass in **real time**, robust to gaps) |
| `troposphere` | Stratified phase-elevation correction (Doin 2009); **`gacos`** (ZTD maps, end-to-end) and `era5` (physical kernel) |
| `postprocess` | Deramp (plane/quadratic), `coherence_mask`, re-exports of referencing and γ_temp |
| `decompose` | LOS → (Up, East) with scalar or **per-pixel** geometry (incidence/heading from `los.rdr`) |
| `features` | Per-pixel ML descriptors (deterministic table ready for smelt-ml, with coordinates for spatial CV) |
| `pipeline` | End-to-end `run_sbas` with the documented physical order and QC products |

## Layout

- `crates/core`: `insar-core`, the engine (table above). Published on
  crates.io.
- `crates/cli`: the `insar` binary: `info`, `ps`, `network`, `run`
  (`--min-quality`, `--unwrap-backend {flood-fill,snaphu}`, `--snaphu-bin`,
  `--deramp`, `--no-closure-correction`), `isce`
  (`--wls`, `--robust`, `--dem-error-range`, `--deramp`, `--ref-region`),
  `decompose` (LOS asc+desc → up/east), `features` (per-pixel ML table,
  `--csv`), `deramp`, `aps`, `tropo-era5` and `tropo-gacos` (standalone on a
  written series). Install it from the repository (`cargo install --path
  crates/cli`); help texts are in Spanish.
- `crates/python`: `insar_rs`, PyO3/numpy bindings (computation releases the
  GIL, idiomatic exceptions): `invert_sbas`,
  `estimate_velocity[_uncertainty]`, `amplitude_dispersion`,
  `temporal_coherence`, `sbas_from_isce`, `decompose_asc_desc`,
  `decompose_per_pixel`, `extract_features`, `remove_ramp`,
  `correct_unwrap_errors`. See `crates/python/README.md`.

The phase-bias correction and the LiCSAR reader are available in the Rust
library only; the CLI and the Python bindings do not expose them yet.

## Validation

Numerical parity with MintPy 1.6.3 on the Fernandina dataset (Sentinel-1,
OLS path): time-series RMSE 0.034 µm and velocity RMSE 0.013 µm/yr,
r = 1.000000, after aligning two MintPy conventions (zero referenced phase
treated as no-data, and its decimal-year time axis). Details in
[`docs/validation.md`](docs/validation.md); performance in
[`docs/benchmarks.md`](docs/benchmarks.md).

## Build

```bash
cargo build --release          # insar binary
cargo test  --workspace        # 242 tests (237 unit + 2 e2e + 3 CLI smoke)
```

Building from the repository requires the sibling repo
[`surtgis`](https://github.com/franciscoparrao/surtgis) at `../surtgis`
(native GeoTIFF reader/writer, no GDAL). Tests and examples also use
`../geostat-rs`, `../smelt` and `../swarm-abm` (ecosystem dev-dependencies).
The published crate depends on `surtgis-core` from crates.io and needs none of
them.

Python bindings:

```bash
pip install maturin
maturin develop -m crates/python/Cargo.toml
pytest crates/python/tests/
```

## Reproducibility and citation

Releases are published on the official registries:

```bash
# Rust (engine): https://crates.io/crates/insar-core
cargo add insar-core

# Python (bindings): https://pypi.org/project/insar-rs/
pip install insar-rs
```

The validation is reproducible: `validation/make_figure_parity.py` recomputes
the MintPy parity on the public Fernandina stack (the regeneration chain is
in `docs/phase_bias/closure_figs.md`), and `validation/bench_fernandina.sh`
reruns the benchmark.

Archived copy with DOI on Zenodo:
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.21924287.svg)](https://doi.org/10.5281/zenodo.21924287)

- **Concept DOI** (all versions, cite this one): [`10.5281/zenodo.21924287`](https://doi.org/10.5281/zenodo.21924287)
- Version DOIs are listed on the Zenodo record.

To cite, see [`CITATION.cff`](CITATION.cff) (GitHub's "Cite this repository"
button) or the metadata in [`.zenodo.json`](.zenodo.json).

## License

MIT OR Apache-2.0 (see `LICENSE-MIT` / `LICENSE-APACHE`).
