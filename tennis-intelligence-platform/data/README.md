# Data Setup

**Last verified live: 2026-07-02.** This document was updated after discovering that Jeff
Sackmann's original `tennis_atp`, `tennis_wta`, and `tennis_slam_pointbypoint` GitHub repos
have been removed/privated from his account very recently (his GitHub profile now shows only
`tennis_MatchChartingProject` as a public repo, confirmed via direct browser check). No public
explanation has been found for the removal. Data sources below were individually re-verified
live before being added to this doc — do not trust older tutorials/blogs citing the original
repo URLs, they will 404.

## Confirmed live sources (use these)

### 1. ATP match-level data → `Tennismylife/TML-Database`

```bash
cd data/raw
git clone https://github.com/Tennismylife/TML-Database.git
cd ../..
```

- One CSV per year, 1968–2026, ~49 columns, same core schema as the original Sackmann
  `tennis_atp` repo (this project explicitly describes itself as built on/compatible with it,
  with some additional corrections/fill-ins).
- Actively maintained; primary "live" version is now hosted at `stats.tennismylife.org`, but
  the GitHub CSVs are still current and are what we'll use for reproducibility.
- **License note:** their README states redistribution/commercial use without permission may
  violate copyright/terms of use — fine for our non-commercial academic/portfolio use, but
  flag this explicitly in the IEEE paper's data/licensing section and do not casually
  redistribute the raw CSVs in the repo. Cite both TML-Database and the original Sackmann
  methodology it's built on.

### 2. Point-by-point / shot-level data (ATP *and* WTA) → `JeffSackmann/tennis_MatchChartingProject`

```bash
cd data/raw
git clone https://github.com/JeffSackmann/tennis_MatchChartingProject.git
cd ../..
```

- Still live and public on Sackmann's account as of today.
- Actually richer than the originally-planned `tennis_slam_pointbypoint` repo: this is
  shot-by-shot data (shot type, direction, depth, errors) for every point of ~5,000+ charted
  matches, not just point winner/loser.
- Files split by gender: `-m-` (men's/ATP) and `-w-` (women's/WTA) prefixes — this is also our
  only current WTA data of any kind, so it partially offsets not having WTA match-level data.
- **License: CC BY-NC-SA 4.0, non-commercial only, and Sackmann has publicly stated he's
  "seriously" enforcing it** after past violations — attribute clearly, do not redistribute
  raw files, cite properly in the paper.
- Coverage is crowdsourced and has selection bias toward popular players / high-profile
  matches (noted explicitly in third-party academic use of this same dataset) — document this
  as a limitation, not just a footnote, since it affects how representative your point-level
  model's training data is.

## Known gap: WTA match-level data

**No confirmed live, actively-maintained equivalent to the original `tennis_wta` repo was
found** as of 2026-07-02 (checked Tennismylife's org directly — no WTA repo there either).

**Scope decision:** build Milestones 1-9 on **ATP only**. This is a legitimate, explicitly
documented scoping choice, not a shortcut — note it plainly in the project README and in the
paper's "Data" / "Limitations" sections. If a live WTA source turns up later (worth
periodically checking Kaggle, or re-checking Sackmann's account in case the repos come back),
extending to WTA is a natural "stretch goal" that reuses the entire pipeline unchanged.

## Verify the clone

```bash
ls data/raw/TML-Database/ | head -10
ls data/raw/tennis_MatchChartingProject/ | head -10
```

Paste the output back before we build the EDA script — Sackmann's Match Charting Project file
naming (`-m-` vs `-w-`, points vs matches vs shots files) needs to be confirmed exactly as it
exists today before I point any code at specific filenames.

## Do not commit raw data to git

`data/raw/`, `data/interim/`, and `data/processed/` are gitignored. DVC comes in Week 2+.