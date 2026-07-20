# %% [markdown]
# # Week 1 — EDA: ATP Match-Level and Point/Shot-Level Data
#
# Data sources (verified live 2026-07-02, see data/README.md for why these differ from the
# original blueprint plan):
# - ATP match-level: data/raw/TML-Database/  (files named YYYY.csv, e.g. 1968.csv)
# - Point/shot-level, ATP + WTA: data/raw/tennis_MatchChartingProject/
#     - charting-m-matches.csv / charting-w-matches.csv  (match metadata)
#     - charting-m-points-to-2009.csv / -2010s.csv / -2020s.csv  (points, chunked by decade —
#       same for charting-w-points-*)
#     - charting-m-stats-*.csv / charting-w-stats-*.csv  (aggregate per-match stats)
#
# Scope: ATP match-level modeling for v1 (no live WTA match-level/season-results source found
# — documented decision in data/README.md). WTA point-level data IS available via Match
# Charting Project, so WTA could plausibly be added to the point-sequence model later even
# before a WTA match-level source appears — worth remembering as a stretch option.
#
# Run in VS Code (Jupyter extension, cells separated by `# %%`) or convert via `jupytext`.

# %%
import pandas as pd
from pathlib import Path

RAW = Path("../data/raw")  # adjust if running from notebooks/ vs project root
TML_DIR = RAW / "TML-Database"
MCP_DIR = RAW / "tennis_MatchChartingProject"

# %% [markdown]
# ## 1. Full inventory

# %%
print("--- TML-Database ---")
tml_files = sorted(TML_DIR.glob("*.csv")) if TML_DIR.exists() else []
print(f"{len(tml_files)} year files: {tml_files[0].name if tml_files else None} ... "
      f"{tml_files[-1].name if tml_files else None}")

print("\n--- Match Charting Project (men's) ---")
mcp_m_matches = MCP_DIR / "charting-m-matches.csv"
mcp_m_points = sorted(MCP_DIR.glob("charting-m-points-*.csv"))
print(f"matches file exists: {mcp_m_matches.exists()}")
print(f"points files: {[f.name for f in mcp_m_points]}")

print("\n--- Match Charting Project (women's) ---")
mcp_w_matches = MCP_DIR / "charting-w-matches.csv"
mcp_w_points = sorted(MCP_DIR.glob("charting-w-points-*.csv"))
print(f"matches file exists: {mcp_w_matches.exists()}")
print(f"points files: {[f.name for f in mcp_w_points]}")

# %% [markdown]
# ## 2. ATP match-level data (TML-Database) — load and inspect schema

# %%
df_sample_year = pd.read_csv(tml_files[-1]) if tml_files else pd.DataFrame()
print(df_sample_year.shape)
print(df_sample_year.columns.tolist())
df_sample_year.head()

# %%
# Load everything — 58 years of data, should still be fast
atp_matches = pd.concat([pd.read_csv(f) for f in tml_files], ignore_index=True) if tml_files else pd.DataFrame()
print(f"Total ATP matches loaded: {len(atp_matches)}")
if not atp_matches.empty:
    print("Missing values per column (top 15):")
    print(atp_matches.isna().sum().sort_values(ascending=False).head(15))

# %% [markdown]
# ## 3. Match Charting Project — men's points (all three decade-chunks concatenated)

# %%
mcp_m_points_df = (
    pd.concat([pd.read_csv(f, low_memory=False) for f in mcp_m_points], ignore_index=True)
    if mcp_m_points else pd.DataFrame()
)
print(f"Total men's charted points: {len(mcp_m_points_df)}")
if not mcp_m_points_df.empty:
    print(mcp_m_points_df.columns.tolist())
    mcp_m_points_df.head(10)

# %%
mcp_m_matches_df = pd.read_csv(mcp_m_matches) if mcp_m_matches.exists() else pd.DataFrame()
print(f"Total men's charted matches: {len(mcp_m_matches_df)}")
if not mcp_m_matches_df.empty:
    print(mcp_m_matches_df.columns.tolist())
    mcp_m_matches_df.head()

# %% [markdown]
# ## 4. Match Charting Project — women's points (same pattern, for later WTA extension)

# %%
mcp_w_points_df = (
    pd.concat([pd.read_csv(f, low_memory=False) for f in mcp_w_points], ignore_index=True)
    if mcp_w_points else pd.DataFrame()
)
print(f"Total women's charted points: {len(mcp_w_points_df)}")

mcp_w_matches_df = pd.read_csv(mcp_w_matches) if mcp_w_matches.exists() else pd.DataFrame()
print(f"Total women's charted matches: {len(mcp_w_matches_df)}")

# %% [markdown]
# ## 5. Key questions to answer before Week 2
#
# - TML-Database schema: does it match the original Sackmann matches_data_dictionary.txt
#   conventions closely enough to reuse existing feature-engineering knowledge, or are there
#   renamed/added/missing columns to account for?
# - Selection bias check: what fraction of ATP matches in TML-Database (all matches) have a
#   corresponding charted match in Match Charting Project (a small, curated subset)? This
#   quantifies exactly how non-representative the point-level training data is relative to
#   the full match population — needed for the paper's limitations section.
# - Match Charting Project points schema: what do point-outcome columns actually look like
#   (server, point winner, rally shot sequence, score state)? This determines how much parsing
#   effort the point-sequence feature pipeline needs.
# - Joinability: is there a shared key between TML-Database matches and Match Charting Project
#   matches, or does this need fuzzy joining on (date, player names, tournament)?
# - Date coverage on both sides — does TML-Database really reach back to 1968 and up to the
#   present, and what's the actual date range of charted matches?

# %% [markdown]
# ## Data Quality Report (fill in after running above)
#
# | Check | Result |
# |---|---|
# | TML-Database years found | |
# | TML-Database total matches | |
# | TML-Database schema vs. expected Sackmann schema | |
# | MCP men's matches / points | |
# | MCP women's matches / points | |
# | % of TML-Database matches with a charted counterpart | |
# | TML <-> MCP joinability | |

# %% [markdown]
# ## 2b. Encoding fix — one or more TML-Database files aren't valid UTF-8
#
# Older data files often have accented characters (player names) saved in Latin-1 rather than
# UTF-8. Identify the culprit file(s), then load everything with a UTF-8-first, Latin-1-fallback
# strategy.

# %%
bad_files = []
for f in tml_files:
    try:
        with open(f, encoding="utf-8") as fh:
            fh.read()
    except UnicodeDecodeError as e:
        bad_files.append((f.name, str(e)))

print(f"{len(bad_files)} file(s) with non-UTF-8 content:")
for name, err in bad_files:
    print(f"  {name}: {err}")

# %%
def read_csv_robust(path):
    try:
        return pd.read_csv(path, encoding="utf-8")
    except UnicodeDecodeError:
        return pd.read_csv(path, encoding="latin-1")

atp_matches = pd.concat([read_csv_robust(f) for f in tml_files], ignore_index=True) if tml_files else pd.DataFrame()
print(f"Total ATP matches loaded: {len(atp_matches)}")
if not atp_matches.empty:
    print("Missing values per column (top 15):")
    print(atp_matches.isna().sum().sort_values(ascending=False).head(15))

# %% [markdown]
# ## 2c. Fix — ATP_Database.csv is a player master file, not a match file
#
# It has a different schema (coaches, backhand, turnedpro, height, birthplace, etc.) and was
# incorrectly concatenated with the match-year files above, producing garbage. Separate it out.

# %%
player_master_file = TML_DIR / "ATP_Database.csv"
match_year_files = [f for f in tml_files if f.name != "ATP_Database.csv"]

print(f"Match-year files: {len(match_year_files)}")
print(f"Player master file found: {player_master_file.exists()}")

# %%
atp_matches = pd.concat([read_csv_robust(f) for f in match_year_files], ignore_index=True)
print(f"Total ATP matches loaded: {len(atp_matches)}")
print(atp_matches.columns.tolist())
print("\nMissing values per column (top 15):")
print(atp_matches.isna().sum().sort_values(ascending=False).head(15))

# %%
atp_players = read_csv_robust(player_master_file) if player_master_file.exists() else pd.DataFrame()
print(f"Total players in master file: {len(atp_players)}")
if not atp_players.empty:
    print(atp_players.columns.tolist())
    atp_players.head()