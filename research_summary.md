# United States Top 50 Playlist Performance and Song Popularity Trend Analysis

## Executive Summary

This project provides a descriptive, historical view of Atlantic Recording Corporation's United States Top 50 playlist. It evaluates ranking stability, chart longevity, artist concentration, popularity patterns, and content attributes. It does not predict future performance or recommend songs algorithmically.

Numeric findings should be populated after `Atlantic_United_States.csv` is placed beside `app.py`. The dashboard reports the observed values directly and does not fabricate missing results.

## EDA Highlights

- Dates are parsed using the required `DD-MM-YYYY` format.
- Position values are checked against the valid Top 50 range.
- Missing values and duplicate `(song, artist, date)` records are reported in the sidebar.
- Artist collaboration text is preserved in `artist_original`, standardized in `artist`, and split into `artist_list` and `primary_artist`.
- Song-level features include days on chart, average rank, best rank, rank volatility, entry and exit dates, end-of-period status, average popularity, and a 3-day rolling popularity trend.
- Analysis is filterable by date, artist, song text, rank, album type, and explicit status.

## Metric Definitions

- **Days on Chart:** number of distinct dates on which a song appears.
- **Average Rank:** arithmetic mean of observed playlist positions; lower is better.
- **Best Rank Achieved:** minimum observed playlist position.
- **Rank Volatility Index:** standard deviation of observed positions; lower indicates greater ranking stability.
- **Popularity Trend Score:** trailing 3-day rolling mean of popularity within each song. The 3-day window is a practical smoothing assumption and can be changed in `app.py`.
- **Artist Dominance Index:** `100 * (0.70 * chart-day share + 0.30 * unique-song share)`. The chart-day component rewards sustained presence, while the song-breadth component rewards having multiple songs represented. Collaboration rows contribute to every listed artist.
- **Explicit Content Share:** percentage of filtered observations marked explicit.

## Key Insights to Populate After Data Load

1. **Stability versus volatility:** identify whether the playlist is mostly stable or experiences frequent movement by comparing daily rank distributions and song-level volatility.
2. **Longevity:** report the longest-charting songs and whether long runs generally coincide with strong best ranks or high popularity.
3. **Risers and decliners:** use the daily rank-change table to identify songs with the fastest upward movement and the slowest deterioration.
4. **Artist concentration:** report the leading artists by Artist Dominance Index, unique songs, and total song-date chart-days.
5. **Popularity alignment:** report the observed popularity-to-rank correlation and compare average popularity across Top 10, 11-20, and 21-50 bands.
6. **Content differences:** compare explicit and non-explicit songs, plus singles and album tracks, on rank, popularity, duration, and longevity.
7. **Catalog attributes:** test whether duration and album size show visible relationships with popularity or rank; treat these as associations, not causal effects.

## Actionable Recommendations

1. Use the leaderboard and chart-day measures to distinguish durable artist momentum from one-song spikes when planning promotion calendars.
2. Review fast-rising songs early, pairing rank movement with popularity trend and cover metadata to prioritize timely campaign decisions.
3. Treat explicit-content and album-type comparisons as audience-segmentation evidence, not universal rules; validate patterns by date range and artist.
4. For release planning, examine whether long chart runs are associated with singles, album tracks, duration, or total album size in the observed history before changing strategy.
5. Monitor concentration in the Artist Dominance Index over time so promotional exposure can balance proven momentum with broader roster visibility.

## Caveats

This is observational playlist history. Rank and popularity associations do not establish causality. Duplicate records, missing attributes, API scoring changes, playlist editorial decisions, release timing, and collaboration crediting can affect comparisons. Conclusions should be reviewed alongside campaign, airplay, streaming, and market-context data.
