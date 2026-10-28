# Additional bullish swing patterns

The library adds four versioned strategies alongside Pullback/Reclaim and Bull
Flag. These are exploratory research heuristics, with no claim of profitability.
Local validation does not prove deployment or a real paper fill.

| Pattern / strategy ID | Formation required by v1 | Structural stop | Measured target |
| --- | --- | --- | --- |
| Ascending Triangle / `ascending_triangle_breakout_v1` | 15–60 sessions; two separated highs within 1%; progressively higher confirmed lows; contracting volume | Below last confirmed reaction low | Resistance plus formation height |
| Double Bottom / `double_bottom_breakout_v1` | Prior decline; troughs 20–80 sessions apart and within 3%; rebound at least 10%; second-trough volume no greater than first | Below lower trough | Neckline plus neckline-to-bottom height |
| Cup and Handle / `cup_handle_breakout_v1` | Prior advance; rounded 30–100-session cup, depth 12–40%, rims within 3%; falling 5–15-session handle in upper third of cup; contracting volume | Below handle low | Resistance plus cup depth |
| Inverse Head and Shoulders / `inverse_head_shoulders_breakout_v1` | Prior decline; three consecutive confirmed troughs; head at least 5% below shoulders; shoulders and neckline peaks within 3% | Below right shoulder | Neckline plus head-to-neckline height |

Every entry requires a bullish completed daily close above resistance by 0.1%,
at most 5% extended, and volume at least 1.5 times the preceding 20 sessions.
All structural stops sit 0.1% below their reference low. Unconfirmed touches do
not authorize proposals. A swing pivot needs two completed bars on both sides;
the breakout bar cannot confirm its own support. The existing 250-session feature
warmup applies. All windows use past bars only. The cup's central region must
remain relatively flat; sharp V formations are rejected. Inverse Head and Shoulders
v1 intentionally requires a nearly horizontal neckline. Quantitative thresholds
are implementation policy, rather than universal definitions or fitted results.

Conceptual references: StockCharts ChartSchool's [Ascending Triangle](https://chartschool.stockcharts.com/table-of-contents/chart-analysis/chart-patterns/ascending-triangle),
[Double Bottom](https://chartschool.stockcharts.com/table-of-contents/chart-analysis/chart-patterns/double-bottom-reversal),
[Cup With Handle](https://chartschool.stockcharts.com/table-of-contents/chart-analysis/chart-patterns/cup-with-handle),
and [Head and Shoulders Bottom](https://chartschool.stockcharts.com/table-of-contents/chart-analysis/chart-patterns/head-and-shoulders-bottom).

`config/chart_pattern_companions.json` controls the additions independently of
the Bull Flag configuration. Missing or disabled configuration omits them. Invalid
configuration makes the new companions unavailable while the original primary
and Bull Flag remain isolated. Each added pattern has a 600-second scan/replay
budget. All producer work stays inside the existing 7,200-second total budget;
a late companion failure cannot reverse an already committed primary archive.
The [runtime recovery contract](strategy_runtime_recovery.md) requires every
enabled screen to complete before the producer reports success, and a complete
unified publication before primary postflight succeeds.

The producer reuses one captured SIP/USD universe, raw and split bars, actions
and calendar, without additional provider requests. Each strategy has its own
`<session>/companions/<strategy-id>` root, immutable archive ID, Backtest/Metrics
records and exact primary handoff. Status appears separately at
`reports/strategy_companions/<strategy-id>/latest.json`. A missing or corrupt
record cannot authorize a fallback to another strategy's inputs. PostgreSQL
remains authoritative, with no schema migration or canonical candidate rewrite.

There is no strategy priority. Every qualifying setup receives the same existing
technical composite score, computed from completed split-adjusted bars. Relative
volume breaks score ties; exact ties use symbol and a deterministic signal hash.
Detector-specific scores and strategy names do not influence the common ranking.
This is an explicit technical ranking policy, not a proven return forecast.

Each strategy retains its three highest finalists using that same ranking key.
Their union contains the true global top three distinct symbols: a fourth-ranked
symbol within one strategy already has three distinct symbols ahead of it under
the identical ordering. The unified scanner merges duplicate symbols, records
all matching strategies, and publishes one global selection. An exact stock-level
tie chooses its exit-plan owner by signal hash, not a preferred strategy. All
three winners may come from one strategy. Scanner, Backtest/Metrics projections
and execution use the same immutable selection record in PostgreSQL.

Every enabled strategy must have a complete exact-session result, including an
honest empty result, before global selection is available. A failed scan cannot
silently grant another strategy preference. The original per-strategy publications
remain auditable compatibility/research records, not the final stock selection.

All strategies share four held/pending symbol slots, three durable new entry
attempts per session, 5% of remaining available cash per entry, account locking
and single symbol ownership. Existing protection, indicator and time exits can
precede a measured target. Targets close the full remaining quantity, without a
percentage partial. Replay resolves ambiguous bars with the prior stop first;
the native monitor requires a fresh observed bid for targets. Historical reports
remain per-strategy research outcomes; no combined portfolio performance is
inferred from selecting setups together.

Pullback/Reclaim's four core rule files and Bull Flag's two versioned rule files
stay unchanged. The default library includes every enabled strategy; a scan uses
the registered strategies declared by its study. The additions
use a separate adapter with the established exit kernel, durable engine,
cancellation and reconciliation paths.

Rollout requires a new merged source package and pinned launcher, all **66**
pipeline bindings verified, a flat native paper account with empty durable
ownership before lifecycle changes, and the existing monitor restart. Fresh
scheduled producer and primary results must precede entries. Never rewrite old
reports, rebind existing positions to new source, or replay consumed deployment
claims. Research qualification stays false; no forced signal or order is needed.
