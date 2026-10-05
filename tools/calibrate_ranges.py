"""Map hand-strength percentile to true combo frequency (calibration helper)."""
from pokerarena.strength import EQUITY_ORDER, _equity_lookup

lk = _equity_lookup()


def weight(token: str) -> int:
    if len(token) == 2:
        return 6
    return 4 if token.endswith("s") else 12


total = sum(weight(t) for t in EQUITY_ORDER)
rows = sorted(((lk[t], t, weight(t)) for t in EQUITY_ORDER), key=lambda r: -r[0])

print("total combos:", total)
print(f"{'strength':>9} {'top % of hands':>15}  example hand")
marks = [0.99, 0.98, 0.97, 0.96, 0.95, 0.94, 0.92, 0.90, 0.88, 0.87,
         0.85, 0.82, 0.80, 0.78, 0.75, 0.72, 0.70, 0.65, 0.62, 0.60,
         0.55, 0.50, 0.45, 0.40, 0.35, 0.30, 0.25, 0.20]
seen = set()
cum = 0
for pct, tok, w in rows:
    cum += w
    for m in marks:
        if pct >= m and m not in seen:
            seen.add(m)
            print(f"{m:>9.2f} {100 * cum / total:>14.1f}%  {tok}")

print()
print("Suggested persona thresholds for target VPIP:")
for target in (8, 12, 18, 25, 35, 50, 65, 80):
    cum = 0
    for pct, tok, w in rows:
        cum += w
        if 100 * cum / total > target:
            print(f"  VPIP ~{target:>3}%  ->  entry threshold {pct:.2f}  (first hand in: {tok})")
            break
