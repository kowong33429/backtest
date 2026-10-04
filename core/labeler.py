"""
labeler.py — Forward-Gain "Entry Point" Labeling (per-bar target for the entry model)

WHAT THIS DOES
--------------
For EVERY candle we open a hypothetical trade at the Open of that candle (the
only unshifted execution price — AGENTS.md Rule #1) and ask a single question:

    Within the next `horizon_days`, does price reach +`tp_pct`
    BEFORE it drops to -`sl_pct`?

If yes  -> y = 1  (this bar was a good ENTRY: a big rally started here and it
                   did not crater first)
If no   -> y = 0

This is "forward-gain per bar" made PATH-AWARE: a bar that first dumps -40% and
only later rallies is NOT labeled a good entry, because a real position would
have been stopped out. This is the triple-barrier UP label specialized for
entry detection, and it reuses the same first-touch logic as labels.py.

Default target (measured on ZEC/DASH to give a ~6% positive rate):
    +100% within 30 days, stop at -40%.

LEAKAGE NOTE (AGENTS.md Rule #1)
-------------------------------
Labels are ALLOWED to look into the future — that is the target Y, never a
feature. The .shift(1) anti-leakage guard applies to FEATURES only (features.py).
Here we deliberately scan forward to build Y.

SAMPLE WEIGHTS (López de Prado, AFML ch. 4)
-------------------------------------------
Forward-looking labels on adjacent bars overlap heavily (bar i and bar i+1 share
almost the same forward window), so naive training over-counts clustered events.
We compute each label's "uniqueness" = the average, over the bars its outcome
window spans, of 1 / (number of concurrent label windows covering that bar).
The resulting weight downweights redundant, overlapping samples.
"""
import numpy as np
import pandas as pd


class EntryLabeler:
    def __init__(self, df, tp_pct=1.00, sl_pct=0.40, horizon_days=30,
                 bars_per_day=None):
        """
        Parameters
        ----------
        df : DataFrame with 'Date','Open','High','Low' (OHLCV).
        tp_pct : take-profit distance as a fraction of entry (1.00 = +100%).
        sl_pct : stop distance as a fraction of entry (0.40 = -40%).
        horizon_days : vertical barrier — max days to reach the TP.
        bars_per_day : bars per calendar day; inferred from Date if None.
        """
        self.df = df.reset_index(drop=True)
        self.tp_pct = tp_pct
        self.sl_pct = sl_pct
        self.horizon_days = horizon_days
        if bars_per_day is None:
            med = pd.to_datetime(self.df['Date']).diff().median()
            bars_per_day = max(1, round(pd.Timedelta(days=1) / med))
        self.bars_per_day = int(bars_per_day)
        self.horizon_bars = int(horizon_days * self.bars_per_day)

    def _first_touch(self):
        """
        For each bar i, race the TP (up) and SL (dn) barriers over the forward
        window (i, i+horizon_bars].

        Returns
        -------
        label   : int array (1 if TP touched before SL, else 0)
        end_idx : int array — the bar index where the outcome resolved (touch
                  bar, or the horizon end if neither was hit). Used for the
                  concurrency/uniqueness weighting and for purging in CV.
        """
        o = self.df['Open'].values.astype(float)
        hi = self.df['High'].values.astype(float)
        lo = self.df['Low'].values.astype(float)
        n = len(self.df)
        H = self.horizon_bars

        label = np.zeros(n, dtype=int)
        end_idx = np.arange(n, dtype=int)  # default: resolves at itself (no window)

        for i in range(n - 1):
            entry = o[i]
            if entry <= 0 or not np.isfinite(entry):
                continue
            up = entry * (1 + self.tp_pct)
            dn = entry * (1 - self.sl_pct)
            end = min(i + 1 + H, n)
            touched_at = end - 1  # if nothing hits, the window ends here
            hit = 0
            for j in range(i + 1, end):
                hit_dn = lo[j] <= dn
                hit_up = hi[j] >= up
                if hit_dn and hit_up:
                    # same-bar ambiguity: assume the stop hit first (conservative)
                    touched_at = j
                    hit = 0
                    break
                if hit_dn:
                    touched_at = j
                    hit = 0
                    break
                if hit_up:
                    touched_at = j
                    hit = 1
                    break
            label[i] = hit
            end_idx[i] = touched_at
        return label, end_idx

    @staticmethod
    def _uniqueness_weights(end_idx):
        """
        AFML average-uniqueness weights.

        Each bar i owns an outcome window [i, end_idx[i]]. Concurrency c_t counts
        how many windows cover bar t. The uniqueness of label i is the mean of
        1/c_t over its own window; rarer (less overlapped) events weigh more.
        Weights are normalized to average 1.0 so they don't rescale the loss.
        """
        n = len(end_idx)
        # Concurrency via a difference array: +1 at start, -1 just after end.
        diff = np.zeros(n + 1, dtype=float)
        for i in range(n):
            diff[i] += 1.0
            diff[min(end_idx[i] + 1, n)] -= 1.0
        concurrency = np.cumsum(diff[:n])
        concurrency[concurrency < 1] = 1.0
        inv = 1.0 / concurrency

        # Average uniqueness over each label's window (prefix sum for speed).
        pref = np.concatenate([[0.0], np.cumsum(inv)])
        w = np.empty(n, dtype=float)
        for i in range(n):
            a, b = i, end_idx[i]
            w[i] = (pref[b + 1] - pref[a]) / (b - a + 1)
        mean_w = w.mean() if w.mean() > 0 else 1.0
        return w / mean_w

    def generate(self):
        """
        Attach three columns and return the DataFrame:
          Entry_Label  : 1 = good entry (TP before SL within horizon), else 0
          Label_EndIdx : bar index where the outcome resolved (for purge/weights)
          Sample_Weight: AFML average-uniqueness weight (mean 1.0)
        """
        label, end_idx = self._first_touch()
        weights = self._uniqueness_weights(end_idx)
        out = self.df.copy()
        out['Entry_Label'] = label
        out['Label_EndIdx'] = end_idx
        # Time-stamp when each label resolves, so purging/embargo in CV can work
        # across stacked coins (positional index collides between symbols).
        dates = pd.to_datetime(out['Date']).values
        out['Label_EndDate'] = dates[end_idx]
        out['Sample_Weight'] = weights
        pos = int(label.sum())
        print(f"[Labeler] +{self.tp_pct*100:.0f}%/{self.horizon_days}d stop -{self.sl_pct*100:.0f}%: "
              f"{pos} positives / {len(out)} bars = {pos/max(1,len(out))*100:.2f}%")
        return out
