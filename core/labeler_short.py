"""
labeler_short.py — Forward-Gain "SHORT Entry Point" labeling (mirror of labeler.py)

WHAT THIS DOES
--------------
The exact MIRROR of EntryLabeler (the long model). For EVERY candle we open a
hypothetical SHORT at the Open of that candle (the only unshifted execution
price — AGENTS.md Rule #1) and ask:

    Within the next `horizon_days`, does the SHORT reach +`tp_pct` profit
    BEFORE it loses `sl_pct`?

A short profits when price FALLS. Using the position-return (reciprocal) P&L that
makes this a clean mirror of the long +100%/-40% model:

    short P&L  =  entry / exit - 1

    * TP (short wins):  exit = entry / (1 + tp_pct)   -> P&L = +tp_pct
        tp_pct = 1.00  ->  price must FALL to entry * 0.5 (price halves) = +100%
    * SL (short loses): exit = entry / (1 - sl_pct)   -> P&L = -sl_pct
        sl_pct = 0.40  ->  price must RISE to entry * 1.667 (+66.7%)      = -40%

So this reflects the long barriers through the reciprocal: the long model needs
price x2.0 (+100%) before x0.6 (-40%); the short model needs price x0.5 (+100%
short profit) before x1.667 (-40% short loss). Same payoff, opposite direction.

If TP (down) is touched first -> y = 1  (good SHORT entry: a big decline started
                                         here and it did not spike up first)
If SL (up) is touched first  -> y = 0

Path-aware, first-touch, same-bar TP&SL ambiguity resolves to SL (conservative),
identical to labeler.py — only the barrier directions are flipped.

LEAKAGE NOTE (AGENTS.md Rule #1)
-------------------------------
Labels are ALLOWED to look into the future — that is the target Y, never a
feature. The .shift(1) anti-leakage guard applies to FEATURES only (features.py).

SAMPLE WEIGHTS (López de Prado, AFML ch. 4)
-------------------------------------------
Identical average-uniqueness weighting as labeler.py — overlapping forward
windows on adjacent bars are downweighted so clustered events aren't over-counted.
"""
import numpy as np
import pandas as pd


class ShortEntryLabeler:
    def __init__(self, df, tp_pct=1.00, sl_pct=0.40, horizon_days=60,
                 bars_per_day=None, no_new_high_bars=0, below_ema=0):
        """
        Parameters
        ----------
        df : DataFrame with 'Date','Open','High','Low' (OHLCV).
        tp_pct : SHORT take-profit as a fraction of the position (1.00 = +100%,
                 reached when price falls to entry/(1+tp_pct)).
        sl_pct : SHORT stop as a fraction of the position (0.40 = -40%, reached
                 when price rises to entry/(1-sl_pct)).
        horizon_days : vertical barrier — max days to reach the TP.
        bars_per_day : bars per calendar day; inferred from Date if None.
        no_new_high_bars : if > 0, a STRICTER short label — a bar can only be a
                 positive short entry if it is NOT making a new high over the
                 trailing `no_new_high_bars` bars (High[i] < max(High[i-N:i])).
                 This drops positives that still sit at fresh highs (price still
                 ripping up), keeping only entries where price has ALREADY rolled
                 over. 0 = off (baseline label). This condition uses only the
                 PAST window [i-N, i-1] so it is leakage-safe on the feature side,
                 but it is applied to the label (Y), which may look forward anyway.
        below_ema : if > 0, a positive short must additionally have the bar's Close
                 BELOW its `below_ema`-span EMA of Close (e.g. 6). Marks only
                 entries where short-term momentum has already rolled below the
                 fast EMA — not bars still bouncing above it. 0 = off.
        """
        self.df = df.reset_index(drop=True)
        self.tp_pct = tp_pct
        self.sl_pct = sl_pct
        self.horizon_days = horizon_days
        self.no_new_high_bars = int(no_new_high_bars)
        self.below_ema = int(below_ema)
        if bars_per_day is None:
            med = pd.to_datetime(self.df['Date']).diff().median()
            bars_per_day = max(1, round(pd.Timedelta(days=1) / med))
        self.bars_per_day = int(bars_per_day)
        self.horizon_bars = int(horizon_days * self.bars_per_day)

    def _first_touch(self):
        """
        For each bar i, race the SHORT TP (price DOWN) and SHORT SL (price UP)
        barriers over the forward window (i, i+horizon_bars].

        Returns
        -------
        label   : int array (1 if TP-down touched before SL-up, else 0)
        end_idx : int array — the bar index where the outcome resolved.
        """
        o = self.df['Open'].values.astype(float)
        hi = self.df['High'].values.astype(float)
        lo = self.df['Low'].values.astype(float)
        n = len(self.df)
        H = self.horizon_bars

        label = np.zeros(n, dtype=int)
        end_idx = np.arange(n, dtype=int)  # default: resolves at itself (no window)

        # Stricter label: precompute, for each bar, the max High over the trailing
        # N bars BEFORE it [i-N, i-1]. A bar making a new high (High[i] >= that
        # max) is disqualified from being a positive short — it is still ripping.
        N = self.no_new_high_bars
        if N > 0:
            prior_high = (pd.Series(hi).rolling(N, min_periods=1)
                          .max().shift(1).values)
        else:
            prior_high = None

        # below_ema veto: a positive short must have Close below its fast EMA.
        cl = self.df['Close'].values.astype(float)
        if self.below_ema > 0:
            ema = pd.Series(cl).ewm(span=self.below_ema, adjust=False).mean().values
            below_ok = cl < ema          # True where price has rolled below the EMA
        else:
            below_ok = None

        for i in range(n - 1):
            entry = o[i]
            if entry <= 0 or not np.isfinite(entry):
                continue
            # MIRROR of labeler.py: TP is BELOW entry (short wins when price falls),
            # SL is ABOVE entry (short loses when price rises).
            dn = entry / (1 + self.tp_pct)   # short take-profit level (price down)
            up = entry / (1 - self.sl_pct)   # short stop level (price up)
            end = min(i + 1 + H, n)
            touched_at = end - 1  # if nothing hits, the window ends here
            hit = 0
            for j in range(i + 1, end):
                hit_up = hi[j] >= up   # price rose into the stop -> short loses
                hit_dn = lo[j] <= dn   # price fell into the target -> short wins
                if hit_up and hit_dn:
                    # same-bar ambiguity: assume the stop hit first (conservative)
                    touched_at = j
                    hit = 0
                    break
                if hit_up:
                    touched_at = j
                    hit = 0
                    break
                if hit_dn:
                    touched_at = j
                    hit = 1
                    break
            # Stricter label: veto a positive if bar i is a fresh N-bar high
            # (price still making new highs -> not yet rolled over).
            if hit == 1 and prior_high is not None:
                ph = prior_high[i]
                if np.isfinite(ph) and hi[i] >= ph:
                    hit = 0
            # below_ema veto: price must already be below its fast EMA.
            if hit == 1 and below_ok is not None and not below_ok[i]:
                hit = 0
            label[i] = hit
            end_idx[i] = touched_at
        return label, end_idx

    @staticmethod
    def _uniqueness_weights(end_idx):
        """AFML average-uniqueness weights (identical to labeler.py)."""
        n = len(end_idx)
        diff = np.zeros(n + 1, dtype=float)
        for i in range(n):
            diff[i] += 1.0
            diff[min(end_idx[i] + 1, n)] -= 1.0
        concurrency = np.cumsum(diff[:n])
        concurrency[concurrency < 1] = 1.0
        inv = 1.0 / concurrency
        pref = np.concatenate([[0.0], np.cumsum(inv)])
        w = np.empty(n, dtype=float)
        for i in range(n):
            a, b = i, end_idx[i]
            w[i] = (pref[b + 1] - pref[a]) / (b - a + 1)
        mean_w = w.mean() if w.mean() > 0 else 1.0
        return w / mean_w

    def generate(self):
        """
        Attach the same three columns as EntryLabeler and return the DataFrame:
          Entry_Label  : 1 = good SHORT entry (TP-down before SL-up within horizon)
          Label_EndIdx : bar index where the outcome resolved (for purge/weights)
          Sample_Weight: AFML average-uniqueness weight (mean 1.0)

        The column name stays 'Entry_Label' so the SAME train_entry_model.py /
        entry_backtest.py pipeline consumes this panel unchanged.
        """
        label, end_idx = self._first_touch()
        weights = self._uniqueness_weights(end_idx)
        out = self.df.copy()
        out['Entry_Label'] = label
        out['Label_EndIdx'] = end_idx
        dates = pd.to_datetime(out['Date']).values
        out['Label_EndDate'] = dates[end_idx]
        out['Sample_Weight'] = weights
        pos = int(label.sum())
        strict = (f" | STRICT: no new high in prior {self.no_new_high_bars} bars"
                  if self.no_new_high_bars > 0 else "")
        strict += (f" | Close<EMA{self.below_ema}" if self.below_ema > 0 else "")
        print(f"[ShortLabeler] SHORT +{self.tp_pct*100:.0f}%/{self.horizon_days}d "
              f"stop -{self.sl_pct*100:.0f}% (price x{1/(1+self.tp_pct):.3f} down "
              f"before x{1/(1-self.sl_pct):.3f} up){strict}: "
              f"{pos} positives / {len(out)} bars = {pos/max(1,len(out))*100:.2f}%")
        return out
