"""The single backtester. Input: signals (timestamp, symbol, score, side). Output: trades + MTM equity.

Rules (config.yaml): a signal's timestamp is the bar whose OPEN is the execution price (the signal itself must
only use information up to the previous bar - adapters are responsible for that). Entry costs: fee + liquidity
dependent slippage per side. At most MAX_POSITIONS open, one per coin, each slot = MTM equity / MAX_POSITIONS,
no leverage. A slot is released the bar AFTER its exit. Exits are resolved per trade (they do not depend on the
rest of the portfolio): 'native' = the variant's own barrier rule, 'unified' = trailing stop + time stop.
"""
import numpy as np
import pandas as pd


class Signals:
    """Per-bar candidate lists sorted by score (desc)."""

    def __init__(self, mk, t, j, score, side):
        t, j, score, side = (np.asarray(x) for x in (t, j, score, side))
        ok = (t >= 0) & (j >= 0)
        t, j, score, side = t[ok], j[ok], score[ok], side[ok]
        ok = mk.elig[t, j]                               # liquid, tradable bar (point in time)
        t, j, score, side = t[ok], j[ok], score[ok], side[ok]
        o = np.lexsort((-score, t))
        self.t, self.j, self.score, self.side = t[o], j[o], score[o], side[o]
        self.start = np.searchsorted(self.t, np.arange(mk.T + 1))
        self.n = len(self.t)

    @classmethod
    def from_frame(cls, mk, df):
        t = np.fromiter((mk.idx(x) for x in df['ts']), int, len(df))
        j = np.fromiter((mk.sym_idx.get(s, -1) for s in df['symbol']), int, len(df))
        side = np.where(df['side'].values == 'short', -1, 1)
        return cls(mk, t, j, df['score'].values.astype(float), side)


def _first(mask):
    k = int(mask.argmax())
    return k if mask[k] else -1


class Engine:
    def __init__(self, mk, cfg, short_pnl='true'):
        self.mk, self.cfg, self.short_pnl = mk, cfg, short_pnl
        self.OT, self.HT, self.LT, self.CT = (np.ascontiguousarray(a.T) for a in (mk.O, mk.H, mk.L, mk.C))
        self.cache = {}

    # ---- exit resolution -------------------------------------------------
    def resolve(self, mode, side, i, j):
        key = (mode, side, i, j)
        r = self.cache.get(key)
        if r is None:
            r = self._resolve(mode, side, i, j)
            self.cache[key] = r
        return r

    def _resolve(self, mode, side, i, j):
        ex = self.cfg['exits']
        o, h, l, c = self.OT[j], self.HT[j], self.LT[j], self.CT[j]
        T = self.mk.T
        px = o[i]
        if mode == 'native':
            p = ex['native_long' if side == 1 else 'native_short']
        else:
            p = ex['unified']
        Hn = p['H_BARS']
        end = min(i + Hn, T - 1)
        hi, lo = h[i:end + 1], l[i:end + 1]
        k_hit, level, why = -1, None, None
        if mode == 'native':
            if side == 1:
                dn, up = px * (1 - p['SL']), px * (1 + p['TP'])
                ka = _first(np.nan_to_num(lo, nan=np.inf) <= dn)
                kb = _first(np.nan_to_num(hi, nan=-np.inf) >= up)
                if ka >= 0 and (kb < 0 or ka <= kb):          # same-bar ambiguity -> stop first
                    k_hit, why, level = ka, 'SL', min(o[i + ka], dn)
                elif kb >= 0:
                    k_hit, why, level = kb, 'TP', up
            else:
                dn, up = px / (1 + p['TP']), px / (1 - p['SL'])   # short wins when price falls
                ka = _first(np.nan_to_num(hi, nan=-np.inf) >= up)
                kb = _first(np.nan_to_num(lo, nan=np.inf) <= dn)
                if ka >= 0 and (kb < 0 or ka <= kb):
                    k_hit, why, level = ka, 'SL', max(o[i + ka], up)
                elif kb >= 0:
                    k_hit, why, level = kb, 'TP', dn
        else:
            tr = p['TRAILING_STOP']
            if side == 1:
                pk = np.maximum.accumulate(np.concatenate(([px], np.nan_to_num(hi, nan=-np.inf)[:-1])))
                pk = np.maximum(pk, px)
                stop = pk * (1 - tr)
                kk = _first(np.nan_to_num(lo, nan=np.inf) <= stop)
                if kk >= 0:
                    ok = o[i + kk]
                    k_hit, why, level = kk, 'TRAIL', (min(ok, stop[kk]) if np.isfinite(ok) else stop[kk])
            else:
                tl = np.minimum.accumulate(np.concatenate(([px], np.nan_to_num(lo, nan=np.inf)[:-1])))
                tl = np.minimum(tl, px)
                stop = tl * (1 + tr)
                kk = _first(np.nan_to_num(hi, nan=-np.inf) >= stop)
                if kk >= 0:
                    ok = o[i + kk]
                    k_hit, why, level = kk, 'TRAIL', (max(ok, stop[kk]) if np.isfinite(ok) else stop[kk])
        if k_hit >= 0:
            return i + k_hit, float(level), why
        # no barrier hit: time stop at the Close of the last bar (or last valid close if the data ended)
        cw = c[i:end + 1]
        last = int(np.nonzero(np.isfinite(cw))[0][-1])
        if i + Hn > T - 1:
            return i + last, float(cw[last]), 'END'            # evaluation window ended while still open
        if last < end - i:
            return i + last, float(cw[last]), 'DELIST'
        return end, float(cw[last]), 'TIME'

    # ---- net trade return ------------------------------------------------
    def net_ret(self, side, e, x, se, sx):
        f = self.cfg['costs']['FEE_PER_SIDE']
        if side == 1:
            return (x * (1 - sx)) / (e * (1 + se)) * (1 - f) ** 2 - 1
        if self.short_pnl == 'legacy_reciprocal':               # LEGACY bug H3: entry/exit - 1
            return e / x - 1 - (2 * f + se + sx)
        s, b = e * (1 - se), x * (1 + sx)
        return (s - b) / s - 2 * f

    def mark(self, side, e, se, c):
        f = self.cfg['costs']['FEE_PER_SIDE']
        if side == 1:
            return c / (e * (1 + se)) * (1 - f) - 1
        if self.short_pnl == 'legacy_reciprocal':
            return e / c - 1 - (f + se)
        s = e * (1 - se)
        return (s - c) / s - f

    # ---- the simulation --------------------------------------------------
    def run(self, sig, exit_mode='native'):
        mk, cfg = self.mk, self.cfg
        MAXP = cfg['portfolio']['MAX_POSITIONS']
        T = mk.T
        cash, eq_prev = 1.0, 1.0
        pos, held, trades = [], set(), []
        eq = np.empty(T)
        for t in range(T):
            a, b = sig.start[t], sig.start[t + 1]
            if len(pos) < MAXP and b > a:                       # entries at the Open of bar t
                for q in range(a, b):
                    if len(pos) >= MAXP:
                        break
                    j = int(sig.j[q])
                    if j in held:
                        continue
                    side = int(sig.side[q])
                    size = min(cash, eq_prev / MAXP)
                    if size <= 1e-9:
                        break
                    e = self.OT[j, t]
                    se = mk.slip[t, j]
                    xb, xp, why = self.resolve(exit_mode, side, t, j)
                    sx = mk.slip[min(xb, T - 1), j]
                    if not np.isfinite(sx):
                        sx = se
                    r = self.net_ret(side, e, xp, se, sx)
                    cash -= size
                    held.add(j)
                    p = dict(j=j, side=side, i=t, e=e, se=se, size=size, xb=xb, xp=xp, why=why, r=r, sx=sx, last=e)
                    pos.append(p)
                    trades.append(p)
            val, keep = 0.0, []                                 # close of bar t: realize exits, mark the rest
            for p in pos:
                if p['xb'] <= t:
                    cash += p['size'] * (1 + p['r'])
                    held.discard(p['j'])
                else:
                    c = self.CT[p['j'], t]
                    if np.isfinite(c):
                        p['last'] = c
                    val += p['size'] * (1 + self.mark(p['side'], p['e'], p['se'], p['last']))
                    keep.append(p)
            pos = keep
            eq[t] = cash + val
            eq_prev = eq[t]
        f = cfg['costs']['FEE_PER_SIDE']
        td = pd.DataFrame([{
            'symbol': mk.symbols[p['j']], 'side': 'long' if p['side'] == 1 else 'short',
            'entry_time': mk.times[p['i']], 'exit_time': mk.times[p['xb']], 'bars': p['xb'] - p['i'],
            'entry_px': p['e'], 'exit_px': p['xp'], 'reason': p['why'], 'ret': p['r'],
            'pnl': p['size'] * p['r'], 'size': p['size'],
            'cost': p['size'] * (2 * f + p['se'] + p['sx'])} for p in trades])
        return td, pd.Series(eq, index=mk.times)
