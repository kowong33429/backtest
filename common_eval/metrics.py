import numpy as np
import pandas as pd


def equity_metrics(eq):
    """CAGR / MaxDD / Sharpe / Sortino / Calmar from a bar-level MTM equity series (starts at 1.0)."""
    d = eq.resample('D').last().dropna()
    d0 = pd.concat([pd.Series([1.0], index=[d.index[0] - pd.Timedelta(days=1)]), d])
    r = d0.pct_change().dropna()
    yrs = max((eq.index[-1] - eq.index[0] + pd.Timedelta(hours=4)).days / 365.25, 1e-9)
    end = float(eq.iloc[-1])
    cagr = end ** (1 / yrs) - 1 if end > 0 else -1.0
    dd = (d0 / d0.cummax() - 1)
    mdd = float(dd.min())
    sd = r.std()
    dn = r[r < 0].std()
    return {
        'final_multiple': end, 'CAGR': cagr, 'MaxDD': mdd,
        'Sharpe': float(r.mean() / sd * np.sqrt(365)) if sd > 0 else np.nan,
        'Sortino': float(r.mean() / dn * np.sqrt(365)) if dn and dn > 0 else np.nan,
        'Calmar': float(cagr / -mdd) if mdd < 0 else np.nan,
    }


def yearly(eq):
    d = eq.resample('D').last().dropna()
    ye = d.resample('YE').last()
    prev = pd.concat([pd.Series([1.0]), ye.iloc[:-1].reset_index(drop=True)]).values
    return pd.Series(ye.values / prev - 1, index=ye.index.year)


def trade_metrics(td):
    if td is None or len(td) == 0:
        return {'trades': 0}
    r = td['ret']
    w, l = r[r > 0], r[r <= 0]
    top3 = td['pnl'].nlargest(3).sum()
    return {
        'trades': len(td), 'win_rate': float((r > 0).mean()),
        'avg_win': float(w.mean()) if len(w) else np.nan, 'avg_loss': float(l.mean()) if len(l) else np.nan,
        'win_loss_ratio': float(w.mean() / -l.mean()) if len(w) and len(l) and l.mean() < 0 else np.nan,
        'expectancy': float(r.mean()),
        'expectancy_ex_top3': float(r.drop(r.nlargest(3).index).mean()),
        'pnl_total': float(td['pnl'].sum()), 'pnl_total_ex_top3': float(td['pnl'].sum() - top3),
        'avg_hold_days': float(td['bars'].mean() / 6),
        'fees_slip_paid': float(td['cost'].sum()),
    }


def full_metrics(td, eq, extra_turnover=True):
    m = equity_metrics(eq)
    m.update(trade_metrics(td))
    if td is not None and len(td):
        yrs = max((eq.index[-1] - eq.index[0]).days / 365.25, 1e-9)
        m['turnover_per_year'] = float((td['size'] * 2).sum() / yrs)      # notional traded / initial equity / year
        m['cost_per_year_pct_equity'] = float(td['cost'].sum() / yrs)
    return m


def window(eq, td, start=None, end=None):
    """Sub-window metrics: equity re-based at the window start; trades by entry time."""
    e = eq
    if start is not None:
        e = e[e.index >= pd.Timestamp(start)]
    if end is not None:
        e = e[e.index <= pd.Timestamp(end) + pd.Timedelta(hours=23)]
    e = e / e.iloc[0]
    t = td
    if td is not None and len(td):
        if start is not None:
            t = t[t['entry_time'] >= pd.Timestamp(start)]
        if end is not None:
            t = t[t['entry_time'] <= pd.Timestamp(end) + pd.Timedelta(hours=23)]
    return full_metrics(t, e)


def model_metrics(y, p, ks=(0.01, 0.05, 0.10)):
    """PR-AUC, ROC, precision@top-k fraction and lift for a probability column."""
    from sklearn.metrics import average_precision_score, roc_auc_score
    y = np.asarray(y); p = np.asarray(p)
    base = float(y.mean())
    out = {'n': len(y), 'base_rate': base,
           'pr_auc': float(average_precision_score(y, p)) if y.sum() > 0 else np.nan,
           'roc_auc': float(roc_auc_score(y, p)) if 0 < y.sum() < len(y) else np.nan}
    order = np.argsort(-p)
    for k in ks:
        n = max(1, int(len(y) * k))
        prec = float(y[order[:n]].mean())
        out[f'precision@{int(k * 100)}%'] = prec
        out[f'lift@{int(k * 100)}%'] = prec / base if base > 0 else np.nan
    return out
