"""
Break-even and sustainability: a Monte Carlo simulation of five launch strategies over 24 months.

Demand, prices, kg and carrier willingness come from the survey. Airport traffic comes from published 2025
figures. Costs and the ranges of uncertain inputs are assumptions, all exposed as parameters.
"""
import numpy as np
import pandas as pd

TRAFFIC = dict(corridor_pax_per_year=5.5e6,      # Delhi-Mumbai passengers, 2025 (IATA World Air Transport Statistics)
               mumbai_domestic_pax=39.2e6,       # Mumbai domestic passengers, 2025 (Moodie Davitt Report)
               delhi_total_pax=78.1e6,           # Delhi IGI total passengers, 2025 (Network Thoughts)
               delhi_domestic_share=0.70,        # assumption
               seats=180, load_factor=0.85)      # A320-class; assumption
CALIB = {5: 0.70, 4: 0.35, 3: 0.10, 2: 0.0, 1: 0.0}  # stated intent -> purchase probability (top-box rule of thumb)

DEFAULTS = dict(
    p2p_price=None, carrier_credit=None, fwd_price=None,   # None = use the survey-optimised price
    var_cost=140,                 # pouch, seal and basic insurance per peer-to-peer transfer (Rs)
    gateway=0.02,                 # payment fee
    courier_cost=150,             # courier cost per kg, most likely value (Rs); range 0.75x-1.4x
    fwd_handling=60,              # per forwarding job (Rs)
    repack_fee=99, repack_cost=15,
    fixed_own=1_200_000,          # two own kiosks, per month (Rs); range 0.85x-1.25x
    fixed_partner=500_000,        # two partner counters, per month (Rs); range 0.8x-1.3x
    overhead=200_000,             # app, cloud, minimum team, per month (Rs)
    partner_share=0.15,           # revenue share to the partner counter
    setup_own=2_500_000, setup_partner=1_000_000,
    awareness=0.25,               # share of passengers who know KiloShare at the right moment, most likely value
    over_share=0.10,              # share of departing passengers over the allowance, most likely value
    paid_budget=300_000, cac=300, churn=0.05,
    incident_cost=1_000_000,
    futures=5000, months=24, seed=7,
)


def survey_inputs(d):
    """Everything the simulation takes from the survey."""
    w = d.times_exceeded_allowance_12m.astype(float)
    cal = d.usage_intent_6m.map(CALIB)
    buy_p2p = cal * d.use
    buy_fwd = cal * np.maximum(d.use, d.F8_courier_forwarding)
    kg = float((d.avg_excess_kg * w).sum() / w.sum())
    return dict(w=w.values, buy_p2p=buy_p2p.values, buy_fwd=buy_fwd.values, wtp_p2p=d.wtp_p2p_per_kg_inr.values,
                wtp_fwd=d.wtp_forwarding_per_kg_inr.values, carry=d.carry.values, credit=d.min_credit_per_kg_inr.values,
                bag=d.avg_checked_bag_kg.values, kg=kg, score=d['score'].values if 'score' in d else np.full(len(d), 0.5))


def conv(S, price, kind='p2p', mask=None):
    w = S['w'] if mask is None else S['w'] * mask
    buy, wtp = (S['buy_p2p'], S['wtp_p2p']) if kind == 'p2p' else (S['buy_fwd'], S['wtp_fwd'])
    return float((w * buy * (wtp >= price)).sum() / max(w.sum(), 1e-9))


def carrier_rate(S, credit):
    return float((S['carry'] * (S['credit'] <= credit) * (S['bag'] <= 15 - S['kg'])).mean())


def expected_min_poisson(r, c, kmax=60):
    r, c = np.asarray(r, float), np.asarray(c, float)
    pr, pc = np.exp(-r), np.exp(-c); sr, sc = 1 - pr, 1 - pc; tot = sr * sc
    for k in range(1, kmax):
        pr = pr * r / k; pc = pc * c / k; sr = sr - pr; sc = sc - pc
        tot = tot + np.clip(sr, 0, 1) * np.clip(sc, 0, 1)
    return tot


def derived():
    T = TRAFFIC
    pax_flight = T['seats'] * T['load_factor']
    return dict(pax_flight=pax_flight, flights_day=T['corridor_pax_per_year'] / 365 / pax_flight,
                departures_day=(T['mumbai_domestic_pax'] + T['delhi_total_pax'] * T['delhi_domestic_share']) / 2 / 365)


def p2p_margin(S, P, p, c):
    return S['kg'] * (p - c) - P['var_cost'] - P['gateway'] * p * S['kg']


def fwd_margin(S, P, pf, courier):
    return S['kg'] * (pf - courier) - P['fwd_handling'] - P['gateway'] * pf * S['kg']


def optimise_prices(S, P):
    """Base-case price search: best peer-to-peer price and credit, and best forwarding price."""
    D = derived(); grid = []
    for p in range(250, 701, 25):
        for c in range(100, 501, 25):
            if c >= p - 50: continue
            r = D['pax_flight'] * P['over_share'] * P['awareness'] * 0.5 * conv(S, p) * 0.8
            k = D['pax_flight'] * P['awareness'] * 0.5 * carrier_rate(S, c)
            t = float(expected_min_poisson(r, k)) * 0.6 * D['flights_day']
            grid.append(dict(price=p, credit=c, transfers_day=t, margin=p2p_margin(S, P, p, c), contribution=30 * t * p2p_margin(S, P, p, c)))
    grid = pd.DataFrame(grid); best = grid.loc[grid.contribution.idxmax()]
    fgrid = []
    for pf in range(100, 601, 25):
        jobs = D['departures_day'] * 0.5 * P['over_share'] * P['awareness'] * conv(S, pf, 'fwd') * 0.8 * 0.8
        fgrid.append(dict(price=pf, jobs_day=jobs, margin=fwd_margin(S, P, pf, P['courier_cost']), contribution=30 * jobs * fwd_margin(S, P, pf, P['courier_cost'])))
    fgrid = pd.DataFrame(fgrid); fbest = fgrid.loc[fgrid.contribution.idxmax()]
    return int(best.price), int(best.credit), int(fbest.price), grid, fgrid


def strategies(p, c, pf):
    return {'S1': dict(name='S1 Plan as first written: peer-to-peer only, own kiosks, Rs 400 / Rs 150 credit', short='S1 Original plan (P2P only)',
                       phrase='the original peer-to-peer plan', p=400, c=150, pf=pf, hybrid=False, lean=False, paid=False),
            'S2': dict(name=f'S2 Peer-to-peer only, own kiosks, best price Rs {p} / Rs {c} credit', short='S2 P2P only, best price',
                       phrase='a peer-to-peer service at the best price', p=p, c=c, pf=pf, hybrid=False, lean=False, paid=False),
            'S3': dict(name='S3 Hybrid (peer-to-peer + forwarding + repack), own kiosks', short='S3 Hybrid, own kiosks',
                       phrase='a hybrid service from its own kiosks', p=p, c=c, pf=pf, hybrid=True, lean=False, paid=False),
            'S4': dict(name='S4 Hybrid, lean: inside partner counters', short='S4 Hybrid, partner counters',
                       phrase='a lean hybrid (peer-to-peer + courier forwarding + repack station) run inside partner counters', p=p, c=c, pf=pf, hybrid=True, lean=True, paid=False),
            'S5': dict(name='S5 Lean hybrid + paid digital acquisition', short='S5 S4 + paid ads',
                       phrase='a lean hybrid with paid digital acquisition', p=p, c=c, pf=pf, hybrid=True, lean=True, paid=True)}


def simulate(S, P):
    P = {**DEFAULTS, **{k: v for k, v in P.items() if v is not None}}
    bp, bc, bf, grid, fgrid = optimise_prices(S, P)
    p = P['p2p_price'] or bp; c = P['carrier_credit'] or bc; pf = P['fwd_price'] or bf
    D = derived(); rng = np.random.default_rng(P['seed']); N, M = P['futures'], P['months']
    tri = lambda lo, mode, hi: rng.triangular(lo, mode, hi, N)
    a, o = P['awareness'], P['over_share']
    U = dict(over_share=tri(o * .5, o, o * 1.8), aware_cap=tri(a * .4, a, a * 1.6), ramp=rng.uniform(6, 12, N), haircut=tri(.5, .8, 1.0),
             carrier_mult=tri(.7, 1.0, 2.0), logistics=tri(.4, .6, .8), coverage=tri(.3, .5, .7), terminal_cov=tri(.35, .5, .65),
             follow_through=tri(.3, .5, .7), courier_cost=tri(P['courier_cost'] * .75, P['courier_cost'], P['courier_cost'] * 1.4),
             repack_rate=tri(.0003, .0008, .0015), fixed_own=tri(P['fixed_own'] * .85, P['fixed_own'], P['fixed_own'] * 1.25),
             fixed_partner=tri(P['fixed_partner'] * .8, P['fixed_partner'], P['fixed_partner'] * 1.3), cac=tri(P['cac'] * .5, P['cac'], P['cac'] * 2),
             corridor_share=tri(.15, .25, .35), home_share=tri(.4, .6, .8),
             incident_rate=np.exp(rng.uniform(np.log(1 / 200_000), np.log(1 / 20_000), N)))
    top30 = S['score'] >= np.quantile(S['score'], 0.7)
    lift30 = float(S['buy_p2p'][top30].mean() / max(S['buy_p2p'].mean(), 1e-9))
    user_over = float(S['w'][top30].mean())
    br_p2p, br_fwd = conv(S, p, 'p2p', top30), conv(S, pf, 'fwd', top30)
    results, fans, finals = [], [], {}
    for code, st in strategies(p, c, pf).items():
        m_p2p = p2p_margin(S, P, st['p'], st['c']); m_fwd = fwd_margin(S, P, pf, U['courier_cost'])
        cr, cv_p, cv_f = carrier_rate(S, st['c']), conv(S, st['p']), conv(S, pf, 'fwd')
        profit = np.zeros((N, M)); T_ = np.zeros((N, M)); F_ = np.zeros((N, M)); MR = np.zeros((N, M))
        users = np.zeros(N); alive = np.ones(N, bool); incident = np.zeros(N, bool)
        for t in range(1, M + 1):
            ramp = np.minimum(1, t / U['ramp']); aware = U['aware_cap'] * ramp
            if st['paid']:
                users = users * (1 - P['churn']) + (P['paid_budget'] / (U['cac'] / lift30) if t >= 4 else 0)
            r = D['pax_flight'] * U['over_share'] * aware * U['coverage'] * cv_p * U['haircut']
            if st['paid']:
                r = r + users * user_over * U['corridor_share'] / 365 * br_p2p * U['haircut'] / D['flights_day']
            k = D['pax_flight'] * aware * U['coverage'] * cr * U['carrier_mult']
            matched = expected_min_poisson(r, k) * U['logistics']
            T = matched * D['flights_day']; MR[:, t - 1] = np.where(r > 0, matched / np.maximum(r, 1e-12), 0)
            rev = T * m_p2p
            if st['hybrid']:
                F = np.maximum(D['departures_day'] * U['terminal_cov'] * U['over_share'] * aware * cv_f * U['haircut'] - T, 0) * U['follow_through']
                if st['paid']:
                    F = F + users * user_over * U['home_share'] / 365 * br_fwd * U['haircut'] * U['follow_through']
                R = D['departures_day'] * U['terminal_cov'] * U['repack_rate'] * ramp
                rev = rev + F * m_fwd + R * (P['repack_fee'] - P['repack_cost'])
                F_[:, t - 1] = np.where(alive, F, 0)
            if st['lean']: rev = rev * (1 - P['partner_share'])
            fixed = (U['fixed_partner'] if st['lean'] else U['fixed_own']) + P['overhead'] + (P['paid_budget'] if st['paid'] and t >= 4 else 0)
            hit = alive & (rng.random(N) < 1 - np.exp(-U['incident_rate'] * T * 30))
            mp = np.where(alive, 30 * rev - fixed - hit * P['incident_cost'], 0)
            incident |= hit; alive &= ~hit
            profit[:, t - 1] = mp; T_[:, t - 1] = np.where(alive, T, 0)
        setup = P['setup_partner'] if st['lean'] else P['setup_own']
        cash = -setup + np.cumsum(profit, axis=1); low = np.minimum(cash.min(axis=1), -setup)
        results.append(dict(code=code, strategy=st['name'], short=st['short'], phrase=st['phrase'], price=st['p'], credit=st['c'], fwd_price=pf,
                            profitable_m18=(profit[:, 17] > 0).mean(), profitable_m24=(profit[:, 23] > 0).mean(),
                            median_profit_m24=np.median(profit[:, 23]), p10_profit_m24=np.percentile(profit[:, 23], 10), p90_profit_m24=np.percentile(profit[:, 23], 90),
                            median_cash_low=np.median(low), p10_cash_low=np.percentile(low, 10), cash_positive_m24=(cash[:, 23] > 0).mean(),
                            p2p_per_day=np.median(T_[:, 23]), fwd_per_day=np.median(F_[:, 23]), match_rate=np.median(MR[:, 23]),
                            incident_share=incident.mean(), margin_per_transfer=m_p2p))
        fans.append(pd.DataFrame({'code': code, 'month': np.arange(1, M + 1), 'p10': np.percentile(profit, 10, axis=0),
                                  'median': np.median(profit, axis=0), 'p90': np.percentile(profit, 90, axis=0)}))
        finals[code] = profit[:, 23]
    res = pd.DataFrame(results); best = res.loc[res.profitable_m24.idxmax(), 'code']
    LAB = dict(over_share='Share of passengers over allowance', aware_cap='Awareness reached', haircut='Stated-to-actual intent',
               carrier_mult='Carrier supply', logistics='Both turn up and pass inspection', coverage='Corridor flights past our counters',
               terminal_cov='Departures past our counters', follow_through='Forwarding follow-through', courier_cost='Courier cost per kg',
               repack_rate='Repack station use', fixed_own='Fixed cost (own kiosks)', fixed_partner='Fixed cost (partner counters)', cac='Cost to acquire a user',
               corridor_share='Corridor share of user trips', home_share='Home-airport share of user trips', ramp='Months to full awareness',
               incident_rate='Security incident rate')
    y = pd.Series(finals[best]).rank()
    sens = pd.DataFrame([dict(input=LAB[k], correlation=pd.Series(v).rank().corr(y)) for k, v in U.items()]).dropna()
    sens = sens.reindex(sens.correlation.abs().sort_values(ascending=False).index).reset_index(drop=True)
    thresholds = pd.DataFrame([dict(cost_structure=cs, price=pp, credit=cc, margin=p2p_margin(S, P, pp, cc) * (1 - (P['partner_share'] if lean else 0)),
                                    monthly_fixed=fx + P['overhead'])
                               for cs, fx, lean in [('Own kiosks (2)', P['fixed_own'], False), ('Partner counters (2)', P['fixed_partner'], True)]
                               for pp, cc in [(400, 150), (p, c)]])
    thresholds['transfers_per_day_needed'] = thresholds.monthly_fixed / 30 / thresholds.margin.clip(lower=1)
    return dict(results=res, fan=pd.concat(fans), sensitivity=sens, thresholds=thresholds, best=best, prices=(p, c, pf),
                grid=grid, fgrid=fgrid, kg=S['kg'], lift30=lift30, params=P)


def verdict(res):
    best = res.loc[res.profitable_m24.idxmax()]
    p2p = res[res.code.isin(['S1', 'S2'])].profitable_m24.max()
    share = best.profitable_m24
    level = 'Sustainable' if share >= 0.6 else 'Marginal' if share >= 0.3 else 'Not sustainable'
    return dict(level=level, best=best, p2p=p2p)
