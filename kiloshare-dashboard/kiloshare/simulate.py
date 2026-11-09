"""
Simulate realistic answers to the 25-question KiloShare survey, then inject deliberate noise and outliers.

Respondents are drawn from four hidden segments with different travel habits and attitudes. Behaviour
(would use, would carry, willingness to pay, features wanted) is generated from those habits and attitudes
with random error, so the algorithms have real but imperfect signal to find.

Noise injected afterwards: duplicate submissions, blanks, impossible values, unit errors, extra zeros,
placeholder 9999s, inconsistent spellings and Yes/No codes, logical contradictions, straight-lining
respondents and flipped answers. Every planted problem is logged.
"""
import random
from datetime import date, timedelta
import numpy as np
import pandas as pd
from .questionnaire import LIKERT_COLS, FEATURE_COLS

SEGMENTS = {'A': 'Frequent overpackers', 'B': 'Budget students & movers', 'C': 'Family visitors', 'D': 'Light travellers & sceptics'}
CITIES = ['Mumbai', 'Delhi', 'Bengaluru', 'Hyderabad', 'Chennai', 'Kolkata', 'Pune', 'Ahmedabad']
CITY_W = [.22, .24, .17, .12, .09, .07, .05, .04]
INCOME = ['<25k', '25k-50k', '50k-1L', '1L-2L', '2L+']
FARES = ['Saver/basic', 'Flexi/value', 'Business/premium']
P = {
    'A': dict(share=.22, age=(36, 8), occ=(['Salaried', 'Business owner', 'Freelancer'], [.6, .3, .1]), inc=3.0, fl=12,
              fare=[.45, .40, .15], bag=(17.5, 2.5), pex=.35, exkg=4.0, paid=.8,
              lk=dict(pos=3.2, tru=2.9, sec=3.5, pri=2.4, tec=4.2, cnt=3.8)),
    'B': dict(share=.25, age=(24, 3.5), occ=(['Student', 'Salaried', 'Freelancer'], [.55, .35, .1]), inc=1.0, fl=4,
              fare=[.85, .13, .02], bag=(17, 3), pex=.45, exkg=5.0, paid=.4,
              lk=dict(pos=2.6, tru=3.4, sec=3.0, pri=4.4, tec=4.5, cnt=3.7)),
    'C': dict(share=.28, age=(42, 11), occ=(['Salaried', 'Homemaker', 'Business owner', 'Retired', 'Freelancer'], [.4, .25, .15, .12, .08]),
              inc=2.0, fl=3, fare=[.75, .22, .03], bag=(18, 2.5), pex=.55, exkg=5.5, paid=.5,
              lk=dict(pos=4.0, tru=2.4, sec=3.9, pri=3.8, tec=2.8, cnt=4.1)),
    'D': dict(share=.25, age=(33, 9), occ=(['Salaried', 'Business owner', 'Freelancer', 'Student'], [.6, .15, .15, .1]), inc=2.5, fl=7,
              fare=[.6, .3, .1], bag=(10, 3), pex=.05, exkg=2.0, paid=.7,
              lk=dict(pos=3.8, tru=2.0, sec=4.2, pri=3.0, tec=3.9, cnt=2.9)),
}
CONCERNS = {
    'sec': ["What if there is something illegal inside?", "scared of getting caught for someone elses stuff", "security issues at airport",
            "drugs or banned items risk", "who is responsible if security finds something", "legal problem if item is not allowed"],
    'pos': ["dont want strangers touching my bag", "my bag is my personal space", "possessive about my luggage tbh",
            "not comfortable mixing my things with others", "what if my stuff gets damaged"],
    'tru': ["how do i know the person is genuine", "trust issue with strangers", "what if they dont give back my stuff",
            "fake profiles", "person might not turn up at airport"],
    'pri': ["price should be much cheaper than airline", "hidden charges?", "if it saves money i'm in", "how much commission you take"],
    'none': ["nothing", "NA", "", "no concerns", "-", "seems fine", "none really"],
}


def simulate(n=5000, seed=2026):
    rng = np.random.default_rng(seed); random.seed(seed)
    sig = lambda z: 1 / (1 + np.exp(-z))
    pick = lambda opts, w: opts[rng.choice(len(opts), p=np.array(w) / np.sum(w))]
    lik = lambda mu: int(np.clip(np.round(rng.normal(mu, 0.9)), 1, 5))
    segs = rng.choice(list(P), size=n, p=[P[s]['share'] for s in P])
    rows = []
    for i, s in enumerate(segs):
        p = P[s]
        age = int(np.clip(round(rng.normal(*p['age'])), 18, 72))
        inc = int(np.clip(round(rng.normal(p['inc'], 0.9)), 0, 4))
        flights = int(rng.poisson(p['fl']) + 1)
        fare = pick(FARES, p['fare'])
        bag = float(np.clip(round(rng.normal(*p['bag']), 1), 3, 28))
        over = int(rng.binomial(flights, p['pex']))
        exkg = float(np.round(np.clip(rng.gamma(4, p['exkg'] / 4), 0.5, 15), 1)) if over else 0.0
        fees = int(round(over * exkg * rng.uniform(550, 800) * p['paid'] * rng.uniform(.6, 1.2), -1)) if over else 0
        lk = {k: lik(v) for k, v in p['lk'].items()}

        z = (-1.0 + 0.30 * min(over, 8) + 0.10 * exkg + 0.45 * (lk['tru'] - 3) - 0.40 * (lk['sec'] - 3) - 0.30 * (lk['pos'] - 3)
             + 0.55 * (lk['cnt'] - 3) + 0.25 * (lk['pri'] - 3) + 0.20 * (lk['tec'] - 3) - 0.02 * (age - 32))
        pu = sig(z); use = rng.random() < pu
        intent = int(np.clip(round(1 + 4 * pu + rng.normal(0, 0.6)), 1, 5))
        zc = (-0.8 + 0.12 * (15 - bag) + 0.40 * (lk['tru'] - 3) - 0.50 * (lk['pos'] - 3) - 0.30 * (lk['sec'] - 3)
              + 0.30 * (lk['cnt'] - 3) + 0.03 * (flights - 5))
        carry = rng.random() < sig(zc)

        b = lambda q: int(rng.random() < np.clip(q, 0.02, 0.97))
        F1 = b(0.45 + 0.35 * use); F2 = b(0.35 + 0.25 * use + 0.10 * (lk['sec'] >= 4))
        F3 = b(0.75 if (F1 and F2) else 0.45 if F1 else 0.15)
        F4 = b(0.35 + 0.15 * use + 0.12 * (lk['sec'] - 3) + 0.10 * (lk['cnt'] - 3))
        F5 = b((0.72 if F4 else 0.18) + (0.08 if s == 'C' else 0))
        F8 = b(0.25 + 0.15 * (s in 'BC') + 0.10 * use)
        F6 = b(0.70 if F8 else 0.25)
        F7 = b(0.65 if (F1 and F3) else 0.20)
        F9 = b((0.50 if F8 else 0.20) + (0.10 if s == 'B' else 0))
        F10 = b(0.20 + 0.40 * carry)

        wtp = (210 + 18 * min(over, 8) + 6 * exkg + 40 * inc + 25 * (lk['cnt'] - 3) - 35 * (lk['pri'] - 3)
               + 30 * (fare == 'Business/premium') + 25 * F5 + 15 * F6 + rng.normal(0, 45))
        wtp = int(round(np.clip(wtp, 60, 720), -1))
        fwd = 0.55 * wtp + 45 * F8 + 15 * (lk['tec'] - 3) - 10 * (lk['pri'] - 3) + 20 * F6 + rng.normal(0, 40)
        fwd = int(round(np.clip(fwd, 50, 600), -1))
        credit = 220 - 30 * (lk['tru'] - 3) + 35 * (lk['pos'] - 3) + 20 * (lk['sec'] - 3) + 15 * inc + rng.normal(0, 40)
        credit = int(round(np.clip(credit, 50, 600), -1))

        top = max([('sec', lk['sec']), ('pos', lk['pos']), ('tru', 6 - lk['tru']), ('pri', lk['pri'])], key=lambda t: t[1] + rng.random())
        concern = random.choice(CONCERNS[top[0] if top[1] >= 4 else 'none'])

        row = dict(respondent_id=f'KS{i + 1:05d}', survey_date=(date(2026, 10, 5) + timedelta(days=int(rng.integers(0, 26)))).isoformat(),
                   age=age, city=pick(CITIES, CITY_W), occupation=pick(*p['occ']), monthly_income_band=INCOME[inc],
                   domestic_flights_12m=flights, usual_fare_type=fare, times_exceeded_allowance_12m=over, avg_excess_kg=exkg,
                   avg_checked_bag_kg=bag, excess_fees_paid_12m_inr=fees,
                   possessiveness_luggage=lk['pos'], trust_strangers=lk['tru'], security_concern=lk['sec'],
                   price_sensitivity=lk['pri'], tech_comfort=lk['tec'], counter_trust_boost=lk['cnt'],
                   would_use_app='Yes' if use else 'No', usage_intent_6m=intent)
        row.update(dict(zip(FEATURE_COLS, [F1, F2, F3, F4, F5, F6, F7, F8, F9, F10])))
        row.update(wtp_p2p_per_kg_inr=wtp, wtp_forwarding_per_kg_inr=fwd, would_carry='Yes' if carry else 'No',
                   min_credit_per_kg_inr=credit, biggest_concern_text=concern)
        rows.append(row)
    clean = pd.DataFrame(rows)
    truth = pd.DataFrame({'respondent_id': clean.respondent_id, 'true_segment': [SEGMENTS[s] for s in segs]})
    raw, log = add_noise(clean, rng, n)
    return raw, log, truth


def add_noise(clean, rng, n):
    df = clean.copy().astype(object); ids = df.respondent_id.values; log, used = [], set()
    k = lambda base: max(1, int(round(base * n / 1200)))
    def sample(m, exclude=()):
        pool = [x for x in range(n) if x not in exclude]
        return sorted(rng.choice(pool, m, replace=False).tolist())
    def note(kind, col, idx, fix):
        log.append(dict(problem=kind, column=col, rows=len(idx), respondent_ids=', '.join(ids[j] for j in idx[:60]) + (' ...' if len(idx) > 60 else ''), how_cleaned=fix))
        used.update(idx)

    idx = sample(k(20))
    for j in idx:
        for c in LIKERT_COLS: df.at[j, c] = 3
        for c in FEATURE_COLS: df.at[j, c] = 1
        df.at[j, 'usage_intent_6m'] = 3
    note('Straight-lining (same answer everywhere)', 'Q12-Q17, Q19, Q20', idx, 'Respondent dropped')
    idx = sample(k(48), used)
    for j in idx: df.at[j, 'would_use_app'] = 'No' if df.at[j, 'would_use_app'] == 'Yes' else 'Yes'
    note('Answer flipped (careless click)', 'would_use_app', idx, 'Cannot be detected; limits model accuracy')
    idx = sample(k(4), used)
    for j, v in zip(idx, [99, 101, 12, 15] * 20): df.at[j, 'age'] = v
    note('Impossible age', 'age', idx, 'Set blank, then imputed')
    idx = sample(k(6), used)
    for j in idx: df.at[j, 'domestic_flights_12m'] = int(rng.integers(120, 300))
    note('Implausible flight count', 'domestic_flights_12m', idx, 'Set blank, then imputed')
    idx = sample(k(8), used)
    for j in idx: df.at[j, 'avg_checked_bag_kg'] = round(float(clean.at[j, 'avg_checked_bag_kg']) * 2.2046, 1)
    note('Bag weight typed in pounds', 'avg_checked_bag_kg', idx, 'Divided by 2.2046')
    idx = sample(k(4), used)
    for j in idx: df.at[j, 'avg_checked_bag_kg'] = 0.0
    note('Zero bag weight', 'avg_checked_bag_kg', idx, 'Set blank, then imputed')
    idx = sample(k(3), used)
    for j in idx: df.at[j, 'avg_checked_bag_kg'] = 85.0
    note('Extreme bag weight', 'avg_checked_bag_kg', idx, 'Set blank, then imputed')
    cand = [j for j in range(n) if clean.at[j, 'excess_fees_paid_12m_inr'] > 0 and j not in used]
    idx = sorted(rng.choice(cand, k(5), replace=False).tolist())
    for j in idx: df.at[j, 'excess_fees_paid_12m_inr'] = int(clean.at[j, 'excess_fees_paid_12m_inr']) * 100
    note('Two extra zeros typed', 'excess_fees_paid_12m_inr', idx, 'Divided by 100')
    for col, val in [('wtp_p2p_per_kg_inr', 9999), ('wtp_forwarding_per_kg_inr', 9999)]:
        idx = sample(k(5), used)
        for j in idx: df.at[j, col] = val
        note('Placeholder 9999', col, idx, 'Set blank, then imputed')
    idx = sample(k(5), used)
    for j in idx: df.at[j, 'wtp_p2p_per_kg_inr'] = int(rng.choice([1500, 2000, 2500, 3000]))
    note('Extreme WTP above the airline rate', 'wtp_p2p_per_kg_inr', idx, 'Capped at Rs 800')
    idx = sample(k(4), used)
    for j in idx: df.at[j, 'min_credit_per_kg_inr'] = int(rng.choice([2000, 5000]))
    note('Extreme carrier ask', 'min_credit_per_kg_inr', idx, 'Capped at Rs 800')
    cand = [j for j in range(n) if clean.at[j, 'times_exceeded_allowance_12m'] > 0 and j not in used]
    idx = sorted(rng.choice(cand, k(12), replace=False).tolist())
    for j in idx: df.at[j, 'times_exceeded_allowance_12m'] = 0
    note('Contradiction: 0 times over but excess kg > 0', 'times_exceeded_allowance_12m', idx, 'Imputed from fees and kg')
    cand = [j for j in range(n) if clean.at[j, 'domestic_flights_12m'] <= 6 and j not in used]
    idx = sorted(rng.choice(cand, k(6), replace=False).tolist())
    for j in idx: df.at[j, 'times_exceeded_allowance_12m'] = int(clean.at[j, 'domestic_flights_12m']) + int(rng.integers(2, 6))
    note('Contradiction: over allowance more often than flights taken', 'times_exceeded_allowance_12m', idx, 'Capped at flights taken')
    var = {'Mumbai': ['Bombay', 'mumbai', 'MUMBAI ', 'Mumbai '], 'Delhi': ['New Delhi', 'Delhi NCR', 'delhi'],
           'Bengaluru': ['Bangalore', 'bangalore', 'Blr', 'Bengaluru '], 'Hyderabad': ['Hyd', 'hyderabad'], 'Chennai': ['Madras', 'chennai'],
           'Kolkata': ['Calcutta', 'kolkata'], 'Pune': ['pune'], 'Ahmedabad': ['Amdavad', 'ahmedabad']}
    idx = sample(k(96))
    for j in idx: df.at[j, 'city'] = random.choice(var[clean.at[j, 'city']])
    note('Inconsistent city spelling', 'city', idx, 'Mapped to 8 standard names')
    for col in ['would_use_app', 'would_carry']:
        idx = sample(k(36))
        for j in idx:
            yes = str(df.at[j, col]) == 'Yes'
            df.at[j, col] = random.choice(['Y', 'yes', 'YES', '1']) if yes else random.choice(['N', 'no', 'NO', '0'])
        note('Inconsistent Yes/No coding', col, idx, 'Mapped to Yes / No')
    for col, rate in [('monthly_income_band', .06), ('avg_checked_bag_kg', .03), ('trust_strangers', .03), ('wtp_p2p_per_kg_inr', .04),
                      ('wtp_forwarding_per_kg_inr', .04), ('age', .015), ('min_credit_per_kg_inr', .035)]:
        idx = sample(int(n * rate))
        for j in idx: df.at[j, col] = np.nan
        note('Blank answer', col, idx, 'Median (numbers) or most common answer (categories)')
    dup = sample(k(15))
    note('Duplicate submission', 'all columns', dup, 'Second copy removed')
    out = pd.concat([df, df.iloc[dup]]).sort_values('respondent_id', kind='stable').reset_index(drop=True)
    return out, pd.DataFrame(log)
