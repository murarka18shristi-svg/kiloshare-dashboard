"""Documented cleaning rules. Returns the clean data and a log of the rows each rule touched."""
import numpy as np
import pandas as pd
from .questionnaire import LIKERT_COLS, FEATURE_COLS

CITY = {'bombay': 'Mumbai', 'mumbai': 'Mumbai', 'new delhi': 'Delhi', 'delhi ncr': 'Delhi', 'delhi': 'Delhi',
        'bangalore': 'Bengaluru', 'blr': 'Bengaluru', 'bengaluru': 'Bengaluru', 'hyd': 'Hyderabad', 'hyderabad': 'Hyderabad',
        'madras': 'Chennai', 'chennai': 'Chennai', 'calcutta': 'Kolkata', 'kolkata': 'Kolkata', 'pune': 'Pune',
        'amdavad': 'Ahmedabad', 'ahmedabad': 'Ahmedabad'}
YN = {'yes': 'Yes', 'y': 'Yes', '1': 'Yes', 'no': 'No', 'n': 'No', '0': 'No'}
NUMERIC = ['age', 'domestic_flights_12m', 'times_exceeded_allowance_12m', 'avg_excess_kg', 'avg_checked_bag_kg', 'excess_fees_paid_12m_inr',
           'usage_intent_6m', 'wtp_p2p_per_kg_inr', 'wtp_forwarding_per_kg_inr', 'min_credit_per_kg_inr'] + LIKERT_COLS + FEATURE_COLS


def clean(raw):
    d = raw.copy(); log = []
    note = lambda step, n: log.append(dict(step=step, rows=int(n)))
    for c in NUMERIC: d[c] = pd.to_numeric(d[c], errors='coerce')
    n0 = len(d); d = d.drop_duplicates('respondent_id'); note('Duplicate submissions removed', n0 - len(d))
    before = d.city.copy(); d['city'] = d.city.astype(str).str.strip().str.lower().map(CITY); note('City spellings standardised', (before != d.city).sum())
    changed = 0
    for col in ['would_use_app', 'would_carry']:
        before = d[col].astype(str); d[col] = before.str.strip().str.lower().map(YN); changed += (before != d[col]).sum()
    note('Yes/No answers standardised', changed)
    straight = (d[LIKERT_COLS].nunique(axis=1, dropna=True) == 1) & (d[FEATURE_COLS].sum(axis=1) == len(FEATURE_COLS))
    note('Straight-lining respondents dropped', straight.sum()); d = d[~straight]
    rules = [
        ('Impossible ages (under 18 or over 80) set blank', (d.age < 18) | (d.age > 80), 'age', 'blank'),
        ('Flight counts over 100 set blank', d.domestic_flights_12m > 100, 'domestic_flights_12m', 'blank'),
    ]
    for label, mask, col, _ in rules:
        note(label, mask.sum()); d.loc[mask, col] = np.nan
    lbs = d.avg_checked_bag_kg.between(30, 65); note('Bag weights in pounds converted to kg', lbs.sum())
    d.loc[lbs, 'avg_checked_bag_kg'] = (d.loc[lbs, 'avg_checked_bag_kg'] / 2.2046).round(1)
    bad = (d.avg_checked_bag_kg == 0) | (d.avg_checked_bag_kg > 65); note('Zero or extreme bag weights set blank', bad.sum()); d.loc[bad, 'avg_checked_bag_kg'] = np.nan
    bad = d.excess_fees_paid_12m_inr > 100000; note('Fees with two extra zeros divided by 100', bad.sum()); d.loc[bad, 'excess_fees_paid_12m_inr'] /= 100
    for col in ['wtp_p2p_per_kg_inr', 'wtp_forwarding_per_kg_inr']:
        bad = d[col] == 9999; note(f'Placeholder 9999 set blank ({col})', bad.sum()); d.loc[bad, col] = np.nan
    for col in ['wtp_p2p_per_kg_inr', 'min_credit_per_kg_inr']:
        bad = d[col] > 800; note(f'Values above Rs 800 capped ({col})', bad.sum()); d.loc[bad, col] = 800
    bad = d.times_exceeded_allowance_12m > d.domestic_flights_12m; note('Times over allowance capped at flights taken', bad.sum())
    d.loc[bad, 'times_exceeded_allowance_12m'] = d.loc[bad, 'domestic_flights_12m']
    bad = (d.times_exceeded_allowance_12m == 0) & (d.avg_excess_kg > 0); note('Zero times over with excess kg: imputed from fees', bad.sum())
    d.loc[bad, 'times_exceeded_allowance_12m'] = np.maximum(1, (d.loc[bad, 'excess_fees_paid_12m_inr'] / (d.loc[bad, 'avg_excess_kg'] * 650)).round())
    d['biggest_concern_text'] = d['biggest_concern_text'].fillna('').astype(str)
    note('Blank cells imputed (median or most common answer)', d.isna().sum().sum())
    for c in d.columns:
        if d[c].isna().any():
            d[c] = d[c].fillna(d[c].median() if c in NUMERIC else d[c].mode()[0])
    for c in ['age', 'domestic_flights_12m', 'times_exceeded_allowance_12m', 'usage_intent_6m'] + LIKERT_COLS + FEATURE_COLS:
        d[c] = d[c].round().astype(int)
    return d.reset_index(drop=True), pd.DataFrame(log)
