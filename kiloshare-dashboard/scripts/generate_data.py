"""Regenerate the simulated survey data in data/ (fixed seed, so the output is identical every time).

Usage:  python scripts/generate_data.py
"""
import sys
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from kiloshare.simulate import simulate
from kiloshare.questionnaire import QUESTIONS

raw, noise, truth = simulate(n=5000, seed=2026)
(ROOT / 'data').mkdir(exist_ok=True)
raw.to_csv(ROOT / 'data' / 'kiloshare_survey_raw.csv', index=False)
noise.to_csv(ROOT / 'data' / 'noise_log.csv', index=False)
truth.to_csv(ROOT / 'data' / 'true_segments.csv', index=False)
q = pd.DataFrame(QUESTIONS, columns=['no', 'section', 'question', 'answer', 'column', 'feeds'])
q['feeds'] = q.feeds.apply('; '.join)
q.to_csv(ROOT / 'data' / 'questionnaire.csv', index=False)
print(f'{len(raw):,} raw rows, {int(noise.rows.sum()):,} planted problems -> data/')
