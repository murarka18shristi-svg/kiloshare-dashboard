"""Shared, cached loaders and formatting for the Streamlit pages."""
from pathlib import Path
import pandas as pd
import streamlit as st
from . import models as M, economics as E
from .clean import clean
from .simulate import simulate

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'data'
BLUE, ORANGE, GREEN, RED, GREY, PURPLE = '#2F6DB5', '#E08A2C', '#3A9A5B', '#C0392B', '#AEB4BE', '#7B5EA7'
SEG_COLORS = {'Budget students & movers': BLUE, 'Frequent overpackers': GREEN, 'Family visitors': ORANGE, 'Light travellers & sceptics': GREY}
STRAT_COLORS = {'S1': GREY, 'S2': RED, 'S3': ORANGE, 'S4': GREEN, 'S5': PURPLE}


def setup(title, icon='🧳'):
    st.set_page_config(page_title=f'KiloShare · {title}', page_icon=icon, layout='wide')
    with st.sidebar:
        st.markdown('### 🧳 KiloShare')
        st.caption('Same-flight baggage-allowance sharing for domestic flyers in India, through a staffed airport counter.')
        st.caption('Synthetic survey: 5,000 simulated respondents with planted noise and outliers.')


@st.cache_data(show_spinner='Loading and cleaning the survey data...')
def load():
    raw_path = DATA / 'kiloshare_survey_raw.csv'
    if raw_path.exists():
        raw = pd.read_csv(raw_path); noise = pd.read_csv(DATA / 'noise_log.csv')
    else:
        raw, noise, _ = simulate()
    d, log = clean(raw)
    return raw, noise, M.prepare(d), log


@st.cache_data(show_spinner='Training and comparing five classifiers...')
def classification(target='use'):
    _, _, d, _ = load()
    return M.classification(d, target)


@st.cache_resource
def scoring_model(target='use'):
    _, _, d, _ = load()
    return M.scoring_model(d, target)


@st.cache_data(show_spinner='Testing k = 2 to 8...')
def k_selection():
    _, _, d, _ = load()
    return M.k_selection(d)


@st.cache_data(show_spinner='Clustering respondents...')
def clustering(k=4):
    _, _, d, _ = load()
    return M.clustering(d, k)


@st.cache_data(show_spinner='Mining association rules...')
def apriori(min_support, min_confidence, min_lift, segment='All respondents'):
    _, _, d, _ = load()
    if segment != 'All respondents':
        d = d[clustering(4)['segment'] == segment]
    return M.apriori(d, min_support, min_confidence, min_lift)


@st.cache_data(show_spinner='Fitting regression models...')
def regression(target):
    _, _, d, _ = load()
    return M.regression(d, target)


@st.cache_data
def feature_premium(target='wtp_p2p_per_kg_inr'):
    _, _, d, _ = load()
    return M.feature_premium(d, target)


@st.cache_data
def price_curves():
    _, _, d, _ = load()
    return M.price_curves(d)


@st.cache_data
def scored():
    """Clean data with the classifier score and the k = 4 segment attached."""
    _, _, d, _ = load()
    return d.assign(score=classification('use')['oof'], segment=clustering(4)['segment'])


@st.cache_data(show_spinner='Simulating 5,000 futures for each of five strategies...')
def sustainability(params: tuple = ()):
    S = E.survey_inputs(scored())
    return E.simulate(S, dict(params))


def inr(v): return ('−' if v < 0 else '') + '₹' + f'{abs(v):,.0f}'
def lakh(v): return ('−' if v < 0 else '') + f'₹{abs(v) / 1e5:,.1f} lakh'
def pct(v, dp=0): return f'{v * 100:.{dp}f}%'


def verdict_box(v):
    best = v['best']
    colour = {'Sustainable': 'success', 'Marginal': 'warning', 'Not sustainable': 'error'}[v['level']]
    text = (f"**{v['level']}, but only as {best.phrase}.** "
            f"In {pct(best.profitable_m24)} of 5,000 simulated futures it makes a monthly profit by month 24 "
            f"(median {lakh(best.median_profit_m24)} a month, after a median cash low point of {lakh(best.median_cash_low)}). "
            f"Pure peer-to-peer bag sharing is profitable in only {pct(v['p2p'])} of futures.")
    getattr(st, colour)(text, icon='✅' if colour == 'success' else '⚠️' if colour == 'warning' else '❌')
