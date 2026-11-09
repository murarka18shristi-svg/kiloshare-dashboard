import pandas as pd
import plotly.express as px
import streamlit as st
from kiloshare import ui

ui.setup('Data and cleaning', '🧹')
raw, noise, d, log = ui.load()
st.title('🧹 Simulated data and cleaning')
st.markdown('The responses are simulated from four hidden customer types, with realistic links between habits, attitudes and answers. '
            'Noise and outliers were then planted on purpose, the way real surveys arrive, and cleaned with the documented rules below.')

c1, c2, c3, c4 = st.columns(4)
c1.metric('Raw rows', f'{len(raw):,}')
c2.metric('Planted problems', f'{int(noise.rows.sum()):,}', help='Cells or rows affected, across all problem types')
c3.metric('Rows removed', f'{len(raw) - len(d):,}', help='Duplicates and straight-lining respondents')
c4.metric('Clean respondents', f'{len(d):,}')

t1, t2, t3 = st.tabs(['What was planted', 'How it was cleaned', 'Before and after'])
with t1:
    st.dataframe(noise[['problem', 'column', 'rows', 'how_cleaned']], hide_index=True, width='stretch')
with t2:
    st.dataframe(log, hide_index=True, width='stretch')
    st.caption('Answers flipped by careless clicks cannot be detected; they are left in and limit how accurate any model can be.')
with t3:
    col = st.selectbox('Column', ['avg_checked_bag_kg', 'wtp_p2p_per_kg_inr', 'excess_fees_paid_12m_inr', 'domestic_flights_12m', 'age', 'min_credit_per_kg_inr'])
    both = pd.concat([pd.DataFrame({'value': pd.to_numeric(raw[col], errors='coerce'), 'version': 'Raw'}),
                      pd.DataFrame({'value': d[col], 'version': 'Clean'})])
    fig = px.box(both, x='value', y='version', color='version', points='outliers', color_discrete_map={'Raw': ui.RED, 'Clean': ui.BLUE})
    fig.update_layout(height=280, showlegend=False, margin=dict(l=10, r=10, t=10, b=10), xaxis_title=col, yaxis_title='')
    st.plotly_chart(fig, width='stretch')
    a, b = st.columns(2)
    a.markdown('**Raw city spellings**'); a.dataframe(raw.city.value_counts().rename('rows'), width='stretch', height=260)
    b.markdown('**Clean city spellings**'); b.dataframe(d.city.value_counts().rename('rows'), width='stretch', height=260)

st.subheader('Data preview')
which = st.radio('Show', ['Clean', 'Raw'], horizontal=True)
st.dataframe((d if which == 'Clean' else raw).head(200), width='stretch', height=300)
x, y = st.columns(2)
x.download_button('Download raw data (CSV)', raw.to_csv(index=False).encode(), 'kiloshare_survey_raw.csv', 'text/csv')
y.download_button('Download clean data (CSV)', d.drop(columns=['income_level', 'premium_fare', 'use', 'carry']).to_csv(index=False).encode(), 'kiloshare_survey_clean.csv', 'text/csv')
