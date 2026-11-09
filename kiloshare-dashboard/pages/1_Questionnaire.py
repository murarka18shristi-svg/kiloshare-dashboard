import pandas as pd
import streamlit as st
from kiloshare import ui
from kiloshare.questionnaire import QUESTIONS, CONCEPT, FEATURES

ui.setup('Questionnaire', '📋')
st.title('📋 The questionnaire')
st.markdown('Twenty-five questions, each chosen because an algorithm needs it: **targets** the models predict, **inputs** that explain them, '
            'and a **feature basket** for association rules. Section D comes after the concept card.')

q = pd.DataFrame(QUESTIONS, columns=['No.', 'Section', 'Question', 'Answer', 'Dataset column', 'Feeds'])
q['Feeds'] = q['Feeds'].apply(lambda x: '; '.join(x) if x else '-')
alg = ['Classification', 'Clustering', 'Association rules', 'Regression']
cols = st.columns(4)
for c, a in zip(cols, alg):
    c.metric(a, f"{q.Feeds.str.contains(a).sum()} questions")

pick = st.multiselect('Show questions that feed', alg, default=[])
view = q if not pick else q[q.Feeds.str.contains('|'.join(pick))]
for section, g in view.groupby('Section', sort=False):
    st.subheader(section)
    if section.startswith('D'):
        with st.container(border=True):
            st.markdown('**Concept card shown before Section D**')
            st.write(CONCEPT)
    st.dataframe(g.drop(columns='Section'), hide_index=True, width='stretch',
                 column_config={'Question': st.column_config.TextColumn(width='large'), 'Feeds': st.column_config.TextColumn(width='large')})

with st.expander('Q20 feature list (the association-rule basket)'):
    st.dataframe(pd.DataFrame({'Code': [k.split('_')[0] for k in FEATURES], 'Feature': list(FEATURES.values()), 'Dataset column': list(FEATURES)}),
                 hide_index=True, width='stretch')

st.subheader('Which questions feed which algorithm')
st.dataframe(pd.DataFrame([
    ['Classification', 'Q18 would use (demand); Q23 would carry (supply)', 'Q2, Q5, Q6, Q8-Q11, Q12-Q17'],
    ['Clustering', 'none (finds groups by itself)', 'Q2, Q6, Q8, Q10, Q12-Q17; segments described with Q4, Q18, Q19, Q21, Q22'],
    ['Association rules', 'none', 'Q20, ten features, one basket per respondent'],
    ['Regression', 'Q21 peer-to-peer price; Q22 forwarding price; Q24 carrier credit', 'Q5, Q7, Q8, Q9, Q11, Q12-Q17 (and Q6, Q10 for carriers)'],
], columns=['Algorithm', 'Target', 'Inputs']), hide_index=True, width='stretch')
st.download_button('Download the questionnaire (CSV)', q.to_csv(index=False).encode(), 'kiloshare_questionnaire.csv', 'text/csv')
