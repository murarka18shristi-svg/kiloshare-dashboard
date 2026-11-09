import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from kiloshare import ui
from kiloshare import models as M

ui.setup('Classification', '🎯')
st.title('🎯 Classification: who will use KiloShare?')
_, _, d, _ = ui.load()
target_label = st.radio('Predict', ['Q18 Would use KiloShare (demand)', 'Q23 Would carry a pouch (supply)'], horizontal=True)
target = 'use' if target_label.startswith('Q18') else 'carry'
C = ui.classification(target)
comp = C['comparison']
st.markdown(f"Five models were trained on 70% of respondents, tested on the other 30% ({C['test_n']:,}), and cross-validated five times. "
            f"Base rate: **{ui.pct(d[target].mean(), 1)}** answered yes.")

best = comp.iloc[0]
st.markdown(f'**Best model: {best.model}**')
m1, m2, m3, m4 = st.columns(4)
m1.metric('Models compared', len(comp))
m2.metric('Cross-validated AUC', f'{best.cv_auc:.3f}', help='Chance the model ranks a random "yes" above a random "no". 0.5 is guessing, 1.0 is perfect.')
m3.metric('Test accuracy', ui.pct(best.accuracy, 1))
m4.metric('Recall', ui.pct(best.recall, 1), help='Share of actual "yes" answers the model catches')

st.subheader('Model comparison')
show = comp.rename(columns={'model': 'Model', 'accuracy': 'Accuracy', 'precision': 'Precision', 'recall': 'Recall', 'f1': 'F1', 'test_auc': 'Test AUC',
                            'cv_auc': 'Cross-validated AUC', 'cv_auc_sd': '± SD'})
st.dataframe(show.style.format({c: '{:.3f}' for c in show.columns if c != 'Model'})
             .highlight_max(subset=['Cross-validated AUC', 'Accuracy', 'F1'], color='#d8ecdf'), hide_index=True, width='stretch')
a, b = st.columns([1, 1])
with a:
    st.subheader('ROC curves (test set)')
    fig = go.Figure()
    for name, r in C['roc'].items():
        fig.add_trace(go.Scatter(x=r.fpr, y=r.tpr, name=f"{name} ({comp.set_index('model').test_auc[name]:.2f})", mode='lines'))
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], line=dict(dash='dash', color='#999'), showlegend=False))
    fig.update_layout(height=320, xaxis_title='False positive rate', yaxis_title='True positive rate', margin=dict(l=10, r=10, t=10, b=10),
                      legend=dict(font=dict(size=10), x=.45, y=.05))
    st.plotly_chart(fig, width='stretch')
with b:
    st.subheader('Confusion matrix')
    pick = st.selectbox('Model', list(C['confusion']), index=list(C['confusion']).index(best.model))
    cm = C['confusion'][pick]
    fig = px.imshow(cm, text_auto=True, color_continuous_scale='Blues', x=['Predicted no', 'Predicted yes'], y=['Actual no', 'Actual yes'])
    fig.update_layout(height=250, coloraxis_showscale=False, margin=dict(l=10, r=10, t=10, b=10))
    st.plotly_chart(fig, width='stretch')

st.subheader('What drives the answer')
a, b = st.columns([1.4, 1])
with a:
    imp = C['importance']['Logistic regression'].sort_values('value')
    fig = go.Figure(go.Bar(x=imp.value, y=imp.driver, orientation='h', marker_color=[ui.BLUE if v > 0 else ui.ORANGE for v in imp.value],
                           customdata=imp.odds_ratio, hovertemplate='%{y}<br>coefficient %{x:.2f}<br>odds × %{customdata:.2f} per SD<extra></extra>'))
    fig.update_layout(height=420, xaxis_title='Logistic regression: standardised coefficient (blue raises the odds, orange lowers them)', margin=dict(l=10, r=10, t=10, b=10))
    st.plotly_chart(fig, width='stretch')
with b:
    if 'Random forest' in C['importance']:
        rf = C['importance']['Random forest'].sort_values('value')
        fig = go.Figure(go.Bar(x=rf.value, y=rf.driver, orientation='h', marker_color=ui.GREEN))
        fig.update_layout(height=420, xaxis_title='Random forest: share of importance', xaxis_tickformat='.0%', margin=dict(l=10, r=10, t=10, b=10))
        st.plotly_chart(fig, width='stretch')

st.subheader('A decision tree, as plain rules')
st.dataframe(C['tree_rules'].style.format({'share_yes': '{:.0%}'}), hide_index=True, width='stretch')

if target == 'use':
    st.subheader('Targeting: why scoring makes marketing cheaper')
    lift = M.lift_table(d, C['oof'])
    fig = go.Figure(go.Bar(x=lift.decile, y=lift.would_use, marker_color=[ui.BLUE if x <= 3 else ui.GREY for x in lift.decile],
                           customdata=np.c_[lift.lift, lift.cumulative_share_of_users],
                           hovertemplate='Decile %{x}<br>Would use %{y:.0%}<br>Lift %{customdata[0]:.2f}x<br>Cumulative share of users %{customdata[1]:.0%}<extra></extra>'))
    fig.add_hline(y=d.use.mean(), line_dash='dash', annotation_text='average')
    fig.update_layout(height=300, yaxis_tickformat='.0%', xaxis_title='Decile of model score (1 = most likely)', yaxis_title='Would use', margin=dict(l=10, r=10, t=10, b=10))
    st.plotly_chart(fig, width='stretch')
    st.caption(f"The top 30% by score hold {ui.pct(lift.cumulative_share_of_users.iloc[2])} of everyone who would use KiloShare: "
               f"spending ad money there reaches {lift.would_use.iloc[:3].mean() / d.use.mean():.1f}x as many likely users per rupee.")

st.subheader('Score a traveller')
with st.form('score'):
    c = st.columns(4)
    age = c[0].number_input('Age', 18, 80, 30); inc = c[1].selectbox('Income', list(M.INCOME), 2); fl = c[2].number_input('Flights a year', 1, 60, 8)
    bag = c[3].number_input('Checked bag kg', 1.0, 30.0, 17.0)
    c = st.columns(4)
    over = c[0].number_input('Times over allowance', 0, 30, 3); ex = c[1].number_input('Average excess kg', 0.0, 15.0, 4.0)
    fees = c[2].number_input('Excess fees paid (₹)', 0, 100000, 6000, step=500)
    c = st.columns(6)
    lk = [c[i].slider(lbl, 1, 5, 3) for i, lbl in enumerate(['Possessive', 'Trusts strangers', 'Security worry', 'Price-sensitive', 'Tech comfort', 'Counter trust'])]
    go_ = st.form_submit_button('Predict')
if go_:
    row = pd.DataFrame([[age, M.INCOME[inc], fl, bag, over, ex, fees] + lk], columns=M.PREDICTORS)
    p = ui.scoring_model(target).predict_proba(row)[0, 1]
    st.metric('Probability of "yes"', ui.pct(p, 1), delta=f'{(p - d[target].mean()) * 100:+.1f} points vs average')
