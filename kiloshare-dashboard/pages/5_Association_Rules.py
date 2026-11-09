import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from kiloshare import ui
from kiloshare.models import FNAME, feature_demand

ui.setup('Association rules', '🛒')
st.title('🛒 Association rules: which features go together?')
st.markdown('Each respondent\'s Q20 answer is a basket of wanted features. Apriori finds rules like **"if they want A and B, they also want C"**. '
            'Rather than launching ten features, we launch the bundles people expect together, so no one gets A and B without the C they assume comes with them.')
st.caption('Support = share of all respondents wanting the whole rule · Confidence = share of those wanting the left side who also want the right side · '
           'Lift = how much more likely than chance (above 1 means a real link).')
_, _, d, _ = ui.load()
c = st.columns(4)
ms = c[0].slider('Minimum support', 0.05, 0.40, 0.10, 0.01)
mc = c[1].slider('Minimum confidence', 0.30, 0.95, 0.60, 0.05)
ml = c[2].slider('Minimum lift', 1.0, 2.5, 1.2, 0.05)
segs = ['All respondents', 'Budget students & movers', 'Frequent overpackers', 'Family visitors', 'Light travellers & sceptics']
seg = c[3].selectbox('Segment', segs)
rules, itemsets = ui.apriori(ms, mc, ml, seg)
m = st.columns(3)
m[0].metric('Frequent itemsets', f'{len(itemsets):,}'); m[1].metric('Rules found', f'{len(rules):,}')
if len(rules): m[2].metric('Strongest rule', rules.iloc[0].rule, f'lift {rules.iloc[0].lift:.2f}', delta_color='off')
st.caption('Feature codes: ' + ' · '.join(f'**{k}** {v}' for k, v in FNAME.items()))

if len(rules):
    a, b = st.columns([1, 1.2])
    with a:
        fig = px.scatter(rules, x='support', y='confidence', size='lift', color='lift', hover_name='rule', color_continuous_scale='Viridis', size_max=18)
        fig.update_layout(height=380, xaxis_tickformat='.0%', yaxis_tickformat='.0%', margin=dict(l=10, r=10, t=10, b=10))
        st.plotly_chart(fig, width='stretch')
    with b:
        st.dataframe(rules[['rule', 'support', 'confidence', 'lift']].style.format({'support': '{:.1%}', 'confidence': '{:.1%}', 'lift': '{:.2f}'})
                     .background_gradient(subset=['lift'], cmap='Greens'), hide_index=True, width='stretch', height=380)
else:
    st.warning('No rules pass these thresholds. Lower the minimum support or confidence.')

st.subheader('Feature demand')
fd = feature_demand(d)
fig = go.Figure([go.Bar(y=fd.code + ' ' + fd.feature, x=fd.all_respondents, name='All respondents', orientation='h', marker_color=ui.BLUE),
                 go.Bar(y=fd.code + ' ' + fd.feature, x=fd.would_use, name='Would use KiloShare', orientation='h', marker_color=ui.ORANGE)])
fig.update_layout(barmode='group', height=460, xaxis_tickformat='.0%', yaxis_autorange='reversed', legend=dict(orientation='h', y=1.06), margin=dict(l=10, r=10, t=10, b=10))
st.plotly_chart(fig, width='stretch')

st.subheader('What to launch, and what to charge for')
prem = ui.feature_premium()
st.markdown('Rules say which features belong together; the feature-premium regression (Regression page) says which ones raise what people pay.')
st.dataframe([
    {'Bundle': 'Trust core: F1 matching, F2 ID check, F3 chat, F7 ratings', 'Evidence': 'F1 + F7 → F3, F2 + F7 → F3, F1 + F3 → F7', 'Launch': 'MVP', 'Price': 'Free in the app'},
    {'Bundle': 'Safety: F4 inspection and seal', 'Evidence': 'Top requested by security-worried users', 'Launch': 'MVP', 'Price': 'Included in the per-kg fee'},
    {'Bundle': 'F5 insurance', 'Evidence': f"F4 → F5; adds ₹{prem.set_index('code').premium['F5']:.0f}/kg to willingness to pay", 'Launch': 'MVP', 'Price': 'Paid add-on'},
    {'Bundle': 'Logistics: F6 tracking, F8 forwarding, F9 repack', 'Evidence': 'F6 + F9 → F8 (strongest rule); F6 adds ' + f"₹{prem.set_index('code').premium['F6']:.0f}/kg", 'Launch': 'MVP for the hybrid', 'Price': 'Forwarding and repack priced separately; tracking paid'},
    {'Bundle': 'F10 wallet credits', 'Evidence': 'Wanted by carriers, not requesters', 'Launch': 'Carrier side', 'Price': 'How carriers are paid'},
], hide_index=True, width='stretch')
