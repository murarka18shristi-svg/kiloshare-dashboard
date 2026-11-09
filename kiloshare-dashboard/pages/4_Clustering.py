import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from kiloshare import ui

ui.setup('Clustering', '🧩')
st.title('🧩 Clustering: which customer segments exist?')
st.markdown('K-means groups respondents on age, flights, bag weight, how often they go over the allowance, and the six attitude questions '
            '(all standardised). Segments are then described with answers they were not clustered on: intent, price and willingness to carry.')
K = ui.k_selection()
best_k = int(K.loc[K.silhouette.idxmax(), 'k'])
a, b = st.columns(2)
with a:
    fig = go.Figure(go.Bar(x=K.k, y=K.silhouette, marker_color=[ui.BLUE if k == best_k else ui.GREY for k in K.k]))
    fig.update_layout(height=260, title='Silhouette score (higher = better separated)', xaxis_title='k', margin=dict(l=10, r=10, t=40, b=10))
    st.plotly_chart(fig, width='stretch')
with b:
    fig = go.Figure(go.Scatter(x=K.k, y=K.inertia, mode='lines+markers', line_color=ui.ORANGE))
    fig.update_layout(height=260, title='Elbow: within-cluster spread', xaxis_title='k', margin=dict(l=10, r=10, t=40, b=10))
    st.plotly_chart(fig, width='stretch')
k = st.slider('Number of segments (k)', 2, 8, best_k, help=f'k = {best_k} has the highest silhouette score')
Cl = ui.clustering(k)
prof = Cl['profile']
m1, m2, m3 = st.columns(3)
m1.metric('Silhouette', f"{Cl['silhouette']:.3f}")
m2.metric('Agreement with hierarchical (Ward) clustering', f"{Cl['ari_ward']:.2f}", help='Adjusted Rand index: 1 = identical groups, 0 = chance')
m3.metric('Variance shown in the 2-D map', ui.pct(Cl['explained']))

colors = {s: ui.SEG_COLORS.get(s, c) for s, c in zip(prof.segment, px.colors.qualitative.Set2)}
a, b = st.columns([1.2, 1])
with a:
    st.subheader('Segment map')
    pc = Cl['pca'].sample(2500, random_state=0)
    fig = px.scatter(pc, x='pc1', y='pc2', color='segment', opacity=.55, color_discrete_map=colors)
    fig.update_traces(marker_size=5); fig.update_layout(height=420, legend=dict(orientation='h', y=-0.15), margin=dict(l=10, r=10, t=10, b=10),
                                                         xaxis_title='Principal component 1', yaxis_title='Principal component 2')
    st.plotly_chart(fig, width='stretch')
with b:
    st.subheader('Who converts')
    fig = go.Figure(go.Bar(x=prof.would_use, y=prof.segment, orientation='h', marker_color=[colors[s] for s in prof.segment],
                           text=[f'{ui.pct(u)} · {c}' for u, c in zip(prof.would_use, prof.conversion)], textposition='auto'))
    fig.update_layout(height=420, xaxis_tickformat='.0%', xaxis_title='Would use KiloShare', yaxis_autorange='reversed', margin=dict(l=10, r=10, t=10, b=10))
    st.plotly_chart(fig, width='stretch')

st.subheader('Segment profiles')
st.caption('Conversion band from average 6-month intent (Q19): High 3.5+, Medium 2.5-3.5, Low below 2.5.')
fmt = {'age': '{:.0f}', 'flights': '{:.1f}', 'bag_kg': '{:.1f}', 'times_over': '{:.1f}', 'possessive': '{:.1f}', 'trust': '{:.1f}', 'security': '{:.1f}',
       'price_sens': '{:.1f}', 'tech': '{:.1f}', 'counter': '{:.1f}', 'would_use': '{:.0%}', 'intent': '{:.1f}', 'would_carry': '{:.0%}',
       'wtp_p2p_median': '₹{:.0f}', 'wtp_fwd_median': '₹{:.0f}', 'credit_ask_median': '₹{:.0f}', 'respondents': '{:,}'}
st.dataframe(prof.style.format(fmt), hide_index=True, width='stretch')

st.subheader('Attitude fingerprint')
att = prof.set_index('segment')[['possessive', 'trust', 'security', 'price_sens', 'tech', 'counter']]
att.columns = ['Possessive', 'Trusts strangers', 'Security worry', 'Price-sensitive', 'Tech comfort', 'Counter trust']
fig = px.imshow(att, text_auto='.1f', color_continuous_scale='RdBu_r', zmin=1, zmax=5, aspect='auto')
fig.update_layout(height=60 + 55 * len(att), margin=dict(l=10, r=10, t=10, b=10), coloraxis_colorbar_title='1-5')
st.plotly_chart(fig, width='stretch')
if k == 4:
    st.info('**What it means for launch:** win **Budget students & movers** and **Frequent overpackers** first (high conversion; overpackers pay the most). '
            '**Light travellers & sceptics** rarely need the service but carry the lightest bags, so recruit them as carriers. '
            '**Family visitors** are lukewarm and possessive: reach them with the counter and inspection, not ads.')
