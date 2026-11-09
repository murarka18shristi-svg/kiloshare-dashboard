import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from kiloshare import ui
from kiloshare.models import REG_TARGETS

ui.setup('Regression and pricing', '💰')
st.title('💰 Regression: what will people pay?')
st.markdown('Three prices decide the business: what requesters pay to send kilos with a co-passenger (Q21), what they pay for courier forwarding (Q22), '
            'and the credit carriers need (Q24). Linear regression explains each from income, fare type, excess-baggage history and attitudes, '
            'and is compared with a random forest.')
target = st.radio('Target', list(REG_TARGETS), format_func=lambda t: REG_TARGETS[t][0], horizontal=True)
R = ui.regression(target)
comp = R['comparison']
m = st.columns(4)
m[0].metric('Linear regression R²', f'{comp.r2.iloc[0]:.2f}'); m[1].metric('Average error', ui.inr(comp.mae.iloc[0]) + ' per kg')
m[2].metric('Random forest R²', f'{comp.r2.iloc[1]:.2f}'); m[3].metric('Respondents', f"{R['n']:,}", help='Carrier credit is modelled only on those who would carry')
a, b = st.columns([1.2, 1])
with a:
    st.subheader('What moves the price')
    cf = R['coefs'].sort_values('per_sd')
    fig = go.Figure(go.Bar(x=cf.per_sd, y=cf.driver, orientation='h', marker_color=[ui.BLUE if v > 0 else ui.ORANGE for v in cf.per_sd],
                           customdata=cf.per_unit, hovertemplate='%{y}<br>₹%{x:.0f} per SD<br>₹%{customdata:.1f} per unit<extra></extra>'))
    fig.update_layout(height=420, xaxis_title='₹ per kg for a one-standard-deviation rise', margin=dict(l=10, r=10, t=10, b=10))
    st.plotly_chart(fig, width='stretch')
with b:
    st.subheader('Actual vs predicted (test set)')
    ap = R['actual_pred']
    fig = px.scatter(ap, x='predicted', y='actual', opacity=.35)
    lo, hi = ap.min().min(), ap.max().max()
    fig.add_trace(go.Scatter(x=[lo, hi], y=[lo, hi], mode='lines', line=dict(dash='dash', color='#999'), showlegend=False))
    fig.update_layout(height=420, margin=dict(l=10, r=10, t=10, b=10), xaxis_title='Predicted ₹/kg', yaxis_title='Actual ₹/kg')
    st.plotly_chart(fig, width='stretch')

st.subheader('Demand meets supply: setting the price')
pc = ui.price_curves(); out = ui.sustainability(); p, c, pf = out['prices']
fig = go.Figure()
fig.add_trace(go.Scatter(x=pc.price, y=pc.p2p_demand, name='Would-users willing to pay this (peer-to-peer)', line=dict(color=ui.BLUE, width=3)))
fig.add_trace(go.Scatter(x=pc.price, y=pc.fwd_demand, name='Respondents willing to pay this (forwarding)', line=dict(color=ui.PURPLE, width=3)))
fig.add_trace(go.Scatter(x=pc.price, y=pc.carrier_supply, name='Carriers who accept this credit', line=dict(color=ui.ORANGE, width=3)))
for x, t, col in [(400, 'planned ₹400', '#999'), (p, f'best price ₹{p}', ui.BLUE), (c, f'best credit ₹{c}', ui.ORANGE), (pf, f'best forwarding ₹{pf}', ui.PURPLE)]:
    fig.add_vline(x=x, line_dash='dot', line_color=col, annotation_text=t, annotation_position='top')
fig.update_layout(height=420, yaxis_tickformat='.0%', xaxis_title='₹ per kg', legend=dict(orientation='h', y=-0.2), margin=dict(l=10, r=10, t=30, b=10))
st.plotly_chart(fig, width='stretch')
st.caption(f'Best prices come from a search over every price and credit pair, maximising monthly contribution (Sustainability page). '
           f'The planned ₹400/kg with a ₹150 credit attracts almost no carriers: only {ui.pct(pc.set_index("price").carrier_supply[150])} accept ₹150.')

st.subheader('Does wanting a feature raise what people pay?')
prem = ui.feature_premium()
fig = go.Figure(go.Scatter(x=prem.premium, y=prem.code + ' ' + prem.feature.str.slice(0, 38), mode='markers',
                           marker=dict(size=11, color=[ui.GREEN if s else ui.GREY for s in prem.significant]),
                           error_x=dict(type='data', symmetric=False, array=prem.high - prem.premium, arrayminus=prem.premium - prem.low)))
fig.add_vline(x=0, line_color='#444')
fig.update_layout(height=420, xaxis_title='Extra ₹ per kg (95% interval), with income, fares, history and attitudes held constant', yaxis_autorange='reversed',
                  margin=dict(l=10, r=10, t=10, b=10))
st.plotly_chart(fig, width='stretch')
sig = prem[prem.significant & (prem.premium > 0)]
st.success('Only ' + ' and '.join(f'**{r.feature.lower()}** (+₹{r.premium:.0f}/kg)' for r in sig.itertuples())
           + ' raise willingness to pay, so they are the paid add-ons. Every other feature should be free or included in the per-kg price.')
