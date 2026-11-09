import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from kiloshare import ui, economics as E

ui.setup('Sustainability', '📈')
st.title('📈 Sustainability: break-even simulation')
st.markdown('Five launch strategies, each run through **5,000 random futures over 24 months**. Every future draws its own value for the uncertain inputs '
            '(how many passengers are over the allowance, awareness, how stated intent turns into purchases, carrier supply, no-shows, costs, '
            'and a small chance of a security incident that shuts the service). Demand and prices come from the survey; traffic from published 2025 figures.')

D = E.DEFAULTS
with st.sidebar:
    st.header('Assumptions')
    st.caption('Change any value; the simulation reruns.')
    auto = st.toggle('Use survey-optimised prices', True)
    p2p = None if auto else st.number_input('Peer-to-peer price, ₹/kg', 100, 800, 450, 25)
    cr = None if auto else st.number_input('Carrier credit, ₹/kg', 50, 600, 300, 25)
    fw = None if auto else st.number_input('Forwarding price, ₹/kg', 50, 600, 300, 25)
    aw = st.slider('Awareness at maturity (most likely)', 0.05, 0.60, D['awareness'], 0.05, help='Share of passengers who know KiloShare at the moment they need it')
    ov = st.slider('Passengers over the allowance (most likely)', 0.03, 0.30, D['over_share'], 0.01)
    courier = st.slider('Courier cost, ₹/kg', 80, 300, D['courier_cost'], 10)
    fo = st.number_input('Fixed cost, two own kiosks (₹/month)', 300_000, 3_000_000, D['fixed_own'], 50_000)
    fp = st.number_input('Fixed cost, two partner counters (₹/month)', 100_000, 2_000_000, D['fixed_partner'], 50_000)
    oh = st.number_input('Overhead: app, cloud, team (₹/month)', 0, 1_000_000, D['overhead'], 25_000)
    share = st.slider('Revenue share to partner counters', 0.0, 0.40, D['partner_share'], 0.05)
    budget = st.number_input('Paid acquisition budget (₹/month, S5)', 0, 2_000_000, D['paid_budget'], 50_000)
    cac = st.number_input('Cost to acquire a user (₹)', 50, 2000, D['cac'], 25)
params = dict(p2p_price=p2p, carrier_credit=cr, fwd_price=fw, awareness=aw, over_share=ov, courier_cost=courier, fixed_own=fo, fixed_partner=fp,
              overhead=oh, partner_share=share, paid_budget=budget, cac=cac)
out = ui.sustainability(tuple(sorted(params.items(), key=lambda kv: kv[0])))
res = out['results']; v = E.verdict(res); p, c, pf = out['prices']
ui.verdict_box(v)
st.caption(f'Prices used: ₹{p}/kg peer-to-peer, ₹{c}/kg carrier credit, ₹{pf}/kg forwarding · {out["kg"]:.1f} kg per transfer (survey) · '
           f'S1 keeps the original ₹400 / ₹150 plan.')

st.subheader('Strategy results')
tbl = res[['strategy', 'profitable_m18', 'profitable_m24', 'median_profit_m24', 'p10_profit_m24', 'median_cash_low', 'cash_positive_m24',
           'p2p_per_day', 'fwd_per_day', 'match_rate', 'incident_share']].copy()
tbl.columns = ['Strategy', 'Profitable m18', 'Profitable m24', 'Median profit m24', 'Bad case (10th pct) m24', 'Median cash low point',
               'Paid back by m24', 'P2P transfers/day', 'Forwarding jobs/day', 'Match rate', 'Security incident']
st.dataframe(tbl.style.format({'Profitable m18': '{:.0%}', 'Profitable m24': '{:.0%}', 'Paid back by m24': '{:.0%}', 'Match rate': '{:.0%}',
                               'Security incident': '{:.1%}', 'P2P transfers/day': '{:.1f}', 'Forwarding jobs/day': '{:.0f}',
                               'Median profit m24': ui.lakh, 'Bad case (10th pct) m24': ui.lakh, 'Median cash low point': ui.lakh})
             .background_gradient(subset=['Profitable m24'], cmap='RdYlGn', vmin=0, vmax=1), hide_index=True, width='stretch')

a, b = st.columns(2)
with a:
    st.subheader('Monthly profit paths')
    pick = st.selectbox('Show the uncertainty band for', list(res.code), index=list(res.code).index(v['best'].code))
    fan = out['fan']; fig = go.Figure(); f = fan[fan.code == pick]
    fig.add_trace(go.Scatter(x=list(f.month) + list(f.month[::-1]), y=list(f.p90 / 1e5) + list(f.p10[::-1] / 1e5), fill='toself',
                             fillcolor='rgba(47,109,181,0.15)', line=dict(width=0), name=f'{pick} 10th-90th pct', hoverinfo='skip'))
    for code, g in fan.groupby('code'):
        fig.add_trace(go.Scatter(x=g.month, y=g['median'] / 1e5, name=code, line=dict(color=ui.STRAT_COLORS[code], width=3 if code == pick else 1.5)))
    fig.add_hline(y=0, line_color='#444')
    fig.update_layout(height=380, yaxis_title='₹ lakh per month', xaxis_title='Month', legend=dict(orientation='h', y=-0.2), margin=dict(l=10, r=10, t=10, b=10))
    st.plotly_chart(fig, width='stretch')
with b:
    st.subheader(f'What matters most ({v["best"].code})')
    s = out['sensitivity'].head(9).iloc[::-1]
    fig = go.Figure(go.Bar(x=s.correlation, y=s.input, orientation='h', marker_color=[ui.BLUE if x > 0 else ui.ORANGE for x in s.correlation]))
    fig.update_layout(height=380, xaxis_title='Rank correlation with month-24 profit', margin=dict(l=10, r=10, t=10, b=10))
    st.plotly_chart(fig, width='stretch')

st.subheader('Break-even thresholds for peer-to-peer alone')
th = out['thresholds'].copy()
st.dataframe(th.style.format({'margin': ui.inr, 'monthly_fixed': ui.lakh, 'transfers_per_day_needed': '{:.0f}', 'price': '₹{:.0f}', 'credit': '₹{:.0f}'}),
             hide_index=True, width='stretch')
st.caption(f"Same-flight matching delivers about {res.set_index('code').p2p_per_day['S2']:.1f} transfers a day on Delhi-Mumbai, "
           f"because only about {ui.pct(res.set_index('code').match_rate['S2'])} of requests find a willing carrier with room on the same flight.")

a, b = st.columns(2)
with a:
    st.subheader('Peer-to-peer price search')
    g = out['grid'].pivot(index='credit', columns='price', values='contribution') / 1e5
    fig = px.imshow(g, color_continuous_scale='RdYlGn', origin='lower', aspect='auto', labels=dict(color='₹ lakh/month'))
    fig.update_layout(height=340, xaxis_title='Price ₹/kg', yaxis_title='Carrier credit ₹/kg', margin=dict(l=10, r=10, t=10, b=10))
    st.plotly_chart(fig, width='stretch')
with b:
    st.subheader('Forwarding price search')
    fg = out['fgrid']
    fig = go.Figure(go.Scatter(x=fg.price, y=fg.contribution / 1e5, mode='lines+markers', line_color=ui.PURPLE))
    fig.update_layout(height=340, xaxis_title='Forwarding price ₹/kg', yaxis_title='Base-case contribution, ₹ lakh/month', margin=dict(l=10, r=10, t=10, b=10))
    st.plotly_chart(fig, width='stretch')

with st.expander('All assumptions and sources'):
    P = out['params']
    st.dataframe(pd.DataFrame([
        ['Delhi-Mumbai passengers, 2025', '5.5 million', 'IATA World Air Transport Statistics (via Free Press Journal)'],
        ['Mumbai domestic passengers, 2025', '39.2 million', 'Moodie Davitt Report'],
        ['Delhi IGI total passengers, 2025', '78.1 million, 70% assumed domestic', 'Network Thoughts; share assumed'],
        ['Airline airport excess rate', '₹800/kg (IndiGo, Sept 2026)', 'Business Today'],
        ['Aircraft', '180 seats, 85% full', 'Assumption'],
        ['Kg per transfer', f"{out['kg']:.1f} kg", 'Survey Q9, weighted by Q8'],
        ['Intent calibration', '5 → 70%, 4 → 35%, 3 → 10%, 1-2 → 0%', 'Top-box rule of thumb'],
        ['Cost per peer-to-peer transfer', f"₹{P['var_cost']} + {P['gateway']:.0%} payment fee", 'Assumption (pouch, seal, basic insurance)'],
        ['Courier cost', f"₹{P['courier_cost']}/kg (0.75x-1.4x)", 'Assumption'],
        ['Fixed costs', f"Own kiosks {ui.lakh(P['fixed_own'])}, partner counters {ui.lakh(P['fixed_partner'])}, overhead {ui.lakh(P['overhead'])} a month", 'Assumption'],
        ['Setup', f"Own {ui.lakh(P['setup_own'])}, partner {ui.lakh(P['setup_partner'])}", 'Assumption'],
        ['Uncertain ranges', 'Over-allowance 0.5x-1.8x, awareness 0.4x-1.6x, intent haircut 50-100%, carrier supply 0.7x-2x, no-shows 20-60%, '
                             'forwarding follow-through 30-70%, security incident 1 in 20,000 to 1 in 200,000 transfers', 'Assumption'],
    ], columns=['Input', 'Value', 'Basis']), hide_index=True, width='stretch')
    st.caption('Regulatory approval of the inspected-and-sealed process is assumed. Without it, only forwarding and repacking remain.')
