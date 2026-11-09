import plotly.graph_objects as go
import streamlit as st
from kiloshare import ui, economics as E

ui.setup('Verdict')
raw, noise, d, log = ui.load()
out = ui.sustainability()
res, v = out['results'], E.verdict(out['results'])
best = v['best']; p, c, pf = out['prices']

st.title('🧳 KiloShare: is the business sustainable?')
st.caption(f'Synthetic consumer survey of {len(raw):,} rows ({len(d):,} respondents after cleaning), analysed with classification, '
           'clustering, association rules and regression, then tested in a 5,000-future break-even simulation.')
ui.verdict_box(v)

k1, k2, k3, k4 = st.columns(4)
k1.metric('Profitable by month 24', ui.pct(best.profitable_m24), help=f'Share of 5,000 simulated futures for {best.strategy}')
k2.metric('Median monthly profit, month 24', ui.lakh(best.median_profit_m24))
k3.metric('Median cash low point', ui.lakh(best.median_cash_low), help='Deepest cumulative cash position, setup costs included: the money needed to get through.')
thr = out['thresholds']; need = thr[(thr.cost_structure == 'Partner counters (2)') & (thr.price == p)].transfers_per_day_needed.iloc[0]
k4.metric(f'P2P transfers a day (need {need:.0f})', f'{best.p2p_per_day:.1f}', help='Same-flight matches at month 24 against the number needed for peer-to-peer to cover costs on its own')

left, right = st.columns([1, 1.3])
with left:
    st.subheader('Which launch strategy survives?')
    r = res.iloc[::-1]
    fig = go.Figure(go.Bar(x=r.profitable_m24, y=r.short, orientation='h',
                           marker_color=[ui.STRAT_COLORS[x] for x in r.code], text=[ui.pct(x) for x in r.profitable_m24], textposition='outside',
                           customdata=r.strategy, hovertemplate='%{customdata}<br>Profitable at month 24: %{x:.0%}<extra></extra>'))
    fig.update_layout(xaxis=dict(tickformat='.0%', range=[0, 1.15], title='Share of futures with a monthly profit in month 24'),
                      height=360, margin=dict(l=10, r=40, t=10, b=40))
    st.plotly_chart(fig, width='stretch')
with right:
    st.subheader('Monthly profit over 24 months')
    fan = out['fan']; fig = go.Figure()
    b = fan[fan.code == best.code]
    fig.add_trace(go.Scatter(x=list(b.month) + list(b.month[::-1]), y=list(b.p90 / 1e5) + list(b.p10[::-1] / 1e5), fill='toself',
                             fillcolor='rgba(58,154,91,0.15)', line=dict(width=0), hoverinfo='skip', name=f'{best.code} 10th-90th percentile'))
    for code, g in fan.groupby('code'):
        fig.add_trace(go.Scatter(x=g.month, y=g['median'] / 1e5, name=code, line=dict(color=ui.STRAT_COLORS[code], width=3 if code == best.code else 2),
                                 hovertemplate=f'{code} month %{{x}}: ₹%{{y:,.1f}} lakh<extra></extra>'))
    fig.add_hline(y=0, line_color='#444', line_width=1)
    fig.update_layout(yaxis_title='₹ lakh per month (median future)', xaxis_title='Month', height=360, margin=dict(l=10, r=10, t=10, b=10),
                      legend=dict(orientation='h', y=-0.22))
    st.plotly_chart(fig, width='stretch')

st.subheader('What each algorithm told us')
cls, clu = ui.classification('use'), ui.clustering(4)
rules, _ = ui.apriori(0.10, 0.60, 1.2)
reg = ui.regression('wtp_p2p_per_kg_inr')
prem = ui.feature_premium()
c1, c2 = st.columns(2)
with c1:
    top = cls['comparison'].iloc[0]
    imp = cls['importance']['Logistic regression'].sort_values('value')
    st.markdown(f"**🎯 Classification: who will use it?**  \n{ui.pct(d.use.mean(), 1)} would use KiloShare and {ui.pct(d.carry.mean(), 1)} would carry for others. "
                f"The best of five models, {top.model.lower()}, predicts use with AUC {top.cv_auc:.2f}. Use rises most with "
                f"**{imp.driver.iloc[-1].lower()}** and **{imp.driver.iloc[-2].lower()}**, and falls with **{imp.driver.iloc[0].lower()}**. "
                f"Targeting the top 30% by model score makes each buyer about {out['lift30']:.1f}x cheaper to reach.")
    prof = clu['profile']
    st.markdown(f"**🧩 Clustering: who are the customers?**  \nFour segments (silhouette {clu['silhouette']:.2f}). "
                + '; '.join(f"**{r.segment}** {ui.pct(r.would_use)} would use ({r.conversion.lower()} conversion)" for r in prof.itertuples())
                + '. Launch with the two high-conversion segments; recruit carriers among light travellers.')
with c2:
    st.markdown(f"**🛒 Association rules: which features go together?**  \n{len(rules)} strong rules. A trust core (matching, ID checks, chat, ratings) "
                f"is wanted together, and a logistics bundle (tracking, forwarding, repack) forms its own cluster; top rule **{rules.iloc[0].rule}** "
                f"(confidence {ui.pct(rules.iloc[0].confidence)}, lift {rules.iloc[0].lift:.2f}). Launch each bundle whole.")
    paid = prem[prem.significant & (prem.premium > 0)]
    st.markdown(f"**💰 Regression: what will people pay?**  \nWillingness to pay is well explained (R² {reg['comparison'].r2.iloc[0]:.2f}). "
                f"Best prices: **{ui.inr(p)}/kg** peer-to-peer with a **{ui.inr(c)}** carrier credit, and **{ui.inr(pf)}/kg** for forwarding. "
                f"Only {', '.join(paid.feature.str.split(' ').str[0:3].str.join(' ').str.lower())} raise willingness to pay, so they are the paid add-ons; everything else is free.")

with st.expander('How to read this, and what it assumes'):
    st.markdown(f"""
- **The survey data is simulated**, with deliberate noise and outliers, to test the method before real fieldwork. Swap in real responses (same columns) in `data/kiloshare_survey_raw.csv` and every page recalculates.
- **Traffic** comes from published 2025 figures: 5.5 million Delhi–Mumbai passengers, 39.2 million domestic passengers at Mumbai, 78.1 million passengers at Delhi.
- **Costs and uncertain ranges are assumptions.** Change them on the **Sustainability** page and the verdict updates.
- **Why peer-to-peer alone fails:** a request needs a willing carrier with spare room on the same flight. That happens for only about {ui.pct(best.match_rate)} of requests, so the counter handles about {best.p2p_per_day:.1f} transfers a day against {need:.0f} needed.
- **Why the hybrid works:** passengers who find no match can courier the extra kilos ({best.fwd_per_day:.0f} jobs a day at month 24) or use the repack station, and partner counters keep fixed costs low.
""")
