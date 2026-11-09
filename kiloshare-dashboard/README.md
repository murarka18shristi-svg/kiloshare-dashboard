# 🧳 KiloShare: is the business sustainable?

An interactive Streamlit dashboard testing a startup idea with four machine-learning algorithms and a break-even simulation.

**The idea.** KiloShare lets domestic air passengers in India who are over their 15 kg baggage allowance send their extra kilos in the spare allowance of a verified co-passenger on the same flight. Every transfer goes through a staffed KiloShare airport counter, which inspects the items, seals them in a tamper-evident pouch and logs both IDs. Launch is planned for tier-1 airports, starting with Delhi and Mumbai.

## What the dashboard does

| Page | What it shows |
|---|---|
| **Verdict** | Is it sustainable, under which launch strategy, and what each algorithm found |
| **Questionnaire** | The 25-question survey, with the algorithm each question feeds |
| **Data and Cleaning** | 5,000 simulated respondents, the noise and outliers planted on purpose, and how each was cleaned |
| **Classification** | Five models (logistic regression, decision tree, random forest, gradient boosting, k-NN) predicting who would use KiloShare and who would carry. Shows ROC curves, drivers, targeting lift, and scoring for a new traveller |
| **Clustering** | K-means with k chosen by silhouette and cross-checked with hierarchical clustering. Shows a segment map, profiles and high / medium / low conversion bands |
| **Association Rules** | Apriori on the 10 wanted features, with adjustable support, confidence and lift; which bundles to launch and which features to charge for |
| **Regression and Pricing** | Willingness to pay (peer-to-peer and forwarding) and carrier credit, linear vs random forest; price curves and a feature-premium test |
| **Sustainability** | Five launch strategies x 5,000 Monte Carlo futures over 24 months. Every assumption is adjustable in the sidebar |

## Run it

```bash
pip install -r requirements.txt
streamlit run Verdict.py
```

To regenerate the simulated data (fixed seed, identical output): `python scripts/generate_data.py`.
To use real survey responses, save them as `data/kiloshare_survey_raw.csv` with the same columns; every page recalculates.

## Project layout

```
Verdict.py                  home page: the sustainability verdict
pages/                      one page per step
kiloshare/questionnaire.py  the 25 questions and what each feeds
kiloshare/simulate.py       simulated respondents + planted noise and outliers
kiloshare/clean.py          documented cleaning rules
kiloshare/models.py         classification, clustering, association rules, regression
kiloshare/economics.py      break-even Monte Carlo simulation
data/                       raw survey data, noise log, true segments, questionnaire
```

## Notes

- **The survey data is simulated.** It demonstrates the method before real fieldwork, not real demand.
- **Traffic figures come from published 2025 data:**
  - 5.5 million Delhi–Mumbai passengers (IATA World Air Transport Statistics)
  - 39.2 million domestic passengers at Mumbai (Moodie Davitt Report)
  - 78.1 million passengers at Delhi (Network Thoughts)
  - IndiGo's ₹800/kg airport excess rate (Business Today, Sept 2026)
- **Costs and uncertainty ranges are assumptions.** They are listed on the Sustainability page and can be changed there.
- **Regulatory approval** of the inspected-and-sealed process is assumed.

Author: Shristi Murarka, Master of Global Business, SP Jain School of Global Management.
