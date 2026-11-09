"""The four algorithms: classification, clustering, association rules and regression."""
import itertools
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score, cross_val_predict
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, RandomForestRegressor
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve,
                             confusion_matrix, silhouette_score, adjusted_rand_score, r2_score, mean_absolute_error)
from .questionnaire import LIKERT_COLS, FEATURE_COLS, FEATURES

SEED = 42
INCOME = {'<25k': 0, '25k-50k': 1, '50k-1L': 2, '1L-2L': 3, '2L+': 4}
LABEL = {'age': 'Age', 'income_level': 'Income band', 'domestic_flights_12m': 'Flights per year', 'avg_checked_bag_kg': 'Checked bag weight',
         'times_exceeded_allowance_12m': 'Times over allowance', 'avg_excess_kg': 'Average excess kg', 'excess_fees_paid_12m_inr': 'Excess fees paid',
         'possessiveness_luggage': 'Luggage possessiveness', 'trust_strangers': 'Trust in verified strangers', 'security_concern': 'Security concern',
         'price_sensitivity': 'Price sensitivity', 'tech_comfort': 'Comfort with travel apps', 'counter_trust_boost': 'Counter boosts trust',
         'premium_fare': 'Books premium fares'}
PREDICTORS = ['age', 'income_level', 'domestic_flights_12m', 'avg_checked_bag_kg', 'times_exceeded_allowance_12m', 'avg_excess_kg',
              'excess_fees_paid_12m_inr'] + LIKERT_COLS
CLUSTER_INPUTS = ['age', 'domestic_flights_12m', 'avg_checked_bag_kg', 'times_exceeded_allowance_12m'] + LIKERT_COLS
CODE = {c: c.split('_')[0] for c in FEATURE_COLS}
FNAME = {c.split('_')[0]: v for c, v in FEATURES.items()}


def prepare(d):
    d = d.copy()
    d['income_level'] = d.monthly_income_band.map(INCOME).astype(int)
    d['premium_fare'] = (d.usual_fare_type == 'Business/premium').astype(int)
    d['use'] = (d.would_use_app == 'Yes').astype(int)
    d['carry'] = (d.would_carry == 'Yes').astype(int)
    return d


# ------------------------------------------------------------------ classification
def classifiers():
    return {
        'Logistic regression': make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000)),
        'Decision tree': DecisionTreeClassifier(max_depth=4, min_samples_leaf=100, random_state=SEED),
        'Random forest': RandomForestClassifier(n_estimators=200, max_depth=8, random_state=SEED, n_jobs=-1),
        'Gradient boosting': GradientBoostingClassifier(n_estimators=120, max_depth=3, learning_rate=0.06, random_state=SEED),
        'K-nearest neighbours': make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=25)),
    }


def classification(d, target='use'):
    X, y = d[PREDICTORS], d[target]
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=SEED, stratify=y)
    cv = StratifiedKFold(5, shuffle=True, random_state=SEED)
    rows, rocs, cms, imps = [], {}, {}, {}
    for name, m in classifiers().items():
        m.fit(Xtr, ytr); p = m.predict_proba(Xte)[:, 1]; yhat = (p >= 0.5).astype(int)
        cvauc = cross_val_score(m, X, y, cv=cv, scoring='roc_auc')
        rows.append(dict(model=name, accuracy=accuracy_score(yte, yhat), precision=precision_score(yte, yhat), recall=recall_score(yte, yhat),
                         f1=f1_score(yte, yhat), test_auc=roc_auc_score(yte, p), cv_auc=cvauc.mean(), cv_auc_sd=cvauc.std()))
        fpr, tpr, _ = roc_curve(yte, p); rocs[name] = pd.DataFrame({'fpr': fpr, 'tpr': tpr})
        cms[name] = confusion_matrix(yte, yhat)
        if name == 'Logistic regression':
            coef = m.named_steps['logisticregression'].coef_[0]
            imps[name] = pd.DataFrame({'driver': [LABEL[c] for c in PREDICTORS], 'value': coef, 'odds_ratio': np.exp(coef)})
        elif hasattr(m, 'feature_importances_'):
            imps[name] = pd.DataFrame({'driver': [LABEL[c] for c in PREDICTORS], 'value': m.feature_importances_})
    comp = pd.DataFrame(rows).sort_values('cv_auc', ascending=False).reset_index(drop=True)
    # out-of-fold scores for every respondent from the logistic model (used for targeting)
    oof = cross_val_predict(classifiers()['Logistic regression'], X, y, cv=cv, method='predict_proba')[:, 1]
    tree = DecisionTreeClassifier(max_depth=3, min_samples_leaf=150, random_state=SEED).fit(X, y)
    return dict(comparison=comp, roc=rocs, confusion=cms, importance=imps, oof=oof, tree_rules=tree_rules(tree, X, y), test_n=len(yte))


def tree_rules(tree, X, y):
    t = tree.tree_; leaf = tree.apply(X); out = []
    def walk(node, conds):
        if t.children_left[node] == -1:
            m = leaf == node
            out.append(dict(rule=' AND '.join(conds), respondents=int(m.sum()), share_yes=float(y[m].mean()))); return
        f, th = PREDICTORS[t.feature[node]], t.threshold[node]
        walk(t.children_left[node], conds + [f'{LABEL[f]} <= {th:,.1f}'])
        walk(t.children_right[node], conds + [f'{LABEL[f]} > {th:,.1f}'])
    walk(0, [])
    return pd.DataFrame(out).sort_values('share_yes', ascending=False).reset_index(drop=True)


def scoring_model(d, target='use'):
    return classifiers()['Logistic regression'].fit(d[PREDICTORS], d[target])


def lift_table(d, score):
    t = d.assign(score=score)
    t['decile'] = pd.qcut(t.score.rank(method='first', ascending=False), 10, labels=range(1, 11)).astype(int)
    g = t.groupby('decile').agg(respondents=('use', 'size'), would_use=('use', 'mean')).reset_index()
    g['lift'] = g.would_use / t.use.mean()
    g['cumulative_share_of_users'] = (t.groupby('decile').use.sum().cumsum() / t.use.sum()).values
    return g


# ------------------------------------------------------------------ clustering
def k_selection(d, kmin=2, kmax=8):
    Z = StandardScaler().fit_transform(d[CLUSTER_INPUTS]); out = []
    for k in range(kmin, kmax + 1):
        km = KMeans(k, n_init=10, random_state=0).fit(Z)
        out.append(dict(k=k, silhouette=silhouette_score(Z, km.labels_, sample_size=3000, random_state=0), inertia=km.inertia_))
    return pd.DataFrame(out)


def name_segments(prof):
    names = {}
    if len(prof) == 4:
        names[prof.flights.idxmax()] = 'Frequent overpackers'
        names[prof.bag_kg.idxmin()] = 'Light travellers & sceptics'
        rest = [c for c in prof.index if c not in names]
        if len(rest) == 2:
            young = min(rest, key=lambda c: prof.loc[c, 'age']); names[young] = 'Budget students & movers'
            names[[c for c in rest if c != young][0]] = 'Family visitors'
            return names
    z = (prof[['age', 'flights', 'bag_kg', 'times_over', 'trust', 'price_sens', 'possessive']] - prof[['age', 'flights', 'bag_kg', 'times_over', 'trust', 'price_sens', 'possessive']].mean())
    z = z / z.std().replace(0, 1)
    words = {'age': ('Younger', 'Older'), 'flights': ('Rare flyers', 'Frequent flyers'), 'bag_kg': ('Light packers', 'Heavy packers'),
             'times_over': ('Rarely over', 'Often over'), 'trust': ('Wary', 'Trusting'), 'price_sens': ('Spend freely', 'Price-sensitive'),
             'possessive': ('Relaxed', 'Possessive')}
    for c in prof.index:
        top = z.loc[c].abs().sort_values(ascending=False).index[:2]
        names[c] = ', '.join(words[t][int(z.loc[c, t] > 0)] for t in top)
    return names


def clustering(d, k=4):
    Z = StandardScaler().fit_transform(d[CLUSTER_INPUTS])
    km = KMeans(k, n_init=10, random_state=0).fit(Z)
    ward = AgglomerativeClustering(n_clusters=k, linkage='ward').fit_predict(Z)
    pcs = PCA(2, random_state=0).fit(Z); xy = pcs.transform(Z)
    t = d.assign(cluster=km.labels_); g = t.groupby('cluster')
    prof = pd.DataFrame({'respondents': g.size(), 'age': g.age.mean(), 'flights': g.domestic_flights_12m.mean(), 'bag_kg': g.avg_checked_bag_kg.mean(),
                         'times_over': g.times_exceeded_allowance_12m.mean(), 'possessive': g.possessiveness_luggage.mean(), 'trust': g.trust_strangers.mean(),
                         'security': g.security_concern.mean(), 'price_sens': g.price_sensitivity.mean(), 'tech': g.tech_comfort.mean(),
                         'counter': g.counter_trust_boost.mean(), 'would_use': g.use.mean(), 'intent': g.usage_intent_6m.mean(), 'would_carry': g.carry.mean(),
                         'wtp_p2p_median': g.wtp_p2p_per_kg_inr.median(), 'wtp_fwd_median': g.wtp_forwarding_per_kg_inr.median(),
                         'credit_ask_median': g.min_credit_per_kg_inr.median(), 'top_occupation': g.occupation.agg(lambda s: s.mode()[0])})
    names = name_segments(prof)
    prof.insert(0, 'segment', prof.index.map(names))
    prof['conversion'] = np.where(prof.intent >= 3.5, 'High', np.where(prof.intent >= 2.5, 'Medium', 'Low'))
    prof = prof.sort_values('intent', ascending=False)
    seg = pd.Series(km.labels_).map(names).values
    return dict(profile=prof.reset_index(drop=True), segment=seg, pca=pd.DataFrame({'pc1': xy[:, 0], 'pc2': xy[:, 1], 'segment': seg}),
                explained=pcs.explained_variance_ratio_.sum(), ari_ward=adjusted_rand_score(km.labels_, ward),
                silhouette=silhouette_score(Z, km.labels_, sample_size=3000, random_state=0))


# ------------------------------------------------------------------ association rules (Apriori)
def apriori(d, min_support=0.10, min_confidence=0.60, min_lift=1.20, max_len=3):
    B = d[FEATURE_COLS].astype(bool)
    support, level = {}, [(c,) for c in FEATURE_COLS]
    while level:                                       # keep growing only frequent itemsets (the Apriori property)
        freq = {}
        for items in level:
            s = B[list(items)].all(axis=1).mean()
            if s >= min_support: freq[items] = s
        support.update(freq)
        keys = sorted(freq)
        level = sorted({tuple(sorted(set(a) | set(b))) for a, b in itertools.combinations(keys, 2)
                        if len(set(a) | set(b)) == len(a) + 1 and len(a) + 1 <= max_len})
    rules = []
    for items, s in support.items():
        if len(items) < 2: continue
        for cons in items:
            ante = tuple(x for x in items if x != cons)
            conf = s / support[ante]; lift = conf / support[(cons,)]
            if conf >= min_confidence and lift >= min_lift:
                rules.append(dict(rule=' + '.join(CODE[a] for a in ante) + ' -> ' + CODE[cons], antecedent=' + '.join(CODE[a] for a in ante),
                                  consequent=CODE[cons], support=s, confidence=conf, lift=lift))
    rules = pd.DataFrame(rules, columns=['rule', 'antecedent', 'consequent', 'support', 'confidence', 'lift'])
    itemsets = pd.DataFrame([dict(itemset=' + '.join(CODE[x] for x in k), size=len(k), support=v) for k, v in support.items()])
    return rules.sort_values(['lift', 'confidence'], ascending=False).reset_index(drop=True), itemsets


def feature_demand(d):
    B = d[FEATURE_COLS].astype(bool)
    return pd.DataFrame({'code': [CODE[c] for c in FEATURE_COLS], 'feature': [FEATURES[c] for c in FEATURE_COLS],
                         'all_respondents': B.mean().values, 'would_use': B[d.use == 1].mean().values}).sort_values('all_respondents', ascending=False)


# ------------------------------------------------------------------ regression
REG_TARGETS = {'wtp_p2p_per_kg_inr': ('Q21 Max price per kg, peer-to-peer', ['income_level', 'premium_fare', 'times_exceeded_allowance_12m', 'avg_excess_kg', 'excess_fees_paid_12m_inr'] + LIKERT_COLS, 'all'),
               'wtp_forwarding_per_kg_inr': ('Q22 Max price per kg, courier forwarding', ['income_level', 'premium_fare', 'times_exceeded_allowance_12m', 'avg_excess_kg', 'excess_fees_paid_12m_inr'] + LIKERT_COLS, 'all'),
               'min_credit_per_kg_inr': ('Q24 Minimum credit per kg to carry', ['income_level', 'avg_checked_bag_kg', 'domestic_flights_12m'] + LIKERT_COLS, 'carriers')}


def regression(d, target):
    title, X_cols, pop = REG_TARGETS[target]
    t = d[d.carry == 1] if pop == 'carriers' else d
    Xa, Xb, ya, yb = train_test_split(t[X_cols], t[target], test_size=0.3, random_state=SEED)
    sc = StandardScaler().fit(Xa)
    lin = LinearRegression().fit(Xa, ya); lin_s = LinearRegression().fit(sc.transform(Xa), ya)
    rf = RandomForestRegressor(n_estimators=200, max_depth=8, random_state=SEED, n_jobs=-1).fit(Xa, ya)
    pl, pr = lin.predict(Xb), rf.predict(Xb)
    comp = pd.DataFrame([dict(model='Linear regression', r2=r2_score(yb, pl), mae=mean_absolute_error(yb, pl)),
                         dict(model='Random forest', r2=r2_score(yb, pr), mae=mean_absolute_error(yb, pr))])
    coefs = pd.DataFrame({'driver': [LABEL[c] for c in X_cols], 'per_unit': lin.coef_, 'per_sd': lin_s.coef_}).sort_values('per_sd', key=abs, ascending=False)
    return dict(title=title, n=len(t), comparison=comp, coefs=coefs, actual_pred=pd.DataFrame({'actual': yb.values, 'predicted': pl}))


def feature_premium(d, target='wtp_p2p_per_kg_inr'):
    """OLS of WTP on the usual drivers plus the ten feature flags: does wanting a feature raise what people pay?"""
    _, X_cols, _ = REG_TARGETS[target]
    u = d[d.use == 1]; cols = X_cols + FEATURE_COLS
    Xm = np.column_stack([np.ones(len(u)), u[cols].values.astype(float)]); yv = u[target].values.astype(float)
    beta, *_ = np.linalg.lstsq(Xm, yv, rcond=None); res = yv - Xm @ beta
    s2 = res @ res / (len(yv) - Xm.shape[1]); se = np.sqrt(np.diag(s2 * np.linalg.inv(Xm.T @ Xm)))
    k = len(X_cols) + 1
    out = pd.DataFrame({'code': [CODE[c] for c in FEATURE_COLS], 'feature': [FEATURES[c] for c in FEATURE_COLS],
                        'premium': beta[k:], 'low': beta[k:] - 1.96 * se[k:], 'high': beta[k:] + 1.96 * se[k:]})
    out['significant'] = (out.low > 0) | (out.high < 0)
    return out


def price_curves(d):
    users, carriers = d[d.use == 1], d[d.carry == 1]
    return pd.DataFrame([dict(price=p, p2p_demand=(users.wtp_p2p_per_kg_inr >= p).mean(), fwd_demand=(d.wtp_forwarding_per_kg_inr >= p).mean(),
                              carrier_supply=(carriers.min_credit_per_kg_inr <= p).mean()) for p in range(50, 801, 25)])


THEMES = {'Legal or security risk': r'illegal|security|caught|drugs|banned|legal|responsible', 'Price or charges': r'price|charges|saves money|commission',
          'Possessiveness or damage': r'touching|personal space|possessive|mixing|damaged', 'Trust in strangers': r'genuine|trust|fake|give back|turn up',
          'No concern': r'^(?:nothing|na|no concerns|-|seems fine|none really)?$'}


def concern_themes(d):
    t = d.biggest_concern_text.fillna('').str.strip().str.lower()
    out = pd.DataFrame([dict(theme=k, respondents=int(t.str.contains(v, regex=True).sum())) for k, v in THEMES.items()])
    out['share'] = out.respondents / len(d)
    return out.sort_values('respondents', ascending=False)
