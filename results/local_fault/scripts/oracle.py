"""In-domain oracle ceilings on the V6 data, same hold-outs as test.py.

UNSW (healthy vs crack): anchor = first half of H1, trained on the rest of H1 and crack
severity s1, tested on H2 (= S, healthy) and the other crack severity.

AGFD (healthy vs missing tooth / mild crack): query healthy gear held out, anchor = mean
of the other clean gears, logistic regression trained on AGFD labels except the query
gear and the query installation.
MCC5 (healthy / crack / missing tooth): trained on the torque run with crack severity
s1, tested on the speed run with the other severity.
Batch-averaged probabilities per (condition, unit), log10 selected bins as model input.
Usage (repo root): python results/local_fault/scripts/oracle.py
"""

import itertools
import sys
import warnings

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, "src/data")
from preprocessing_individual import get_important_frequencies  # noqa: E402

warnings.filterwarnings("ignore")


def clf():
    return make_pipeline(StandardScaler(), LogisticRegression(C=0.1, class_weight="balanced", max_iter=3000))


def agfd(sensor, clean):
    a = pd.read_feather("data/AGFD_V6.feather", columns=["speed", "load", "installation", "severity", "healthy_GP", "class", sensor])
    X = np.log10(np.stack(a[sensor].to_numpy()))[:, get_important_frequencies(15, 8, 5)]
    unit = np.where(a["class"] == "healthy", "H", np.where((a["class"] == "crack") & (a.severity == "L"), "MT",
                    np.where((a["class"] == "crack") & (a.severity == "S"), "CS", "other")))
    gp, inst = a.healthy_GP.to_numpy(), a.installation.to_numpy()
    cm = {c: ((a.speed == c[0]) & (a.load == c[1])).to_numpy() for c in itertools.product([500, 750, 1000, 1250, 1500], [1, 6, 11])}
    for fault in ["MT", "CS"]:
        hits = []
        for q, qi in itertools.product(clean, [1, 2, 3]):
            D = X.copy()
            for m in cm.values():
                D[m] -= np.mean([X[m & (unit == "H") & (gp == g)].mean(0) for g in clean if g != q], 0)
            tr = ((unit == "H") & (gp != q)) | ((unit == fault) & (inst != qi))
            model = clf().fit(D[tr], unit[tr] == fault)
            for m in cm.values():
                ph = model.predict_proba(D[m & (unit == "H") & (gp == q)])[:, 1].mean()
                pf = model.predict_proba(D[m & (unit == fault) & (inst == qi)])[:, 1].mean()
                hits.append([ph < 0.5, pf > 0.5])
        h = np.array(hits)
        print(f"AGFD {sensor} gears {clean} healthy/{fault}: {h.mean():.1%} (H {h[:, 0].mean():.0%}, fault {h[:, 1].mean():.0%})", flush=True)


def mcc5(sensor):
    m = pd.read_feather("data/MCC5_V6.feather", columns=["speed", "load", "severity", "class", "circulation", sensor])
    m["X"] = list(np.log10(np.stack(m[sensor].to_numpy()))[:, get_important_frequencies(36, 8, 5)])
    conds = [(s, l) for s in (1000, 2000, 3000) for l in (10, 20)]

    def unit(c, sev, circ, s, l):
        return np.stack(m[(m["class"] == c) & (m.severity == sev) & (m.circulation == circ) & (m.speed == s) & (m.load == l)].X.to_numpy())

    res = []
    for s1, s2 in [("M", "L"), ("L", "M")]:
        Xtr, ytr, tests = [], [], []
        for s, l in conds:
            th = unit("healthy", "-", "torque", s, l)
            anchor = th[:7].mean(0)
            for Xu, y in [(th[7:], 0), (unit("crack", s1, "torque", s, l), 1), (unit("missing_tooth", "S", "torque", s, l), 2)]:
                Xtr.append(Xu - anchor)
                ytr += [y] * len(Xu)
            tests.append([unit("healthy", "-", "speed", s, l) - anchor, unit("crack", s2, "speed", s, l) - anchor,
                          unit("missing_tooth", "S", "speed", s, l) - anchor])
        model = clf().fit(np.concatenate(Xtr), ytr)
        for qs in tests:
            res.append([np.argmax(model.predict_proba(q).mean(0)) == c for c, q in enumerate(qs)])
    r = np.array(res)
    print(f"MCC5 {sensor}: {r.mean():.1%} (H {r[:, 0].mean():.0%}, crack {r[:, 1].mean():.0%}, MT {r[:, 2].mean():.0%})", flush=True)


def unsw():
    u = pd.read_feather("data/UNSW_V6.feather")
    u = u[(u.OT_method == "signal") & u.speed.isin([5, 10, 15, 20]) & u.load.isin([5, 10, 15])]
    u["X"] = list(np.log10(np.stack(u.signal.to_numpy()))[:, get_important_frequencies(27, 8, 5)])
    res = []
    for s1, s2 in [("M", "L"), ("L", "M")]:
        Xtr, ytr, tests = [], [], []
        for _, g in u.groupby(["speed", "load"]):
            h1 = np.stack(g[g.severity == "H"].X.to_numpy())
            n = len(h1) // 2
            anchor = h1[:n].mean(0)
            for Xu, y in [(h1[n:], 0), (np.stack(g[g.severity == s1].X.to_numpy()), 1)]:
                Xtr.append(Xu - anchor)
                ytr += [y] * len(Xu)
            tests.append([np.stack(g[g.severity == "S"].X.to_numpy()) - anchor,
                          np.stack(g[g.severity == s2].X.to_numpy()) - anchor])
        model = clf().fit(np.concatenate(Xtr), ytr)
        for qs in tests:
            res.append([np.argmax(model.predict_proba(q).mean(0)) == c for c, q in enumerate(qs)])
    r = np.array(res)
    print(f"UNSW signal: {r.mean():.1%} (H {r[:, 0].mean():.0%}, crack {r[:, 1].mean():.0%})", flush=True)


unsw()
for sensor in ["gearbox_vibration_x", "gearbox_vibration_y", "gearbox_vibration_z"]:
    mcc5(sensor)
for sensor in ["acc4", "acc3"]:
    for clean in ([1, 2, 3, 4, 6, 7], [1, 2, 3, 4]):
        agfd(sensor, clean)
