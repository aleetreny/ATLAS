"""Data for the ATLAS 'Bought Together' web article, module 2.4.

Every other article in this module works on points: rows of numbers with
distances between them and a middle somewhere. This one does not. A basket is a
SET, two baskets have no distance, and there is nothing to average. The only
operation available is counting how often things happen together, and that is
enough for a whole branch of the map.

Counting is trivial. The problem is that there are 2^d subsets to count for d
distinct items, and d is 169 here, so the number of candidate patterns is a 51
digit integer. Two algorithms make it feasible with two completely different
tricks, and both are measured rather than described:

  * APRIORI prunes. If a set is frequent then every subset of it is frequent,
    so a set with one infrequent subset can be discarded without ever being
    counted. The article counts exactly how many that is.
  * FP-GROWTH compresses. It puts the database into a prefix tree with the most
    common items nearest the root, and then mines the tree instead of the
    database. The article measures how much that ordering buys by building the
    same tree three ways.

Results worth knowing before reading the code:

  * on the twenty most common items, all 1,048,575 non-empty subsets are counted
    exhaustively here, so the pruning is checked against the truth rather than
    trusted: downward closure holds in every one of them, and Apriori touches a
    tiny fraction of the lattice.
  * the FP-tree's item ordering is the algorithm. By frequency it is the
    smallest; by the reverse it is much larger, and the article prints both.
  * confidence is a trap, and the trap is measurable: the highest confidence
    rule whose lift is below one is a rule that makes its own conclusion LESS
    likely while looking like a strong rule.
  * a permutation test over the same thresholds says how many of the rules found
    would have been found in data with the associations destroyed and the
    marginals kept. That number is not zero.

No wall clock anywhere: the cost section counts candidates, database passes and
tree nodes.

Run from anywhere: `python src/utils/generate_baskets_data.py`.
"""
import base64
import itertools
import json
import math
from collections import Counter
from pathlib import Path

import numpy as np
import scipy
import sklearn
from sklearn.datasets import load_wine

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from basket_data import groceries  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "baskets" / "data"
OUT.mkdir(parents=True, exist_ok=True)
rnd = lambda v, n=4: round(float(v), n)
SEED = 19

print("=" * 70)
print("Bought Together: counting is easy, the subsets are not")
print("=" * 70)

# ================================================================== the baskets
TX, DIGEST = groceries()
N = len(TX)
counts = Counter(i for t in TX for i in t)
ITEMS = [i for i, _ in counts.most_common()]
SLOTS = sum(len(t) for t in TX)
print(f"\n{N:,} transactions, {len(ITEMS)} item categories, {SLOTS:,} item slots")
print(f"  the average basket holds {SLOTS / N:.3f} items and the largest holds "
      f"{max(len(t) for t in TX)}")
print(f"  the five most common: "
      + ", ".join(f"{i} ({counts[i] / N:.2%})" for i in ITEMS[:5]))
print(f"  sha256 of the file: {DIGEST[:16]}")

# A vertical layout: one bitset per item, bit t set when transaction t holds it.
# Support of any set is then one AND and one population count, which is what
# makes the exhaustive enumeration below affordable and what the browser will
# use as well.
MASK = {}
for t, basket in enumerate(TX):
    for i in set(basket):
        MASK[i] = MASK.get(i, 0) | (1 << t)
support = lambda mask: mask.bit_count()


def sup_of(names):
    m = (1 << N) - 1
    for name in names:
        m &= MASK[name]
    return m.bit_count()


# ---- the size of the question
total_subsets = 2 ** len(ITEMS) - 1
print(f"\n  subsets of {len(ITEMS)} items: {total_subsets:,}")
print(f"  which is {len(str(total_subsets))} digits, so nothing counts them one at a time")

# ======================================================= exhaustive, on twenty
# Twenty items is 1,048,575 non-empty subsets, which IS countable, so the
# pruning below can be checked against the truth instead of trusted.
TOP = 20
top_items = ITEMS[:TOP]
top_masks = [MASK[i] for i in top_items]
FULL = (1 << N) - 1

exhaustive = {}


def enumerate_all():
    """Depth first over every subset of the top items, carrying the intersection
    down so that each node costs one AND and one population count."""
    out = {}
    stack = [(0, FULL, ())]
    while stack:
        start, mask, chosen = stack.pop()
        for j in range(start, TOP):
            m = mask & top_masks[j]
            key = chosen + (j,)
            out[key] = m.bit_count()
            stack.append((j + 1, m, key))
    return out


exhaustive = enumerate_all()
print(f"\n  every one of the {len(exhaustive):,} non-empty subsets of the {TOP} most common "
      f"items counted exactly")
by_size = Counter(len(k) for k in exhaustive)
print("  by size: " + ", ".join(f"{s}: {by_size[s]:,}" for s in sorted(by_size)))

# ================================================================== apriori
def apriori(masks, names, min_count, max_len=None, count_work=True):
    """Levelwise, with the prune step written out so that it can be counted.

    Returns the frequent sets by level and a tally of the work: how many
    candidates were generated by joining, how many survived the subset check
    without being counted at all, and how many were actually counted against
    the database."""
    d = len(masks)
    work = {"generated": 0, "pruned": 0, "counted": 0, "frequent": 0, "passes": 0}
    levels = []
    freq = {}
    level = []
    for j in range(d):
        work["generated"] += 1
        work["counted"] += 1
        c = masks[j].bit_count()
        if c >= min_count:
            level.append(((j,), masks[j], c))
    work["passes"] += 1
    while level:
        work["frequent"] += len(level)
        levels.append([(tuple(names[q] for q in k), c) for k, _, c in level])
        freq.update({k: c for k, _, c in level})
        if max_len and len(level[0][0]) >= max_len:
            break
        nxt = []
        for a in range(len(level)):
            for b in range(a + 1, len(level)):
                ka, kb = level[a][0], level[b][0]
                if ka[:-1] != kb[:-1]:
                    continue                       # the standard join on a shared prefix
                cand = ka + (kb[-1],)
                work["generated"] += 1
                # downward closure: every subset of size k-1 has to be frequent
                ok = all(tuple(s) in freq for s in itertools.combinations(cand, len(cand) - 1))
                if not ok:
                    work["pruned"] += 1
                    continue
                work["counted"] += 1
                m = level[a][1] & masks[cand[-1]]
                c = m.bit_count()
                if c >= min_count:
                    nxt.append((cand, m, c))
        work["passes"] += 1
        level = nxt
    return levels, work, freq


MIN_SUP = 0.01                     # one basket in a hundred, the usual starting point
min_count = math.ceil(MIN_SUP * N)
levels, work, freq_all = apriori([MASK[i] for i in ITEMS], ITEMS, min_count)
print(f"\napriori on all {len(ITEMS)} items at support {MIN_SUP:.0%} ({min_count} baskets):")
for L, sets in enumerate(levels, start=1):
    print(f"  size {L}: {len(sets):,} frequent")
print(f"  candidates generated {work['generated']:,}, of which {work['pruned']:,} were thrown "
      f"away by the subset check without being counted")
print(f"  so {work['counted']:,} were counted against the database, in "
      f"{work['passes']} passes")
longest = len(levels)
reachable = sum(math.comb(len(ITEMS), k) for k in range(1, longest + 1))
print(f"  the lattice up to size {longest} holds {reachable:,} sets; apriori counted "
      f"{work['counted'] / reachable:.3e} of it")

# ---- and the same thing on the twenty, where the truth is known
lev20, work20, freq20 = apriori(top_masks, top_items, min_count)
truth20 = {k: v for k, v in exhaustive.items() if v >= min_count}
found20 = set()
for sets in lev20:
    for names, _ in sets:
        found20.add(tuple(sorted(top_items.index(x) for x in names)))
same = found20 == set(tuple(sorted(k)) for k in truth20)
print(f"\n  on the {TOP} most common items, where all {len(exhaustive):,} subsets were counted:")
print(f"    frequent sets by exhaustion {len(truth20):,}, by apriori {len(found20):,}: "
      f"{'the same sets' if same else 'THEY DIFFER'}")
print(f"    apriori counted {work20['counted']:,} of the {len(exhaustive):,}, which is "
      f"{work20['counted'] / len(exhaustive):.2%}")

# downward closure, verified on every frequent set rather than assumed
violations = 0
for k in truth20:
    for r in range(1, len(k)):
        for s in itertools.combinations(k, r):
            if exhaustive[s] < exhaustive[k]:
                violations += 1
print(f"    every subset of a frequent set is at least as frequent: "
      f"{'holds in all of them' if violations == 0 else f'{violations} violations'}")

# ---- the support sweep, which is the parameter that decides everything
sweep = []
for s in [0.05, 0.03, 0.02, 0.015, 0.01, 0.0075, 0.005, 0.003, 0.002, 0.001]:
    mc = math.ceil(s * N)
    lv, wk, _ = apriori([MASK[i] for i in ITEMS], ITEMS, mc, max_len=4)
    sweep.append({"support": s, "count": mc, "frequent": wk["frequent"],
                  "generated": wk["generated"], "pruned": wk["pruned"],
                  "counted": wk["counted"], "levels": len(lv),
                  "sizes": [len(x) for x in lv]})
print("\n  the support threshold, swept (capped at size 4 so the sweep terminates):")
for r in sweep:
    print(f"    {r['support']:.4f} ({r['count']:>3} baskets): {r['frequent']:>6,} frequent, "
          f"{r['counted']:>8,} counted, {r['pruned']:>8,} pruned")

# ==================================================================== fp-growth
def fp_tree(transactions, order):
    """Insert every transaction into a prefix tree, its items sorted by `order`.

    Returns the node count and the tree itself. The node count is the whole
    point: it is what the compression buys, and it depends entirely on the
    order."""
    rank = {name: i for i, name in enumerate(order)}
    root = {"item": None, "count": 0, "children": {}}
    nodes = 1
    for t in transactions:
        items = sorted((i for i in set(t) if i in rank), key=lambda i: rank[i])
        node = root
        for i in items:
            if i not in node["children"]:
                node["children"][i] = {"item": i, "count": 0, "children": {}}
                nodes += 1
            node = node["children"][i]
            node["count"] += 1
    return nodes, root


frequent1 = [i for i in ITEMS if counts[i] >= min_count]
orders = {
    "by frequency, commonest first": frequent1,
    "by frequency, rarest first": list(reversed(frequent1)),
    "alphabetical": sorted(frequent1),
}
rng = np.random.default_rng(SEED)
shuffled = list(frequent1)
rng.shuffle(shuffled)
orders["a random order"] = shuffled
tree_rows = []
for name, order in orders.items():
    nodes, _ = fp_tree(TX, order)
    slots = sum(len([i for i in set(t) if i in set(order)]) for t in TX)
    tree_rows.append({"order": name, "nodes": nodes, "slots": slots,
                      "ratio": rnd(slots / nodes, 3)})
print(f"\nthe FP-tree over the {len(frequent1)} items frequent at {MIN_SUP:.0%}:")
for r in tree_rows:
    print(f"  {r['order']:<32} {r['nodes']:>7,} nodes for {r['slots']:,} item slots "
          f"({r['ratio']:.3f} slots per node)")
best_tree = min(tree_rows, key=lambda r: r["nodes"])
worst_tree = max(tree_rows, key=lambda r: r["nodes"])
print(f"  the algorithm's own order is {'the smallest' if best_tree['order'].startswith('by frequency, commonest') else 'NOT the smallest'}"
      f", and the worst is {worst_tree['nodes'] / best_tree['nodes']:.3f} times it")

# ======================================================================= rules
def rules_from(freq, n, names, min_conf=0.0, min_lift=None):
    """`freq` is keyed by tuples of item INDICES, which is what apriori works in;
    `names` translates them on the way out, because a rule the reader cannot
    read is not a rule."""
    out = []
    for k, c in freq.items():
        if len(k) < 2:
            continue
        for r in range(1, len(k)):
            for ante in itertools.combinations(k, r):
                cons = tuple(x for x in k if x not in ante)
                sa = freq[ante]
                sc = freq[cons] if cons in freq else None
                if sc is None:
                    continue
                conf = c / sa
                lift = conf / (sc / n)
                if conf < min_conf:
                    continue
                if min_lift is not None and lift < min_lift:
                    continue
                out.append({
                    "a": [names[x] for x in ante], "c": [names[x] for x in cons],
                    "support": c / n, "confidence": conf, "lift": lift,
                    "leverage": c / n - (sa / n) * (sc / n),
                    "conviction": (1 - sc / n) / (1 - conf) if conf < 1 else float("inf"),
                    "count": c,
                })
    return out


MIN_CONF = 0.2
all_rules = rules_from(freq_all, N, ITEMS, MIN_CONF)
print(f"\n{len(all_rules):,} rules at support {MIN_SUP:.0%} and confidence {MIN_CONF:.0%}")
below_one = [r for r in all_rules if r["lift"] < 1]
trap = max(below_one, key=lambda r: r["confidence"]) if below_one else None
print(f"  {len(below_one):,} of them have lift below one, which means the rule makes its own "
      f"conclusion LESS likely")
if trap:
    print(f"  the worst of those: {{{', '.join(trap['a'])}}} => {{{', '.join(trap['c'])}}} at "
          f"confidence {trap['confidence']:.4f} and lift {trap['lift']:.4f}")
    print(f"    because {trap['c'][0]} is in {counts[trap['c'][0]] / N:.2%} of all baskets "
          f"anyway, so the rule is BELOW that")
# the one pair the opening paragraph names, counted rather than guessed
pair_names = ITEMS[:2]
pair_count = sup_of(pair_names)
print(f"  {pair_names[0]} and {pair_names[1]} share a basket {pair_count:,} times "
      f"({pair_count / N:.2%})")

best_conf = max(all_rules, key=lambda r: r["confidence"])
best_lift = max(all_rules, key=lambda r: r["lift"])
print(f"  the highest confidence rule is {{{', '.join(best_conf['a'])}}} => "
      f"{{{', '.join(best_conf['c'])}}} at {best_conf['confidence']:.4f}, lift "
      f"{best_conf['lift']:.4f}")
print(f"  the highest lift rule is {{{', '.join(best_lift['a'])}}} => "
      f"{{{', '.join(best_lift['c'])}}} at lift {best_lift['lift']:.4f}, confidence "
      f"{best_lift['confidence']:.4f}")

# lift is symmetric and confidence is not: measured on the same rules
pairs = 0
worst_asym = None
index = {(tuple(sorted(r["a"])), tuple(sorted(r["c"]))): r for r in all_rules}
for key, r in index.items():
    back = index.get((key[1], key[0]))
    if back is None:
        continue
    pairs += 1
    gap = abs(r["confidence"] - back["confidence"])
    if worst_asym is None or gap > worst_asym[0]:
        worst_asym = (gap, r, back)
lift_gap = max(abs(index[(a, c)]["lift"] - index[(c, a)]["lift"])
               for (a, c) in index if (c, a) in index) if pairs else 0.0
print(f"  of the {len(all_rules):,} rules, {pairs:,} have their reverse in the list too; across "
      f"those pairs the two lifts differ by at most {lift_gap:.2e} and the two confidences by up "
      f"to {worst_asym[0]:.4f}" if pairs else "  no reversible pairs")

# ---- the permutation test
# Shuffle each item's column independently. Every item keeps exactly the support
# it had; every association between items is destroyed. Anything found after
# that is the search itself talking.
PERMS = 20
perm_counts = []
g = np.random.default_rng(SEED)
for p in range(PERMS):
    perm_masks = []
    for i in ITEMS:
        c = counts[i]
        idx = g.permutation(N)[:c]
        m = 0
        for t in idx:
            m |= 1 << int(t)
        perm_masks.append(m)
    lv, wk, fr = apriori(perm_masks, ITEMS, min_count, max_len=4)
    rl = rules_from(fr, N, ITEMS, MIN_CONF)
    perm_counts.append({"frequent": wk["frequent"], "rules": len(rl),
                        "pairs": len(lv[1]) if len(lv) > 1 else 0})
real_capped, real_work, real_freq = apriori([MASK[i] for i in ITEMS], ITEMS, min_count, max_len=4)
real_rules = rules_from(real_freq, N, ITEMS, MIN_CONF)
perm_freq = [r["frequent"] for r in perm_counts]
perm_rules = [r["rules"] for r in perm_counts]
print(f"\n  a permutation test, {PERMS} shuffles that keep every item's support and destroy "
      f"every association:")
print(f"    frequent sets: {real_work['frequent']:,} in the real data, "
      f"{np.mean(perm_freq):.1f} plus or minus {np.std(perm_freq):.1f} in the shuffles "
      f"(worst {max(perm_freq)})")
print(f"    rules:         {len(real_rules):,} in the real data, "
      f"{np.mean(perm_rules):.1f} plus or minus {np.std(perm_rules):.1f} in the shuffles "
      f"(worst {max(perm_rules)})")

# ============================================================= the browser's copy
# Vertical bitsets for the items the page will let the reader play with, so it
# can compute the support of any subset of them exactly. One AND and one
# population count per subset, the same as here.
SHIP = 24
ship_items = ITEMS[:SHIP]
nbytes = (N + 7) // 8


def to_b64(mask):
    return base64.b64encode(mask.to_bytes(nbytes, "little")).decode("ascii")


bitsets = {i: to_b64(MASK[i]) for i in ship_items}

# The widget mines these 24 items live, so the guard needs the same numbers from
# here. Its figures are NOT comparable with the tables above, which use all 169
# items: two protocols, and the page says so where it prints them.
SHIP_SUPPORTS = [0.05, 0.03, 0.02, 0.015, 0.01, 0.0075, 0.005]
ship_ref = []
ship_masks = [MASK[i] for i in ship_items]
for s_ in SHIP_SUPPORTS:
    mc = math.ceil(s_ * N)
    lv, wk, fr = apriori(ship_masks, ship_items, mc)
    rl = rules_from(fr, N, ship_items, MIN_CONF)
    ship_ref.append({"support": s_, "count": mc, "work": wk,
                     "levels": [len(x) for x in lv], "rules": len(rl),
                     "below_one": sum(1 for r in rl if r["lift"] < 1)})
print(f"  and its apriori tallies at {len(SHIP_SUPPORTS)} thresholds, for the guard")
for r in ship_ref:
    print(f"    {r['support']:.4f}: {r['work']['frequent']:>4} frequent, "
          f"{r['work']['counted']:>5} counted, {r['work']['pruned']:>5} pruned, "
          f"{r['rules']:>4} rules")
print(f"\n  shipping vertical bitsets for the {SHIP} most common items: "
      f"{SHIP * nbytes / 1024:.1f} kB before base64")

# ================================================================ the wine again
# The same 178 bottles the previous article binned to terciles for k-modes, as
# baskets: thirteen items each, "alcohol is high" and so on. Dense where the
# groceries are sparse, and small enough that everything can be checked.
wine = load_wine()
codes = np.zeros_like(wine.data, dtype=int)
for j in range(wine.data.shape[1]):
    qs = np.quantile(wine.data[:, j], [1 / 3, 2 / 3])
    codes[:, j] = np.digitize(wine.data[:, j], qs)
shared = None
prev = ROOT / "assumptions" / "data" / "relaxations.json"
if prev.exists():
    ref = np.array(json.loads(prev.read_text(encoding="utf-8"))["modes"]["codes"])
    shared = {"gap": int(np.abs(codes - ref).max()), "rows": int(ref.shape[0])}
    print(f"\nthe wine binned to terciles reproduces the codes `assumptions/` ships "
          f"{'exactly' if shared['gap'] == 0 else f'to {shared[chr(39)]}'}")
BINS = ["low", "middle", "high"]
wine_tx = [[f"{wine.feature_names[j]} {BINS[codes[r, j]]}" for j in range(codes.shape[1])]
           for r in range(codes.shape[0])]
wine_items = sorted({i for t in wine_tx for i in t})
wine_mask = {}
for t, basket in enumerate(wine_tx):
    for i in basket:
        wine_mask[i] = wine_mask.get(i, 0) | (1 << t)
WN = len(wine_tx)
wine_min = math.ceil(0.2 * WN)
wine_lev, wine_work, wine_freq = apriori([wine_mask[i] for i in wine_items], wine_items,
                                         wine_min, max_len=4)
wine_rules = rules_from(wine_freq, WN, wine_items, 0.7)
print(f"  {WN} baskets, {len(wine_items)} items, every basket exactly "
      f"{len(wine_tx[0])} items long")
print(f"  at support 20% and confidence 70%: {wine_work['frequent']:,} frequent sets and "
      f"{len(wine_rules):,} rules, from {wine_work['counted']:,} counted candidates")
print(f"  the same thresholds on the groceries: average basket {SLOTS / N:.2f} items against "
      f"{len(wine_tx[0])} here, which is why the two behave nothing alike")
wine_best = max(wine_rules, key=lambda r: r["lift"]) if wine_rules else None
if wine_best:
    print(f"  highest lift: {{{', '.join(wine_best['a'])}}} => {{{', '.join(wine_best['c'])}}} "
          f"at {wine_best['lift']:.4f}")

# ==================================================================== the cost
cost_rows = []
for d in [10, 20, 50, 169, 1000]:
    cost_rows.append({
        "items": d,
        "subsets": str(2 ** d - 1),
        "digits": len(str(2 ** d - 1)),
        "pairs": d * (d - 1) // 2,
        "upto3": d + d * (d - 1) // 2 + d * (d - 1) * (d - 2) // 6,
    })
print("\ncost, counted:")
for r in cost_rows:
    print(f"  {r['items']:>5} items: {r['subsets'] if r['digits'] <= 12 else str(r['digits']) + ' digits':>12} "
          f"subsets, {r['pairs']:,} pairs, {r['upto3']:,} sets of up to three")


def slim(rules, k=40, key="lift"):
    top = sorted(rules, key=lambda r: -r[key])[:k]
    return [{"a": r["a"], "c": r["c"], "support": rnd(r["support"], 6),
             "confidence": rnd(r["confidence"], 6), "lift": rnd(r["lift"], 6),
             "leverage": rnd(r["leverage"], 6),
             "conviction": rnd(r["conviction"], 4) if r["conviction"] != float("inf") else None,
             "count": r["count"]} for r in top]


out = {
    "groceries": {
        "n": N, "items": len(ITEMS), "slots": SLOTS,
        "avg_basket": rnd(SLOTS / N, 4), "max_basket": max(len(t) for t in TX),
        "names": ship_items,
        "counts": [counts[i] for i in ship_items],
        "bitsets": [bitsets[i] for i in ship_items],
        "bytes": nbytes,
        "all_names": ITEMS, "all_counts": [counts[i] for i in ITEMS],
        "digest": DIGEST,
        "ship_ref": ship_ref, "ship_supports": SHIP_SUPPORTS,
        "total_subsets_digits": len(str(total_subsets)),
        "total_subsets": str(total_subsets),
    },
    "apriori": {
        "min_support": MIN_SUP, "min_count": min_count, "min_conf": MIN_CONF,
        "levels": [len(x) for x in levels], "work": work,
        "reachable": str(reachable), "longest": longest,
        "sweep": sweep,
        "top": {
            "n_items": TOP, "names": top_items,
            "subsets": len(exhaustive), "frequent": len(truth20),
            "counted": work20["counted"], "generated": work20["generated"],
            "pruned": work20["pruned"], "same": same, "violations": violations,
            "by_size": {str(k): v for k, v in sorted(by_size.items())},
        },
    },
    "tree": {"rows": tree_rows, "best": best_tree, "worst": worst_tree,
             "items": len(frequent1), "order": frequent1},
    "rules": {
        "n": len(all_rules), "below_one": len(below_one),
        "pair_names": pair_names, "pair_count": pair_count,
        "trap": {k: (rnd(v, 6) if isinstance(v, float) else v) for k, v in trap.items()}
        if trap else None,
        "trap_prior": rnd(counts[trap["c"][0]] / N, 6) if trap else None,
        "best_conf": {k: (rnd(v, 6) if isinstance(v, float) else v)
                      for k, v in best_conf.items()},
        "best_lift": {k: (rnd(v, 6) if isinstance(v, float) else v)
                      for k, v in best_lift.items()},
        "top_lift": slim(all_rules, 40, "lift"),
        "top_conf": slim(all_rules, 40, "confidence"),
        "pairs": pairs, "lift_gap": lift_gap,
        "worst_asym": {"gap": rnd(worst_asym[0], 6),
                       "a": worst_asym[1]["a"], "c": worst_asym[1]["c"],
                       "conf_ab": rnd(worst_asym[1]["confidence"], 6),
                       "conf_ba": rnd(worst_asym[2]["confidence"], 6),
                       "lift": rnd(worst_asym[1]["lift"], 6)} if pairs else None,
    },
    "permutation": {
        "shuffles": PERMS,
        "real_frequent": real_work["frequent"], "real_rules": len(real_rules),
        "frequent": perm_freq, "rules": perm_rules,
        "mean_frequent": rnd(float(np.mean(perm_freq)), 3),
        "sd_frequent": rnd(float(np.std(perm_freq)), 3),
        "mean_rules": rnd(float(np.mean(perm_rules)), 3),
        "sd_rules": rnd(float(np.std(perm_rules)), 3),
        "max_rules": int(max(perm_rules)), "max_frequent": int(max(perm_freq)),
        "capped_at": 4,
    },
    "wine": {
        "n": WN, "items": wine_items, "basket_size": len(wine_tx[0]),
        "bitsets": [base64.b64encode(wine_mask[i].to_bytes((WN + 7) // 8, "little")).decode("ascii")
                    for i in wine_items],
        "bytes": (WN + 7) // 8,
        "min_support": 0.2, "min_conf": 0.7,
        "frequent": wine_work["frequent"], "counted": wine_work["counted"],
        "generated": wine_work["generated"], "pruned": wine_work["pruned"],
        "rules": len(wine_rules), "top": slim(wine_rules, 25, "lift"),
        "shared": shared,
    },
    "cost": {"rows": cost_rows},
    "meta": {"seed": SEED, "sklearn": sklearn.__version__, "scipy": scipy.__version__,
             "numpy": np.__version__, "source": "arules Groceries via a public mirror"},
}
path = OUT / "baskets.json"
path.write_text(json.dumps(out, separators=(",", ":")), encoding="utf-8")
print(f"\nwrote {path}  ({path.stat().st_size / 1024:.1f} kB)")
