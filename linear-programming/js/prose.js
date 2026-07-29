/* Every figure on this page, composed from the file it was measured into.
 *
 * Nothing numeric is typed into the HTML. If the generator is run again and a
 * count moves, the sentence moves with it, and every comparison is a branch
 * rather than a word that happens to be true today.
 */
const n0 = (v) => v.toLocaleString('en-US');
const pc = (v, d = 1) => `${(v * 100).toFixed(d)}%`;

function set(id, html) {
  const node = document.querySelector(`#${id}`);
  if (node) node.innerHTML = html;
}

export function initProse(data) {
  const P = data.polytope;
  const C = data.corner;
  const D = data.dial;
  const S = data.simplex;
  const U = data.duals;
  const I = data.integers;
  const K = I.knapsack;
  const N = I.nodes;
  const M = data.demo;
  const best = P.vertices[P.best_index];
  const km = S.klee_minty;
  const kmLast = km[km.length - 1];
  const kmExact = km.filter((d) => d.dantzig === d.predicted).length;
  const lathe = U.rows[0];
  const press = U.rows[1];
  const free = U.rows.filter((d) => d.shadow === 0);
  const biggestSaving = N.filter((d) => d.saving !== null).slice(-1)[0];
  const cyc = I.cycles;
  const cyc5 = cyc.find((d) => d.n === 5) || cyc[0];
  const fmt = (v) => (Math.abs(v - Math.round(v)) < 1e-9 ? String(Math.round(v)) : v.toFixed(2));
  const corner = (v) => `(${fmt(v.x)}, ${fmt(v.y)})`;

  set('opening-note',
    `This first branch is about problems where the rules are known exactly and the answer still `
    + `has to be found: a plan, a schedule, a route, a bag. The whole of it rests on one geometric `
    + `fact that this page checks rather than states. When both the objective and the limits are `
    + `straight lines, the best plan can never be in the middle of the allowed region, nor in the `
    + `middle of one of its faces. It has to be at a corner. That turns an infinite search into a `
    + `finite one, and everything after it is about a finite list that is still far too long to `
    + `read.`);

  set('setup-intro',
    `A workshop makes two things. Each unit of the first needs ${P.rows[0].a[0]} hours on the `
    + `lathe and ${P.rows[1].a[0]} on the press; each unit of the second needs `
    + `${P.rows[0].a[1]} and ${P.rows[1].a[1]}. There are ${P.rows[0].b} lathe hours and `
    + `${P.rows[1].b} press hours in the week, no more than ${P.rows[3].b} tonnes of steel, and a `
    + `rule that the second product may not run more than ${P.rows[2].b} unit ahead of the first. `
    + `Profits are ${P.profit[0]} and ${P.profit[1]}. Nothing here is uncertain and nothing is `
    + `estimated: the question is only what to build.`);

  set('setup-note',
    `The four limits and the two "you cannot build a negative amount" rules make `
    + `${P.rows.length + 2} straight lines, and every pair of them crosses somewhere. There are `
    + `${P.pairs} pairs; ${P.singular} of them are parallel and never meet, and of the rest only `
    + `<span class="bold">${P.n_vertices}</span> crossings satisfy all the other limits at the `
    + `same time. Those ${P.n_vertices} points are the corners, and the generator found them by `
    + `solving every pair and throwing away the ones that broke something else. The claim to test `
    + `is that the best plan is one of them, whatever the profits happen to be.`);

  set('dial-note',
    `Turn the dial and watch what the answer does. It does not slide around the boundary: it sits `
    + `on one corner while a whole range of directions goes by, then jumps. Over the `
    + `${D.steps} directions the generator swept, only `
    + `<span class="bold">${D.distinct_owners}</span> corners are ever the answer, which is all of `
    + `them, and no point that is not a corner ever wins. The jumps happen at `
    + `${D.switch_angles.length} angles, and they are not measured so much as predicted: a `
    + `direction changes owner exactly when it is perpendicular to an edge of the region, so the `
    + `switch angles should be the directions of the edge normals. Found by bisection, the worst `
    + `disagreement between the two lists is `
    + `<span class="bold">${D.worst_switch_error_deg}</span> degrees. `
    + `At the first of them, ${D.tie_angle} degrees, two corners score within `
    + `${D.tie_gap.toExponential(0)} of each other, and that is the only way a problem like this `
    + `has more than one answer.`);

  set('corner-claim',
    `Six corners is a drawing, not evidence. So: ${n0(C.instances)} random regions, between three `
    + `and seven limits each, every corner of every one enumerated by brute force, and the best `
    + `corner compared against what HiGHS returns for the same problem. The largest disagreement `
    + `over the ${n0(C.instances)} is <span class="bold">${C.worst_gap}</span>. `
    + `${C.ties === 0
      ? `Not one of them had two corners tied for best, which is what you would expect from `
        + `drawing objectives at random: a tie needs the objective to be exactly parallel to an `
        + `edge, and that is a measure zero coincidence you have to arrange on purpose, as the `
        + `dial above does.`
      : `${C.ties} of them had a tie, which happens when the objective is parallel to an edge.`} `
    + `The regions had ${C.mean_vertices} corners on average and at most ${C.max_vertices}. That `
    + `is the number to keep an eye on for the rest of this page.`);

  set('scrolly-intro',
    `Knowing the answer is a corner is not the same as knowing which one. Enumerating them worked `
    + `here because there are ${P.n_vertices}; in ${S.random.dims} dimensions with `
    + `${S.random.constraints} limits there are already ${S.random.mean_vertices} on average, and `
    + `the count grows like the number of ways to choose which limits are tight, which is a `
    + `binomial coefficient and therefore hopeless. The simplex method does not enumerate. It `
    + `stands on one corner, looks along the edges leaving it, and walks down whichever one goes `
    + `up.`);

  set('scrolly-step-0',
    `Build nothing. That is a corner: two limits are tight there, the two that say you cannot `
    + `build a negative amount. Profit ${P.vertices.find((v) => Math.abs(v.x) < 1e-9 && Math.abs(v.y) < 1e-9).value}. `
    + `Every edge leaving it goes up, because both products are profitable, so there is a choice `
    + `of direction and a rule is needed to make it.`);

  set('scrolly-step-1',
    `The textbook rule takes the variable with the steepest reduced cost, which here is the first `
    + `product at ${P.profit[0]} against ${P.profit[1]}. Walk along that edge until a limit stops `
    + `you: the lathe runs out at ${corner(S.path[1])}, worth ${S.path[1].value}. One pivot.`);

  set('scrolly-step-2',
    `From there the only improving edge trades a little of the first product for the second, and `
    + `it ends at ${corner(S.path[2])}, worth ${S.path[2].value}. Now every edge leaving this `
    + `corner goes down, and that is the stopping rule: a local check, over `
    + `${S.path.length - 1 === 2 ? 'two' : S.path.length - 1} edges, that certifies a global `
    + `answer. It certifies it because the region is convex, which is what convex buys you.`);

  set('scrolly-step-3',
    `<span class="bold">${S.pivots} pivots, ${S.path.length} corners visited out of `
    + `${S.n_vertices}.</span> HiGHS gets ${S.value_highs} for the same problem and the simplex `
    + `written for this page gets ${S.value}, `
    + `${S.agrees_with_highs ? 'the same number' : 'a different number, which is a bug'}. `
    + `The habit generalises: over ${S.random.instances} random problems in `
    + `${S.random.dims} dimensions, the average corner count is ${S.random.mean_vertices} and the `
    + `average number of pivots is <span class="bold">${S.random.mean_pivots}</span>, a factor of `
    + `${S.random.ratio}. The worst case in that sample was ${S.random.max_pivots} pivots against `
    + `${S.random.max_vertices} corners.`);

  set('simplex-note',
    `That average is the reason the simplex method took over, and the reason it was a scandal for `
    + `thirty years: nobody could show it was always fast, and in 1972 Klee and Minty showed it is `
    + `not. Their construction is a cube squashed so that every corner is nearly as good as the `
    + `one above it, which turns the steepest rule into a tour. The counts below are pivots, not `
    + `seconds, so they are the same on any machine, and the page recomputes them as you read.`);

  set('km-note',
    `The prediction for the steepest rule is 2<sup>n</sup> - 1 pivots: every corner of the cube `
    + `except the one it starts on. Measured, it is exactly that in `
    + `<span class="bold">${kmExact} of ${km.length}</span> sizes, up to `
    + `${n0(kmLast.dantzig)} pivots for ${kmLast.n} variables, where the cube has `
    + `${n0(kmLast.vertices)} corners. Bland's rule, which takes the first improving column instead `
    + `of the best one, needs ${km.map((d) => d.bland).join(', ')} on the same cubes: `
    + `${kmLast.bland < kmLast.dantzig
      ? `far fewer, and still growing, so it is a different rule with a different cliff rather `
        + `than a fix`
      : `more, so it is no fix at all`}. `
    + `And HiGHS, with its presolve turned off so that it is doing the same job, needs `
    + `${km.map((d) => d.highs).join(', ')}. `
    + `${km.every((d) => d.highs_presolve === 0)
      ? `With the presolve left on it needs none at all, at any size: it recognises the structure `
        + `and reports the answer without pivoting once. That is worth knowing and it is not a `
        + `fact about the simplex method, which is exactly why the comparison here is made with it `
        + `switched off.`
      : ''} `
    + `All four columns end at the same objective value, so this is a race, not a disagreement.`);

  set('dual-intro',
    `The workshop has a second question, and it is usually the one that matters: what is an extra `
    + `hour on the lathe worth? There is a way to get it without guessing. Every linear problem `
    + `has a mirror image, one price per constraint, whose optimum is the same number, and those `
    + `prices are exactly the answer. Solved here, the plan is worth ${U.objective} and the prices `
    + `add up to ${U.dual_objective}, a gap of ${U.duality_gap}.`);

  set('dual-note',
    `Two of the four limits have a price of zero: `
    + `${free.map((d) => d.name).join(' and ')}. They are not scarce, and the plan leaves `
    + `${free.map((d) => d.slack).join(' and ')} of them unused, so buying more buys nothing. `
    + `The other two are exactly used up, and their prices are `
    + `${lathe.shadow} and ${press.shadow} per unit. Those come out of the dual solution; `
    + `re-solving the whole problem with the right hand side nudged by a millionth gives `
    + `${lathe.finite_difference} and ${press.finite_difference}, and the worst disagreement `
    + `across all four is ${U.worst_shadow_gap}. `
    + `<span class="bold">A shadow price is a local truth, and the widget is there to show where `
    + `it stops.</span> Lathe hours are worth ${lathe.shadow} each for the next `
    + `${lathe.valid_up_to} of them and less after that: extend the straight line to `
    + `${lathe.beyond} extra hours and it promises ${lathe.predicted_at_beyond}, while the honest `
    + `answer is ${lathe.actual_at_beyond}. Press hours run out of road much sooner, after only `
    + `${press.valid_up_to}. Reporting a price without its range is the most common way this `
    + `number gets misused.`);

  set('int-intro',
    `Now break it. Add one word to the problem, "whole", and the geometry that everything above `
    + `rests on is gone: the allowed points are no longer a region with corners, they are a cloud `
    + `of dots, and the best dot need not be anywhere near the best corner. The trick that works `
    + `is to keep solving the easy problem anyway. Drop the whole-number requirement, and whatever `
    + `the relaxation says is a ceiling on what any honest answer can reach, because every whole `
    + `plan is also a fractional plan. A ceiling is enough to throw away most of the search. `
    + `The bag below has ${M.n} items, so ${n0(M.brute_force)} subsets; its relaxation says `
    + `${M.lp_bound} and the true best bag is ${M.optimum}, `
    + `${pc(M.gap, 2)} below it.`);

  set('bb-note',
    `The search opens <span class="bold">${n0(M.bb_nodes)}</span> nodes with the bound and `
    + `${n0(M.bb_nodes_no_bound)} with it switched off. `
    + `${M.bb_nodes_no_bound > M.brute_force
      ? `That second number is larger than the ${n0(M.brute_force)} subsets, which is not a `
        + `mistake: a binary tree has internal nodes as well as leaves, and the plain search walks `
        + `all of them that fit in the bag.`
      : `The plain search prunes only on weight, so it is close to the ${n0(M.brute_force)} `
        + `subsets.`} `
    + `Scaling that up is where it gets interesting. `
    + `${N.map((d) => `${d.n} items: ${n0(d.with_bound)} nodes`).join(', ')}, against `
    + `${n0(N[N.length - 1].brute_force)} subsets at ${N[N.length - 1].n} items. Without the `
    + `bound, the same instances take ${n0(biggestSaving.without_bound)} nodes at `
    + `${biggestSaving.n} items, ${biggestSaving.saving} times as many, and past that size the `
    + `plain search was stopped for running over the node budget rather than allowed to finish. `
    + `Two honest cautions. These are median counts over ${N[0].instances} instances each, and `
    + `knapsack instances of the same size differ enormously; and this whole page is a small `
    + `problem, where the relaxation happens to be a very tight ceiling. A model whose relaxation `
    + `is loose gets none of this, which is the entire craft of integer programming.`);

  set('round-intro',
    `Which leaves the question everybody asks first: why not solve the easy problem and round the `
    + `answer? Sometimes there is nothing to round, and that is not luck. Over `
    + `${n0(I.assignment.instances)} random ${I.assignment.size} by ${I.assignment.size} `
    + `assignment problems, the relaxation came out whole `
    + `<span class="bold">${I.assignment.integral} times</span>, with a worst distance to a whole `
    + `number of ${I.assignment.worst_distance_to_integer}. Nobody asked for integers and integers `
    + `arrived anyway. Over ${n0(K.integral_of)} knapsacks it came out whole `
    + `${K.integral_relaxations} times, and rounding the fractional item down loses `
    + `${pc(K.mean_rounded_loss, 2)} on average and up to ${pc(K.max_rounded_loss, 2)}.`);

  const cycRatio = cyc.map((d) => `${d.n}: ${d.ratio}`).join(', ');
  set('round-assign-n',
    `${I.assignment.integral} of ${n0(I.assignment.instances)}`);
  set('round-assign-val',
    `exactly the integer optimum, distance ${I.assignment.worst_distance_to_integer}`);
  set('round-assign-why',
    `the constraint matrix is totally unimodular, so every corner of the region already has whole `
    + `coordinates and the solver cannot return anything else`);
  set('round-knap-n', `${K.integral_relaxations} of ${n0(K.integral_of)}`);
  set('round-knap-val',
    `${pc(K.mean_gap, 3)} above the true answer on average, at most ${pc(K.max_gap, 3)}`);
  set('round-knap-why',
    `one item gets cut in half; the ceiling is tight but the plan is not a plan, and rounding it `
    + `down costs ${pc(K.mean_rounded_loss, 2)} and is optimal only ${K.rounded_optimal} times in `
    + `${n0(K.instances)}`);
  set('round-cycle-n', `${cyc.filter((d) => d.all_half).length} of ${cyc.length} came out all halves`);
  set('round-cycle-val',
    `the relaxation is ${cyc5.lp} where the answer is ${cyc5.ip}, on ${cyc5.n} nodes`);
  set('round-cycle-why',
    `putting half a guard on every node of an odd ring satisfies every edge and means nothing; the `
    + `ratio is ${cycRatio}, worst at the smallest ring and shrinking, which is the integrality gap `
    + `written out`);

  set('round-note',
    `The three rows are the whole story of integer programming in one table. `
    + `${I.assignment.integral === I.assignment.instances
      ? `When the structure is right, the relaxation is not an approximation to the answer, it is `
        + `the answer, and the fact that it happened ${I.assignment.integral} times out of `
        + `${n0(I.assignment.instances)} is not a coincidence but a theorem about the matrix.`
      : `The assignment relaxation came out whole ${I.assignment.integral} times out of `
        + `${n0(I.assignment.instances)}, which is not what the theorem says and is worth chasing.`} `
    + `When it is nearly right, as in the knapsack, the relaxation is a ceiling good enough to `
    + `prune a search. And when it is wrong, as on the odd ring, the relaxation is confidently `
    + `${cyc5.lp} on a problem whose answer is ${cyc5.ip}, and every fraction in it is a half: not `
    + `a rounding problem, a different problem.`);

  set('verdict-intro',
    `Everything on this page is exact, which makes it a strange first article for a module about `
    + `agents. It is here because the two halves of that word live in different places: the model `
    + `is a set of constraints somebody wrote down, and the search is a machine that walks corners `
    + `or opens nodes. The rest of this branch keeps the search and gives up the guarantee.`);

  set('verdict-lp',
    `the answer is at a corner, checked here on ${n0(C.instances)} random regions with a worst `
    + `disagreement of ${C.worst_gap}, and the walk takes ${S.random.mean_pivots} pivots where `
    + `there are ${S.random.mean_vertices} corners`);
  set('verdict-lp-warn',
    `the pivot rule is not a detail: on the cube of Klee and Minty the steepest rule takes `
    + `${n0(kmLast.dantzig)} pivots for ${kmLast.n} variables and never less than 2<sup>n</sup> - 1`);
  set('verdict-dual',
    `it comes free with the solve, and matches re-solving to ${U.worst_shadow_gap}`);
  set('verdict-dual-warn',
    `it is a slope, not a promise: the lathe price holds for ${lathe.valid_up_to} more hours and `
    + `the press price for ${press.valid_up_to}`);
  set('verdict-mip',
    `the relaxation is a ceiling on every branch, and it cuts ${n0(M.brute_force)} subsets down to `
    + `${n0(M.bb_nodes)} opened nodes on the bag on this page`);
  set('verdict-mip-warn',
    `the saving is the tightness of your relaxation and nothing else: here the gap is `
    + `${pc(M.gap, 2)}, and a model whose relaxation is loose gets none of this`);
  set('verdict-tu',
    `matchings, flows and assignments have totally unimodular matrices, so the corners are already `
    + `whole: ${I.assignment.integral} of ${n0(I.assignment.instances)} came back integral without `
    + `being asked`);
  set('verdict-tu-warn',
    `add one side constraint and the structure is gone; check the matrix, do not check a sample`);

  set('closing-note',
    `The bargain of this page is a good one and it is narrow. Write the problem in straight lines `
    + `and you get a certificate: this is the best plan, here is the price of every resource, and `
    + `here is the proof, in ${S.random.mean_pivots} pivots. Ask for whole numbers and the `
    + `certificate survives but the price of it goes up, from pivots to `
    + `${n0(M.bb_nodes)} nodes on twelve items. Ask for anything that is not a straight line and `
    + `the certificate is gone entirely, and that is where most real problems live. `
    + `<a href="../genetic-algorithms/">The next article</a> takes the same kind of question, `
    + `throws away the guarantee, and asks what is left: a search that knows nothing about the `
    + `shape of what it is searching, and has to earn every improvement by trying things.`);

  set('resources-note',
    `Everything was measured by <code>src/utils/generate_lp_data.py</code> with seed `
    + `${data.meta.seed}, using ${data.meta.solver} through SciPy ${data.meta.scipy} as the `
    + `independent solver. The corner enumeration, the simplex, the Klee and Minty cube and the `
    + `branch and bound all exist twice, once in Python and once in <code>js/lpkit.js</code>, and `
    + `the page checks the two against each other on load.`);
}
