/* Level-wise against leaf-wise growth: the one picture that separates the
 * classic XGBoost tree from the LightGBM tree. Same budget of splits, spent
 * two different ways. */
import { makeChart } from '../../assets/js/chart.js';

/* A toy tree whose nodes carry a plausible loss-reduction score. Leaf-wise
 * growth always spends the next split on the highest score anywhere; level-wise
 * finishes a whole depth before descending. */
const NODES = [
  { id: 0, parent: null, gain: 100, side: null },
  { id: 1, parent: 0, gain: 62, side: 'L' },
  { id: 2, parent: 0, gain: 18, side: 'R' },
  { id: 3, parent: 1, gain: 40, side: 'L' },
  { id: 4, parent: 1, gain: 9, side: 'R' },
  { id: 5, parent: 2, gain: 6, side: 'L' },
  { id: 6, parent: 2, gain: 4, side: 'R' },
  { id: 7, parent: 3, gain: 27, side: 'L' },
  { id: 8, parent: 3, gain: 5, side: 'R' },
  { id: 9, parent: 7, gain: 15, side: 'L' },
  { id: 10, parent: 7, gain: 3, side: 'R' },
];

function depthOf(id) {
  let d = 0;
  let cur = NODES[id];
  while (cur.parent !== null) {
    d++;
    cur = NODES[cur.parent];
  }
  return d;
}

/* Order in which each strategy would add nodes, given a budget. */
function order(strategy, budget) {
  const added = [0];
  while (added.length < budget) {
    const candidates = NODES.filter((nd) => nd.parent !== null && !added.includes(nd.id) && added.includes(nd.parent));
    if (!candidates.length) break;
    let pick;
    if (strategy === 'leaf') {
      pick = candidates.reduce((a, b) => (b.gain > a.gain ? b : a));
    } else {
      const minDepth = Math.min(...candidates.map((c) => depthOf(c.id)));
      const shallow = candidates.filter((c) => depthOf(c.id) === minDepth);
      pick = shallow.reduce((a, b) => (b.gain > a.gain ? b : a));
    }
    added.push(pick.id);
  }
  return added;
}

export function initGrowthWidget() {
  const { g, w, h } = makeChart('#growth-chart', {
    width: 680, height: 320, margin: { top: 34, right: 20, bottom: 30, left: 20 },
  });

  const maxDepth = 4;
  const pos = {};
  // lay out by depth and by in-order horizontal position
  const perDepth = {};
  NODES.forEach((nd) => {
    const d = depthOf(nd.id);
    perDepth[d] = perDepth[d] || [];
    perDepth[d].push(nd.id);
  });
  Object.entries(perDepth).forEach(([d, ids]) => {
    ids.forEach((id, i) => {
      pos[id] = { x: ((i + 1) / (ids.length + 1)) * w, y: (+d / maxDepth) * h };
    });
  });

  const linkG = g.append('g');
  const nodeG = g.append('g');
  const title = g.append('text').attr('class', 'chart-annotation')
    .attr('x', w / 2).attr('y', -16).attr('text-anchor', 'middle');

  const slider = document.querySelector('#growth-budget');
  const label = document.querySelector('#growth-budget-label');
  const stats = document.querySelector('#growth-stats');
  const state = { strategy: 'level' };

  function render() {
    const budget = +slider.value;
    label.textContent = budget;
    const added = order(state.strategy, budget);
    const set = new Set(added);

    linkG.selectAll('line').data(NODES.filter((nd) => nd.parent !== null)).join('line')
      .attr('x1', (nd) => pos[nd.parent].x).attr('y1', (nd) => pos[nd.parent].y)
      .attr('x2', (nd) => pos[nd.id].x).attr('y2', (nd) => pos[nd.id].y)
      .attr('stroke', (nd) => (set.has(nd.id) ? 'var(--squidink)' : 'var(--stone)'))
      .attr('stroke-width', (nd) => (set.has(nd.id) ? 2.4 : 1))
      .attr('stroke-dasharray', (nd) => (set.has(nd.id) ? null : '3 4'));

    const nodes = nodeG.selectAll('g.node').data(NODES).join(
      (enter) => {
        const gg = enter.append('g').attr('class', 'node');
        gg.append('circle');
        gg.append('text').attr('class', 'chart-annotation').attr('text-anchor', 'middle')
          .style('font-size', '0.6rem').style('letter-spacing', '0').style('text-transform', 'none');
        return gg;
      }
    );
    nodes.attr('transform', (nd) => `translate(${pos[nd.id].x},${pos[nd.id].y})`);
    nodes.select('circle')
      .attr('r', (nd) => (set.has(nd.id) ? 15 : 10))
      .attr('fill', (nd) => (set.has(nd.id) ? 'var(--primary)' : 'var(--paper)'))
      .attr('stroke', (nd) => (set.has(nd.id) ? 'var(--squidink)' : 'var(--stone)'))
      .attr('stroke-width', 2);
    nodes.select('text')
      .attr('y', 4)
      .attr('fill', (nd) => (set.has(nd.id) ? 'white' : 'var(--squidink)'))
      .attr('opacity', (nd) => (set.has(nd.id) ? 1 : 0.45))
      .text((nd) => nd.gain);

    const depth = Math.max(...added.map(depthOf));
    const captured = added.reduce((s, id) => s + NODES[id].gain, 0);
    const total = NODES.reduce((s, nd) => s + nd.gain, 0);

    title.text(state.strategy === 'level' ? 'level-wise: finish a depth before descending' : 'leaf-wise: always split the most promising leaf');
    stats.innerHTML =
      `<div class="slider-label">depth reached: <span class="value">${depth}</span>` +
      ` · loss reduction captured: <span class="value">${((captured / total) * 100).toFixed(0)}%</span></div>` +
      `<div class="slider-label" style="margin-top:.45rem;line-height:1.45">${
        state.strategy === 'leaf'
          ? 'Leaf-wise reaches the valuable nodes sooner and buys more improvement per split, at the cost of a lopsided, deeper tree that overfits small datasets unless the number of leaves is capped.'
          : 'Level-wise keeps the tree balanced and predictable, and spends splits on nodes that were never worth much. Safer, and slower to improve.'
      }</div>`;
  }

  for (const [k, id] of [['level', '#growth-level'], ['leaf', '#growth-leaf']]) {
    document.querySelector(id).addEventListener('click', () => {
      state.strategy = k;
      document.querySelector('#growth-level').classList.toggle('ghost', k !== 'level');
      document.querySelector('#growth-leaf').classList.toggle('ghost', k !== 'leaf');
      render();
    });
  }
  slider.addEventListener('input', render);
  render();
}
