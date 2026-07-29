/* One layer of a graph network, run in the page, on a real piece of Cora.
 *
 * Seventy papers and their citations. Each round replaces every node's value
 * with the average of itself and its neighbours, which is the whole of the
 * propagation half of a GCN layer, and the colours are that value. Two or
 * three rounds spread the labelled nodes usefully; ten turn the picture into
 * one colour, which is what over-smoothing looks like before anybody measures
 * it.
 */
import { makeChart } from '../../assets/js/chart.js';

const CLASS_COLOURS = ['#003181', '#df2a5d', '#0e8a55', '#ff9900', '#7c5aed', '#007faa', '#a8410f'];

export function initLiveWidget(data) {
  const W = data.widget;
  const slider = document.querySelector('#live-slider');
  const label = document.querySelector('#live-value');
  const buttons = document.querySelector('#live-buttons');
  const readout = document.querySelector('#live-readout');
  const note = document.querySelector('#live-note');
  const st = { view: 'labels' };

  const chart = makeChart('#live-chart', {
    width: 640, height: 420, margin: { top: 30, right: 20, bottom: 24, left: 20 }, clip: true,
  });
  const { g, plot, w, h, clear } = chart;

  const n = W.nodes.length;
  const x = d3.scaleLinear().domain([-1.05, 1.05]).range([0, w]);
  const y = d3.scaleLinear().domain([-1.05, 1.05]).range([h, 0]);

  /* the same operator the generator uses: A + I, row averaged. The edges
     arrive as pairs of positions in this node list, not as node ids, which is
     what the drawing below indexes with too. */
  const nb = W.nodes.map(() => []);
  W.edges.forEach(([a, b]) => {
    if (a < 0 || b < 0 || a >= n || b >= n || a === b) return;
    nb[a].push(b);
    nb[b].push(a);
  });

  const btns = {};
  for (const [key, text] of [['labels', 'the labelled nodes spreading'],
    ['features', 'the word vectors averaging']]) {
    const b = document.createElement('button');
    b.className = 'atlas-btn ghost';
    b.textContent = text;
    b.addEventListener('click', () => { st.view = key; render(); });
    buttons.appendChild(b);
    btns[key] = b;
  }

  function initial() {
    if (st.view === 'labels') {
      const F = W.nodes.map((d) => {
        const row = new Array(W.classes).fill(0);
        if (d.labelled) row[d.y] = 1;
        return row;
      });
      return F;
    }
    /* one number per node: how many words its abstract has, scaled, which is a
       stand in for the feature vector that a reader can actually see move */
    const vals = W.feature_sum;
    const lo = Math.min(...vals);
    const hi = Math.max(...vals);
    return vals.map((v) => [(v - lo) / Math.max(hi - lo, 1e-9)]);
  }

  function propagate(F, steps) {
    let cur = F.map((row) => row.slice());
    for (let s = 0; s < steps; s++) {
      const next = cur.map((row, i) => {
        const out = row.slice();
        nb[i].forEach((j) => { for (let c = 0; c < out.length; c++) out[c] += cur[j][c]; });
        const d = nb[i].length + 1;
        return out.map((v) => v / d);
      });
      cur = next;
    }
    return cur;
  }

  function spread(F) {
    /* how far apart two nodes still are, averaged over the edges, which is the
       quantity the depth section measures on the trained models */
    let s = 0;
    let m = 0;
    for (let i = 0; i < n; i++) {
      nb[i].forEach((j) => {
        let d = 0;
        for (let c = 0; c < F[i].length; c++) d += (F[i][c] - F[j][c]) ** 2;
        s += d;
        m += 1;
      });
    }
    let norm = 0;
    F.forEach((row) => row.forEach((v) => { norm += v * v; }));
    return s / Math.max(m, 1) / Math.max(norm / n, 1e-12);
  }

  function render() {
    Object.entries(btns).forEach(([k, b]) => b.classList.toggle('ghost', k !== st.view));
    const steps = Number(slider.value);
    label.textContent = steps === 0 ? 'none' : String(steps);
    const F = propagate(initial(), steps);

    clear();
    plot.selectAll(null).data(W.edges).enter().append('line')
      .attr('x1', (e) => x(W.nodes[e[0]].x)).attr('y1', (e) => y(W.nodes[e[0]].y2))
      .attr('x2', (e) => x(W.nodes[e[1]].x)).attr('y2', (e) => y(W.nodes[e[1]].y2))
      .attr('stroke', 'var(--stone)').attr('stroke-width', 0.8);

    const colour = (i) => {
      if (st.view === 'features') {
        const v = Math.max(0, Math.min(1, F[i][0]));
        return d3.interpolateLab('#f1f3f3', '#591c5c')(v);
      }
      const row = F[i];
      const tot = row.reduce((a, v) => a + v, 0);
      if (tot < 1e-9) return '#f1f3f3';
      let best = 0;
      row.forEach((v, c) => { if (v > row[best]) best = c; });
      const conf = row[best] / tot;
      return d3.interpolateLab('#f1f3f3', CLASS_COLOURS[best % CLASS_COLOURS.length])(
        Math.min(1, 0.25 + 0.75 * conf));
    };

    plot.selectAll(null).data(W.nodes).enter().append('circle')
      .attr('cx', (d) => x(d.x)).attr('cy', (d) => y(d.y2))
      .attr('r', (d) => (d.labelled ? 6 : 4.2))
      .attr('fill', (d, i) => colour(i))
      .attr('stroke', (d) => (d.labelled ? 'var(--squidink)' : 'var(--stone)'))
      .attr('stroke-width', (d) => (d.labelled ? 1.8 : 0.7));

    const lab = W.nodes.filter((d) => d.labelled).length;
    const reached = F.filter((row) => row.reduce((a, v) => a + v, 0) > 1e-9).length;
    const agree = W.nodes.filter((d, i) => {
      const row = F[i];
      const tot = row.reduce((a, v) => a + v, 0);
      if (tot < 1e-9 || st.view === 'features') return false;
      let best = 0;
      row.forEach((v, c) => { if (v > row[best]) best = c; });
      return best === d.y;
    }).length;

    readout.innerHTML = st.view === 'labels'
      ? `${n} papers from the citation graph, <span class="value">${lab}</span> of them labelled. `
        + `After <span class="value">${steps}</span> round${steps === 1 ? '' : 's'} of averaging, `
        + `<span class="value">${reached}</span> of the ${n} have received anything at all, and `
        + `<span class="value">${agree}</span> would be classified correctly by taking the largest `
        + `share. The spread between neighbours is `
        + `<span class="value">${spread(F).toExponential(2)}</span>: it falls with every round, and `
        + `when it reaches zero every node is identical and no layer after that can tell them apart.`
      : `The same graph, averaging the word counts instead of the labels. After `
        + `<span class="value">${steps}</span> round${steps === 1 ? '' : 's'} the spread between `
        + `neighbours is <span class="value">${spread(F).toExponential(2)}</span>. Nothing is being `
        + `trained here: this is the fixed half of a graph layer, the part that happens before any `
        + `weights are applied, and it is where both the benefit and the failure come from.`;

    note.textContent = steps <= 3
      ? 'Two or three rounds is what a working model uses: far enough to reach a labelled node, '
        + 'near enough that the neighbourhood still has an edge.'
      : 'Past about four rounds the picture stops being about any particular node. This is not a '
        + 'bug in the drawing: it is the fixed point of averaging.';
  }

  slider.addEventListener('input', render);
  render();
  return { render, propagate, spread, state: st };
}
