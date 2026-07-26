/* The accuracy trap, told on a waffle of ten thousand squares.
 *
 * Each square stands for ~11 cells of the real test set (109,880 / 10,000),
 * and the counts of coloured squares are scaled from the real confusion
 * matrix, so the picture is proportional while every number printed is exact.
 * The squares are created once; steps only restyle them, which keeps
 * scrolling cheap. The confusion matrix lives beside the waffle so the wide
 * chart survives being scaled down on short viewports.
 */
import { makeChart } from '../../assets/js/chart.js';
import { scrolly } from '../../assets/js/scrolly.js';
import { makeMatrix } from './matrix.js';

/* Deterministic small PRNG so the sprinkle of positives is stable. */
function rng(seed) {
  let a = seed >>> 0;
  return () => {
    a |= 0; a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

const SIDE = 100;               // 100 x 100 = 10,000 squares
const N_POS_DOTS = 100;         // 1% of them are the tree

export function initScrollyTrap(data) {
  const { g, w, h } = makeChart('#trap-chart', {
    width: 690,
    height: 470,
    margin: { top: 34, right: 10, bottom: 10, left: 10 },
  });
  const waffleSize = Math.min(h, 420);
  const cell = waffleSize / SIDE;

  const caption = g.append('text').attr('class', 'chart-annotation')
    .attr('x', w / 2).attr('y', -14).attr('text-anchor', 'middle').style('font-size', '0.66rem');

  /* Which squares are positive, and of those, which get found at 0.5; which
   * negatives raise false alarms. Scaled counts from the real matrix.
   * Seeded Fisher-Yates: d3.shuffle would use its own random source. */
  const r = rng(29);
  const idx = d3.range(SIDE * SIDE);
  for (let i = idx.length - 1; i > 0; i--) {
    const j = Math.floor(r() * (i + 1));
    [idx[i], idx[j]] = [idx[j], idx[i]];
  }
  const posSet = new Set(idx.slice(0, N_POS_DOTS));
  const b = data.baseline;
  const scale = SIDE * SIDE / data.trap.n_test;                 // squares per cell
  const nTP = Math.round(b.tp * scale);                         // ~52
  const nFP = Math.round(b.fp * scale);                         // ~20
  const posArr = idx.slice(0, N_POS_DOTS);
  const tpSet = new Set(posArr.slice(0, nTP));
  const fpSet = new Set(idx.slice(N_POS_DOTS, N_POS_DOTS + nFP));

  const waffle = g.append('g').attr('transform', `translate(0,${(h - waffleSize) / 2})`);
  const sq = cell * 0.78;
  const dots = waffle.selectAll('rect').data(d3.range(SIDE * SIDE)).join('rect')
    .attr('x', (i) => (i % SIDE) * cell + (cell - sq) / 2)
    .attr('y', (i) => Math.floor(i / SIDE) * cell + (cell - sq) / 2)
    .attr('width', sq).attr('height', sq);

  const matrix = makeMatrix(g, waffleSize + 70, h / 2 - 70, 176, 116, { numSize: 13 });
  matrix.hide();

  /* Styling passes. Each returns squares to a complete state: idempotent. */
  function paint(mode) {
    dots
      .attr('fill', (i) => {
        if (mode === 'plain' || mode === 'lazy') return posSet.has(i) ? 'var(--primary)' : 'var(--stone)';
        if (posSet.has(i)) return tpSet.has(i) ? 'var(--primary)' : 'var(--cosmos)';
        return fpSet.has(i) ? 'white' : 'var(--stone)';
      })
      .attr('stroke', (i) => {
        if (mode !== 'plain' && mode !== 'lazy' && !posSet.has(i) && fpSet.has(i)) return 'var(--cosmos)';
        return 'none';
      })
      .attr('stroke-width', 1.1)
      .attr('opacity', (i) => {
        if (mode === 'plain') return posSet.has(i) ? 1 : 0.45;
        if (mode === 'lazy') return posSet.has(i) ? 1 : 0.7;
        return posSet.has(i) || fpSet.has(i) ? 1 : 0.3;
      });
  }

  const t = data.trap;
  const states = {
    0: () => {
      paint('plain');
      matrix.hide();
      caption.text(`${t.n_test.toLocaleString('en-US')} test cells, ${t.n_pos.toLocaleString('en-US')} are the tree (1%)`);
    },
    1: () => {
      paint('lazy');
      matrix.show();
      matrix.update({ tp: 0, fn: t.n_pos, fp: 0, tn: t.n_test - t.n_pos });
      caption.text(`"never say tree": accuracy ${(t.constant_acc * 100).toFixed(2)}%, trees found: 0`);
    },
    2: () => {
      paint('model');
      matrix.show();
      matrix.update({ tp: b.tp, fn: b.fn, fp: b.fp, tn: b.tn });
      caption.text(`logistic at t = 0.5: accuracy ${(b.acc * 100).toFixed(2)}%, found ${b.tp}, missed ${b.fn}, false alarms ${b.fp}`);
    },
    3: () => {
      states[2]();
      caption.text(`precision ${b.tp}/${b.tp + b.fp} = ${(b.precision * 100).toFixed(1)}%, recall ${b.tp}/${b.tp + b.fn} = ${(b.recall * 100).toFixed(1)}%`);
    },
    4: () => {
      states[2]();
      caption.text('the threshold is a dial, not a fact: next section');
    },
  };

  scrolly('#trap-scrolly .step', states);
}
