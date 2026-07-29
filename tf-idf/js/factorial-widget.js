/* The twelve cells, drawn so the three decisions can be told apart.
 *
 * Each row is one combination of what-you-do-with-the-count and
 * do-you-weight-by-rarity. The two dots on the row are the same combination
 * with and without the length normalisation, joined by a line, so the length
 * of that line IS the effect of normalising and no arithmetic is needed to
 * read it off. The shaded band behind everything is the noise floor: how far
 * the score moves when nothing about the representation changes at all.
 *
 * Both classifiers are available, and they are not decoration. The claim the
 * article makes is about the representation, so it has to survive being read
 * by two different models; where they disagree, the readout says so instead of
 * quietly showing the flattering one.
 */
import { makeChart, drawAxes, drawGrid, axisLabels } from '../../assets/js/chart.js';
import { seriesLabel, legend } from '../../assets/js/textdraw.js';

const CLFS = [
  { key: 'svm', label: 'linear SVM' },
  { key: 'logreg', label: 'logistic regression' },
];
const METRICS = [
  { key: 'f1', label: 'macro F1' },
  { key: 'acc', label: 'accuracy' },
];
const TF_LABEL = { binary: 'present or not', raw: 'raw count', sublinear: '1 + log count' };

export function initFactorialWidget(data) {
  const F = data.factorial;
  const N = data.noise;
  let clf = 'svm';
  let metric = 'f1';

  const chart = makeChart('#factorial-chart', {
    width: 640, height: 400, margin: { top: 42, right: 30, bottom: 56, left: 168 },
  });
  const { g, w, h } = chart;
  const readout = d3.select('#factorial-readout');

  d3.select('#factorial-clf').selectAll('button').data(CLFS).join('button')
    .attr('class', (d) => `atlas-btn${d.key === clf ? '' : ' ghost'}`)
    .text((d) => d.label)
    .on('click', (event, d) => { clf = d.key; sync(); render(); });
  d3.select('#factorial-metric').selectAll('button').data(METRICS).join('button')
    .attr('class', (d) => `atlas-btn${d.key === metric ? '' : ' ghost'}`)
    .text((d) => d.label)
    .on('click', (event, d) => { metric = d.key; sync(); render(); });

  function sync() {
    d3.select('#factorial-clf').selectAll('button')
      .attr('class', (d) => `atlas-btn${d.key === clf ? '' : ' ghost'}`);
    d3.select('#factorial-metric').selectAll('button')
      .attr('class', (d) => `atlas-btn${d.key === metric ? '' : ' ghost'}`);
  }

  const val = (cell) => cell[`${clf}_${metric}`];
  const floor = () => (metric === 'f1' ? N.f1_floor : N.acc_floor);

  function rows() {
    const out = [];
    ['binary', 'raw', 'sublinear'].forEach((tf) => {
      [false, true].forEach((useIdf) => {
        const off = F.cells.find((c) => c.tf === tf && c.idf === useIdf && !c.norm);
        const on = F.cells.find((c) => c.tf === tf && c.idf === useIdf && c.norm);
        out.push({ tf, useIdf, off, on });
      });
    });
    return out;
  }

  function render() {
    g.selectAll('*').remove();
    const R = rows();
    const all = F.cells.map(val);
    const lo = Math.min(...all);
    const hi = Math.max(...all);
    const pad = 0.06 * (hi - lo || 1);
    const x = d3.scaleLinear().domain([lo - pad, hi + pad]).range([0, w]);
    const band = h / R.length;

    g.append('g').attr('class', 'grid')
      .call((s) => drawGrid(s, x, d3.scaleLinear().domain([0, 1]).range([h, 0]), w, h,
        { xTicks: 6, yTicks: 0 }));
    const axes = g.append('g');
    drawAxes(axes, x, d3.scaleLinear().domain([0, 1]).range([h, 0]), w, h,
      { xTicks: 6, yTicks: 0, xFmt: (v) => v.toFixed(2) });
    axisLabels(axes, w, h, { x: metric === 'f1' ? 'macro F1 on the held-out split' : 'accuracy on the held-out split' });

    /* The noise floor, anchored at the best cell: anything whose line ends
     * inside this band is a cell the test split cannot tell from the winner. */
    const best = Math.max(...all);
    g.append('rect')
      .attr('x', x(best - floor())).attr('y', 0)
      .attr('width', Math.max(0, x(best) - x(best - floor()))).attr('height', h)
      .attr('fill', 'var(--squidink)').attr('opacity', 0.09);

    R.forEach((r, i) => {
      const y = i * band + band / 2;
      const a = val(r.off);
      const b = val(r.on);
      g.append('line')
        .attr('x1', x(a)).attr('x2', x(b)).attr('y1', y).attr('y2', y)
        .attr('stroke', 'var(--primary)').attr('stroke-width', 2).attr('opacity', 0.55);
      g.append('circle').attr('cx', x(a)).attr('cy', y).attr('r', 5)
        .attr('fill', 'none').attr('stroke', 'var(--squidink)').attr('stroke-width', 2);
      g.append('circle').attr('cx', x(b)).attr('cy', y).attr('r', 5)
        .attr('fill', 'var(--primary)');
      seriesLabel(g, -12, y + 4,
        `${TF_LABEL[r.tf]}${r.useIdf ? ' + idf' : ''}`, { anchor: 'end', size: 11.5 });
    });

    legend(g, [
      { label: 'length not normalised', colour: 'var(--squidink)', shape: 'ring' },
      { label: 'length normalised', colour: 'var(--primary)', shape: 'dot' },
      { label: 'what the test split cannot resolve', colour: 'var(--squidink)', shape: 'square' },
    ], { x0: 0, y0: -20, gap: 200, size: 10.5, cols: 2 });

    /* Every number in the sentence below is measured here, including the
     * directions: an effect that comes out negative has to read as one. */
    const normGaps = R.map((r) => val(r.on) - val(r.off));
    const idfGaps = ['binary', 'raw', 'sublinear'].map((tf) => {
      const a = F.cells.find((c) => c.tf === tf && !c.idf && c.norm);
      const b = F.cells.find((c) => c.tf === tf && c.idf && c.norm);
      return val(b) - val(a);
    });
    const fl = floor();
    const normMin = Math.min(...normGaps);
    const normMax = Math.max(...normGaps);
    const idfBig = idfGaps.filter((v) => Math.abs(v) > fl).length;
    const winner = F.cells.reduce((p, c) => (val(c) > val(p) ? c : p), F.cells[0]);

    readout.html(
      `<div>Normalising the length moves this score by between `
      + `<span class="value">${normMin.toFixed(4)}</span> and `
      + `<span class="value">${normMax.toFixed(4)}</span> across the six rows, and it is the same `
      + `sign in every one of them. The floor is `
      + `<span class="value">${fl.toFixed(4)}</span>, so all six clear it.</div>`
      + `<div>Weighting by rarity, with the length already normalised, moves it by `
      + `${idfGaps.map((v) => `<span class="value">${v >= 0 ? '+' : ''}${v.toFixed(4)}</span>`).join(', ')} `
      + `for the three counts. `
      + (idfBig === 0
        ? `None of the three clears the floor, in either direction.`
        : `${idfBig} of the three clears the floor.`)
      + `</div>`
      + `<div>Best cell here: <span class="bold">${TF_LABEL[winner.tf]}`
      + `${winner.idf ? ' + idf' : ''}${winner.norm ? ', normalised' : ', not normalised'}</span> `
      + `at <span class="value">${val(winner).toFixed(4)}</span>.</div>`
    );
  }

  sync();
  render();
  return { render };
}
