/* Six training diets, one metric spotlight at a time.
 *
 * A horizontal bar per method, the baseline always outlined, and a verdict
 * paragraph per metric, because the same six models look heroic under recall
 * and damning under Brier and the whole article is about which number you
 * choose to look at.
 */
import { makeChart } from '../../assets/js/chart.js';

const METRICS = [
  {
    key: 'recall', label: 'recall @0.5', fmt: (v) => v.toFixed(3), max: 1,
    text: 'The number that sells resampling tutorials: every intervention rockets from 0.524 to above ' +
      '0.99. Keep a finger on this button and press the others.',
  },
  {
    key: 'precision', label: 'precision @0.5', fmt: (v) => v.toFixed(3), max: 1,
    text: 'The bill for the rocket: from 0.727 down to roughly 0.22. Where the baseline raised 792 ' +
      'alarms, class weights raises 4,727, and three in four of them are false.',
  },
  {
    key: 'f1', label: 'F1 @0.5', fmt: (v) => v.toFixed(3), max: 1,
    text: 'Combine the two and every intervention lands near 0.37, below the baseline\'s 0.609 at the ' +
      'same threshold, and far below the 0.683 the untouched model reaches by simply moving its own ' +
      'threshold to 0.258. The recall miracle, priced honestly, is a loss.',
  },
  {
    key: 'auroc', label: 'AUROC', fmt: (v) => v.toFixed(4), max: 1,
    text: 'From 0.9935 to 0.9956: a spread of two thousandths across all six. Resampling barely changed which cells outrank which: ' +
      'the model\'s ordering survived every diet. What moved was only where the probabilities sit, ' +
      'which is exactly what a threshold move does.',
  },
  {
    key: 'auprc', label: 'AUPRC', fmt: (v) => v.toFixed(4), max: 0.8,
    text: 'The honest summary of the whole trade curve, and the quiet scandal of this benchmark: every ' +
      'single intervention made it worse. Baseline 0.723; the five fixes, 0.602 to 0.667. On this ' +
      'problem the famous methods bought nothing and paid for it.',
  },
  {
    key: 'brier', label: 'Brier score (lower is better)', fmt: (v) => v.toFixed(4), max: 0.03,
    text: 'Squared error on the probabilities themselves. The baseline: 0.0047. Everything else: five to ' +
      'six times worse. This is the damage the next widget dissects.',
  },
  {
    key: 'mean_p', label: 'mean predicted probability', fmt: (v) => (v * 100).toFixed(2) + '%', max: 0.05,
    text: 'The model\'s implied base rate. Reality is 1.00%; the untouched model says 0.97%; every ' +
      'intervention says 4.0% to 4.5%. A model that has the base rate wrong by a factor of four is not ' +
      'a probability model any more, it is a scoring function wearing one\'s clothes.',
  },
];

export function initBenchWidget(data) {
  const bench = data.benchmark;
  const { g, w, h } = makeChart('#bench-chart', {
    width: 700,
    height: 330,
    margin: { top: 22, right: 84, bottom: 30, left: 150 },
  });

  const y = d3.scaleBand().domain(bench.map((b) => b.name)).range([0, h]).padding(0.3);
  g.append('g').selectAll('text').data(bench).join('text')
    .attr('x', -12).attr('y', (b) => y(b.name) + y.bandwidth() / 2 + 4)
    .attr('text-anchor', 'end')
    .style('font-family', 'var(--font-mono)').style('font-size', '12px')
    .attr('fill', 'var(--squidink)')
    .text((b) => b.name);

  const barG = g.append('g');
  const valG = g.append('g');
  const axisG = g.append('g').attr('transform', `translate(0,${h})`);

  const btnRow = document.querySelector('#bench-buttons');
  const readout = document.querySelector('#bench-readout');
  const state = { i: 0 };

  const btns = METRICS.map((m, i) => {
    const b = document.createElement('button');
    b.className = 'atlas-btn' + (i === 0 ? '' : ' ghost');
    b.textContent = m.key === 'mean_p' ? 'mean p' : m.key;
    b.addEventListener('click', () => { state.i = i; render(); });
    btnRow.appendChild(b);
    return b;
  });

  function render() {
    const m = METRICS[state.i];
    const x = d3.scaleLinear().domain([0, m.max]).range([0, w]);
    axisG.call(d3.axisBottom(x).ticks(5).tickFormat(m.key === 'mean_p' ? d3.format('.0%') : d3.format('~g')))
      .selectAll('text').style('font-family', 'var(--font-mono)').style('font-size', '10px');

    barG.selectAll('rect').data(bench).join('rect')
      .attr('x', 0).attr('y', (b) => y(b.name)).attr('height', y.bandwidth())
      .attr('fill', (b) => (b.name === 'baseline' ? 'var(--primary)' : 'var(--stone)'))
      .attr('stroke', 'var(--squidink)').attr('stroke-width', (b) => (b.name === 'baseline' ? 1.6 : 0.8))
      .transition().duration(300)
      .attr('width', (b) => x(Math.min(b[m.key], m.max)));

    valG.selectAll('text').data(bench).join('text')
      .attr('y', (b) => y(b.name) + y.bandwidth() / 2 + 4)
      .style('font-family', 'var(--font-mono)').style('font-size', '11px')
      .attr('fill', 'var(--squidink)')
      .transition().duration(300)
      .attr('x', (b) => x(Math.min(b[m.key], m.max)) + 6)
      .text((b) => m.fmt(b[m.key]));

    btns.forEach((b, i) => b.classList.toggle('ghost', i !== state.i));
    readout.innerHTML =
      `<div class="slider-label" style="line-height:1.5"><span class="bold">${m.label}.</span> ${m.text}</div>` +
      (m.key === 'auprc'
        ? `<table class="imb-table" style="margin-top:.6rem">
             <tr><th>method</th><th>training rows</th><th>fit time</th></tr>
             ${bench.map((b) => `<tr><td>${b.name}</td><td>${b.n_train.toLocaleString('en-US')}</td><td>${b.seconds.toFixed(2)}s</td></tr>`).join('')}
           </table>
           <div class="slider-label" style="margin-top:.5rem;line-height:1.5">The one defensible purchase ` +
          `is in this table: undersampling kept 2% of the data and fitted 20 times faster than the ` +
          `baseline. On 165k rows that is a rounding error; on 165 million it is the whole argument.</div>`
        : '');
  }

  render();
}
