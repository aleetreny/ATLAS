/* The reliability diagram: what was promised against what happened.
 *
 * Dot area is bin size, because the middle buckets on this problem hold one or
 * two patients each and a reader who cannot see that will over-read them. The
 * two big buckets at the ends are where the argument is, and they are the ones
 * drawn largest.
 */
import { makeChart, drawAxes, drawGrid, axisLabels, tooltip } from '../../assets/js/chart.js';

const SERIES = {
  logistic: { label: 'logistic regression', colour: 'var(--primary)' },
  naive_bayes: { label: 'naive Bayes', colour: 'var(--cosmos)' },
};

export function initCalibWidget(data) {
  const C = data.calibration;
  const { g, w, h } = makeChart('#calib-chart', {
    width: 620,
    height: 470,
    margin: { top: 34, right: 30, bottom: 58, left: 68 },
  });

  /* Padded away from the origin so the two "0.0" ticks do not sit on top of
   * each other where the axes meet. */
  const x = d3.scaleLinear().domain([-0.03, 1.03]).range([0, w]);
  const y = d3.scaleLinear().domain([-0.03, 1.03]).range([h, 0]);
  drawGrid(g, x, y, w, h, { xTicks: 5, yTicks: 5 });
  drawAxes(g, x, y, w, h, { xTicks: 5, yTicks: 5, xFmt: d3.format('.1f'), yFmt: d3.format('.1f') });
  axisLabels(g, w, h, { x: 'probability the model promised', y: 'share that were actually malignant' });

  g.append('line').attr('x1', x(0)).attr('y1', y(0)).attr('x2', x(1)).attr('y2', y(1))
    .attr('stroke', 'var(--squidink)').attr('stroke-width', 1.6).attr('stroke-dasharray', '7 6').attr('opacity', 0.6);
  /* Anchored at the end and set back from the corner: a label riding the
   * diagonal runs off the top right of the canvas otherwise. */
  g.append('text').attr('class', 'chart-annotation')
    .attr('x', x(0.62)).attr('y', y(0.62) - 8).attr('text-anchor', 'end')
    .attr('transform', `rotate(-45,${x(0.62)},${y(0.62) - 8})`)
    .style('font-size', '0.6rem').style('text-transform', 'none').attr('opacity', 0.75)
    .text('a model that means what it says');

  const size = d3.scaleSqrt().domain([1, d3.max([...C.logistic, ...C.naive_bayes], (b) => b.n)]).range([4, 17]);
  const linesG = g.append('g');
  const dotsG = g.append('g');
  const tip = tooltip();

  const title = g.append('text').attr('class', 'chart-annotation')
    .attr('x', w / 2).attr('y', -14).attr('text-anchor', 'middle').style('font-size', '0.66rem');

  const state = { mode: 'log' };
  const readout = document.querySelector('#calib-readout');
  const btns = {
    log: document.querySelector('#calib-log'),
    nb: document.querySelector('#calib-nb'),
    both: document.querySelector('#calib-both'),
  };
  const KEYS = { log: ['logistic'], nb: ['naive_bayes'], both: ['logistic', 'naive_bayes'] };

  const line = d3.line().x((b) => x(b.predicted)).y((b) => y(b.observed));

  function render() {
    const keys = KEYS[state.mode];
    const rows = keys.flatMap((k) => C[k].map((b) => ({ ...b, key: k })));

    linesG.selectAll('path').data(keys, (k) => k).join('path')
      .attr('fill', 'none')
      .attr('stroke', (k) => SERIES[k].colour)
      .attr('stroke-width', 2.4)
      .attr('opacity', 0.75)
      .attr('d', (k) => line(C[k]));

    dotsG.selectAll('circle').data(rows, (b) => b.key + b.lo).join(
      (e) => e.append('circle').attr('stroke', 'white').attr('stroke-width', 1.6),
      (u) => u,
      (ex) => ex.remove()
    )
      .attr('cx', (b) => x(b.predicted))
      .attr('cy', (b) => y(b.observed))
      .attr('r', (b) => size(b.n))
      .attr('fill', (b) => SERIES[b.key].colour)
      .attr('opacity', 0.85)
      .on('mousemove', (event, b) => tip.show(
        `<span style="color:${SERIES[b.key].colour === 'var(--primary)' ? '#a8410f' : '#df2a5d'}">${SERIES[b.key].label}</span><br/>` +
        `bucket ${b.lo.toFixed(1)} to ${b.hi.toFixed(1)}<br/>` +
        `${b.n} patient${b.n === 1 ? '' : 's'}<br/>` +
        `promised ${b.predicted.toFixed(3)}, observed ${b.observed.toFixed(3)}`,
        event
      ))
      .on('mouseleave', () => tip.hide());

    title.text('dot area is how many patients landed in that bucket');
    for (const [k, b] of Object.entries(btns)) b.classList.toggle('ghost', k !== state.mode);

    /* The readout speaks about the biggest bucket of the series on screen,
     * because that is where almost all the patients are and where a promise
     * being broken actually costs something. */
    const lines = keys.map((k) => {
      const big = C[k].reduce((a, b) => (b.n > a.n ? b : a));
      const missed = Math.round(big.n * big.observed);
      const s = SERIES[k];
      return (
        `<div class="slider-label" style="line-height:1.5;margin-bottom:.45rem">` +
        `<span class="bold" style="color:${s.colour}">${s.label}</span> put ` +
        `<span class="value">${big.n}</span> of the 171 held-out patients in the ` +
        `${big.lo.toFixed(1)} to ${big.hi.toFixed(1)} bucket, at an average promise of ` +
        `<span class="value">${big.predicted.toFixed(4)}</span>. ` +
        `<span class="value">${missed}</span> of them ` +
        `${missed === 1 ? 'was' : 'were'} malignant, an observed rate of ` +
        `<span class="value">${big.observed.toFixed(3)}</span>.</div>`
      );
    });
    const extra =
      `<div class="slider-label" style="line-height:1.5">` +
      `Across the whole test set, naive Bayes is past 99% certainty on ` +
      `<span class="value">${(C.extreme_nb * 100).toFixed(0)}%</span> of patients ` +
      `against <span class="value">${(C.extreme_logistic * 100).toFixed(0)}%</span> for logistic regression. ` +
      `It has two answers and calls them probabilities.</div>`;
    readout.innerHTML = lines.join('') + (state.mode === 'log' ? '' : extra);
  }

  for (const [k, b] of Object.entries(btns)) {
    b.addEventListener('click', () => {
      state.mode = k;
      render();
    });
  }
  render();
}
