/* Sparsemax: the one idea in TabNet worth stealing.
 *
 * TabNet's pitch is that it selects features the way a tree does, one subset at
 * a time. The mechanism that makes that more than a slogan is sparsemax, which
 * replaces softmax as the normalising step. Softmax is strictly positive
 * everywhere: exp of anything is above zero, so every feature always gets some
 * weight, however small, and "this model ignored that column" is never literally
 * true. Sparsemax projects the score vector onto the simplex instead, and a
 * projection can land on a face of it, where some coordinates are exactly zero.
 * Whether it does depends on how unequal the scores are, not merely on the fact
 * that the raw scores lie outside the simplex, which they always do.
 *
 *   sparsemax(z) = argmin_{p in simplex} ||p - z||^2
 *
 * The closed form is a sort and a threshold: find how many of the top scores
 * survive, then subtract the same tau from each and clip the rest to zero.
 *
 * The scores are not invented. Each feature starts with a score proportional to
 * how far the additive model on this page moves the price across that feature's
 * range, so the ordering is the one the data actually produced.
 */
import { makeChart, axisLabels } from '../../assets/js/chart.js';

function softmax(z) {
  const m = Math.max(...z);
  const e = z.map((v) => Math.exp(v - m));
  const s = e.reduce((a, b) => a + b, 0);
  return e.map((v) => v / s);
}

/* Condon-style closed form, exactly as in Martins & Astudillo (2016). */
function sparsemax(z) {
  const sorted = [...z].sort((a, b) => b - a);
  let cum = 0;
  let k = 0;
  let tau = 0;
  for (let j = 0; j < sorted.length; j++) {
    cum += sorted[j];
    if (1 + (j + 1) * sorted[j] > cum) {
      k = j + 1;
      tau = (cum - 1) / k;
    }
  }
  return z.map((v) => Math.max(0, v - tau));
}

/* Rounding the smallest softmax weight to two decimals printed "0.00%" next to
 * the words "not zero", which is the one thing this readout must never do. */
const pct = (v) => {
  const p = v * 100;
  if (p >= 0.01) return p.toFixed(2) + '%';
  const e = Math.floor(Math.log10(p));
  return `${(p / 10 ** e).toFixed(1)} × 10^${e} %`;
};

export function initAttentionWidget(data) {
  const feats = data.shapes.map((s) => ({ name: s.feature, raw: s.range }));
  const mx = d3.max(feats, (f) => f.raw);
  /* centred so the slider scales spread rather than level: multiplying every
   * score by the same constant is what a temperature actually does */
  const mean = d3.mean(feats, (f) => f.raw / mx);
  feats.forEach((f) => (f.z = f.raw / mx - mean));

  const { g, w, h } = makeChart('#attn-chart', {
    width: 660,
    height: 360,
    margin: { top: 30, right: 24, bottom: 82, left: 62 },
  });

  const x = d3.scaleBand().domain(feats.map((f) => f.name)).range([0, w]).padding(0.28);
  const y = d3.scaleLinear().domain([0, 1]).range([h, 0]);
  const inner = x.bandwidth() / 2;

  g.append('g')
    .attr('class', 'axis x')
    .attr('transform', `translate(0,${h})`)
    .call(d3.axisBottom(x).tickSizeOuter(0))
    .selectAll('text')
    .attr('transform', 'rotate(-38)')
    .attr('text-anchor', 'end')
    .attr('dx', '-0.4em')
    .attr('dy', '0.5em');
  const gy = g.append('g').attr('class', 'axis y');
  axisLabels(g, w, h, { y: 'attention weight' });

  const softG = g.append('g');
  const sparseG = g.append('g');
  const zeroTags = g.append('g');

  const legend = g.append('g').attr('transform', `translate(0,${-16})`);
  [
    { label: 'softmax', colour: 'var(--stone)', dx: 0 },
    { label: 'sparsemax', colour: 'var(--primary)', dx: 96 },
  ].forEach((d) => {
    legend.append('rect').attr('x', d.dx).attr('y', -9).attr('width', 11).attr('height', 11).attr('fill', d.colour).attr('stroke', 'var(--squidink)').attr('stroke-width', 0.8);
    legend
      .append('text')
      .attr('class', 'chart-annotation')
      .attr('x', d.dx + 16)
      .attr('y', 0)
      .style('font-size', '0.6rem')
      .style('text-transform', 'none')
      .text(d.label);
  });

  const slider = document.querySelector('#attn-slider');
  const readout = document.querySelector('#attn-readout');

  function render() {
    const scale = +slider.value;
    const z = feats.map((f) => f.z * scale);
    const sm = softmax(z);
    const sp = sparsemax(z);

    y.domain([0, Math.max(d3.max(sm), d3.max(sp)) * 1.12]);
    gy.transition().duration(180).call(d3.axisLeft(y).ticks(5).tickFormat(d3.format('.0%')).tickSizeOuter(0));

    const bar = (sel, vals, dx, colour) =>
      sel
        .selectAll('rect')
        .data(feats.map((f, i) => ({ name: f.name, v: vals[i] })))
        .join((e) => e.append('rect').attr('stroke', 'var(--squidink)').attr('stroke-width', 0.8))
        .attr('x', (d) => x(d.name) + dx)
        .attr('width', inner)
        .attr('fill', colour)
        .transition()
        .duration(180)
        .attr('y', (d) => y(d.v))
        .attr('height', (d) => Math.max(0, h - y(d.v)));

    bar(softG, sm, 0, 'var(--stone)');
    bar(sparseG, sp, inner, 'var(--primary)');

    zeroTags
      .selectAll('text')
      .data(feats.map((f, i) => ({ name: f.name, off: sp[i] === 0 })))
      .join((e) => e.append('text').attr('class', 'chart-annotation').attr('text-anchor', 'middle').style('font-size', '0.55rem').attr('fill', 'var(--cosmos)'))
      .attr('x', (d) => x(d.name) + inner * 1.5)
      .attr('y', h - 5)
      .text((d) => (d.off ? 'off' : ''));

    const nZero = sp.filter((v) => v === 0).length;
    const smMin = Math.min(...sm);
    readout.innerHTML =
      `<div class="slider-label">Sparsemax switches <span class="value" style="color:var(--cosmos)">${nZero}</span> of the ` +
      `${feats.length} features fully off. Softmax switches off <span class="value">none</span>: its smallest weight is ` +
      `<span class="value">${pct(smMin)}</span>, small but not zero.</div>` +
      `<div class="slider-label" style="margin-top:.5rem">${
        nZero === 0
          ? 'At low sharpness the two agree: nothing is far enough from the pack to be dropped, so the projection lands in the interior of the simplex and behaves like a soft average.'
          : 'This is what lets TabNet claim it selects features rather than merely weighting them. Whether that selection is worth the training cost is the question the benchmark above answers.'
      }</div>`;
  }

  slider.addEventListener('input', render);
  render();
}
