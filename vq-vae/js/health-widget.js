/* How much of the alphabet survives training, and what keeps it alive.
 *
 * Four variants of the same model, differing in exactly one thing each: how the
 * codebook is updated (gradient or exponential moving average) and how it is
 * started (from noise or from k-means on the encoder's own first outputs).
 * Then the commitment cost, swept, which is the knob everyone tunes without
 * ever seeing what it does to the alphabet.
 */
import { makeChart, drawAxes, drawGrid, axisLabels, wrapLabel } from '../../assets/js/chart.js';

export function initHealthWidget(data) {
  const C = data.collapse;
  const chart = makeChart('#health-chart', {
    width: 620, height: 350, margin: { top: 52, right: 30, bottom: 58, left: 70 },
  });
  const { g, w, h } = chart;
  const buttons = document.querySelector('#health-buttons');
  const readout = document.querySelector('#health-readout');
  const st = { view: 'variants' };

  const btns = {};
  for (const [key, text] of [['variants', 'four ways to keep a codebook'],
    ['beta', 'the commitment cost, swept'],
    ['history', 'how fast it dies']]) {
    const b = document.createElement('button');
    b.className = 'atlas-btn ghost';
    b.textContent = text;
    b.addEventListener('click', () => { st.view = key; render(); });
    buttons.appendChild(b);
    btns[key] = b;
  }

  const layer = g.append('g');
  const title = g.append('text').attr('class', 'chart-note')
    .attr('x', w / 2).attr('y', -32).attr('text-anchor', 'middle')
    .style('font-size', '12px');

  function render() {
    Object.entries(btns).forEach(([k, b]) => b.classList.toggle('ghost', k !== st.view));
    layer.selectAll('*').remove();

    if (st.view === 'variants') {
      const rows = C.variants;
      const x = d3.scaleBand().domain(rows.map((r) => r.id)).range([0, w]).padding(0.28);
      const y = d3.scaleLinear().domain([0, C.k]).range([h, 0]);
      drawGrid(g, x, y, w, h, { xValues: [], yTicks: 5 });
      drawAxes(g, x, y, w, h, { xValues: [], yTicks: 5 });
      axisLabels(g, w, h, { x: '', y: `entries of ${C.k} still in use` });
      layer.selectAll('rect').data(rows).join('rect')
        .attr('x', (r) => x(r.id)).attr('width', x.bandwidth())
        .attr('y', (r) => y(r.active.mean)).attr('height', (r) => h - y(r.active.mean))
        .attr('fill', (r) => (r.id === C.winner ? 'var(--primary)' : 'var(--squidink)'))
        .attr('opacity', (r) => (r.id === C.winner ? 1 : 0.55));
      rows.forEach((r) => {
        layer.append('line')
          .attr('x1', x(r.id) + x.bandwidth() / 2).attr('x2', x(r.id) + x.bandwidth() / 2)
          .attr('y1', y(r.active.min)).attr('y2', y(r.active.max))
          .attr('stroke', 'var(--squidink)').attr('stroke-width', 1.5);
        layer.append('text').attr('class', 'chart-note')
          .attr('x', x(r.id) + x.bandwidth() / 2).attr('y', y(r.active.mean) - 10)
          .attr('text-anchor', 'middle').style('font-size', '11.5px')
          .text(`${r.active.mean}`);
        const words = r.label.split(' ');
        const half = Math.ceil(words.length / 2);
        [words.slice(0, half).join(' '), words.slice(half).join(' ')].forEach((line, li) => {
          layer.append('text').attr('class', 'chart-note')
            .attr('x', x(r.id) + x.bandwidth() / 2).attr('y', h + 18 + li * 18)
            .attr('text-anchor', 'middle').style('font-size', '11px').text(line);
        });
      });
      wrapLabel(title, 'one thing changed at a time, three seeds each', w - 8);
    } else if (st.view === 'beta') {
      const rows = C.beta;
      const x = d3.scalePoint().domain(rows.map((r) => String(r.beta)))
        .range([0, w]).padding(0.5);
      const y = d3.scaleLinear().domain([0, C.k]).range([h, 0]);
      drawGrid(g, x, y, w, h, { xValues: [], yTicks: 5 });
      drawAxes(g, x, y, w, h, { yTicks: 5 });
      axisLabels(g, w, h, { x: 'commitment cost', y: `entries of ${C.k} still in use` });
      layer.append('path')
        .attr('d', d3.line().x((r) => x(String(r.beta))).y((r) => y(r.active.mean))(rows))
        .attr('fill', 'none').attr('stroke', 'var(--primary)').attr('stroke-width', 2);
      layer.selectAll('circle').data(rows).join('circle')
        .attr('cx', (r) => x(String(r.beta))).attr('cy', (r) => y(r.active.mean)).attr('r', 4)
        .attr('fill', 'var(--primary)');
      rows.forEach((r) => {
        layer.append('line')
          .attr('x1', x(String(r.beta))).attr('x2', x(String(r.beta)))
          .attr('y1', y(r.active.min)).attr('y2', y(r.active.max))
          .attr('stroke', 'var(--squidink)').attr('stroke-width', 1.2).attr('opacity', 0.7);
      });
      wrapLabel(title, 'the knob everyone tunes, and what it does to the alphabet', w - 8);
    } else {
      const rows = C.variants;
      const hist = rows[0].history;
      const x = d3.scaleLinear().domain([hist[0].epoch, hist[hist.length - 1].epoch]).range([0, w]);
      const y = d3.scaleLinear().domain([0, C.k]).range([h, 0]);
      drawGrid(g, x, y, w, h, { xTicks: 5, yTicks: 5 });
      drawAxes(g, x, y, w, h, { xTicks: 5, yTicks: 5 });
      axisLabels(g, w, h, { x: 'epoch', y: `entries of ${C.k} in use` });
      const colours = ['var(--anchor)', 'var(--primary)', 'var(--aqua)', 'var(--galaxy)'];
      rows.forEach((r, i) => {
        layer.append('path')
          .attr('d', d3.line().x((p) => x(p.epoch)).y((p) => y(p.active))(r.history))
          .attr('fill', 'none').attr('stroke', colours[i % colours.length])
          .attr('stroke-width', r.id === C.winner ? 2.4 : 1.5)
          .attr('opacity', r.id === C.winner ? 1 : 0.7);
        /* four curves ending close together, so the labels are stacked on a
           fixed ladder instead of on the curve ends they would collide on */
        layer.append('text').attr('class', 'chart-note')
          .attr('x', w - 4).attr('y', 14 + i * 15)
          .attr('text-anchor', 'end').style('font-size', '11px')
          .attr('fill', colours[i % colours.length]).text(r.label);
      });
      wrapLabel(title, 'the alphabet does not die slowly', w - 8);
    }

    const worst = C.variants.reduce((a, b) => (b.active.mean < a.active.mean ? b : a));
    const best = C.variants.reduce((a, b) => (b.active.mean > a.active.mean ? b : a));
    const b0 = C.beta[0];
    const bLast = C.beta[C.beta.length - 1];
    readout.innerHTML = `
      <table class="gen-table">
        <tr>
          <th>variant</th><th>entries alive</th><th>perplexity</th>
          <th>entropy of the usage</th><th>held out error</th>
        </tr>
        ${C.variants.map((r) => `
        <tr>
          <td>${r.id === C.winner ? '<span class="bold">' + r.label + '</span>' : r.label}</td>
          <td><span class="value">${r.active.mean}</span> of ${C.k} &plusmn; ${r.active.spread}</td>
          <td>${r.perplexity.mean}</td>
          <td>${r.usage_bits.mean} bits of ${Math.round(Math.log2(C.k))}</td>
          <td>${r.test.mean}</td>
        </tr>`).join('')}
      </table>
      <div class="gen-note">
        ${st.view === 'beta'
    ? `The commitment cost pulls the encoder's output towards whichever entry it picked. At
       ${b0.beta} the model keeps ${b0.active.mean} entries alive and rebuilds at ${b0.test.mean};
       at ${bLast.beta}, ${bLast.active.mean} and ${bLast.test.mean}. The seed range is drawn as
       a bar on every point, and it is wide enough that the middle of this sweep is one result
       rather than several.`
    : `Started from noise and updated by gradient, the codebook loses
       ${worst.dead.mean} of its ${C.k} entries and ends with a perplexity of
       ${worst.perplexity.mean}: the model is using
       ${worst.active.mean === 1 ? 'a single entry' : `about ${worst.active.mean} entries`}
       and paying for ${Math.round(Math.log2(C.k))} bits. Started from k-means on the encoder's
       own first outputs it keeps ${best.active.mean}. That is one line of initialisation, and
       it is worth ${(worst.test.mean - best.test.mean).toFixed(6)} of held out error against a
       seed floor of ${C.floor}.`}
      </div>`;
  }

  render();
  return st;
}
