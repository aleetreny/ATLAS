/* The front door, with the confounder drawn and unavailable.
 *
 * The graph is drawn with the unmeasured variable in outline rather than
 * filled, because the whole content of the section is that it exists, it
 * matters, and no formula on this page is allowed to read it.
 */
import { makeChart, drawAxes, drawGrid, axisLabels, annotate } from '../../assets/js/chart.js';

const f4 = (v) => v.toFixed(4);
const f6 = (v) => v.toFixed(6);

export function initFrontWidget(data) {
  const host = document.querySelector('#front-widget');
  if (!host) return;
  const F = data.frontdoor;
  const chart = makeChart('#front-chart', {
    width: 640, height: 400, margin: { top: 128, right: 30, bottom: 56, left: 168 },
  });
  const { g, svg, w, h, margin } = chart;
  const readout = document.querySelector('#front-readout');
  const views = d3.select('#front-views');
  let view = 'estimates';

  const buttons = [
    ['estimates', 'what each route returns'],
    ['steps', 'the two steps of the formula'],
  ];
  views.selectAll('button')
    .data(buttons)
    .join('button')
    .attr('class', (d) => (d[0] === view ? 'atlas-btn' : 'atlas-btn ghost'))
    .text((d) => d[1])
    .on('click', (_, d) => { view = d[0]; render(); });

  const LAYOUT = [[130, 8], [20, 80], [130, 80], [240, 80]];

  function drawGraph(highlight) {
    svg.selectAll('g.dag').remove();
    const dag = svg.append('g').attr('class', 'dag')
      .attr('transform', `translate(${Math.max(10, margin.left + w / 2 - 130)}, 12)`);
    const defs = dag.append('defs');
    defs.append('marker')
      .attr('id', 'arrow-front').attr('viewBox', '0 0 10 10')
      .attr('refX', 9).attr('refY', 5)
      .attr('markerWidth', 6).attr('markerHeight', 6).attr('orient', 'auto')
      .append('path').attr('d', 'M 0 0 L 10 5 L 0 10 z').attr('fill', 'var(--squidink)');
    F.parents.forEach((pa, j) => {
      pa.forEach((i) => {
        const a = LAYOUT[i];
        const b = LAYOUT[j];
        const dx = b[0] - a[0];
        const dy = b[1] - a[1];
        const len = Math.hypot(dx, dy) || 1;
        const lit = highlight && highlight.some(([p, q]) => p === i && q === j);
        dag.append('line')
          .attr('x1', a[0] + (dx / len) * 19).attr('y1', a[1] + (dy / len) * 19)
          .attr('x2', b[0] - (dx / len) * 25).attr('y2', b[1] - (dy / len) * 25)
          .attr('stroke', lit ? 'var(--primary)' : 'var(--squidink)')
          .attr('stroke-width', lit ? 3.2 : 1.6)
          .attr('stroke-dasharray', i === 0 ? '5 4' : null)
          .attr('marker-end', 'url(#arrow-front)');
      });
    });
    F.names.forEach((name, i) => {
      const hidden = i === 0;
      dag.append('circle')
        .attr('cx', LAYOUT[i][0]).attr('cy', LAYOUT[i][1]).attr('r', 18)
        .attr('fill', hidden ? 'var(--paper)' : 'var(--squidink)')
        .attr('stroke', hidden ? 'var(--cosmos)' : 'var(--squidink)')
        .attr('stroke-width', hidden ? 2.4 : 1.6)
        .attr('stroke-dasharray', hidden ? '4 3' : null);
      dag.append('text')
        .attr('x', LAYOUT[i][0]).attr('y', LAYOUT[i][1] + 5)
        .attr('text-anchor', 'middle').style('font-size', '13px')
        .style('font-weight', '700')
        .attr('fill', hidden ? 'var(--cosmos)' : 'var(--paper)')
        .text(name);
    });
    dag.append('text').attr('class', 'chart-note')
      .attr('x', 130).attr('y', 116).attr('text-anchor', 'middle')
      .style('font-size', '11.5px')
      .text('the dashed circle is never observed');
  }

  function render() {
    views.selectAll('button')
      .attr('class', (d) => (d[0] === view ? 'atlas-btn' : 'atlas-btn ghost'));
    g.selectAll('*').remove();
    drawGraph(view === 'steps' ? [[1, 2], [2, 3]] : null);

    const bars = [
      ['the truth', F.truth, 'var(--anchor)'],
      ['front door formula', F.frontdoor, 'var(--primary)'],
      ['difference in means', F.naive, 'var(--cosmos)'],
      ...F.backdoor_sets.map((s) => [
        `adjust for ${s.S.length ? s.S.join(', ') : 'nothing'}`, s.estimate, 'var(--cosmos)']),
    ];
    const seen = new Set();
    const unique = bars.filter((b) => {
      const k = b[0];
      if (seen.has(k)) return false;
      seen.add(k);
      return true;
    });
    const y = d3.scaleBand().domain(unique.map((b) => b[0])).range([0, h]).padding(0.3);
    const x = d3.scaleLinear()
      .domain([0, d3.max(unique, (b) => b[1]) * 1.18]).nice().range([0, w]);
    drawGrid(g, x, y, w, h);
    drawAxes(g, x, y, w, h);
    axisLabels(g, w, h, { x: 'effect of the treatment on the outcome' });
    unique.forEach(([label, v, colour]) => {
      g.append('rect')
        .attr('x', 0).attr('y', y(label)).attr('width', Math.max(1, x(v)))
        .attr('height', y.bandwidth())
        .attr('fill', colour).attr('fill-opacity', 0.85);
      g.append('text').attr('class', 'chart-note')
        .attr('x', x(v) + 8).attr('y', y(label) + y.bandwidth() / 2 + 4)
        .style('font-size', '12px')
        .text(f6(v));
    });
    g.append('line')
      .attr('x1', x(F.truth)).attr('x2', x(F.truth)).attr('y1', 0).attr('y2', h)
      .attr('stroke', 'var(--anchor)').attr('stroke-width', 1.5)
      .attr('stroke-dasharray', '4 4');

    readout.innerHTML = view === 'steps'
      ? `<p>The two highlighted arrows are the whole formula. The first step asks how much `
        + `the treatment moves the mediator, and nothing confounds that pair because the `
        + `hidden variable does not point at the mediator. The second asks how much the `
        + `mediator moves the outcome, holding the treatment fixed, and that pair is not `
        + `confounded either once the treatment is conditioned on. Multiply the two and sum, `
        + `and the answer is <span class="bold">${f6(F.frontdoor)}</span> against a truth of `
        + `${f6(F.truth)}. Neither step ever reads the dashed circle.</p>`
      : `<p>Four ways of answering, and only one of them is right. The true effect, computed `
        + `with the hidden variable in full view, is <span class="bold">${f6(F.truth)}</span>. `
        + `The front door formula, which reads only what an analyst can see, returns `
        + `<span class="bold">${f6(F.frontdoor)}</span>, an error of `
        + `${F.error.toExponential(0)}. Everything else available is wrong: the plain `
        + `difference gives ${f6(F.naive)}, and `
        + `${F.backdoor_sets.map((s) => `adjusting for ${s.S.length ? s.S.join(', ') : 'nothing'} `
          + `gives ${f6(s.estimate)}`).join(', ')}. None of those sets satisfies the backdoor `
        + `criterion, and the criterion is right about all of them.</p>`;
  }

  render();
}
