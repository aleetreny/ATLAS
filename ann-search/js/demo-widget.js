/* The graph, small enough to look at, with a real walk drawn on it.
 *
 * Two hundred and fifty of the same newswire vectors, projected to their first
 * two coordinates and rebuilt as a graph there, so what is drawn is a real
 * index over real points rather than a diagram of one. The walk is the search
 * the generator ran: enter at the top node, take the best neighbour, repeat
 * until nothing is closer, then widen at the bottom layer.
 *
 * The projection is honest about being a projection: it is two of the sixty
 * four coordinates, and the graph was built on those two, so the picture is a
 * complete account of the search it shows rather than a shadow of another one.
 */
import { makeChart, drawAxes, axisLabels } from '../../assets/js/chart.js';

export function initDemoWidget(data) {
  const D = data.demo;
  const slider = document.querySelector('#demo-slider');
  const label = document.querySelector('#demo-value');
  const readout = document.querySelector('#demo-readout');
  const st = { q: 0 };

  const chart = makeChart('#demo-chart', {
    width: 640, height: 420, margin: { top: 44, right: 30, bottom: 56, left: 60 },
  });
  const { g, w, h } = chart;
  const layer = g.append('g');

  slider.max = String(D.walks.length - 1);
  slider.addEventListener('input', () => { st.q = Number(slider.value); render(); });

  const xs = D.points.map((p) => p[0]);
  const ys = D.points.map((p) => p[1]);
  const pad = 0.15;
  const x = d3.scaleLinear()
    .domain([Math.min(...xs) - pad, Math.max(...xs) + pad]).range([0, w]);
  const y = d3.scaleLinear()
    .domain([Math.min(...ys) - pad, Math.max(...ys) + pad]).range([h, 0]);

  function render() {
    label.textContent = `${st.q + 1} of ${D.walks.length}`;
    layer.selectAll('*').remove();
    drawAxes(g, x, y, w, h, { xTicks: 5, yTicks: 5, xFmt: d3.format('.1f'), yFmt: d3.format('.1f') });
    axisLabels(g, w, h, { x: 'first coordinate', y: 'second coordinate' });

    const walk = D.walks[st.q];
    const onPath = new Set(walk.path.map((p) => p.node));
    const found = new Set(walk.found);

    D.edges.forEach(([a, b]) => {
      layer.append('line')
        .attr('x1', x(D.points[a][0])).attr('y1', y(D.points[a][1]))
        .attr('x2', x(D.points[b][0])).attr('y2', y(D.points[b][1]))
        .attr('stroke', 'var(--stone)').attr('stroke-width', 0.7).attr('opacity', 0.8);
    });

    D.points.forEach((p, i) => {
      const lvl = D.levels[i];
      layer.append('circle').attr('cx', x(p[0])).attr('cy', y(p[1]))
        .attr('r', lvl > 0 ? 3.4 : 2)
        .attr('fill', lvl > 0 ? 'var(--anchor)' : 'var(--squidink)')
        .attr('opacity', lvl > 0 ? 0.9 : 0.4);
    });

    /* The descent through the layers, drawn as the arrows it is, with the node
       where each new layer is entered recorded for the readout. */
    let prev = null;
    let lastLevel = null;
    const entries = [];
    walk.path.forEach((step) => {
      const p = D.points[step.node];
      if (prev) {
        layer.append('line').attr('x1', x(prev[0])).attr('y1', y(prev[1]))
          .attr('x2', x(p[0])).attr('y2', y(p[1]))
          .attr('stroke', 'var(--primary)').attr('stroke-width', 2)
          .attr('stroke-dasharray', '4 3');
      }
      layer.append('circle').attr('cx', x(p[0])).attr('cy', y(p[1])).attr('r', 6)
        .attr('fill', 'none').attr('stroke', 'var(--primary)').attr('stroke-width', 2);
      if (step.level !== lastLevel) {
        entries.push({ node: step.node, p, level: step.level });
        lastLevel = step.level;
      }
      prev = p;
    });

    /* The entry points are NOT labelled on the canvas. They move with the
       slider, so any placement rule that works for one query collides for
       another: against each other where the walk converges, against "the query"
       where it lands close. This is the lesson the consistency article already
       paid for, and the answer is the same one: a series that moves gets named
       in the readout, not on top of itself. */
    const grouped = [];
    entries.forEach((e) => {
      const last = grouped[grouped.length - 1];
      if (last && last.node === e.node) last.levels.push(e.level);
      else grouped.push({ node: e.node, p: e.p, levels: [e.level] });
    });

    found.forEach((i) => {
      layer.append('circle').attr('cx', x(D.points[i][0])).attr('cy', y(D.points[i][1]))
        .attr('r', 4.2).attr('fill', 'var(--cosmos)');
    });
    const t = D.points[walk.truth];
    layer.append('path').attr('d', d3.symbol().type(d3.symbolDiamond).size(120)())
      .attr('transform', `translate(${x(t[0])},${y(t[1])})`)
      .attr('fill', 'none').attr('stroke', 'var(--smile)').attr('stroke-width', 2.4);
    layer.append('circle').attr('cx', x(walk.q[0])).attr('cy', y(walk.q[1])).attr('r', 5)
      .attr('fill', 'var(--squidink)');
    layer.append('text').attr('class', 'chart-note').attr('x', x(walk.q[0]) + 9)
      .attr('y', y(walk.q[1]) + 4).style('font-size', '11px').text('the query');

    [['larger dots: nodes that also live in an upper layer', 'var(--anchor)'],
      ['what the search returned', 'var(--cosmos)'],
      ['the true nearest', 'var(--smile)']].forEach(([txt, colour], i) => {
      layer.append('text').attr('class', 'chart-note').attr('x', 0).attr('y', -26 + i * 13)
        .style('font-size', '10.5px').attr('fill', colour).text(txt);
    });

    const counts = D.upper.map((nodes, L) => `${nodes.length} at layer ${L}`).reverse();
    const qCat = D.labels[walk.truth];
    const agree = walk.found.filter((i) => D.labels[i] === qCat).length;
    readout.innerHTML = `<div class="gen-note">${D.points.length} of the same newswire `
      + `vectors, projected onto two coordinates and indexed there, with `
      + `${D.edges.length} edges at the bottom layer and at most ${D.m} new neighbours per `
      + `insertion. Every walk starts at node ${D.entry}, the single node of the top layer. `
      + `The layer sizes are ${counts.join(', ')}: each level up keeps a `
      + `geometrically smaller sample, which is what makes the first hop long and the last `
      + `ones short. This query takes ${walk.path.length} `
      + `${walk.path.length === 1 ? 'step' : 'steps'}, entering `
      + `${grouped.map((gr) => (gr.levels.length === 1
        ? `layer ${gr.levels[0]}`
        : `layers ${gr.levels.join(' and ')} on the same node`)).join(', then ')}`
      + `. Then it widens at the bottom, and it `
      + (walk.hit
        ? `lands on the true nearest point.`
        : `does not land on the true nearest point, which is marked with a diamond. That is `
          + `what approximate means, and on this small a graph it happens often enough to `
          + `see.`)
      + ` Either way ${agree} of the ${walk.found.length} it returned share the true `
      + `neighbour's category, which is the measure the previous widget argues matters more `
      + `than whether the walk landed on the exact point.</div>`;
  }

  render();
  return { render, state: st };
}
