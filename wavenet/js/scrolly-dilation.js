/* Where a stack of dilated causal convolutions can see.
 *
 * The figure is the arithmetic: each layer doubles the reach and the total is
 * one plus the sum of the dilations, so the number is known before any weights
 * exist. The last step puts one pitch period of the sound the models are
 * trained on next to it, which is what turns an architecture detail into a
 * prediction about what the model can possibly learn.
 */
import { makeChart, drawAxes, axisLabels, wrapLabel } from '../../assets/js/chart.js';
import { scrolly } from '../../assets/js/scrolly.js';

export function initDilationScrolly(data) {
  const R = data.receptive;
  /* The left margin has to hold the widest row label ("d=16") on the first
     three steps AND a rotated axis label on the fourth, and 34 px was only
     enough for the first of those: the tick "0.15" and the words "held out
     loss" landed on top of each other. The top margin holds a title that wraps
     to two lines on step 2, which is why it is not 46 either. */
  const chart = makeChart('#dilation-chart', {
    width: 640, height: 430, margin: { top: 64, right: 24, bottom: 52, left: 58 },
  });
  const { g, w, h, margin } = chart;
  const title = g.append('text').attr('class', 'chart-note')
    .attr('x', w / 2).attr('y', -44).attr('text-anchor', 'middle').style('font-size', '12px');
  const layer = g.append('g');
  const N = 33;                       // samples drawn, newest on the right
  const x = d3.scaleLinear().domain([-(N - 1), 0]).range([0, w]);

  function draw(levels, note) {
    layer.selectAll('*').remove();
    const rowH = Math.min(52, (h - 30) / (levels + 1));
    const yOf = (l) => h - 20 - l * rowH;
    /* the dots: one row of samples per layer */
    for (let l = 0; l <= levels; l++) {
      layer.selectAll(`circle.l${l}`).data(d3.range(-(N - 1), 1)).join('circle')
        .attr('class', `l${l}`)
        .attr('cx', (d) => x(d)).attr('cy', yOf(l)).attr('r', 3)
        .attr('fill', 'var(--squidink)').attr('opacity', 0.25);
      layer.append('text').attr('class', 'chart-note')
        .attr('x', -6).attr('y', yOf(l) + 4).attr('text-anchor', 'end')
        .style('font-size', '10.5px')
        .text(l === 0 ? 'in' : `d=${2 ** (l - 1)}`);
    }
    /* the cone that actually feeds the newest output */
    let reach = [0];
    for (let l = 1; l <= levels; l++) {
      const d = 2 ** (l - 1);
      const next = [];
      reach.forEach((p) => {
        [p, p - d].forEach((q) => { if (!next.includes(q)) next.push(q); });
      });
      reach.forEach((p) => {
        [[p, p], [p, p - d]].forEach(([from, to]) => {
          layer.append('line')
            .attr('x1', x(from)).attr('y1', yOf(l - 1))
            .attr('x2', x(Math.max(to, -(N - 1)))).attr('y2', yOf(l))
            .attr('stroke', 'var(--primary)').attr('stroke-width', 1).attr('opacity', 0.5);
        });
      });
      reach = next.filter((p) => p >= -(N - 1));
    }
    for (let l = 0; l <= levels; l++) {
      const d = l === 0 ? 1 : 2 ** (l - 1);
      const lit = [];
      let cur = [0];
      for (let k = levels; k > l; k--) {
        const dd = 2 ** (k - 1);
        const nxt = [];
        cur.forEach((p) => { [p, p - dd].forEach((q) => { if (!nxt.includes(q)) nxt.push(q); }); });
        cur = nxt;
      }
      cur.filter((p) => p >= -(N - 1)).forEach((p) => lit.push(p));
      layer.selectAll(`circle.lit${l}`).data(lit).join('circle').attr('class', `lit${l}`)
        .attr('cx', (d2) => x(d2)).attr('cy', yOf(l)).attr('r', 4)
        .attr('fill', 'var(--primary)');
    }
    const rf = 1 + d3.sum(d3.range(levels).map((i) => 2 ** i));
    layer.append('line').attr('x1', x(-(rf - 1))).attr('x2', x(0))
      .attr('y1', h - 6).attr('y2', h - 6)
      .attr('stroke', 'var(--anchor)').attr('stroke-width', 2);
    /* anchored to the right end of the bar, which is where the newest sample
       always is. Anchored to the left end it runs off the plot whenever the
       reach is short, which is exactly the first two steps. */
    layer.append('text').attr('class', 'chart-note')
      .attr('x', x(0)).attr('y', h - 10).attr('text-anchor', 'end').style('font-size', '11px')
      .attr('fill', 'var(--anchor)').text(`${rf} samples`);
    layer.append('text').attr('class', 'chart-note').attr('x', 0).attr('y', -8)
      .style('font-size', '11px').text(note);
    return rf;
  }

  const steps = {
    0: () => {
      draw(1, 'every output depends on the sample under it and the one before');
      wrapLabel(title, 'one causal layer: a reach of two samples, a quarter of a millisecond',
        w - 8);
    },
    1: () => {
      draw(2, 'the second layer skips one, so its two inputs are two apart');
      wrapLabel(title, 'two layers, dilations 1 and 2: a reach of four', w - 8);
    },
    2: () => {
      draw(5, 'dilations 1, 2, 4, 8, 16: each layer doubles the distance it skips');
      wrapLabel(title, 'five layers: a reach of 32, and the count is exactly '
        + 'one plus the sum of the dilations', w - 8);
    },
    3: () => {
      layer.selectAll('*').remove();
      const rows = R.rows;
      const xr = d3.scaleLog().domain([rows[0].receptive * 0.8,
        rows[rows.length - 1].receptive * 1.3]).range([0, w]);
      const y = d3.scaleLinear()
        .domain([d3.min(rows, (d) => d.nll) * 0.92, d3.max(rows, (d) => d.nll) * 1.05])
        .range([h - 30, 10]);
      drawAxes(layer, xr, y, w, h - 20, {
        xValues: rows.map((d) => d.receptive), xFmt: d3.format('d'), yTicks: 4,
      });
      axisLabels(layer, w, h - 20, {
        x: 'samples the model can see', y: 'held out loss', leftBudget: margin.left, xOffset: 34,
      });
      layer.append('line').attr('x1', xr(R.period)).attr('x2', xr(R.period))
        .attr('y1', 6).attr('y2', h - 20).attr('stroke', 'var(--cosmos)')
        .attr('stroke-width', 1.6).attr('stroke-dasharray', '5 4');
      layer.append('text').attr('class', 'chart-note').attr('x', xr(R.period) + 5).attr('y', 16)
        .style('font-size', '11px').attr('fill', 'var(--cosmos)')
        .text(`one pitch period: ${R.period} samples`);
      layer.append('path').datum(rows).attr('fill', 'none').attr('stroke', 'var(--primary)')
        .attr('stroke-width', 2.4)
        .attr('d', d3.line().x((d) => xr(d.receptive)).y((d) => y(d.nll)));
      layer.selectAll('circle').data(rows).join('circle')
        .attr('cx', (d) => xr(d.receptive)).attr('cy', (d) => y(d.nll)).attr('r', 4.5)
        .attr('fill', (d) => (d.covers_period ? 'var(--primary)' : '#c9ced0'));
      wrapLabel(title, `five stacks, ${rows[0].layers} to ${rows[rows.length - 1].layers} layers, `
        + `trained on the same six seconds`, w - 8);
    },
  };

  steps[0]();
  scrolly('#dilation-scrolly .step', steps);
  return { render: (i) => steps[i]() };
}
