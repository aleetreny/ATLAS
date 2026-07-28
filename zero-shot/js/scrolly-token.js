/* One window of a real series turning into a string of letters.
 *
 * The whole interface between a series and a language model is on this figure:
 * subtract, divide, round. Everything the model can ever know about your data
 * arrives through it, which is why the alphabet size below is not a detail of
 * the implementation but a hard floor on the answer.
 */
import { makeChart, drawAxes, drawGrid, axisLabels, wrapLabel } from '../../assets/js/chart.js';
import { scrolly } from '../../assets/js/scrolly.js';

export function encodeWindow(vals, mean, sd, bins, clip) {
  return vals.map((v) => {
    const z = Math.max(-clip, Math.min(clip - 1e-9, (v - mean) / sd));
    return Math.max(0, Math.min(bins - 1, Math.floor(((z + clip) / (2 * clip)) * bins)));
  });
}

export function decodeTokens(tokens, mean, sd, bins, clip) {
  return tokens.map((t) => (((t + 0.5) / bins) * 2 * clip - clip) * sd + mean);
}

export function initTokenScrolly(data) {
  const Q = data.quantisation;
  const D = Q.demo;
  const bins = data.meta.bins;
  const clip = data.meta.clip;
  const chart = makeChart('#token-chart', {
    width: 640, height: 440, margin: { top: 46, right: 24, bottom: 50, left: 70 },
  });
  const { g, w, h, margin } = chart;
  const title = g.append('text').attr('class', 'chart-note')
    .attr('x', w / 2).attr('y', -26).attr('text-anchor', 'middle').style('font-size', '12px');
  const subtitle = g.append('text').attr('class', 'chart-note')
    .attr('x', 2).attr('y', -8).style('font-size', '11px');
  const layer = g.append('g');
  const n = D.window.length;
  const x = d3.scaleLinear().domain([0, n - 1]).range([0, w]);

  function draw(vals, { yLabel, ticks = null, grid = null, steps = false, second = null }) {
    layer.selectAll('*').remove();
    const all = second ? vals.concat(second) : vals;
    const y = d3.scaleLinear()
      .domain([d3.min(all) - (d3.max(all) - d3.min(all)) * 0.08,
        d3.max(all) + (d3.max(all) - d3.min(all)) * 0.08]).range([h, 0]);
    drawGrid(layer, x, y, w, h, { xTicks: 5, yValues: grid, yTicks: 5 });
    drawAxes(layer, x, y, w, h, { xTicks: 5, yValues: ticks, yTicks: 5 });
    axisLabels(layer, w, h, {
      x: 'months of the window', y: yLabel, leftBudget: margin.left,
    });
    if (grid) {
      layer.selectAll('line.edge').data(grid).join('line').attr('class', 'edge')
        .attr('x1', 0).attr('x2', w).attr('y1', (d) => y(d)).attr('y2', (d) => y(d))
        .attr('stroke', 'var(--squidink)').attr('opacity', 0.25);
    }
    if (second) {
      layer.append('path').datum(second).attr('fill', 'none')
        .attr('stroke', 'var(--squidink)').attr('stroke-width', 1.2).attr('opacity', 0.45)
        .attr('d', d3.line().x((d, i) => x(i)).y((d) => y(d)));
    }
    const line = steps
      ? d3.line().curve(d3.curveStepAfter).x((d, i) => x(i)).y((d) => y(d))
      : d3.line().x((d, i) => x(i)).y((d) => y(d));
    layer.append('path').datum(vals).attr('fill', 'none').attr('stroke', 'var(--primary)')
      .attr('stroke-width', 2).attr('d', line);
    layer.selectAll('circle').data(vals).join('circle')
      .attr('cx', (d, i) => x(i)).attr('cy', (d) => y(d)).attr('r', 2.4)
      .attr('fill', 'var(--primary)');
  }

  const z = D.window.map((v) => (v - D.mean) / D.sd);
  const tokens = encodeWindow(D.window, D.mean, D.sd, bins, clip);
  const back = decodeTokens(tokens, D.mean, D.sd, bins, clip);

  const steps = {
    0: () => {
      draw(D.window, { yLabel: 'parts per million' });
      wrapLabel(title, `${n} months of carbon dioxide ending ${D.label}, in its own units`,
        w - 8);
      subtitle.text(`mean ${D.mean}, standard deviation ${D.sd}`);
    },
    1: () => {
      draw(z, { yLabel: 'standard deviations' });
      wrapLabel(title, 'the same window, centred and scaled by its own mean and spread', w - 8);
      subtitle.text('now it could be anything: a temperature, a price, a heart rate');
    },
    2: () => {
      const edges = d3.range(-clip, clip + 0.001, (2 * clip) / 8);
      draw(z.map((v) => Math.max(-clip, Math.min(clip, v))), {
        yLabel: 'standard deviations', grid: edges, ticks: edges.map((e) => +e.toFixed(1)),
      });
      wrapLabel(title, `chopped at +-${clip} and rounded into ${bins} bins`, w - 8);
      subtitle.text('the lines are eight of the bin edges, of which there are '
        + `${bins}`);
    },
    3: () => {
      draw(tokens, { yLabel: 'letter', steps: true });
      wrapLabel(title, `what the model actually receives: ${n} integers between 0 and `
        + `${bins - 1}`, w - 8);
      subtitle.text(`first ten: ${tokens.slice(0, 10).join(', ')}`);
    },
  };

  steps[0]();
  scrolly('#token-scrolly .step', steps);
  return { render: (i) => steps[i](), tokens, back };
}
