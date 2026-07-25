/* Sticky scrollytelling chart: from raw cloud to OLS and its systematic flaw.
 * Each step handler is an idempotent, full declaration of chart state. */
import { scrolly } from '../../assets/js/scrolly.js';
import { makeChart, drawAxes, drawGrid, axisLabels, ols1d } from '../../assets/js/chart.js';

const DUR = 650;

export function initScrollyOLS(mpg) {
  const pts = mpg.points;
  const xs = pts.map((d) => d.hp);
  const ys = pts.map((d) => d.mpg);
  const yMean = d3.mean(ys);
  const fit = ols1d(xs, ys); // {intercept, slope} — matches mpg.ref.ols

  const mseFor = (predict) => d3.mean(pts, (d) => (d.mpg - predict(d.hp)) ** 2);
  const mseMean = mseFor(() => yMean);
  const mseOLS = mseFor((x) => fit.intercept + fit.slope * x);

  const { g, w, h } = makeChart('#scrolly-chart', {
    width: 640,
    height: 560,
    margin: { top: 46, right: 30, bottom: 62, left: 64 },
  });

  const x = d3.scaleLinear().domain([35, 235]).range([0, w]);
  const y = d3.scaleLinear().domain([0, 50]).range([h, 0]);

  drawGrid(g, x, y, w, h);
  drawAxes(g, x, y, w, h);
  axisLabels(g, w, h, { x: 'horsepower', y: 'MPG' });

  // ---- layers (order matters) ----
  const residualsG = g.append('g');
  const dotsG = g.append('g');
  const lineG = g.append('g');
  const predG = g.append('g');
  const annG = g.append('g');

  const dots = dotsG
    .selectAll('circle')
    .data(pts)
    .join('circle')
    .attr('cx', (d) => x(d.hp))
    .attr('cy', (d) => y(d.mpg))
    .attr('r', 4.5)
    .attr('fill', 'var(--smile)')
    .attr('stroke', 'white')
    .attr('stroke-width', 1)
    .attr('opacity', 0.9);

  // regression line with white halo (MLU convention)
  const halo = lineG.append('line').attr('stroke', 'white').attr('stroke-width', 7).attr('opacity', 0);
  const line = lineG.append('line').attr('stroke', 'var(--squidink)').attr('stroke-width', 3.5).attr('opacity', 0);

  // top readout
  const readout = annG
    .append('text')
    .attr('class', 'chart-annotation')
    .attr('x', w / 2)
    .attr('y', -18)
    .attr('text-anchor', 'middle');

  const t = () => d3.transition().duration(DUR).ease(d3.easeCubicOut);

  function setLine(slope, intercept, visible) {
    const [x0, x1] = x.domain();
    for (const l of [halo, line]) {
      l.transition(t())
        .attr('x1', x(x0))
        .attr('y1', y(intercept + slope * x0))
        .attr('x2', x(x1))
        .attr('y2', y(intercept + slope * x1))
        .attr('opacity', visible ? 1 : 0);
    }
  }

  function setResiduals(predict, { colorBySign = false } = {}) {
    if (!predict) {
      residualsG.selectAll('line').transition(t()).attr('opacity', 0).remove();
      return;
    }
    residualsG
      .selectAll('line')
      .data(pts, (d, i) => i)
      .join(
        (enter) =>
          enter
            .append('line')
            .attr('x1', (d) => x(d.hp))
            .attr('x2', (d) => x(d.hp))
            .attr('y1', (d) => y(d.mpg))
            .attr('y2', (d) => y(d.mpg))
            .attr('opacity', 0),
        (update) => update,
        (exit) => exit.remove()
      )
      .transition(t())
      .attr('x1', (d) => x(d.hp))
      .attr('x2', (d) => x(d.hp))
      .attr('y1', (d) => y(d.mpg))
      .attr('y2', (d) => y(predict(d.hp)))
      .attr('stroke', (d) =>
        colorBySign ? (d.mpg - predict(d.hp) >= 0 ? 'var(--sky)' : 'var(--cosmos)') : 'var(--cosmos)'
      )
      .attr('stroke-width', 1.6)
      .attr('opacity', 0.55);
  }

  function setDotColors(predict, colorBySign) {
    dots
      .transition(t())
      .attr('fill', (d) => {
        if (!colorBySign || !predict) return 'var(--smile)';
        return d.mpg - predict(d.hp) >= 0 ? 'var(--sky)' : 'var(--cosmos)';
      });
  }

  function setPrediction(visible) {
    predG.selectAll('*').remove();
    if (!visible) return;
    const hp0 = 150;
    const yhat = fit.intercept + fit.slope * hp0;
    const dash = predG.append('g').attr('opacity', 0);
    dash
      .append('line')
      .attr('x1', x(hp0)).attr('y1', y(0))
      .attr('x2', x(hp0)).attr('y2', y(yhat))
      .attr('stroke', 'var(--anchor)').attr('stroke-width', 2).attr('stroke-dasharray', '6 5');
    dash
      .append('line')
      .attr('x1', x(hp0)).attr('y1', y(yhat))
      .attr('x2', x(35)).attr('y2', y(yhat))
      .attr('stroke', 'var(--anchor)').attr('stroke-width', 2).attr('stroke-dasharray', '6 5');
    dash
      .append('circle')
      .attr('cx', x(hp0)).attr('cy', y(yhat)).attr('r', 8)
      .attr('fill', 'var(--anchor)').attr('stroke', 'white').attr('stroke-width', 2.5);
    dash
      .append('text')
      .attr('class', 'chart-annotation')
      .attr('x', x(42)).attr('y', y(yhat) - 10)
      .text(`${yhat.toFixed(1)} mpg`);
    dash
      .append('text')
      .attr('class', 'chart-annotation')
      .attr('x', x(hp0) + 8).attr('y', y(2))
      .text('150 hp');
    dash.transition(t()).attr('opacity', 1);
  }

  function setFlawAnnotations(visible) {
    annG.selectAll('.flaw').remove();
    if (!visible) return;
    const a1 = annG.append('text').attr('class', 'chart-annotation flaw')
      .attr('x', x(48)).attr('y', y(45)).attr('opacity', 0);
    a1.append('tspan').text('too pessimistic here');
    const a2 = annG.append('text').attr('class', 'chart-annotation flaw')
      .attr('x', x(120)).attr('y', y(6)).attr('opacity', 0);
    a2.append('tspan').text('too optimistic here');
    annG.selectAll('.flaw').transition(t()).attr('opacity', 1);
  }

  function setReadout(text) {
    readout.text(text);
  }

  const olsPredict = (v) => fit.intercept + fit.slope * v;
  const meanPredict = () => yMean;

  const handlers = {
    0: () => {
      setLine(0, yMean, false);
      setResiduals(null);
      setDotColors(null, false);
      setPrediction(false);
      setFlawAnnotations(false);
      setReadout('the auto mpg dataset · n = 392');
    },
    1: () => {
      setLine(0, yMean, true);
      setResiduals(null);
      setDotColors(null, false);
      setPrediction(false);
      setFlawAnnotations(false);
      setReadout(`model: y = ${yMean.toFixed(1)} (the mean)`);
    },
    2: () => {
      setLine(0, yMean, true);
      setResiduals(meanPredict);
      setDotColors(null, false);
      setPrediction(false);
      setFlawAnnotations(false);
      setReadout(`mse = ${mseMean.toFixed(1)}`);
    },
    3: () => {
      setLine(fit.slope, fit.intercept, true);
      setResiduals(olsPredict);
      setDotColors(null, false);
      setPrediction(false);
      setFlawAnnotations(false);
      setReadout(`mse = ${mseOLS.toFixed(1)} · was ${mseMean.toFixed(1)}`);
    },
    4: () => {
      setLine(fit.slope, fit.intercept, true);
      setResiduals(null);
      setDotColors(null, false);
      setPrediction(false);
      setFlawAnnotations(false);
      setReadout(`y = ${fit.intercept.toFixed(2)} − ${Math.abs(fit.slope).toFixed(3)}·hp · r² = 0.61`);
    },
    5: () => {
      setLine(fit.slope, fit.intercept, true);
      setResiduals(null);
      setDotColors(null, false);
      setPrediction(true);
      setFlawAnnotations(false);
      setReadout('prediction is just plugging in');
    },
    6: () => {
      setLine(fit.slope, fit.intercept, true);
      setResiduals(olsPredict, { colorBySign: true });
      setDotColors(olsPredict, true);
      setPrediction(false);
      setFlawAnnotations(true);
      setReadout('the errors have a pattern');
    },
  };

  handlers[0]();
  scrolly('#scrolly .step', handlers);
}
