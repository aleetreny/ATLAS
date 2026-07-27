/* The obvious explanation, tested.
 *
 * Ask anyone why a deep network trains badly and you will hear "the gradients
 * vanish". It is a real phenomenon and it is not what is happening here, and
 * the only way to know that is to measure the gradient rather than assume it.
 * These are the norms of the gradient arriving at every convolution, in the
 * networks whose training accuracy the chart above showed going the wrong way.
 */
import { makeChart, drawAxes, drawGrid, axisLabels } from '../../assets/js/chart.js';

export function initGradientWidget(data) {
  const profiles = data.grad_profiles;
  const keys = Object.keys(profiles);
  const deepest = data.ladder[data.ladder.length - 1].depth;

  const W = 700;
  const H = 380;
  const margin = { top: 30, right: 172, bottom: 54, left: 74 };
  const { g, w, h } = makeChart('#gradient-chart', { width: W, height: H, margin });

  const series = [
    { key: `plain${deepest}`, label: `${deepest} plain`, colour: '#232f3e', dash: null },
    { key: `residual${deepest}`, label: `${deepest} with shortcuts`, colour: 'var(--primary)', dash: null },
    { key: 'plain20', label: '20 plain', colour: '#8a94a6', dash: '5 4' },
  ].filter((s) => profiles[s.key]);

  const maxLen = Math.max(...series.map((s) => profiles[s.key].length));
  const all = series.flatMap((s) => profiles[s.key]).filter((v) => v > 0);

  const x = d3.scaleLinear().domain([1, maxLen]).range([0, w]);
  const y = d3.scaleLog().domain([Math.min(...all) * 0.7, Math.max(...all) * 1.4]).range([h, 0]);

  drawGrid(g, x, y, w, h, { xTicks: 6, yTicks: 5 });
  drawAxes(g, x, y, w, h, { xTicks: 6, yTicks: 5, xFmt: d3.format('d'), yFmt: d3.format('.0e') });
  axisLabels(g, w, h, { x: 'convolution, counted from the input', y: 'gradient norm' });

  const line = d3.line().x((d, i) => x(i + 1)).y((d) => y(d));

  series.forEach((s, i) => {
    g.append('path')
      .datum(profiles[s.key])
      .attr('fill', 'none')
      .attr('stroke', s.colour)
      .attr('stroke-width', 2.2)
      .attr('stroke-dasharray', s.dash)
      .attr('d', line);
    g.append('text')
      .attr('x', w + 10)
      .attr('y', 16 + i * 20)
      .attr('class', 'annotation')
      .style('font-size', '12px')
      .style('fill', s.colour)
      .text(s.label);
  });

  /* The number the vanishing-gradient story needs: what happens to the
     gradient at layer one as the network deepens, at initialisation, which is
     the moment the hypothesis is about. It can go three ways and the readout
     has to be able to say all three. */
  const shallow = data.ladder[0];
  const deep = data.ladder[data.ladder.length - 1];
  const gLo = shallow.plain.grad_first_init;
  const gHi = deep.plain.grad_first_init;
  const shrink = gLo / gHi;

  let claim;
  if (shrink > 5) {
    claim = `That is a genuine collapse, ${shrink.toFixed(1)} times smaller, and the `
      + `vanishing-gradient story survives this test.`;
  } else if (shrink > 1.2) {
    claim = `Smaller, by ${shrink.toFixed(1)} times, but nothing like the orders of magnitude `
      + `the vanishing-gradient story needs.`;
  } else if (shrink > 0.8) {
    claim = `Which is to say it barely moved. The signal is arriving, and something else is wrong.`;
  } else {
    claim = `It did not shrink at all: it <span class="bold">grew</span>, by `
      + `${(1 / shrink).toFixed(0)} times. Whatever is stopping these networks, it is not that `
      + `the gradient fails to reach the early layers, because at initialisation the deeper the `
      + `network the louder that gradient is.`;
  }

  /* After training the picture can be different again, and it is: whatever
     the number turns out to be, it gets described rather than assumed. */
  const tHi = deep.plain.grad_first;
  const tLo = shallow.plain.grad_first;
  const tRatio = tLo / tHi;
  const trainedWord = tRatio > 2
    ? `${tRatio.toFixed(0)} times smaller at the deep end, which is a real shrinking, though it is `
      + `a description of a network that has already stopped descending rather than an explanation `
      + `of why it stopped`
    : (tRatio < 0.5
      ? `${(1 / tRatio).toFixed(0)} times larger at the deep end`
      : `within a factor of two of each other`);

  document.querySelector('#gradient-readout').innerHTML =
    `<span class="bold">At initialisation, the gradient reaching the first convolution</span> averages `
    + `${gHi.toExponential(2)} in the ${deep.depth}-layer plain network against `
    + `${gLo.toExponential(2)} in the ${shallow.depth}-layer one. ${claim} After training the same `
    + `measurement gives ${tHi.toExponential(2)} and ${tLo.toExponential(2)}: ${trainedWord}. `
    + `Batch normalisation is in every one of these networks, plain and residual alike, which is `
    + `exactly the point: it was already the standard answer to vanishing gradients by 2015, and the `
    + `degradation happened anyway.`;
}
