/* Five ways to spend bits on the same held out pictures, revealed one at a time.
 *
 * Every bar is a real code length measured on the same 8 bit images: the raw
 * file, the entropy of the pixel histogram, an independent model per pixel,
 * gzip and PNG as they actually compress the bytes, and the flow's own exact
 * likelihood converted to bits. Nothing here is a published figure from
 * somewhere else.
 */
import { makeChart, drawAxes, drawGrid, axisLabels, wrapLabel } from '../../assets/js/chart.js';
import { scrolly } from '../../assets/js/scrolly.js';

export function initScrollyGlow(data) {
  const C = data.compress;
  const bars = [
    { key: 'raw', label: 'the file, uncompressed', v: C.raw },
    { key: 'histogram', label: 'the entropy of the greys', v: C.histogram },
    { key: 'per_pixel', label: 'a model per pixel, independent', v: C.per_pixel },
    { key: 'gzip', label: 'gzip', v: C.gzip },
    { key: 'png', label: 'PNG', v: C.png },
    { key: 'flow', label: 'this flow', v: C.flow },
  ];
  const chart = makeChart('#glow-chart', {
    width: 560, height: 440, margin: { top: 52, right: 26, bottom: 62, left: 200 },
  });
  const { g, w, h } = chart;
  const y = d3.scaleBand().domain(bars.map((b) => b.key)).range([0, h]).padding(0.3);
  const x = d3.scaleLinear().domain([0, Math.max(...bars.map((b) => b.v)) * 1.08]).range([0, w]);

  const layer = g.append('g');
  const title = g.append('text').attr('class', 'chart-note')
    .attr('x', w / 2).attr('y', -32).attr('text-anchor', 'middle')
    .style('font-size', '12px');

  /* which bars are visible at each step: the last one adds gzip and PNG together
     because they are the same idea, and the flow arrives alone */
  const REVEAL = [1, 2, 3, 5, 6];

  function render(step) {
    const n = REVEAL[Math.min(step, REVEAL.length - 1)];
    const shown = bars.slice(0, n);
    drawGrid(g, x, y, w, h, { xTicks: 5, yValues: [] });
    drawAxes(g, x, y, w, h, { xTicks: 5, yValues: [] });
    /* the axis label is set in bold uppercase mono with a point of letter
       spacing, so it costs about nine pixels a character: the long version of
       this sentence runs off the panel, and the held out set is said in prose */
    axisLabels(g, w, h, { x: 'bits per pixel, held out', y: '' });

    layer.selectAll('rect').data(shown, (b) => b.key).join('rect')
      .attr('x', 0).attr('y', (b) => y(b.key)).attr('height', y.bandwidth())
      .attr('width', (b) => x(b.v))
      .attr('fill', (b) => (b.key === 'flow' ? 'var(--primary)' : 'var(--squidink)'))
      .attr('opacity', (b) => (b.key === 'flow' ? 1 : 0.55));
    layer.selectAll('text.v').data(shown, (b) => b.key).join('text')
      .attr('class', 'chart-note v')
      .attr('x', (b) => x(b.v) + 7).attr('y', (b) => y(b.key) + y.bandwidth() / 2 + 4)
      .style('font-size', '11.5px').text((b) => b.v.toFixed(4));
    layer.selectAll('text.l').data(shown, (b) => b.key).join('text')
      .attr('class', 'chart-note l')
      .attr('x', -10).attr('y', (b) => y(b.key) + y.bandwidth() / 2 + 4)
      .attr('text-anchor', 'end').style('font-size', '11.5px').text((b) => b.label)
      /* Measured against the margin that actually exists rather than assumed to
       * fit it. "a model per pixel, independent" needs 219 units and the left
       * margin is 200, so it hung 29 units off the left edge of the canvas at
       * every width. A row label is data, so its length is not something the
       * margin can be planned around: place it, ask, and drop words. */
      .each(function wrapRowLabel(b) {
        const room = chart.margin.left - 12;
        if (this.getComputedTextLength() <= room) return;
        const parts = String(b.label).split(' ');
        while (parts.length > 1) {
          parts.pop();
          this.textContent = parts.join(' ');
          if (this.getComputedTextLength() <= room) return;
        }
      });

    const labels = [
      'eight bits a pixel, which is what the file costs before anyone looks at it',
      'most greys never appear, and counting them is already worth something',
      'and counting them separately for each pixel is worth a lot more',
      'then two programs that have been shipping since the nineties',
      'and the flow, whose bits are an exact log likelihood divided by the pixels',
    ];
    wrapLabel(title, labels[step], w - 8);
  }

  render(0);
  const handlers = {};
  for (let i = 0; i < 5; i++) handlers[i] = () => render(i);
  const ctl = scrolly('#glow-scrolly .step', handlers);
  return { render, goTo: ctl.goTo };
}
