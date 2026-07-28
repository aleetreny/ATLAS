/* Six steps from a table of numbers back to a map, on one canvas that changes
 * what it is showing rather than six charts stacked up.
 *
 * The two matrix steps are drawn as an <image> built from an offscreen canvas
 * rather than as 31,684 SVG rectangles. A 178 by 178 heatmap is a picture, and
 * asking the browser to lay out one node per cell costs about a second for
 * something the reader looks at for four.
 */
import { scrolly } from '../../assets/js/scrolly.js';
import { makeChart, drawAxes, drawGrid, axisLabels, wrapLabel, equalScales } from '../../assets/js/chart.js';

const CULTIVAR = ['var(--primary)', 'var(--aqua)', 'var(--smile)'];

/* An n by n matrix as a data URL, one pixel per cell, drawn at whatever size
 * the SVG wants it. `image-rendering: pixelated` keeps the cells square instead
 * of letting the browser smooth them into a cloud. */
function matrixImage(M, colour) {
  const n = M.length;
  const cv = document.createElement('canvas');
  cv.width = n;
  cv.height = n;
  const ctx = cv.getContext('2d');
  const img = ctx.createImageData(n, n);
  for (let i = 0; i < n; i++) {
    for (let j = 0; j < n; j++) {
      const c = d3.rgb(colour(M[i][j]));
      const o = (i * n + j) * 4;
      img.data[o] = c.r;
      img.data[o + 1] = c.g;
      img.data[o + 2] = c.b;
      img.data[o + 3] = 255;
    }
  }
  ctx.putImageData(img, 0, 0);
  return cv.toDataURL();
}

export function initScrollyMds(data, computed) {
  const W = data.wine;
  const n = W.n;
  const chart = makeChart('#mds-chart', {
    width: 640, height: 516, margin: { top: 62, right: 26, bottom: 88, left: 62 },
  });
  const { g, w, h } = chart;

  const layers = {
    table: g.append('g').attr('opacity', 0),
    heat: g.append('g').attr('opacity', 0),
    bars: g.append('g').attr('opacity', 0),
    map: g.append('g').attr('opacity', 0),
  };
  const title = g.append('text').attr('class', 'chart-annotation')
    .attr('x', w / 2).attr('y', -42).attr('text-anchor', 'middle')
    .style('font-size', '12px').style('text-transform', 'none');
  const caption = g.append('text').attr('class', 'chart-annotation')
    .attr('x', w / 2).attr('y', h + 64).attr('text-anchor', 'middle')
    .style('font-size', '11.5px').style('text-transform', 'none');

  /* ------------------------------------------------------- step 0: the table */
  {
    const labels = W.small_labels;
    const m = labels.length;
    const cell = Math.min(64, (w - 70) / m);
    const x0 = (w - cell * m) / 2 + 34;
    const y0 = 34;
    const t = layers.table;
    labels.forEach((lab, j) => {
      t.append('text').attr('x', x0 + cell * (j + 0.5)).attr('y', y0 - 10)
        .attr('text-anchor', 'middle').style('font-size', '12px')
        .style('font-family', 'var(--font-mono)').attr('fill', CULTIVAR[Math.floor(j / 2)])
        .text(lab);
      t.append('text').attr('x', x0 - 10).attr('y', y0 + cell * (j + 0.5) + 4)
        .attr('text-anchor', 'end').style('font-size', '12px')
        .style('font-family', 'var(--font-mono)').attr('fill', CULTIVAR[Math.floor(j / 2)])
        .text(lab);
    });
    const maxD = d3.max(W.small_D.flat());
    for (let i = 0; i < m; i++) {
      for (let j = 0; j < m; j++) {
        const v = W.small_D[i][j];
        t.append('rect').attr('x', x0 + cell * j).attr('y', y0 + cell * i)
          .attr('width', cell - 2).attr('height', cell - 2)
          .attr('fill', i === j ? 'var(--paper)' : d3.interpolateRgb('#ffffff', 'var(--primary)')(v / maxD * 0.85))
          .attr('stroke', 'var(--stone)').attr('stroke-width', 0.6);
        t.append('text').attr('x', x0 + cell * j + cell / 2 - 1)
          .attr('y', y0 + cell * i + cell / 2 + 3).attr('text-anchor', 'middle')
          .style('font-size', '11px').style('font-family', 'var(--font-mono)')
          .attr('fill', v / maxD > 0.55 ? 'white' : 'var(--squidink)')
          .text(v.toFixed(2));
      }
    }
  }

  /* ------------------------------------------- steps 1 and 2: the two heatmaps */
  const heatSize = Math.min(w, h) - 12;
  const heatX = (w - heatSize) / 2;
  const imgNode = layers.heat.append('image')
    .attr('x', heatX).attr('y', 4)
    .attr('width', heatSize).attr('height', heatSize)
    .attr('preserveAspectRatio', 'none')
    .style('image-rendering', 'pixelated');
  /* Cultivar ribbons down the side, so the block structure has a name. */
  {
    const bandW = 7;
    const cnt = [0, 0, 0];
    W.classes.forEach((c) => { cnt[c] += 1; });
    let acc = 0;
    cnt.forEach((c, i) => {
      const y = 4 + (acc / n) * heatSize;
      const hh = (c / n) * heatSize;
      layers.heat.append('rect').attr('x', heatX - bandW - 2).attr('y', y)
        .attr('width', bandW).attr('height', hh).attr('fill', CULTIVAR[i]);
      layers.heat.append('rect').attr('x', heatX + (acc / n) * heatSize).attr('y', 4 + heatSize + 2)
        .attr('width', hh).attr('height', bandW).attr('fill', CULTIVAR[i]);
      acc += c;
    });
  }

  const maxD = d3.max(computed.D.map((r) => d3.max(r)));
  const imgD = matrixImage(computed.D, (v) => d3.interpolateRgb('#ffffff', '#7e22ce')(v / maxD));
  const maxB = d3.max(computed.B.map((r) => d3.max(r.map(Math.abs))));
  const imgB = matrixImage(computed.B, (v) => (v >= 0
    ? d3.interpolateRgb('#ffffff', '#7e22ce')(v / maxB)
    : d3.interpolateRgb('#ffffff', '#003181')(-v / maxB)));

  /* ------------------------------------------------- step 3: the eigenvalues */
  {
    const K = 24;
    const vals = Array.from(computed.values.slice(0, K));
    const x = d3.scalePoint().domain(d3.range(1, K + 1)).range([18, w - 18]);
    const y = d3.scaleLinear().domain([0, d3.max(vals) * 1.06]).range([h, 0]);
    const bw = Math.min(18, (w - 36) / K - 3);
    layers.bars.append('g').attr('class', 'eig-axes');
    layers.bars.selectAll('rect').data(vals).join('rect')
      .attr('x', (d, i) => x(i + 1) - bw / 2).attr('width', bw)
      .attr('y', (d) => y(Math.max(d, 0)))
      .attr('height', (d) => h - y(Math.max(d, 0)))
      .attr('fill', (d, i) => (i < 2 ? 'var(--primary)' : 'var(--stone)'))
      .attr('stroke', 'var(--squidink)').attr('stroke-width', 0.6);
    layers.barsScales = { x, y, K };
  }

  /* ---------------------------------------------- steps 4 and 5: the two maps */
  const mapDots = layers.map.append('g');
  const mapRings = layers.map.append('g').attr('opacity', 0);
  layers.mapReady = false;

  function drawMap(showPca) {
    const Z = computed.mds2;
    const P = computed.pca2;
    const { x, y } = equalScales(Z, w, h);
    drawGrid(layers.map, x, y, w, h, { xTicks: 5, yTicks: 5 });
    drawAxes(layers.map, x, y, w, h, { xTicks: 5, yTicks: 5 });
    axisLabels(layers.map, w, h, { x: 'first coordinate', y: 'second coordinate' });
    mapDots.selectAll('circle').data(Z).join('circle')
      .attr('cx', (d) => x(d[0])).attr('cy', (d) => y(d[1])).attr('r', 4)
      .attr('fill', (d, i) => CULTIVAR[W.classes[i]])
      .attr('opacity', 0.82);
    mapRings.selectAll('circle').data(P).join('circle')
      .attr('cx', (d) => x(d[0])).attr('cy', (d) => y(d[1])).attr('r', 8)
      .attr('fill', 'none').attr('stroke', 'var(--squidink)').attr('stroke-width', 1.4);
    mapRings.transition('rings').duration(600).attr('opacity', showPca ? 0.95 : 0);
    layers.mapReady = true;
  }

  function show(which) {
    for (const [k, layer] of Object.entries(layers)) {
      if (!layer || typeof layer.transition !== 'function') continue;
      layer.transition(`fade-${k}`).duration(450).attr('opacity', k === which ? 1 : 0);
    }
  }

  const steps = [
    () => {
      show('table');
      wrapLabel(title, 'the distance between six of the 178 bottles, in standardised units', w - 8);
      wrapLabel(caption, 'two bottles from each cultivar, and nothing else about them', w - 8);
    },
    () => {
      show('heat');
      imgNode.attr('href', imgD);
      wrapLabel(title, `all ${n} by ${n} distances, dark for far apart`, w - 8);
      wrapLabel(caption, 'rows and columns ordered by cultivar, which is the only thing the colours know', w - 8);
    },
    () => {
      show('heat');
      imgNode.attr('href', imgB);
      wrapLabel(title, 'the same table double centred: inner products, not distances', w - 8);
      wrapLabel(caption, 'purple for a positive inner product, blue for a negative one', w - 8);
    },
    () => {
      show('bars');
      const L = layers.barsScales;
      drawGrid(layers.bars, L.x, L.y, w, h, { xTicks: 0, yTicks: 5 });
      drawAxes(layers.bars, L.x, L.y, w, h, { xTicks: L.K, yTicks: 5 });
      axisLabels(layers.bars, w, h, { x: 'eigenvalue', y: 'size' });
      wrapLabel(title, `the ${L.K} largest eigenvalues of that matrix`, w - 8);
      wrapLabel(caption, 'not one of the 178 comes back negative, which is a fact and not a coincidence', w - 8);
    },
    () => {
      show('map');
      if (!layers.mapReady) drawMap(false);
      else mapRings.transition('rings').duration(400).attr('opacity', 0);
      wrapLabel(title, 'the first two columns, drawn', w - 8);
      wrapLabel(caption, 'colour is the cultivar, which nothing in the calculation was ever told', w - 8);
    },
    () => {
      show('map');
      drawMap(true);
      wrapLabel(title, 'rings: the same 178 bottles put here by PCA on the original thirteen columns', w - 8);
      wrapLabel(caption, 'every ring has a dot inside it', w - 8);
    },
  ];

  steps[0]();
  return scrolly('#mds-scrolly .step', steps);
}
