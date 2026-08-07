/* Wine chemistry: correlation heatmap + PCA-vs-PLS latent space morph. */
import { makeChart, axisLabels, tooltip } from '../../assets/js/chart.js';

const ABBREV = {
  alcohol: 'alcohol',
  malic_acid: 'malic',
  ash: 'ash',
  alcalinity_of_ash: 'alcal.',
  magnesium: 'mg',
  total_phenols: 'phenols',
  flavanoids: 'flavan.',
  nonflavanoid_phenols: 'nonflav.',
  proanthocyanins: 'proanth.',
  color_intensity: 'color',
  hue: 'hue',
  'od280/od315_of_diluted_wines': 'od280',
  proline: 'proline',
};

const CULTIVARS = ['Barolo', 'Grignolino', 'Barbera'];
const CLASS_COLORS = ['var(--anchor)', 'var(--primary)', 'var(--cosmos)'];

export function initPLSWidget(wine) {
  drawHeatmap(wine);
  drawLatentSpace(wine);
}

/* ---------------- correlation heatmap ---------------- */

function drawHeatmap(wine) {
  const n = wine.features.length;
  const labels = wine.features.map((f) => ABBREV[f] || f);
  // The column labels are rotated 45 degrees, so the rightmost one reaches well
  // past its own column. The canvas is wider than the grid to give it room:
  // at equal width the last label ("proline") was clipped by the svg edge.
  const W = 625;
  const H = 580;
  const pad = { top: 78, left: 78 };
  const cell = (560 - pad.left - 12) / n;

  const svg = d3
    .select('#corr-widget')
    .append('svg')
    .attr('viewBox', `0 0 ${W} ${H}`)
    .attr('preserveAspectRatio', 'xMidYMid meet')
    .style('width', '100%')
    .style('height', 'auto');

  svg
    .append('text')
    .attr('class', 'chart-title')
    .attr('x', pad.left + (cell * n) / 2)
    .attr('y', 18)
    .attr('text-anchor', 'middle')
    .text('13 features, one story told twice');

  const gg = svg.append('g').attr('transform', `translate(${pad.left},${pad.top})`);
  const color = (v) => d3.interpolateRdBu((1 - v) / 2);
  const tip = tooltip();

  for (let i = 0; i < n; i++) {
    for (let j = 0; j < n; j++) {
      gg.append('rect')
        .attr('x', j * cell)
        .attr('y', i * cell)
        .attr('width', cell - 1)
        .attr('height', cell - 1)
        .attr('fill', color(wine.corr[i][j]))
        .on('mousemove', (event) => {
          const v = wine.corr[i][j];
          tip.show(
            `<span style="color:var(--primary)">${labels[i]} × ${labels[j]}</span><br/>corr = ${v >= 0 ? '+' : ''}${v.toFixed(2)}`,
            event
          );
        })
        .on('mouseleave', () => tip.hide());
    }
  }

  // row labels (left) & column labels (rotated, top)
  gg.selectAll('text.row')
    .data(labels)
    .join('text')
    .attr('class', 'row')
    .attr('x', -6)
    .attr('y', (d, i) => i * cell + cell / 2 + 3)
    .attr('text-anchor', 'end')
    .style('font-family', 'var(--font-mono)')
    .style('font-size', '10.5px')
    .text((d) => d);

  gg.selectAll('text.col')
    .data(labels)
    .join('text')
    .attr('class', 'col')
    .attr('transform', (d, i) => `translate(${i * cell + cell / 2 + 3},-6) rotate(-65)`)
    .attr('text-anchor', 'start')
    .style('font-family', 'var(--font-mono)')
    .style('font-size', '10.5px')
    .text((d) => d);
}

/* ---------------- PCA vs PLS morphing scatter ---------------- */

function drawLatentSpace(wine) {
  const { g, w, h } = makeChart('#pls-chart', {
    width: 700,
    height: 520,
    margin: { top: 40, right: 24, bottom: 58, left: 58 },
  });

  const domains = {};
  for (const key of ['pca', 'pls']) {
    const xs = wine.points.map((d) => d[key][0]);
    const ys = wine.points.map((d) => d[key][1]);
    domains[key] = {
      x: [d3.min(xs) - 0.5, d3.max(xs) + 0.5],
      y: [d3.min(ys) - 0.5, d3.max(ys) + 0.5],
    };
  }

  const x = d3.scaleLinear().range([0, w]);
  const y = d3.scaleLinear().range([h, 0]);

  const gx = g.append('g').attr('class', 'axis x').attr('transform', `translate(0,${h})`);
  const gy = g.append('g').attr('class', 'axis y');
  axisLabels(g, w, h, { x: 'component 1', y: 'component 2' });

  const caption = g
    .append('text')
    .attr('class', 'chart-annotation')
    .attr('x', w / 2)
    .attr('y', -16)
    .attr('text-anchor', 'middle');

  const dots = g
    .append('g')
    .selectAll('circle')
    .data(wine.points)
    .join('circle')
    .attr('r', 5.5)
    .attr('fill', (d) => CLASS_COLORS[d.cls])
    .attr('stroke', 'white')
    .attr('stroke-width', 1.2)
    .attr('opacity', 0.9);

  // legend
  const legend = g.append('g').attr('transform', `translate(${w - 150},${h - 84})`);
  CULTIVARS.forEach((name, i) => {
    const row = legend.append('g').attr('transform', `translate(0,${i * 22})`);
    row.append('circle').attr('r', 6).attr('fill', CLASS_COLORS[i]).attr('stroke', 'white').attr('stroke-width', 1);
    row
      .append('text')
      .attr('x', 12)
      .attr('y', 4)
      .style('font-family', 'var(--font-mono)')
      .style('font-size', '12.5px')
      .style('fill', 'var(--squidink)')
      .text(name);
  });

  const btnPCA = document.querySelector('#pls-toggle-pca');
  const btnPLS = document.querySelector('#pls-toggle-pls');

  function show(key) {
    x.domain(domains[key].x);
    y.domain(domains[key].y);
    const t = d3.transition().duration(900).ease(d3.easeCubicInOut);
    gx.transition(t).call(d3.axisBottom(x).ticks(6).tickSizeOuter(0));
    gy.transition(t).call(d3.axisLeft(y).ticks(6).tickSizeOuter(0));
    dots
      .transition(t)
      .attr('cx', (d) => x(d[key][0]))
      .attr('cy', (d) => y(d[key][1]));
    caption.text(
      key === 'pca'
        ? 'pca: axes of maximum variance, never saw the labels'
        : 'pls: axes of maximum covariance with the label'
    );
    btnPCA.classList.toggle('ghost', key !== 'pca');
    btnPLS.classList.toggle('ghost', key === 'pca');
  }

  btnPCA.addEventListener('click', () => show('pca'));
  btnPLS.addEventListener('click', () => show('pls'));
  show('pca');
}
