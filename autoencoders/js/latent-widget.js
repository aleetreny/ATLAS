/* Two numbers, and the digit they decode to.
 *
 * This is the widget the whole module has been building towards, and the reason
 * it can exist at all is that an autoencoder leaves a function behind. Drag the
 * crosshair anywhere in the plane, including places no digit ever was, and the
 * decoder answers. Nothing else in module 2.2 can be asked that question: PCA
 * can, but its answer is a blur; Isomap, non-metric scaling and t-SNE have no
 * inverse to ask.
 */
import { makeChart, drawAxes, drawGrid, axisLabels, wrapLabel } from '../../assets/js/chart.js';
import { makeCanvas, paintGray } from '../../assets/js/imagery.js';
import { toRows } from '../../assets/js/embed.js';

const DIGIT = d3.scaleOrdinal(d3.schemeCategory10).domain(d3.range(10));

export function initLatentWidget(data, net) {
  const N = data.net;
  const side = data.meta.side;
  const Z = toRows(N.latent);
  const chart = makeChart('#latent-chart', {
    width: 460, height: 420, margin: { top: 56, right: 18, bottom: 52, left: 52 },
  });
  const { svg, g, w, h } = chart;
  svg.style('max-width', '470px').style('display', 'inline-block')
    .style('vertical-align', 'top');

  const holder = d3.select('#latent-canvas');
  const cv = makeCanvas(holder.node(), side * 12, side * 12);
  const gridHolder = d3.select('#latent-grid');
  const GRID = 7;
  const gcv = makeCanvas(gridHolder.node(), side * GRID * 3, side * GRID * 3);

  const buttons = document.querySelector('#latent-buttons');
  const readout = document.querySelector('#latent-readout');
  const lo = N.latent_lo;
  const hi = N.latent_hi;
  const pad = 0.12;
  const x = d3.scaleLinear()
    .domain([lo[0] - (hi[0] - lo[0]) * pad, hi[0] + (hi[0] - lo[0]) * pad]).range([0, w]);
  const y = d3.scaleLinear()
    .domain([lo[1] - (hi[1] - lo[1]) * pad, hi[1] + (hi[1] - lo[1]) * pad]).range([h, 0]);

  const st = { z: [(lo[0] + hi[0]) / 2, (lo[1] + hi[1]) / 2], show: 'all' };

  const btns = {};
  for (const [key, text] of [['all', 'every digit'], ['test', 'only the held out ones'],
    ['grid', 'sweep the whole square']]) {
    const b = document.createElement('button');
    b.className = 'atlas-btn' + (key === st.show ? '' : ' ghost');
    b.textContent = text;
    b.addEventListener('click', () => {
      st.show = key;
      render();
    });
    buttons.appendChild(b);
    btns[key] = b;
  }

  const dots = g.append('g');
  const cross = g.append('g');
  const title = g.append('text').attr('class', 'chart-annotation')
    .attr('x', w / 2).attr('y', -38).attr('text-anchor', 'middle')
    .style('font-size', '12px').style('text-transform', 'none');

  /* the decoded image at the crosshair */
  function paintOne() {
    const px = net.decode(st.z);
    const gray = new Float64Array(px.length);
    for (let i = 0; i < px.length; i++) gray[i] = 1 - px[i];
    paintGray(cv.ctx, gray, side, side, { zoom: 12 });
    return px;
  }

  /* the whole square at once, which is the picture that shows it is a function
     and not a lookup table */
  function paintGrid() {
    const big = new Float64Array(side * GRID * side * GRID);
    for (let r = 0; r < GRID; r++) {
      for (let c = 0; c < GRID; c++) {
        const zz = [
          lo[0] + ((hi[0] - lo[0]) * c) / (GRID - 1),
          hi[1] - ((hi[1] - lo[1]) * r) / (GRID - 1),
        ];
        const px = net.decode(zz);
        for (let a = 0; a < side; a++) {
          for (let b = 0; b < side; b++) {
            big[(r * side + a) * (side * GRID) + c * side + b] = 1 - px[a * side + b];
          }
        }
      }
    }
    paintGray(gcv.ctx, big, side * GRID, side * GRID, { zoom: 3 });
  }

  function render() {
    for (const [k, b] of Object.entries(btns)) b.classList.toggle('ghost', k !== st.show);
    drawGrid(g, x, y, w, h, { xTicks: 5, yTicks: 5 });
    drawAxes(g, x, y, w, h, { xTicks: 5, yTicks: 5 });
    axisLabels(g, w, h, { x: 'first number', y: 'second number' });
    const keep = st.show === 'test'
      ? Z.map((p, i) => [p, i]).filter(([, i]) => N.is_test[i])
      : Z.map((p, i) => [p, i]);
    dots.selectAll('circle').data(keep).join('circle')
      .attr('cx', (d) => x(d[0][0])).attr('cy', (d) => y(d[0][1]))
      .attr('r', st.show === 'test' ? 3 : 2.2)
      .attr('fill', (d) => DIGIT(N.labels[d[1]])).attr('opacity', 0.72);
    cross.selectAll('*').remove();
    cross.append('line').attr('x1', x(st.z[0])).attr('x2', x(st.z[0]))
      .attr('y1', 0).attr('y2', h).attr('stroke', 'var(--squidink)').attr('stroke-width', 1);
    cross.append('line').attr('y1', y(st.z[1])).attr('y2', y(st.z[1]))
      .attr('x1', 0).attr('x2', w).attr('stroke', 'var(--squidink)').attr('stroke-width', 1);
    cross.append('circle').attr('cx', x(st.z[0])).attr('cy', y(st.z[1])).attr('r', 8)
      .attr('fill', 'none').attr('stroke', 'var(--primary)').attr('stroke-width', 3);
    wrapLabel(title, st.show === 'test'
      ? `the ${data.meta.test} digits the fit never saw, put here by the encoder`
      : `all ${data.meta.n.toLocaleString('en-US')} digits, and one point you choose`, w - 8);

    paintOne();
    gridHolder.style('display', st.show === 'grid' ? 'block' : 'none');
    holder.style('display', st.show === 'grid' ? 'none' : 'block');
    if (st.show === 'grid') paintGrid();

    const G = data.generalisation;
    readout.innerHTML = `
      <table class="ae-table">
        <tr><th>the two numbers</th><th>fitted on</th><th>error there</th><th>error held out</th><th>5-nn on the code</th></tr>
        <tr>
          <td><span class="value">${st.z[0].toFixed(3)}, ${st.z[1].toFixed(3)}</span></td>
          <td>${data.meta.train.toLocaleString('en-US')} digits</td>
          <td>${G.train_mse}</td>
          <td><span class="value">${G.test_mse}</span></td>
          <td><span class="value">${(G.code_accuracy * 100).toFixed(1)}%</span></td>
        </tr>
      </table>
      <div class="ae-note">
        ${st.show === 'grid'
    ? `Every square is the decoder's answer at that point of the plane, and most of those points `
      + `are not any digit. The map is a <span class="bold">function</span>: it has a value `
      + `everywhere, it is smooth, and walking across it walks through shapes. That is the thing `
      + `no other method in this module produces, and it is the whole reason to accept a worse `
      + `reconstruction than PCA's cousin would give. The corners are worth looking at too: the `
      + `square is the box that holds the middle 98% of the codes, so its corners are places the `
      + `data never goes, and the decoder saturates there into solid ink. A function has a value `
      + `everywhere, and having a value is not the same as having a sensible one.`
    : `Drag the crosshair. The picture on the right is what the decoder makes of those two numbers, `
      + `and it will answer for coordinates that no digit in the dataset ever occupied. `
      + `${st.show === 'test'
    ? `These ${data.meta.test} digits were not in the fit: the encoder placed them by running a `
      + `function it had already learned, which is what "out of sample" means and what none of the `
      + `neighbour or manifold methods in this module can do at all.`
    : `Five nearest neighbours on these two numbers alone identifies `
      + `${(G.code_accuracy * 100).toFixed(1)}% of the held out digits, against `
      + `${(G.pixel_accuracy * 100).toFixed(1)}% on all ${data.meta.pixels} pixels.`}`}
      </div>`;
  }

  /* dragging: on the svg, in data coordinates, clamped to the drawn square */
  const surface = g.append('rect').attr('x', 0).attr('y', 0).attr('width', w).attr('height', h)
    .attr('fill', 'transparent').style('cursor', 'crosshair');
  const move = (event) => {
    const [px, py] = d3.pointer(event, g.node());
    st.z = [
      Math.max(x.domain()[0], Math.min(x.domain()[1], x.invert(px))),
      Math.max(y.domain()[1], Math.min(y.domain()[0], y.invert(py))),
    ];
    render();
  };
  surface.call(d3.drag().on('start drag', move));
  surface.on('click', move);

  render();
  return st;
}
