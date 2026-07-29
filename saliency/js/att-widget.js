/* Attention: the map, and the two things it can and cannot be read for.
 *
 * The scatter of a map statistic against measured importance is the honest
 * version of the usual figure. A heatmap of one sentence tells the reader what
 * a head looks at; only the scatter says whether looking at it would have told
 * them anything.
 */
import { makeChart, drawAxes, drawGrid, axisLabels } from '../../assets/js/chart.js';

/* Ver la nota de score-widget.js: annotate() va en versalitas y con espaciado,
 * así que las frases largas se salen del lienzo. */
function note(g, x, y, text, { anchor = 'start', size = 11.5 } = {}) {
  return g.append('text').attr('class', 'chart-note')
    .attr('x', x).attr('y', y).attr('text-anchor', anchor)
    .style('font-size', `${size}px`)
    .text(text);
}

const f3 = (v) => v.toFixed(3);
const f4 = (v) => v.toFixed(4);
const sgn = (v, d = 3) => (v >= 0 ? `+${v.toFixed(d)}` : v.toFixed(d));

/* El tamaño del grupo más grande de cabezas que caben dentro de una sola marca
 * del retículo de 1/fichas: contarlo evita escribir "siete de ocho" a mano y
 * que deje de ser verdad en la siguiente tirada. */
function crowded(A) {
  const step = 1 / A.tokens;
  const v = A.rows.map((r) => r.published).sort((a, b) => a - b);
  let best = 1;
  for (let i = 0; i < v.length; i++) {
    /* Hacia arriba solamente. Contando también las diferencias negativas, la
     * ventana que arranca en el valor más alto se traga las ocho cabezas y el
     * texto decía "8 de 8" sobre una tabla cuyo rango es tres marcas. */
    let k = 0;
    for (let j = i; j < v.length; j++) if (v[j] - v[i] <= step) k += 1;
    best = Math.max(best, k);
  }
  return best;
}

export function initAttWidget(data) {
  const host = document.querySelector('#att-widget');
  if (!host) return;
  const A = data.attention;
  const chart = makeChart('#att-chart', {
    width: 640, height: 430, margin: { top: 48, right: 34, bottom: 62, left: 84 },
  });
  const { g, w, h } = chart;
  const readout = document.querySelector('#att-readout');
  const views = d3.select('#att-views');
  let view = 'map';

  const buttons = [
    ['map', 'one head, one sentence'],
    ['read', 'can the map be read for importance'],
    ['small', 'the same on eight sentences'],
  ];
  views.selectAll('button')
    .data(buttons)
    .join('button')
    .attr('class', (d) => (d[0] === view ? 'atlas-btn' : 'atlas-btn ghost'))
    .text((d) => d[1])
    .on('click', (_, d) => { view = d[0]; render(); });

  /* La cabeza importante es la que el 48 midió sobre su test entero, no la que
   * sale de ablacionar sobre las ocho frases de ejemplo: con 95 fichas la
   * exactitud sólo se mueve de 0,0105 en 0,0105 y las caídas publicadas son
   * más pequeñas que eso. La vista "small" es exactamente ese error, dibujado. */
  const best = [...A.rows].sort((a, b) => b.published - a.published)[0];
  const accent = getComputedStyle(document.documentElement)
    .getPropertyValue('--primary').trim() || '#853c6f';

  function render() {
    views.selectAll('button')
      .attr('class', (d) => (d[0] === view ? 'atlas-btn' : 'atlas-btn ghost'));
    g.selectAll('*').remove();

    if (view === 'map') {
      const key = `${best.layer}-${best.head}`;
      const mat = A.example.attn[key];
      const words = A.example.words;
      const n = Math.min(words.length, mat.length);
      const cell = Math.min(w, h) / n;
      const x0 = (w - cell * n) / 2;
      const scale = d3.scaleSequential(d3.interpolateLab('#f1f3f3', accent)).domain([0, 1]);
      for (let i = 0; i < n; i++) {
        for (let j = 0; j < n; j++) {
          g.append('rect')
            .attr('x', x0 + j * cell).attr('y', i * cell)
            .attr('width', cell - 0.5).attr('height', cell - 0.5)
            .attr('fill', scale(Math.min(1, mat[i][j])));
        }
      }
      /* Only every other word gets a label: at eighteen tokens the ticks
       * collide with each other long before they leave the canvas. */
      for (let i = 0; i < n; i += 2) {
        g.append('text').attr('class', 'chart-note')
          .attr('x', x0 - 6).attr('y', i * cell + cell / 2 + 4)
          .attr('text-anchor', 'end').style('font-size', '10.5px')
          .text(words[i]);
      }
      note(g, x0, -16, `layer ${best.layer}, head ${best.head}: the one that matters`,
        { size: 12 });
      note(g, x0, h + 24, 'row: the word doing the looking');
      note(g, x0, h + 39, 'column: the word looked at');
      readout.innerHTML =
        `<p>This is the picture that gets published: one head, one sentence, dark where the `
        + `weight is large. It is the head that matters most in this model, and the article `
        + `that trained it knows that by a different route, which is removing it and watching `
        + `the accuracy fall by <span class="bold">${f4(best.published)}</span> over `
        + `${A.published_n.toLocaleString('en-US')} test sentences. Reading the map, you would `
        + `say it looks mostly at the previous word (${f3(best.previous)} of its weight) and `
        + `that it is the most peaked of the ${A.rows.length} (${f3(best.peak)}). Both of `
        + `those readings are correct, and this is the head they point at. The next view asks `
        + `whether they point at the right head when there are eight to rank.</p>`;
    } else if (view === 'read' || view === 'small') {
      const small = view === 'small';
      const pts = A.rows.map((r) => [r.peak, small ? -r.ablation : r.published, r]);
      const x = d3.scaleLinear().domain(d3.extent(pts, (p) => p[0])).nice().range([0, w]);
      const y = d3.scaleLinear().domain([Math.min(0, d3.min(pts, (p) => p[1])) * 1.1,
        d3.max(pts, (p) => p[1]) * 1.12]).nice().range([h, 0]);
      drawGrid(g, x, y, w, h);
      drawAxes(g, x, y, w, h);
      axisLabels(g, w, h, {
        x: 'how peaked the head attention is',
        y: small ? 'cost on eight sentences' : 'cost on the whole test set',
      });
      g.append('line').attr('x1', 0).attr('x2', w).attr('y1', y(0)).attr('y2', y(0))
        .attr('stroke', 'var(--squidink)').attr('stroke-width', 1.1)
        .attr('stroke-dasharray', '4 4');
      /* Seis cabezas caen en un pañuelo, así que la etiqueta busca sitio en vez
       * de tener dos posiciones. La primera versión sólo sabía subir o bajar y
       * dejó "1-1" encima de "1-2": las dos habían bajado para esquivar a
       * "1-0". Se prueban candidatos por orden y se acepta el primero que no
       * pise una caja ya colocada, con el ancho medido y no supuesto. */
      const placed = [];
      const CAND = [[0, -13], [0, 20], [27, 4], [-27, 4], [0, -28], [0, 35], [27, -13],
        [-27, -13]];
      pts.forEach(([px, py, r]) => {
        g.append('circle').attr('cx', x(px)).attr('cy', y(py)).attr('r', 6)
          .attr('fill', r === best ? 'var(--cosmos)' : 'var(--primary)')
          .attr('fill-opacity', 0.85);
        const t = note(g, x(px), y(py) - 13, `${r.layer}-${r.head}`,
          { anchor: 'middle', size: 11 });
        const tw = t.node().getComputedTextLength();
        const boxAt = ([dx, dy]) => ({
          x0: x(px) + dx - tw / 2 - 2, x1: x(px) + dx + tw / 2 + 2,
          y0: y(py) + dy - 11, y1: y(py) + dy + 3,
        });
        const free = (b) => !placed.some((p) => b.x0 < p.x1 && p.x0 < b.x1
          && b.y0 < p.y1 && p.y0 < b.y1);
        const spot = CAND.find((c) => free(boxAt(c))) ?? CAND[0];
        placed.push(boxAt(spot));
        t.attr('x', x(px) + spot[0]).attr('y', y(py) + spot[1]);
      });
      const C = small ? A.corrs_small : A.corrs;
      note(g, 0, -16, `rank correlation ${sgn(C.peak)}`, { size: 12 });
      if (small) {
        /* Las líneas del retículo de 1/95 son el argumento entero de esta
         * vista: los ocho puntos no pueden caer entre ellas. */
        const step = 1 / A.tokens;
        d3.range(Math.ceil(y.domain()[0] / step), Math.floor(y.domain()[1] / step) + 1)
          .forEach((k) => {
            g.append('line').attr('x1', 0).attr('x2', w)
              .attr('y1', y(k * step)).attr('y2', y(k * step))
              .attr('stroke', 'var(--cosmos)').attr('stroke-width', 1)
              .attr('stroke-opacity', 0.45).attr('stroke-dasharray', '3 5');
          });
        note(g, w, -16, `the ruler has ${A.tokens} marks`, { anchor: 'end' })
          .attr('fill', 'var(--cosmos)');
      }
      readout.innerHTML = small
        ? `<p>Same heads, same statistic, importance measured by ablating on the `
          + `${A.sentences} example sentences the attention article's file ships: `
          + `${A.tokens} tags. Now the rank correlation is `
          + `<span class="bold">${sgn(A.corrs_small.peak)}</span> and the picture looks like `
          + `evidence. The dashed rules are why it is not. Accuracy over ${A.tokens} tags can `
          + `only take values ${(1 / A.tokens).toFixed(4)} apart, so every dot has to sit on `
          + `a rule, while the differences being ranked run from `
          + `${f4(Math.min(...A.rows.map((r) => r.published)))} to `
          + `${f4(Math.max(...A.rows.map((r) => r.published)))} and `
          + `<span class="bold">${crowded(A)} of the ${A.rows.length} heads</span> sit inside `
          + `a single mark of each other. The correlation is measuring where the rounding `
          + `happened to land.</p>`
        : `<p>Each dot is a head: how peaked its attention is against how much removing it `
          + `costs, that second number measured by the attention article over `
          + `${A.published_n.toLocaleString('en-US')} test sentences. If a map could be read `
          + `for importance these would lie on a line. The rank correlation is `
          + `<span class="bold">${sgn(A.corrs.peak)}</span> over ${A.rows.length} heads, with `
          + `entropy at ${sgn(A.corrs.entropy)} and weight on the previous word at `
          + `${sgn(A.corrs.previous)}. The one head everybody would get right is the one in `
          + `red, top right: most peaked and most important. Every other pair is a coin flip. `
          + `${A.adversarial ? `And there is a harder limit underneath it. A pattern `
            + `<span class="bold">${f3(A.adversarial.tv)}</span> away in total variation from `
            + `the real one, which is most of the way across the space, leaves every predicted `
            + `tag unchanged and moves the largest logit by ${f3(A.adversarial.logit_shift)}.`
            : ''}</p>`;
    }
  }

  render();
}
