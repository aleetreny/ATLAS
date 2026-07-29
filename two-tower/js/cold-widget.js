/* Warm and cold, on the same axes, for three towers that differ in one thing.
 *
 * The cold bars are the experiment: three hundred books with every training
 * rating deleted, ranked among themselves. An id embedding for a book nobody
 * has read is whatever the initialisation left there, so the ids only arm has
 * to score at chance, and it does; the interesting number is not that one, it
 * is how far above chance the content arms get, and what that costs them on
 * the warm half.
 */
import { makeChart, drawAxes, drawGrid, axisLabels } from '../../assets/js/chart.js';

const ARMS = [['ids', 'ids only'], ['content', 'content only'], ['both', 'both, id dropped half the time']];

export function initColdWidget(data) {
  const A = data.arms.arms;
  const buttons = document.querySelector('#cold-buttons');
  const readout = document.querySelector('#cold-readout');
  const st = { view: 'cold' };

  const chart = makeChart('#cold-chart', {
    width: 640, height: 380, margin: { top: 46, right: 30, bottom: 76, left: 74 }, clip: true,
  });
  const { g, plot, w, h, clear } = chart;

  const btns = {};
  for (const [key, text] of [['cold', 'the books nobody has read'], ['warm', 'the ordinary catalogue']]) {
    const b = document.createElement('button');
    b.className = 'atlas-btn ghost';
    b.textContent = text;
    b.addEventListener('click', () => { st.view = key; render(); });
    buttons.appendChild(b);
    btns[key] = b;
  }

  const pct = (v, d = 1) => `${(100 * v).toFixed(d)}%`;

  function render() {
    Object.entries(btns).forEach(([k, b]) => b.classList.toggle('ghost', k !== st.view));
    clear();
    g.selectAll('text.chart-note').remove();

    const cold = st.view === 'cold';
    const rows = ARMS.map(([key, label]) => ({
      key, label, v: cold ? A[key].cold.hits['10'] : A[key].warm.hits['10'],
    }));
    if (cold) rows.push({ key: 'popular', label: 'most rated of the three hundred', v: A.popular.cold.hits['10'] });

    const x = d3.scaleBand().domain(rows.map((r) => r.label)).range([0, w]).padding(0.34);
    const y = d3.scaleLinear()
      .domain([0, d3.max(rows, (r) => r.v) * 1.25]).range([h, 0]);
    drawGrid(g, x, y, w, h, { xValues: [], yTicks: 5 });

    plot.selectAll(null).data(rows).enter().append('rect')
      .attr('x', (d) => x(d.label)).attr('width', x.bandwidth())
      .attr('y', (d) => y(d.v)).attr('height', (d) => h - y(d.v))
      .attr('fill', (d) => (d.key === 'both' ? 'var(--primary)'
        : d.key === 'popular' ? 'var(--stone)' : 'var(--anchor)'));

    rows.forEach((d) => {
      g.append('text').attr('class', 'chart-note')
        .attr('x', x(d.label) + x.bandwidth() / 2).attr('y', y(d.v) - 8)
        .attr('text-anchor', 'middle').style('font-size', '11.5px')
        .text(pct(d.v));
      const words = d.label.split(' ');
      const lines = [];
      let cur = '';
      words.forEach((word) => {
        if ((cur + ' ' + word).trim().length > 16) { lines.push(cur.trim()); cur = word; }
        else cur += ` ${word}`;
      });
      lines.push(cur.trim());
      lines.forEach((ln, i) => {
        g.append('text').attr('class', 'chart-note')
          .attr('x', x(d.label) + x.bandwidth() / 2).attr('y', h + 18 + i * 13)
          .attr('text-anchor', 'middle').style('font-size', '10.5px').text(ln);
      });
    });

    if (cold) {
      const ch = data.arms.chance_at_10;
      plot.append('line').attr('x1', 0).attr('x2', w).attr('y1', y(ch)).attr('y2', y(ch))
        .attr('stroke', 'var(--squidink)').attr('stroke-width', 1.4).attr('stroke-dasharray', '4 4');
      g.append('text').attr('class', 'chart-note')
        .attr('x', 4).attr('y', y(ch) - 6).style('font-size', '11px')
        .text(`chance, ${pct(ch)}`);
    }

    drawAxes(g, x, y, w, h, { xValues: [], yTicks: 5, yFmt: d3.format('.0%') });
    axisLabels(g, w, h, { y: 'held out favourite in a list of ten' });
    g.append('text').attr('class', 'chart-note')
      .attr('x', w / 2).attr('y', -26).attr('text-anchor', 'middle').style('font-size', '12px')
      .text(cold
        ? `${data.arms.cold_books} books with every training rating deleted, ranked among themselves`
        : `the ordinary catalogue of ${data.meta.books.toLocaleString('en-US')} books`);

    const ids = A.ids[cold ? 'cold' : 'warm'].hits['10'];
    const both = A.both[cold ? 'cold' : 'warm'].hits['10'];
    const content = A.content[cold ? 'cold' : 'warm'].hits['10'];
    readout.innerHTML = cold
      ? `With every rating of these <span class="value">${data.arms.cold_books}</span> books removed `
        + `(<span class="value">${data.arms.cold_cells.toLocaleString('en-US')}</span> training `
        + `cells deleted), an id embedding is whatever its initialisation left behind, and the ids `
        + `only tower scores <span class="value">${pct(ids)}</span> against a chance of `
        + `<span class="value">${pct(data.arms.chance_at_10)}</span>. Reading the author, the decade `
        + `and the words of the title instead gets <span class="value">${pct(content)}</span>, and `
        + `carrying both with the id dropped half the time `
        + `<span class="value">${pct(both)}</span>. The bar to compare them against is not zero, it `
        + `is the grey one: recommending the most rated of the three hundred scores `
        + `<span class="value">${pct(A.popular.cold.hits['10'])}</span> without any model at all.`
      : `On the ordinary catalogue: ids alone score <span class="value">${pct(ids)}</span>, content `
        + `alone <span class="value">${pct(content)}</span>, and the two together `
        + `<span class="value">${pct(both)}</span>. `
        + `${ids > content
          ? 'The order is the reverse of the cold view, and for the same reason: a book that '
            + 'thousands of people have rated does not need its author read out, because what '
            + 'those readers did is a better description of it than any feature anybody would '
            + 'write down.'
          : 'The features are not the weaker signal here even on well rated books, which is worth '
            + 'flagging rather than smoothing over: at this budget the id embeddings have not '
            + 'learned more than the content ones have.'}`;
  }

  render();
  return { render, state: st };
}
