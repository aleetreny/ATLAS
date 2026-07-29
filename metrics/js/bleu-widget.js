/* What BLEU sees, measured by showing it things whose difference is known.
 *
 * There is no translation system on this page, and that is a decision rather
 * than a shortcut. A system's output differs from a reference in a way nobody
 * can describe, so a score on it says only "worse than a reference by an
 * unknown amount". Perturbing a real sentence in a known way and reading the
 * score is the experiment that says what the statistic responds to: the same
 * words in a different order, the same order with words missing, the same
 * length with words replaced.
 */
import { makeChart, drawAxes, drawGrid, axisLabels } from '../../assets/js/chart.js';

export function initBleuWidget(data) {
  const B = data.bleu;
  const buttons = document.querySelector('#bleu-buttons');
  const readout = document.querySelector('#bleu-readout');
  const st = { view: 'perturb' };

  const chart = makeChart('#bleu-chart', {
    width: 640, height: 400, margin: { top: 52, right: 34, bottom: 106, left: 78 },
  });
  const { g, w, h } = chart;
  const layer = g.append('g');

  const btns = {};
  for (const [key, text] of [['perturb', 'what it responds to'],
    ['brevity', 'the length penalty'], ['segment', 'the tokenisation']]) {
    const b = document.createElement('button');
    b.className = 'atlas-btn ghost';
    b.textContent = text;
    b.addEventListener('click', () => { st.view = key; render(); });
    buttons.appendChild(b);
    btns[key] = b;
  }

  function render() {
    Object.entries(btns).forEach(([k, b]) => b.classList.toggle('ghost', k !== st.view));
    layer.selectAll('*').remove();

    if (st.view === 'perturb') {
      const rows = B.rows;
      const x = d3.scaleBand().domain(rows.map((r) => r.kind)).range([0, w]).padding(0.28);
      const y = d3.scaleLinear().domain([0, 1.02]).range([h, 0]);
      drawGrid(g, x, y, w, h, { xValues: [], yTicks: 5 });
      drawAxes(g, x, y, w, h, { xValues: [], yTicks: 5, yFmt: d3.format('.1f') });
      axisLabels(g, w, h, { x: '', y: 'BLEU over the corpus' });
      rows.forEach((r) => {
        const keeps = r.word_overlap > 0.99;
        layer.append('rect').attr('x', x(r.kind)).attr('width', x.bandwidth())
          .attr('y', y(r.corpus)).attr('height', h - y(r.corpus))
          .attr('fill', keeps ? 'var(--cosmos)' : 'var(--primary)');
        layer.append('text').attr('class', 'chart-note')
          .attr('x', x(r.kind) + x.bandwidth() / 2).attr('y', y(r.corpus) - 7)
          .attr('text-anchor', 'middle').style('font-size', '10.5px')
          .text(r.corpus.toFixed(3));
        layer.append('text').attr('class', 'chart-note')
          .attr('transform', `translate(${x(r.kind) + x.bandwidth() / 2},${h + 8}) rotate(-40)`)
          .attr('text-anchor', 'end').style('font-size', '10.5px').text(r.kind);
      });
      layer.append('text').attr('class', 'chart-note').attr('x', 0).attr('y', -34)
        .style('font-size', '11px').attr('fill', 'var(--cosmos)')
        .text('darker: every word of the reference is still there, only moved');
    } else if (st.view === 'brevity') {
      const rows = B.brevity;
      const x = d3.scaleLinear().domain([rows[0].frac, rows[rows.length - 1].frac]).range([0, w]);
      const y = d3.scaleLinear().domain([0, 1.02]).range([h, 0]);
      drawGrid(g, x, y, w, h, { xValues: rows.map((r) => r.frac), yTicks: 5 });
      drawAxes(g, x, y, w, h, {
        xValues: rows.map((r) => r.frac), yTicks: 5, xFmt: d3.format('.2f'), yFmt: d3.format('.1f'),
      });
      axisLabels(g, w, h, { x: 'candidate length as a fraction of the reference', y: '' });
      [['bleu', 'var(--primary)', 'BLEU'],
        ['bp', 'var(--anchor)', 'the brevity penalty on its own']]
        .forEach(([key, colour, name], i) => {
          layer.append('path').attr('d', d3.line().x((r) => x(r.frac)).y((r) => y(r[key]))(rows))
            .attr('fill', 'none').attr('stroke', colour).attr('stroke-width', 2.2);
          rows.forEach((r) => layer.append('circle').attr('cx', x(r.frac)).attr('cy', y(r[key]))
            .attr('r', 3.8).attr('fill', colour));
          layer.append('text').attr('class', 'chart-note').attr('x', 0).attr('y', -34 + i * 15)
            .style('font-size', '11px').attr('fill', colour).text(name);
        });
      layer.append('line').attr('x1', x(1)).attr('x2', x(1)).attr('y1', 0).attr('y2', h)
        .attr('stroke', 'var(--squidink)').attr('stroke-width', 1).attr('stroke-dasharray', '4 4');
    } else {
      const rows = B.segmentation;
      const x = d3.scaleBand().domain(rows.map((r) => r.scheme)).range([0, w]).padding(0.35);
      const y = d3.scaleLinear().domain([0, d3.max(rows.map((r) => r.bleu)) * 1.2]).range([h, 0]);
      drawGrid(g, x, y, w, h, { xValues: [], yTicks: 5 });
      drawAxes(g, x, y, w, h, { xValues: [], yTicks: 5, yFmt: d3.format('.2f') });
      axisLabels(g, w, h, { x: '', y: 'BLEU on the same pairs' });
      rows.forEach((r) => {
        layer.append('rect').attr('x', x(r.scheme)).attr('width', x.bandwidth())
          .attr('y', y(r.bleu)).attr('height', h - y(r.bleu)).attr('fill', 'var(--primary)');
        layer.append('text').attr('class', 'chart-note')
          .attr('x', x(r.scheme) + x.bandwidth() / 2).attr('y', y(r.bleu) - 8)
          .attr('text-anchor', 'middle').style('font-size', '11.5px').text(r.bleu.toFixed(4));
        layer.append('text').attr('class', 'chart-note')
          .attr('transform', `translate(${x(r.scheme) + x.bandwidth() / 2},${h + 8}) rotate(-24)`)
          .attr('text-anchor', 'end').style('font-size', '10.5px').text(r.scheme);
      });
    }

    const rows = B.rows;
    const same = rows.find((r) => r.kind === 'identical');
    const shuf = rows.find((r) => r.kind === 'shuffled');
    const rev = rows.find((r) => r.kind === 'reversed');
    const drop1 = rows.find((r) => r.kind === 'drop 1');
    const seg = B.segmentation;
    const segSpread = Math.max(...seg.map((s) => s.bleu)) - Math.min(...seg.map((s) => s.bleu));
    const tbl = rows.map((r) => `<tr><td>${r.kind}</td><td>${r.corpus.toFixed(4)}</td>`
      + `<td>${r.sentence_mean.toFixed(4)}</td>`
      + `<td>${r.precisions.map((p) => p.toFixed(2)).join(' ')}</td>`
      + `<td>${(r.word_overlap * 100).toFixed(0)}%</td></tr>`).join('');
    readout.innerHTML = `<table class="gen-table"><thead><tr><th>what was done</th>`
      + `<th>corpus BLEU</th><th>mean sentence BLEU</th>`
      + `<th>1 to 4 gram precision</th><th>words kept</th></tr></thead>`
      + `<tbody>${tbl}</tbody></table>`
      + `<div class="gen-note">Shuffling every word of a sentence keeps `
      + `${(shuf.word_overlap * 100).toFixed(0)}% of its words and scores `
      + `<span class="bold">${shuf.corpus.toFixed(4)}</span> against `
      + `${same.corpus.toFixed(4)} for leaving it alone; reversing it scores `
      + `${rev.corpus.toFixed(4)}, with a unigram precision of `
      + `${rev.precisions[0].toFixed(2)} that is untouched by the reversal because unigrams `
      + `have no order. Removing a single word scores ${drop1.corpus.toFixed(4)}. `
      + `Two more things this table says that are usually left out. The corpus figure is not `
      + `the mean of the sentence figures: for the identical arm they are `
      + `${same.corpus.toFixed(4)} and ${same.sentence_mean.toFixed(4)}, and the difference `
      + `is bigger for the damaged arms, because a corpus BLEU pools the counts before `
      + `dividing. And the tokenisation moves the number by ${segSpread.toFixed(4)} on the `
      + `<span class="bold">same</span> pairs of sentences, which is larger than most of the `
      + `differences that get reported between systems.</div>`;
  }

  render();
  return { render, state: st };
}
