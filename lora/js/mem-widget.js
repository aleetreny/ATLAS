/* The memory bill, as a table, because it is arithmetic and not a measurement.
 *
 * There is no chart here on purpose. Three rows of exact byte counts read
 * better than three bars, and a bar chart would invite the reader to compare
 * areas when the interesting quantity is a ratio between two numbers that
 * differ by an order of magnitude.
 */
const mb = (v) => `${(v / (1024 * 1024)).toFixed(2)} MB`;

export function initMemWidget(data) {
  const M = data.memory;
  const readout = document.querySelector('#mem-readout');
  const full = M.rows.find((r) => r.arm === 'full fine tune');

  const rows = M.rows.map((r) => `<tr><td>${r.arm}</td>`
    + `<td>${r.trainable.toLocaleString('en-US')}</td>`
    + `<td>${mb(r.weights)}</td><td>${mb(r.grads)}</td><td>${mb(r.optimiser)}</td>`
    + `<td>${mb(r.total)}</td>`
    + `<td>${(full.total / r.total).toFixed(1)} times less</td></tr>`).join('');

  readout.innerHTML = `<table class="gen-table"><thead><tr><th>recipe</th>`
    + `<th>trainable</th><th>weights</th><th>gradients</th><th>optimiser</th>`
    + `<th>total</th><th>against full</th></tr></thead><tbody>${rows}</tbody></table>`
    + `<div class="gen-note">Counted on this network's ${M.params.toLocaleString('en-US')} `
    + `parameters with an adapter of rank ${M.rank}: four bytes per weight and per gradient, `
    + `eight for the two moments Adam keeps, and for the last row the four bit base at the `
    + `bits per weight the quantiser really costs, scales included. The column that does the `
    + `work is the optimiser: it is twice the gradients and it only exists for parameters `
    + `that move, which is why freezing the base removes almost the whole bill rather than a `
    + `third of it. None of this is measured on a machine, and it should not be: it is `
    + `arithmetic on parameter counts, so it is the same number everywhere.</div>`;

  return { render: () => {} };
}
