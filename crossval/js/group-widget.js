/* Groups that nobody declared, found rather than assumed.
 *
 * There is no chart here because the finding is four numbers and a count, and a
 * chart of four numbers is a table with extra steps. What matters is that the
 * groups were not invented for the demonstration: they are near duplicate
 * newswires in a corpus this atlas has already used twice, found by thresholding
 * cosine similarity and joined transitively, and the count of them is part of
 * the result.
 */
export function initGroupWidget(data) {
  const G = data.groups;
  const readout = document.querySelector('#group-readout');

  const gap = G.by_row - G.by_group;
  const floor = G.by_row_sd + G.by_group_sd;
  readout.innerHTML = `<table class="gen-table"><thead><tr><th>how the folds were made</th>`
    + `<th>accuracy</th><th>spread over seeds</th></tr></thead><tbody>`
    + `<tr><td>five folds of rows</td><td>${G.by_row.toFixed(4)}</td>`
    + `<td>${G.by_row_sd.toFixed(4)}</td></tr>`
    + `<tr><td>five folds of groups</td><td>${G.by_group.toFixed(4)}</td>`
    + `<td>${G.by_group_sd.toFixed(4)}</td></tr>`
    + `</tbody></table>`
    + `<div class="gen-note">Among ${G.docs.toLocaleString('en-US')} single label newswires, `
    + `${G.pairs.toLocaleString('en-US')} pairs sit above a cosine of ${G.threshold} of each `
    + `other in ${G.dims} dimensions. Joined transitively they leave `
    + `${G.groups.toLocaleString('en-US')} groups, of which `
    + `${G.multi_groups.toLocaleString('en-US')} hold more than one document and `
    + `${G.in_groups.toLocaleString('en-US')} documents live inside one; the largest holds `
    + `${G.biggest}. Splitting by row puts members of the same group on both sides of a fold, `
    + `and it reports ${G.by_row.toFixed(4)}. Splitting by group reports `
    + `${G.by_group.toFixed(4)}, `
    + (Math.abs(gap) > floor
      ? `<span class="bold">${Math.abs(gap).toFixed(4)} ${gap > 0 ? 'lower' : 'higher'}</span>, `
        + `against ${floor.toFixed(4)} of spread across seeds.`
      : `a difference of ${Math.abs(gap).toFixed(4)} that the ${floor.toFixed(4)} of seed `
        + `spread cannot separate: on this corpus the duplicates are too few a share of the `
        + `whole to move the average, which is worth publishing because it is the case in `
        + `which the usual advice does not pay.`)
    + ` Nobody labelled these groups. They are a property of the corpus that a split by row `
    + `cannot see, and the only reason they are visible here is that somebody went looking.`
    + `</div>`;

  return { render: () => {} };
}
