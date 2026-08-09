/* Shared accessibility pass for the static ATLAS articles.
 * Widgets are deliberately small and independent, so this observer supplies
 * the common semantics without changing their measured data or visual state.
 */

function replaceTag(node, tagName) {
  if (!node || node.tagName.toLowerCase() === tagName) return node;
  const replacement = document.createElement(tagName);
  for (const attr of node.attributes) replacement.setAttribute(attr.name, attr.value);
  replacement.innerHTML = node.innerHTML;
  node.replaceWith(replacement);
  return replacement;
}

function decorateButtons(root) {
  root.querySelectorAll('button.atlas-btn').forEach((button) => {
    /* Widgets change the active state by toggling `.ghost`; keep the semantic
     * state in lockstep even when the button was already decorated. */
    button.setAttribute('aria-pressed', String(!button.classList.contains('ghost')));
  });
}

function decorateRanges(root) {
  root.querySelectorAll('input[type="range"]').forEach((range) => {
    const label = document.getElementById(`${range.id}-label`)
      || range.closest('.controls-row')?.querySelector('.value');
    const valueText = label?.textContent?.trim() || range.value;
    range.setAttribute('aria-valuetext', valueText);
    if (!range.dataset.atlasA11yBound) {
      range.addEventListener('input', () => {
        const liveLabel = document.getElementById(`${range.id}-label`)
          || range.closest('.controls-row')?.querySelector('.value');
        range.setAttribute('aria-valuetext', liveLabel?.textContent?.trim() || range.value);
      });
      range.dataset.atlasA11yBound = 'true';
    }
  });
}

function decorateGraphics(root) {
  root.querySelectorAll('svg:not([aria-hidden="true"])').forEach((svg) => {
    if (!svg.hasAttribute('role')) svg.setAttribute('role', 'img');
    if (!svg.hasAttribute('aria-label')) {
      const id = svg.closest('[id]')?.id || 'ATLAS interactive chart';
      svg.setAttribute('aria-label', `${id.replace(/-/g, ' ')} visualization`);
    }
    if (!svg.querySelector(':scope > title')) {
      const title = document.createElementNS('http://www.w3.org/2000/svg', 'title');
      title.textContent = svg.getAttribute('aria-label');
      svg.prepend(title);
    }
  });
  root.querySelectorAll('canvas:not([aria-hidden="true"])').forEach((canvas) => {
    if (!canvas.hasAttribute('role')) canvas.setAttribute('role', 'img');
    if (!canvas.hasAttribute('aria-label')) {
      canvas.setAttribute('aria-label', `${canvas.closest('[id]')?.id || 'ATLAS'} visualization`);
    }
  });
}

function decorateTables(root) {
  root.querySelectorAll('table').forEach((table) => {
    if (!table.hasAttribute('aria-label')) table.setAttribute('aria-label', 'ATLAS data table');
    const firstRow = table.querySelector('tr');
    firstRow?.querySelectorAll('th').forEach((th) => {
      if (!th.hasAttribute('scope')) th.setAttribute('scope', 'col');
    });
    if (!table.querySelector(':scope > caption')) {
      const caption = document.createElement('caption');
      caption.className = 'sr-only';
      caption.textContent = 'Interactive data table';
      table.prepend(caption);
    }
  });
}

function decorateHeadings() {
  const introSub = document.querySelector('#intro-sub');
  if (introSub) replaceTag(introSub, 'p');
  document.querySelectorAll('h1:not(#intro-hed)').forEach((heading) => replaceTag(heading, 'h2'));
  const date = document.querySelector('#intro__date');
  if (date) replaceTag(date, 'p');
}

function decorate() {
  decorateButtons(document);
  decorateRanges(document);
  decorateGraphics(document);
  decorateTables(document);
  decorateHeadings();
}

function boot() {
  decorate();
  const observer = new MutationObserver(() => decorate());
  observer.observe(document.body, { childList: true, subtree: true, attributes: true, attributeFilter: ['class'] });
}

if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', boot, { once: true });
else boot();
