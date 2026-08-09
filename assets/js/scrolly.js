/* ATLAS scrollytelling helper.
 *
 * Handlers must be IDEMPOTENT state declarations: each one fully specifies the
 * chart state rather than nudging it, so scrolling backwards works for free.
 *
 * On the trigger geometry. MLU-Explain observes steps with {threshold: 0.7},
 * meaning "70% of the step's area is on screen". That couples the trigger to the
 * ratio between step height and viewport height: a 110vh step can reach at most
 * 0.909, and the 130vh step this stylesheet uses on narrow screens can reach at
 * most 0.769. Measured on a 910px viewport that leaves only 0.07 of headroom,
 * and anything that shrinks the visual viewport (a mobile URL bar sliding into
 * place, for instance) pushes it under the threshold, at which point the story
 * silently stops advancing.
 *
 * So trigger on a thin band across the middle of the viewport instead:
 * rootMargin collapses the observation region to the centre slice, and whichever
 * step covers that band is the active one. True for any step height and any
 * viewport, with no arithmetic to get wrong.
 */
export function scrolly(stepSelector, handlers, { band = 45 } = {}) {
  const steps = document.querySelectorAll(stepSelector);
  let current = null;
  const visible = new Map();

  const activate = (index) => {
    if (index === null || index === current || !(index in handlers)) return;
    current = index;
    handlers[index]();
  };

  const observer = new IntersectionObserver(
    (entries) => {
      entries.forEach((entry) => {
        const index = entry.target.getAttribute('data-index');
        if (entry.isIntersecting) visible.set(index, entry.target);
        else visible.delete(index);
      });
      if (!visible.size) return;
      const mid = window.innerHeight / 2;
      const closest = [...visible.entries()].sort((a, b) => {
        const da = Math.abs(a[1].getBoundingClientRect().top + a[1].getBoundingClientRect().height / 2 - mid);
        const db = Math.abs(b[1].getBoundingClientRect().top + b[1].getBoundingClientRect().height / 2 - mid);
        return da - db;
      })[0];
      activate(closest[0]);
    },
    { rootMargin: `-${band}% 0px -${band}% 0px`, threshold: 0 }
  );

  steps.forEach((s) => observer.observe(s));

  /* Escape hatch: select a step without scrolling. Used to check chart states
   * outside a live browser, where IntersectionObserver does not run. */
  const goTo = (index) => activate(String(index));
  steps.forEach((s) => {
    s.addEventListener('atlas:activate', () => goTo(s.getAttribute('data-index')));
  });

  return { observer, goTo };
}
