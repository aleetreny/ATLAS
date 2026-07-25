/* ATLAS scrollytelling helper.
 *
 * Same mechanism MLU-Explain uses: a raw IntersectionObserver (threshold 0.7)
 * watches every step element; when a step becomes visible its data-index is
 * dispatched into a map of index -> handler. Handlers must be IDEMPOTENT state
 * declarations (fully specify the chart state), so scrolling backwards works
 * for free.
 */
export function scrolly(stepSelector, handlers, { threshold = 0.7 } = {}) {
  const steps = document.querySelectorAll(stepSelector);
  const observer = new IntersectionObserver(
    (entries) => {
      entries.forEach((entry) => {
        if (!entry.isIntersecting) return;
        const i = entry.target.getAttribute('data-index');
        if (i in handlers) handlers[i]();
      });
    },
    { threshold }
  );
  steps.forEach((s) => observer.observe(s));
  return observer;
}
