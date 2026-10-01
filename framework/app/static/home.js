"use strict";
// Decorative, event-driven sculpture. Home never calls analysis or monitoring APIs.
(() => {
  const visual = document.querySelector(".heading-visual");
  const stage = document.querySelector(".network-stage");
  if (!visual || !stage) return;
  const preference = matchMedia("(prefers-reduced-motion: reduce)");
  let frame = null;
  const reset = () => {
    if (frame !== null) cancelAnimationFrame(frame);
    frame = null;
    stage.style.setProperty("--pointer-x", "0deg");
    stage.style.setProperty("--pointer-y", "0deg");
  };
  visual.addEventListener("pointermove", (event) => {
    if (event.pointerType === "touch" || preference.matches || document.hidden) return;
    const rect = visual.getBoundingClientRect();
    const x = Math.max(-12, Math.min(12, (event.clientX - rect.left - rect.width / 2) / rect.width * 24));
    const y = Math.max(-9, Math.min(9, (rect.top + rect.height / 2 - event.clientY) / rect.height * 18));
    if (frame !== null) cancelAnimationFrame(frame);
    frame = requestAnimationFrame(() => {
      frame = null;
      if (preference.matches || document.hidden) return;
      stage.style.setProperty("--pointer-x", `${x.toFixed(2)}deg`);
      stage.style.setProperty("--pointer-y", `${y.toFixed(2)}deg`);
    });
  }, {passive: true});
  visual.addEventListener("pointerleave", reset, {passive: true});
  preference.addEventListener("change", reset);
  document.addEventListener("visibilitychange", reset);
  window.addEventListener("blur", reset);
})();
