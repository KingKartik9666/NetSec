"use strict";
(() => {
  const desktop = window.netsecDesktop;
  const screen = document.getElementById("lifecycle-screen");
  const title = document.getElementById("lifecycle-title");
  const description = document.getElementById("lifecycle-description");
  const primary = document.getElementById("lifecycle-primary");
  const cancel = document.getElementById("lifecycle-cancel");
  const home = document.getElementById("lifecycle-home");
  const details = document.getElementById("lifecycle-details");
  const returnURL = document.body.dataset.returnUrl || "/";
  let current = document.body.dataset.lifecycleState || (document.body.dataset.workspaceMode ? "loading" : null);
  let currentDetail = "";
  let saved = null, savedFocus = null, exitActive = false;
  const copy = {
    loading: ["01 / INITIALIZING", "Bringing things into focus.", "Connecting to your local analysis service. Monitoring and language generation start only when you request them."],
    error: ["02 / CONNECTION PAUSED", "A pause in the signal.", "The local analysis service is unavailable. Try again when it is ready. No substitute prediction was returned."],
    exit: ["03 / END SESSION", "Close this workspace?", desktop ? "Export anything you want to keep before leaving. Closing NetSec stops local monitoring and closes the analysis runtime." : "Export anything you want to keep before leaving. End this workspace and stop its monitoring session."],
    closing: ["03 / CLOSING", "Wrapping up.", desktop ? "Stopping monitoring and closing the local analysis runtime." : "Stopping this workspace’s monitoring session."],
    ended: ["03 / SESSION ENDED", "All wrapped up.", "Monitoring has stopped. You can close this tab or return to home when you’re ready."]
  };
  function render() {
    screen.hidden = !current;
    document.body.classList.toggle("state-open", Boolean(current));
    document.querySelectorAll("#workspace, .studio-header").forEach(element => {element.inert = Boolean(current);});
    if (!current) return;
    screen.dataset.state = current;
    const content = copy[current];
    document.getElementById("lifecycle-code").textContent = content[0];
    title.textContent = content[1];
    description.textContent = current === "error" && document.body.dataset.errorCode === "404"
      ? "This page could not be found. Head home to choose a NetSec workspace." : content[2];
    document.getElementById("lifecycle-detail").textContent = currentDetail;
    details.hidden = current !== "error" || !currentDetail;
    if (details.hidden) details.open = false;
    const missing = document.body.dataset.errorCode === "404";
    primary.hidden = !["error", "exit"].includes(current) || (current === "error" && missing);
    primary.disabled = false;
    primary.textContent = current === "exit" ? (desktop ? "Exit NetSec ↗" : "End workspace ↗") : "Try again ↗";
    cancel.hidden = current !== "exit";
    cancel.disabled = false;
    home.hidden = current === "exit" || current === "closing";
    home.textContent = current === "ended" ? "Open home ↗" : "Back to home ↗";
    document.getElementById("lifecycle-footnote").textContent = current === "exit"
      ? "Your saved files stay on your device." : "Private by default. Local by design.";
  }
  function setState(next, detail = "") {
    if (exitActive) {saved = {state: next, detail}; return;}
    const wasOpen = Boolean(current);
    current = next;
    currentDetail = detail;
    if (!wasOpen && next) savedFocus = document.activeElement;
    render();
    if (next === "error") title.focus();
    if (!next && savedFocus?.isConnected) savedFocus.focus({preventScroll: true});
  }
  function requestExit() {
    if (desktop) {desktop.requestClose().catch(error => setState("error", error.message)); return;}
    enterExit();
  }
  function enterExit() {
    if (exitActive) return;
    saved = {state: current, detail: currentDetail};
    savedFocus = document.activeElement;
    exitActive = true;
    current = "exit";
    currentDetail = "";
    render();
    cancel.focus();
  }
  function leaveExit() {
    exitActive = false;
    current = saved?.state || null;
    currentDetail = saved?.detail || "";
    saved = null;
    render();
    if (savedFocus?.isConnected) savedFocus.focus({preventScroll: true});
  }
  cancel.addEventListener("click", () => {
    if (desktop) desktop.cancelClose().then(() => {
      if (document.body.dataset.lifecycleState === "exit") location.assign(returnURL);
    }).catch(error => {exitActive = false; setState("error", error.message);});
    else if (document.body.dataset.lifecycleState === "exit") location.assign(returnURL);
    else leaveExit();
  });
  primary.addEventListener("click", async () => {
    if (current === "error") {
      if (location.pathname === "/error") location.assign(returnURL);
      else location.reload();
      return;
    }
    if (current !== "exit") return;
    current = "closing";
    render();
    if (desktop) {
      try {await desktop.confirmClose();} catch (error) {exitActive = false; setState("error", error.message);}
      return;
    }
    try {
      const response = await fetch("/api/live/stop", {method: "POST", headers: {"Content-Type": "application/json"}, body: "{}"});
      if (!response.ok) throw new Error("The local service did not confirm that monitoring stopped. Restart the local app to end its session.");
      exitActive = false;
      setState("ended");
    } catch (error) {exitActive = false; setState("error", error.message);}
  });
  document.querySelectorAll(".session-exit").forEach(link => link.addEventListener("click", event => {event.preventDefault(); requestExit();}));
  document.addEventListener("keydown", event => {
    if (!current) return;
    if (event.key === "Escape" && current === "exit") {event.preventDefault(); cancel.click();}
    if (event.key !== "Tab") return;
    const focusable = [...screen.querySelectorAll('a[href], button, summary')].filter(element => !element.hidden && !element.disabled && element.getClientRects().length);
    if (!focusable.length) {event.preventDefault(); title.focus(); return;}
    const index = focusable.indexOf(document.activeElement);
    if (event.shiftKey && index <= 0) {event.preventDefault(); focusable.at(-1).focus();}
    else if (!event.shiftKey && (index === -1 || index === focusable.length - 1)) {event.preventDefault(); focusable[0].focus();}
  });
  if (desktop) {
    document.body.classList.add("desktop-app");
    document.querySelector(".window-chrome").hidden = false;
    document.querySelector('[data-window-action="minimize"]').addEventListener("click", () => desktop.minimize());
    document.querySelector('[data-window-action="maximize"]').addEventListener("click", () => desktop.toggleMaximize());
    document.querySelector('[data-window-action="close"]').addEventListener("click", requestExit);
    const onState = state => {
      const maximize = document.querySelector('[data-window-action="maximize"]');
      maximize.setAttribute("aria-label", state.maximized ? "Restore window" : "Maximize window");
      maximize.classList.toggle("is-maximized", state.maximized);
      if (state.exitRequested && !exitActive) enterExit();
      if (!state.exitRequested && exitActive) leaveExit();
      if (state.closing) {current = "closing"; render();}
    };
    desktop.onState(onState);
    desktop.getState().then(state => {
      onState(state);
      if (document.body.dataset.lifecycleState === "exit" && !state.exitRequested) requestExit();
    }).catch(error => setState("error", error.message));
  }
  window.netsecUI = Object.freeze({setState, requestExit});
  render();
  if (current === "exit") {current = null; enterExit();}
})();
