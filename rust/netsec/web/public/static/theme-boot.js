// Applies the saved or system theme before first paint (a classic script, so it runs before styles render).
// The desktop shell maps the saved choice onto prefers-color-scheme, which covers its non-persistent storage.
(function () {
  var theme;
  try { theme = localStorage.getItem('netsec-theme'); } catch (error) { theme = null; }
  if (theme !== 'light' && theme !== 'dark') theme = window.matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light';
  document.documentElement.dataset.theme = theme;
  document.documentElement.style.colorScheme = theme;
})();
