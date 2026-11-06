'use strict';
const {app, BrowserWindow, dialog, session, shell, Menu, ipcMain, nativeTheme, powerMonitor} = require('electron');
const {NativeRuntime} = require('./native-runtime.cjs');
const crypto = require('node:crypto');
const fs = require('node:fs');
const path = require('node:path');
const {pathToFileURL} = require('node:url');
let child, window, backendOrigin, starting = false, shuttingDown = false, allowExit = false;
let exitRequested = false, serviceStatus = 'loading', serviceMessage = '';
const root = path.resolve(__dirname, '../../..');
const token = crypto.randomBytes(32).toString('hex');
const loader = path.join(__dirname, 'loading.html');
// Chromium may report the drive letter in either case; compare file URLs with it normalized.
const fileKey = url => url.split('#')[0].replace(/^file:\/\/\/([a-z]):/i, (_match, drive) => 'file:///' + drive.toUpperCase() + ':');
const localFiles = new Set(['loading.html', 'shell.js'].map(name => fileKey(pathToFileURL(path.join(__dirname, name)).href)));
const allowedExternal = new Set(['https://npcap.com/#download', 'https://npcap.com/']);
app.setName('NetSec');

// Hardware acceleration. Chromium already composites and rasterizes on the GPU wherever its driver blocklist
// allows, and falls back to software where it does not; the default keeps exactly that. NETSEC_GPU_BOOST=1
// forces GPU rasterization and zero-copy uploads (bypassing the raster blocklist, as Discord-style flags do)
// and NETSEC_HIGH_PERFORMANCE_GPU=1 prefers the discrete GPU on hybrid laptops at the cost of battery.
// Opt out entirely with --disable-gpu-acceleration or NETSEC_DISABLE_GPU=1.
const gpuSwitches = [];
const hardwareAcceleration = !process.argv.includes('--disable-gpu-acceleration') && process.env.NETSEC_DISABLE_GPU !== '1';
if (!hardwareAcceleration) app.disableHardwareAcceleration();
else {
  if (process.env.NETSEC_GPU_BOOST === '1') gpuSwitches.push('enable-gpu-rasterization', 'enable-zero-copy');
  if (process.env.NETSEC_HIGH_PERFORMANCE_GPU === '1') gpuSwitches.push('force_high_performance_gpu');
  for (const name of gpuSwitches) app.commandLine.appendSwitch(name);
}
// Theme preference lives with the app's own settings; the renderer's storage partition is per session.
const preferencesPath = () => path.join(app.getPath('userData'), 'preferences.json');
function readPreferences() {
  try {
    const value = JSON.parse(fs.readFileSync(preferencesPath(), 'utf8'));
    return value && typeof value === 'object' && !Array.isArray(value) ? value : {};
  } catch {
    return {};
  }
}
function savedTheme() {
  const theme = readPreferences().theme;
  return theme === 'light' || theme === 'dark' ? theme : 'system';
}
const windowBackground = () => nativeTheme.shouldUseDarkColors ? '#161512' : '#f6f5f2';

function publicState() {
  return {desktop: true, maximized: Boolean(window && !window.isDestroyed() && window.isMaximized()),
    exitRequested, closing: shuttingDown, serviceStatus, message: serviceMessage,
    powerSaver: app.isReady() && powerMonitor.isOnBatteryPower()};
}
function broadcast() {
  child?.sendState(publicState());
  if (window && !window.isDestroyed()) window.webContents.send('netsec:state', publicState());
}
function trustedURL(url) {
  return localFiles.has(fileKey(url)) || Boolean(backendOrigin && url.startsWith(backendOrigin + '/'));
}
function trustedSender(event) {
  return window && !window.isDestroyed() && event.sender === window.webContents &&
    event.senderFrame === window.webContents.mainFrame && trustedURL(event.senderFrame.url);
}
function requestExit() {
  if (shuttingDown) return;
  exitRequested = true;
  broadcast();
}
async function finishExit() {
  if (shuttingDown || !exitRequested) return;
  shuttingDown = true;
  broadcast();
  await stopBackend();
  allowExit = true;
  app.quit();
}
async function serviceError(error) {
  serviceStatus = 'error';
  serviceMessage = String(error.message || error).slice(-1800);
  if (window && !window.isDestroyed() && !shuttingDown) {
    await window.loadFile(loader);
    broadcast();
  }
}

async function spawnBackend() {
  const runtimeRoot = app.isPackaged ? path.join(process.resourcesPath, 'runtime') : (process.env.NETSEC_ROOT || root);
  const owned = child = new NativeRuntime({root: runtimeRoot, token});
  owned.on('failure', error => {if (!shuttingDown && child === owned) serviceError(error).catch(() => {});});
  owned.on('window', action => {
    if (!window || window.isDestroyed()) return;
    if (action === 'minimize') window.minimize();
    else if (action === 'toggle_maximize') window.isMaximized() ? window.unmaximize() : window.maximize();
    else if (action === 'request_close') requestExit();
    else if (action === 'cancel_close') {exitRequested = false; broadcast();}
    else if (action === 'confirm_close' && exitRequested) void finishExit();
  });
  return owned.start();
}

async function startService() {
  if (starting || shuttingDown) return;
  starting = true;
  serviceStatus = 'loading';
  serviceMessage = '';
  try {
    await stopBackend();
    backendOrigin = null;
    await window.loadFile(loader);
    broadcast();
    const origin = await spawnBackend();
    if (shuttingDown) return;
    backendOrigin = origin;
    serviceStatus = 'ready';
    await window.loadURL(origin + '/bootstrap?session=' + token);
    broadcast();
  } catch (error) {
    await stopBackend();
    await serviceError(error);
  } finally {
    starting = false;
  }
}

async function openWindow() {
  const partition = session.fromPartition('imdai-' + crypto.randomBytes(8).toString('hex'));
  partition.setPermissionRequestHandler((_contents, _permission, callback) => callback(false));
  partition.setPermissionCheckHandler(() => false);
  partition.webRequest.onBeforeRequest((details, callback) => {
    const permitted = trustedURL(details.url) || Boolean(backendOrigin && details.url.startsWith('blob:' + backendOrigin + '/'));
    callback({cancel: !permitted});
  });
  window = new BrowserWindow({width: 1480, height: 980, minWidth: 740, minHeight: 620,
    title: 'NetSec', backgroundColor: windowBackground(), show: false, frame: false, autoHideMenuBar: true,
    icon: path.join(__dirname, 'assets', 'icon.png'),
    webPreferences: {session: partition, preload: path.join(__dirname, 'preload.cjs'),
      contextIsolation: true, nodeIntegration: false, sandbox: true, webSecurity: true, spellcheck: false}});
  window.removeMenu();
  window.once('ready-to-show', () => window.show());
  // A first load that never paints must not leave an invisible, unclosable process behind.
  setTimeout(() => {if (window && !window.isDestroyed() && !window.isVisible()) window.show();}, 8000);
  window.on('maximize', broadcast);
  window.on('unmaximize', broadcast);
  window.on('close', event => {if (!allowExit) {event.preventDefault(); requestExit();}});
  window.webContents.setWindowOpenHandler(({url}) => {
    if (allowedExternal.has(url)) shell.openExternal(url);
    return {action: 'deny'};
  });
  window.webContents.on('will-navigate', (event, url) => {if (!trustedURL(url)) event.preventDefault();});
  window.webContents.on('will-attach-webview', event => event.preventDefault());
  window.webContents.on('did-finish-load', broadcast);
  window.webContents.on('render-process-gone', () => {
    stopBackend().then(() => serviceError(new Error('The interface stopped. Restart this workspace to continue.'))).catch(() => {});
  });
  partition.on('will-download', (_event, item) => {
    item.setSaveDialogOptions({defaultPath: path.join(app.getPath('downloads'), path.basename(item.getFilename()))});
  });
  await startService();
}

async function stopBackend() {
  const owned = child;
  if (owned) await owned.stop();
  if (child === owned) child = null;
}

ipcMain.handle('netsec:window-state', event => {
  if (!trustedSender(event)) throw new Error('Window controls are unavailable');
  return publicState();
});
ipcMain.handle('netsec:window-action', (event, action) => {
  if (!trustedSender(event)) throw new Error('Window controls are unavailable');
  if (action === 'minimize') window.minimize();
  else if (action === 'maximize') window.isMaximized() ? window.unmaximize() : window.maximize();
  else if (action === 'request-close') requestExit();
  else if (action === 'cancel-close' && !shuttingDown) {exitRequested = false; broadcast();}
  else if (action === 'confirm-close') {finishExit().catch(() => {allowExit = true; app.quit();});}
  else if (action === 'retry' && serviceStatus === 'error' && !exitRequested) {startService().catch(serviceError);}
  else throw new Error('Unsupported window action');
});

function savePreferences(values) {
  const file = preferencesPath(), staging = file + '.tmp';
  const handle = fs.openSync(staging, 'w');
  try { fs.writeSync(handle, JSON.stringify(values)); fs.fsyncSync(handle); } finally { fs.closeSync(handle); }
  // Windows can briefly lock the target (indexers, antivirus); retry the atomic replace a few times.
  for (let attempt = 0; ; attempt += 1) {
    try { fs.renameSync(staging, file); return; }
    catch (error) {
      if (attempt >= 4 || !['EPERM', 'EBUSY', 'EACCES'].includes(error.code)) { fs.rmSync(staging, {force: true}); throw error; }
      Atomics.wait(new Int32Array(new SharedArrayBuffer(4)), 0, 0, 25 * (attempt + 1));
    }
  }
}
ipcMain.handle('netsec:set-theme', (event, theme) => {
  if (!trustedSender(event)) throw new Error('Theme control is unavailable');
  if (theme !== 'light' && theme !== 'dark') throw new Error('Unsupported theme');
  // The session always follows the choice; saved reports whether it will also survive a restart.
  let saved = true;
  try { savePreferences({...readPreferences(), theme}); } catch { saved = false; }
  nativeTheme.themeSource = theme;
  if (window && !window.isDestroyed()) window.setBackgroundColor(windowBackground());
  return {theme, saved};
});
ipcMain.handle('netsec:graphics', event => {
  if (!trustedSender(event)) throw new Error('Graphics status is unavailable');
  return {hardwareAcceleration, switches: hardwareAcceleration ? [...gpuSwitches] : [], features: app.getGPUFeatureStatus()};
});

if (!app.requestSingleInstanceLock()) app.quit();
else {
  app.on('second-instance', () => {if (window) {if (window.isMinimized()) window.restore(); window.focus();}});
  app.whenReady().then(() => {
    Menu.setApplicationMenu(null); nativeTheme.themeSource = savedTheme();
    // Battery power switches the interface to its low-power presentation (no ambient motion, simpler glass).
    powerMonitor.on('on-battery', broadcast); powerMonitor.on('on-ac', broadcast);
    return openWindow();
  })
    .catch(error => {if (window && !window.isDestroyed()) serviceError(error).catch(() => {});
      else {dialog.showErrorBox('Unable to open NetSec', error.message); allowExit = true; app.quit();}});
  app.on('window-all-closed', () => {shuttingDown = true; stopBackend().finally(() => {allowExit = true; app.quit();});});
  app.on('before-quit', event => {if (!allowExit) {event.preventDefault(); requestExit();}});
}
