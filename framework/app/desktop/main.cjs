'use strict';
const {app, BrowserWindow, dialog, session, shell, Menu, ipcMain} = require('electron');
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
const localFiles = new Set(['loading.html', 'shell.js'].map(name => pathToFileURL(path.join(__dirname, name)).href));
const allowedExternal = new Set(['https://npcap.com/#download', 'https://npcap.com/']);
app.setName('NetSec');

function publicState() {
  return {desktop: true, maximized: Boolean(window && !window.isDestroyed() && window.isMaximized()),
    exitRequested, closing: shuttingDown, serviceStatus, message: serviceMessage};
}
function broadcast() {
  child?.sendState(publicState());
  if (window && !window.isDestroyed()) window.webContents.send('netsec:state', publicState());
}
function trustedURL(url) {
  return localFiles.has(url.split('#')[0]) || Boolean(backendOrigin && url.startsWith(backendOrigin + '/'));
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
    title: 'NetSec', backgroundColor: '#f6f5f2', show: false, frame: false, autoHideMenuBar: true,
    icon: path.join(__dirname, 'assets', 'icon.png'),
    webPreferences: {session: partition, preload: path.join(__dirname, 'preload.cjs'),
      contextIsolation: true, nodeIntegration: false, sandbox: true, webSecurity: true, spellcheck: false}});
  window.removeMenu();
  window.once('ready-to-show', () => window.show());
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

if (!app.requestSingleInstanceLock()) app.quit();
else {
  app.on('second-instance', () => {if (window) {if (window.isMinimized()) window.restore(); window.focus();}});
  app.whenReady().then(() => {Menu.setApplicationMenu(null); return openWindow();})
    .catch(error => {if (window && !window.isDestroyed()) serviceError(error).catch(() => {});
      else {dialog.showErrorBox('Unable to open NetSec', error.message); allowExit = true; app.quit();}});
  app.on('window-all-closed', () => {shuttingDown = true; stopBackend().finally(() => {allowExit = true; app.quit();});});
  app.on('before-quit', event => {if (!allowExit) {event.preventDefault(); requestExit();}});
}
