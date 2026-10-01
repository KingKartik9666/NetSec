'use strict';
const {app, BrowserWindow, dialog, session, shell, Menu, ipcMain} = require('electron');
const {spawn} = require('node:child_process');
const crypto = require('node:crypto');
const fs = require('node:fs');
const path = require('node:path');
const readline = require('node:readline');
const {pathToFileURL} = require('node:url');
let child, window, backendOrigin, starting = false, shuttingDown = false, allowExit = false;
let exitRequested = false, serviceStatus = 'loading', serviceMessage = '';
const expectedStops = new WeakSet();
const root = path.resolve(__dirname, '../../..');
const token = crypto.randomBytes(32).toString('hex');
const loader = path.join(__dirname, 'loading.html');
const localFiles = new Set(['loading.html', 'shell.js'].map(name => pathToFileURL(path.join(__dirname, name)).href));
const allowedExternal = new Set(['https://npcap.com/#download', 'https://npcap.com/']);
app.setName('NetSec');

function publicState() {
  return {maximized: Boolean(window && !window.isDestroyed() && window.isMaximized()),
    exitRequested, closing: shuttingDown, serviceStatus, message: serviceMessage};
}
function broadcast() {
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

function spawnBackend() {
  return new Promise((resolve, reject) => {
    const binary = process.platform === 'win32' ? 'imdai-backend.exe' : 'imdai-backend';
    const python = process.env.IMDAI_PYTHON || path.join(root, '.venv', process.platform === 'win32' ? 'Scripts/python.exe' : 'bin/python');
    const executable = app.isPackaged ? path.join(process.resourcesPath, 'backend', binary) : python;
    if (!fs.existsSync(executable)) return reject(new Error('The local analysis runtime is missing. Restore the project environment or reinstall NetSec, then try again.'));
    const args = app.isPackaged ? [] : ['-B', '-u', '-m', 'framework.app.desktop_backend'];
    const owned = spawn(executable, args, {cwd: app.isPackaged ? process.resourcesPath : root,
      windowsHide: true, stdio: ['pipe', 'pipe', 'pipe'],
      env: {...process.env, PYTHONDONTWRITEBYTECODE: '1', IMDAI_SESSION_TOKEN: token,
        IMDAI_LLM_DIR: app.isPackaged ? path.join(process.resourcesPath, 'models/qwen') : path.join(root, 'artifacts/llm/qwen')}});
    child = owned;
    let settled = false;
    const finish = (error, value) => {if (!settled) {settled = true; clearTimeout(timer); error ? reject(error) : resolve(value);}};
    const timer = setTimeout(() => finish(new Error('The local analysis runtime did not become ready within 60 seconds.')), 60000);
    const lines = readline.createInterface({input: owned.stdout});
    lines.on('line', line => {
      try {const message = JSON.parse(line); if (message.type === 'ready' && Number.isInteger(message.port) && message.port > 0 && message.port < 65536) finish(null, 'http://127.0.0.1:' + message.port);} catch {}
    });
    // Keep diagnostics local and bounded. Never log the session token or evidence.
    let diagnostics = '';
    owned.stderr.on('data', chunk => {diagnostics = (diagnostics + chunk.toString()).slice(-12000);});
    owned.on('error', error => finish(error));
    owned.on('exit', code => {
      if (!settled) finish(new Error('Local analysis runtime exited (' + code + '). ' + diagnostics.slice(-1500)));
      else if (!shuttingDown && !expectedStops.has(owned) && child === owned) {
        serviceError(new Error('The local analysis service stopped. Previous observations are no longer current. Restart the workspace to continue.')).catch(() => {});
      }
    });
  });
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
    await window.loadURL(origin);
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
  partition.webRequest.onBeforeSendHeaders((details, callback) => {
    if (backendOrigin && details.url.startsWith(backendOrigin + '/')) details.requestHeaders['X-IMDAI-Session'] = token;
    callback({requestHeaders: details.requestHeaders});
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
  partition.on('will-download', (_event, item) => {
    item.setSaveDialogOptions({defaultPath: path.join(app.getPath('downloads'), path.basename(item.getFilename()))});
  });
  await startService();
}

function stopBackend() {
  return new Promise(resolve => {
    const owned = child;
    if (!owned || owned.exitCode !== null || owned.signalCode !== null) return resolve();
    expectedStops.add(owned);
    const timer = setTimeout(() => {owned.kill(); resolve();}, 10000);
    owned.once('exit', () => {clearTimeout(timer); resolve();});
    owned.stdin.on('error', () => {});
    if (!owned.stdin.destroyed && !owned.stdin.writableEnded) owned.stdin.end('shutdown\n');
  });
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
  app.on('window-all-closed', () => {allowExit = true; app.quit();});
  app.on('before-quit', event => {if (!allowExit) {event.preventDefault(); requestExit();}});
}
