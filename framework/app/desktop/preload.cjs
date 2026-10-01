'use strict';
const {contextBridge, ipcRenderer} = require('electron');
// Fixed window actions only. No raw IPC, filesystem or Node access.
contextBridge.exposeInMainWorld('netsecDesktop', Object.freeze({
  minimize: () => ipcRenderer.invoke('netsec:window-action', 'minimize'),
  toggleMaximize: () => ipcRenderer.invoke('netsec:window-action', 'maximize'),
  requestClose: () => ipcRenderer.invoke('netsec:window-action', 'request-close'),
  confirmClose: () => ipcRenderer.invoke('netsec:window-action', 'confirm-close'),
  cancelClose: () => ipcRenderer.invoke('netsec:window-action', 'cancel-close'),
  retry: () => ipcRenderer.invoke('netsec:window-action', 'retry'),
  getState: () => ipcRenderer.invoke('netsec:window-state'),
  onState: callback => {
    if (typeof callback !== 'function') return () => {};
    const listener = (_event, state) => callback(state);
    ipcRenderer.on('netsec:state', listener);
    return () => ipcRenderer.removeListener('netsec:state', listener);
  }
}));
