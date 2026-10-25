'use strict';
const {spawn} = require('node:child_process');
const {EventEmitter} = require('node:events');
const readline = require('node:readline');
const path = require('node:path');
const fs = require('node:fs');
const actions = new Set(['minimize', 'toggle_maximize', 'request_close', 'cancel_close', 'confirm_close', 'start_drag']);
class NativeRuntime extends EventEmitter {
  constructor({root, token, executable, args, readyTimeout = 60000, stopTimeout = 10000}) {
    super(); Object.assign(this, {root, token, readyTimeout, stopTimeout});
    this.executable = executable || (fs.existsSync(path.join(root, 'bin/netsec-runtime.exe'))
      ? path.join(root, 'bin/netsec-runtime.exe') : path.join(root, 'rust/netsec/target/release/netsec-runtime.exe'));
    this.args = args || ['--root', root, '--parent-stdin'];
    this.stopping = false; this.stopPromise = null;
  }
  start() {
    return new Promise((resolve, reject) => {
      const owned = this.process = spawn(this.executable, this.args, {cwd: this.root,
        windowsHide: true, stdio: ['pipe', 'pipe', 'pipe'],
        env: {...process.env, IMDAI_SESSION_TOKEN: this.token, NETSEC_DESKTOP: '1'}});
      let settled = false, diagnostics = '';
      const finish = (error, origin) => {if (!settled) {settled = true; clearTimeout(timer); error ? reject(error) : resolve(origin);}};
      const timer = setTimeout(() => finish(new Error('The native runtime did not become ready within its startup deadline.')), this.readyTimeout);
      owned.stdin.on('error', () => {});
      owned.stderr.on('data', data => {diagnostics = (diagnostics + data.toString()).replaceAll(this.token, '[redacted]').slice(-1500);});
      const lines = readline.createInterface({input: owned.stdout});
      lines.on('line', line => {
        if (line.length > 8192) return;
        try {
          const message = JSON.parse(line);
          if (message.type === 'ready' && Number.isInteger(message.port) && message.port > 0 && message.port < 65536) finish(null, `http://127.0.0.1:${message.port}`);
          else if (message.type === 'window' && actions.has(message.action)) this.emit('window', message.action);
        } catch {}
      });
      owned.once('error', error => finish(new Error(error.message.replaceAll(this.token, '[redacted]'))));
      owned.once('exit', code => {
        lines.close();
        if (!settled) finish(new Error(`Native runtime exited (${code}). ${diagnostics}`));
        else if (!this.stopping) this.emit('failure', new Error('The native analysis service stopped. Restart the workspace to continue.'));
      });
    });
  }
  sendState(state) {
    const input = this.process?.stdin;
    if (input && !input.destroyed && !input.writableEnded) input.write(JSON.stringify({type:'window_state', state}) + '\n');
  }
  stop() {
    if (this.stopPromise) return this.stopPromise;
    this.stopping = true;
    this.stopPromise = new Promise(resolve => {
      const owned = this.process;
      if (!owned || owned.exitCode !== null || owned.signalCode !== null || !owned.pid) return resolve();
      const timer = setTimeout(() => owned.kill(), this.stopTimeout);
      owned.once('exit', () => {clearTimeout(timer); resolve();});
      if (!owned.stdin.destroyed && !owned.stdin.writableEnded) owned.stdin.end('shutdown\n');
    });
    return this.stopPromise;
  }
}
module.exports = {NativeRuntime};
