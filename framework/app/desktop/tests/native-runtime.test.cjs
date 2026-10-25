'use strict';
const test = require('node:test');
const assert = require('node:assert/strict');
const {NativeRuntime} = require('../native-runtime.cjs');

function runtime(script, overrides = {}) {
  return new NativeRuntime({executable: process.execPath, args: ['-e', script], root: process.cwd(),
    token: 'a'.repeat(64), readyTimeout: 700, stopTimeout: 700, ...overrides});
}
test('native protocol admits one bounded port and forwards fixed shell actions', async () => {
  const owned = runtime("console.log(JSON.stringify({type:'ready',port:12345}));process.stdin.on('data',d=>{if(String(d).includes('shutdown'))process.exit(0)});setTimeout(()=>console.log(JSON.stringify({type:'window',action:'request_close'})),25)");
  let action;
  owned.on('window', value => {action = value;});
  assert.equal(await owned.start(), 'http://127.0.0.1:12345');
  await new Promise(resolve => setTimeout(resolve, 80));
  assert.equal(action, 'request_close');
  await owned.stop();
  assert.equal(owned.process.exitCode, 0);
});
test('invalid ready messages time out and startup errors redact session credentials', async () => {
  const owned = runtime("console.error(process.env.IMDAI_SESSION_TOKEN);console.log(JSON.stringify({type:'ready',port:65536}));setInterval(()=>{},100)");
  await assert.rejects(owned.start(), error => !error.message.includes('a'.repeat(64)) && /ready/.test(error.message));
  await owned.stop();
  assert.ok(owned.process.exitCode !== null || owned.process.signalCode !== null);
});
test('unexpected exit is observable and stop is idempotent', async () => {
  const owned = runtime("console.log(JSON.stringify({type:'ready',port:12345}));setTimeout(()=>process.exit(7),80)");
  const failure = new Promise(resolve => owned.once('failure', resolve));
  await owned.start();
  assert.match((await failure).message, /stopped/);
  await Promise.all([owned.stop(), owned.stop()]);
  assert.equal(owned.process.exitCode, 7);
});
test('a hung runtime is terminated after bounded graceful shutdown', async () => {
  const owned = runtime("console.log(JSON.stringify({type:'ready',port:12345}));setInterval(()=>{},100)");
  await owned.start();
  await owned.stop();
  assert.ok(owned.process.exitCode !== null || owned.process.signalCode !== null);
});
