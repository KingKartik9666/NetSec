const fs = require('node:fs');
const path = require('node:path');
const crypto = require('node:crypto');
function verifyModelAssets(root, backend, manifest) {
  const checksum = file => crypto.createHash('sha256').update(fs.readFileSync(file)).digest('hex');
  const names = ['xgboost_app', 'packet_live', 'xgboost_partial', 'xgboost_partial_enhanced'];
  for (const name of names) {
    const source = path.join(root, 'framework/models', name);
    const sourceBundle = path.join(source, 'bundle.json');
    if (!fs.existsSync(sourceBundle)) continue;
    if (!Object.hasOwn(manifest.checkpoint_sha256 || {}, name)) throw new Error(`Model missing from frozen backend: ${name}. Rebuild before packaging.`);
    const frozen = path.join(backend, '_internal/framework/models', name);
    try {
      const original = JSON.parse(fs.readFileSync(sourceBundle, 'utf8'));
      const expected = {...original, checkpoint: 'model.ubj'};
      const actual = JSON.parse(fs.readFileSync(path.join(frozen, 'bundle.json'), 'utf8'));
      if (JSON.stringify(expected) !== JSON.stringify(actual)) throw new Error('bundle metadata changed');
      const recorded = manifest.checkpoint_sha256[name];
      if (original.checkpoint_sha256 !== recorded || checksum(path.resolve(source, original.checkpoint)) !== recorded || checksum(path.join(frozen, 'model.ubj')) !== recorded) throw new Error('checkpoint changed');
      for (const file of ['xai_reference.json', 'xai_reference.npz']) {
        if (checksum(path.join(source, file)) !== checksum(path.join(frozen, file))) throw new Error('explanation references changed');
      }
    } catch (error) {
      throw new Error(`Stale model assets for ${name}: ${error.message}. Rebuild before packaging.`);
    }
  }
}
module.exports = async (context) => {
  const manifest = JSON.parse(fs.readFileSync(path.resolve(__dirname, '../../../artifacts/desktop/backend/imdai-backend/build-manifest.json'), 'utf8'));
  verifyModelAssets(path.resolve(__dirname, '../../..'), path.resolve(__dirname, '../../../artifacts/desktop/backend/imdai-backend'), manifest);
  if (manifest.platform !== context.electronPlatformName) throw new Error('Build the Python backend on the same OS as the desktop target. Do not package a Windows backend for Linux.');
  if (manifest.architecture !== 'x64' || !manifest.smoke_test_passed) throw new Error('A verified x64 native backend build is required.');
  if (!manifest.compiled_module_sha256 || !Object.hasOwn(manifest.compiled_module_sha256, 'framework.llm.local_model')) throw new Error('Cleanly rebuild and verify the embedded Python code before packaging.');
  const backendFile = path.resolve(__dirname, '../../../artifacts/desktop/backend/imdai-backend/imdai-backend.exe');
  const nativeBackend = context.electronPlatformName === 'win32' ? backendFile : backendFile.slice(0, -4);
  if (crypto.createHash('sha256').update(fs.readFileSync(nativeBackend)).digest('hex') !== manifest.executable_sha256) throw new Error('The frozen executable changed after compiled-code verification. Rebuild before packaging.');
  if (!manifest.runtime_source_sha256) throw new Error('Rebuild the backend with a recorded Python source revision before packaging.');
  for (const [relative, expected] of Object.entries(manifest.runtime_source_sha256)) {
    const filename = path.resolve(__dirname, '../../..', relative);
    if (crypto.createHash('sha256').update(fs.readFileSync(filename)).digest('hex') !== expected) throw new Error('Python runtime changed. Refreeze and verify the backend before packaging.');
  }
  if (!manifest.ui_assets_sha256) throw new Error('Refresh and verify the bundled UI assets before packaging.');
  for (const [relative, expected] of Object.entries(manifest.ui_assets_sha256)) {
    const source = path.resolve(__dirname, '../../..', relative);
    const bundled = path.resolve(__dirname, '../../../artifacts/desktop/backend/imdai-backend/_internal', relative);
    for (const filename of [source, bundled]) {
      if (crypto.createHash('sha256').update(fs.readFileSync(filename)).digest('hex') !== expected) throw new Error('UI assets changed. Run python -m experiment.build_desktop --refresh-assets before packaging.');
    }
  }
  const modelDir = path.resolve(__dirname, '../../../artifacts/llm/qwen');
  const file = path.join(modelDir, 'qwen2.5-0.5b-instruct-q4_k_m.gguf');
  const hash = crypto.createHash('sha256');
  for await (const chunk of fs.createReadStream(file)) hash.update(chunk);
  if (hash.digest('hex') !== '74a4da8c9fdbcd15bd1f6d01d621410d31c6fc00986f5eb687824e7b93d7a9db') throw new Error('The packaged small language model checksum does not match.');
};
module.exports.verifyModelAssets = verifyModelAssets;
