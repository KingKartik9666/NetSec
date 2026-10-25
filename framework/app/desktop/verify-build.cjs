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
module.exports = async context => {
  if (context.electronPlatformName !== 'win32') throw new Error('This native release is verified for Windows x64 only.');
  await require('./verify-native-assets.cjs').verifyNativeAssets(path.resolve(__dirname, '../../../artifacts/application/package_root'), path.resolve(__dirname, '../../..'));
};
module.exports.verifyModelAssets = verifyModelAssets;
