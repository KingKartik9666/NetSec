'use strict';
const fs = require('node:fs');
const path = require('node:path');
const crypto = require('node:crypto');
async function digest(file) {const hash=crypto.createHash('sha256');for await(const chunk of fs.createReadStream(file))hash.update(chunk);return hash.digest('hex');}
async function verifyNativeAssets(root, source) {
  const manifest=JSON.parse(fs.readFileSync(path.join(root,'runtime_assets.json'),'utf8'));
  if(manifest.version!==1||manifest.runtime!=='rust'||manifest.cpu_only!==true||!Array.isArray(manifest.files)||!manifest.files.length||manifest.files.length>2048)throw new Error('Stage the sealed native runtime before packaging.');
  const admitted=new Set();
  for(const entry of manifest.files){
    if(typeof entry.path!=='string'||entry.path.split(/[\\/]/).some(part=>part==='..'||part==='.'||!part))throw new Error('Invalid runtime asset path');
    const file=fs.realpathSync(path.resolve(root,entry.path));
    if(!file.startsWith(fs.realpathSync(root)+path.sep))throw new Error('Runtime asset escapes package');
    if(/(^|[\\/])(node_modules|target|__pycache__)([\\/]|$)|\.(py|pyc|pyd)$/i.test(entry.path))throw new Error('Development or Python artifacts cannot be packaged');
    if(await digest(file)!==entry.sha256)throw new Error(`Staged native asset changed: ${entry.path}`);
    if(source){const original=path.join(source,entry.path==='bin/netsec-runtime.exe'?'rust/netsec/target/release/netsec-runtime.exe':entry.path);if(fs.existsSync(original)&&await digest(original)!==entry.sha256)throw new Error(`Restage current native resources: ${entry.path}`);}
    admitted.add(entry.path);
  }
  for(const required of ['bin/netsec-runtime.exe','bin/vcruntime140.dll','rust/netsec/web/dist/index.html','artifacts/application/native_models_v1/catalog.json'])if(!admitted.has(required))throw new Error(`Native package missing ${required}`);
  function checkContents(directory){for(const entry of fs.readdirSync(directory,{withFileTypes:true})){const file=path.join(directory,entry.name);if(entry.isSymbolicLink())throw new Error('Package cannot contain symbolic links');if(entry.isDirectory())checkContents(file);else{const relative=path.relative(root,file).split(path.sep).join('/');if(relative!=='runtime_assets.json'&&!admitted.has(relative))throw new Error(`Unsealed package file: ${relative}`);}}}
  checkContents(root);
  const catalog=JSON.parse(fs.readFileSync(path.join(root,'artifacts/application/native_models_v1/catalog.json'),'utf8'));
  if(catalog.models.filter(model=>model.id!=='tabpack').length!==10)throw new Error('The visible native model scope must contain ten models');
  return manifest;
}
module.exports={verifyNativeAssets};
