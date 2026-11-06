// Generates the Graphite & Iris palette from the original light stylesheets:
//   public/static/theme-light.css  cool porcelain neutrals; the brand coral becomes iris
//   public/static/theme-dark.css   the same palette inverted into deep graphite
// Every colour-bearing declaration is re-emitted in its original order under a zero-specificity scope, so
// specificity and cascade order match the source. Semantic colours (errors, negative SHAP, suspicious
// windows, warnings, success) keep their hue in both themes; only neutrals and the brand accent move.
// Run: node scripts/build-dark-theme.mjs  (runs on every `npm run build`).
import {readFileSync, writeFileSync} from 'node:fs';
import {fileURLToPath} from 'node:url';
import path from 'node:path';
import postcss from 'postcss';

const here = path.dirname(fileURLToPath(import.meta.url));
const staticDir = path.join(here, '..', 'public', 'static');
const sources = ['style.css', 'studio.css', 'models.css', 'native.css'];
const colourProps = /^(--.+|color|background(-color|-image)?|border(-(top|right|bottom|left))?(-color)?|outline(-color)?|box-shadow|text-shadow|fill|stroke|caret-color|accent-color|text-decoration(-color)?|column-rule(-color)?|scrollbar-color|filter|-webkit-text-stroke(-color)?)$/;
const literal = /#[0-9a-fA-F]{3,8}\b|\b(white|black)\b/g;
// Rules whose colour carries meaning rather than brand.
const semantic = /error|negative|suspicious|decision-badge|failed|unavailable|warning|danger|attack|malicious|invalid|clos(e|ing)|positive|success|benign/i;
const deg = d => d * Math.PI / 180;
const NEUTRAL_HUE = deg(268), IRIS_HUE = deg(277);
// sRGB <-> OKLCH (Björn Ottosson's OKLab).
const toLinear = c => c <= 0.04045 ? c / 12.92 : ((c + 0.055) / 1.055) ** 2.4;
const toGamma = c => c <= 0.0031308 ? 12.92 * c : 1.055 * c ** (1 / 2.4) - 0.055;
function rgbToOklch([r, g, b]) {
  [r, g, b] = [r, g, b].map(toLinear);
  const l = Math.cbrt(0.4122214708 * r + 0.5363325363 * g + 0.0514459929 * b);
  const m = Math.cbrt(0.2119034982 * r + 0.6806995451 * g + 0.1073969566 * b);
  const s = Math.cbrt(0.0883024619 * r + 0.2817188376 * g + 0.6299787005 * b);
  const L = 0.2104542553 * l + 0.793617785 * m - 0.0040720468 * s;
  const A = 1.9779984951 * l - 2.428592205 * m + 0.4505937099 * s;
  const B = 0.0259040371 * l + 0.7827717662 * m - 0.808675766 * s;
  return [L, Math.hypot(A, B), Math.atan2(B, A)];
}
function oklchToRgb([L, C, H]) {
  const A = C * Math.cos(H), B = C * Math.sin(H);
  const l = (L + 0.3963377774 * A + 0.2158037573 * B) ** 3;
  const m = (L - 0.1055613458 * A - 0.0638541728 * B) ** 3;
  const s = (L - 0.0894841775 * A - 1.291485548 * B) ** 3;
  return [4.0767416621 * l - 3.3077115913 * m + 0.2309699292 * s,
    -1.2684380046 * l + 2.6097574011 * m - 0.3413193965 * s,
    -0.0041960863 * l - 0.7034186147 * m + 1.707614701 * s].map(v => toGamma(Math.min(1, Math.max(0, v))));
}
function parse(token) {
  if (token === 'white') return [[1, 1, 1], 1];
  if (token === 'black') return [[0, 0, 0], 1];
  let hex = token.slice(1);
  if (hex.length <= 4) hex = [...hex].map(c => c + c).join('');
  const rgb = [0, 2, 4].map(i => parseInt(hex.slice(i, i + 2), 16) / 255);
  return [rgb, hex.length === 8 ? parseInt(hex.slice(6, 8), 16) / 255 : 1];
}
const hex2 = v => Math.round(Math.min(1, Math.max(0, v)) * 255).toString(16).padStart(2, '0');
const format = (rgb, alpha) => '#' + rgb.map(hex2).join('') + (alpha < 1 ? hex2(alpha) : '');
const hueDeg = H => (H * 180 / Math.PI + 360) % 360;

// Light palette: warm neutrals turn cool porcelain/graphite at the same lightness and the brand coral becomes
// iris. Semantic rules keep their colour exactly, and amber/sand tones (hue 46-90) are left alone everywhere:
// in these stylesheets they mark warnings, missing inputs and non-benign predictions.
function toLight([L, C, H], isSemantic) {
  const h = hueDeg(H);
  if (isSemantic || (h > 46 && h < 90 && C >= 0.015)) return [L, C, H];
  if (C < 0.05) {
    const chroma = L > 0.985 || L < 0.03 ? 0.002 : Math.min(0.006 + C * 0.4, 0.022);
    return [L, chroma, NEUTRAL_HUE];
  }
  if (h >= 20 && h <= 46) return [L > 0.8 ? L : Math.min(L, 0.6) - 0.04, Math.min(C * 1.08, 0.2), IRIS_HUE];
  return [L, C, H];
}
// Dark palette: pale surfaces become graphite (lighter surfaces stay slightly lighter, preserving elevation),
// hairlines become mid-dark rules, ink inverts to cool off-white, accents stay vivid.
function mapLightness(L) {
  if (L >= 0.93) return 0.17 + (L - 0.93) / 0.07 * 0.04;
  if (L >= 0.80) return 0.34 - (L - 0.80) / 0.13 * 0.09;
  return Math.min(0.94, 1.05 - 0.62 * L);
}
function toDark([L, C, H], prop, isSemantic) {
  if (/shadow/.test(prop) || prop === 'filter') return [Math.min(L, 0.07), Math.min(C, 0.02), H];  // depth stays dark
  // Pale semantic tints (error, warning, failed surfaces) become dark tints that keep their hue.
  if (isSemantic && C >= 0.012 && C < 0.06 && L > 0.8) return [Math.min(0.3, Math.max(0.22, mapLightness(L))), Math.max(C, 0.035), H];
  if (C >= 0.06) {
    if (L > 0.85) return [0.27, Math.min(C, 0.05), H];                                             // pale tint -> dark tint
    return [Math.max(L, 0.72), Math.min(C * 1.05, 0.19), H];                                       // accent stays vivid
  }
  return [mapLightness(L), Math.min(Math.max(C, 0.008) * 1.2, 0.03), H];
}
function remap(token, prop, isSemantic, dark) {
  let [rgb, alpha] = parse(token);
  let colour = toLight(rgbToOklch(rgb), isSemantic);
  if (dark) {
    colour = toDark(colour, prop, isSemantic);
    if (/shadow/.test(prop) || prop === 'filter') alpha = Math.min(1, alpha * 1.8);
  }
  return format(oklchToRgb(colour), alpha);
}

function build(dark) {
  const scope = dark ? ':where(:root[data-theme="dark"])' : ':where(:root)';
  let out = `/* Generated by scripts/build-dark-theme.mjs from ${sources.join(', ')}. Do not edit by hand. */\n`;
  for (const file of sources) {
    const rootNode = postcss.parse(readFileSync(path.join(staticDir, file), 'utf8'), {from: file});
    const result = postcss.root();
    rootNode.walkRules(rule => {
      if (rule.parent.type === 'atrule' && /keyframes$/i.test(rule.parent.name)) return;
      // Every colour-bearing declaration is re-emitted, literal or not (none, transparent, var(), url()), so a
      // later rule overridden by an earlier one keeps losing in both generated themes too.
      const decls = rule.nodes.filter(node => node.type === 'decl' && colourProps.test(node.prop));
      if (!decls.length) return;
      const selector = rule.selectors.map(sel => {
        const trimmed = sel.trim();
        if (/^(:root|html)\b/.test(trimmed)) return dark ? trimmed.replace(/^(:root|html)/, ':root[data-theme="dark"]') : trimmed;
        return `${scope} ${trimmed}`;
      }).join(', ');
      const copy = postcss.rule({selector});
      for (const decl of decls) {
        const isSemantic = semantic.test(`${rule.selector} ${decl.prop}`);
        copy.append(postcss.decl({prop: decl.prop, important: decl.important,
          value: /url\(/i.test(decl.value) ? decl.value : decl.value.replace(literal, token => remap(token, decl.prop, isSemantic, dark))}));
      }
      // Preserve enclosing @media/@supports so responsive overrides keep their cascade position.
      let target = result;
      const chain = [];
      for (let parent = rule.parent; parent && parent.type === 'atrule'; parent = parent.parent) chain.unshift(parent);
      for (const at of chain) {
        const wrapper = postcss.atRule({name: at.name, params: at.params});
        target.append(wrapper); target = wrapper;
      }
      target.append(copy);
    });
    out += `/* ${file} */\n` + result.toString() + '\n';
  }
  if (dark) out += ':root[data-theme="dark"] {color-scheme: dark;}\n';
  return out;
}
writeFileSync(path.join(staticDir, 'theme-light.css'), build(false));
writeFileSync(path.join(staticDir, 'theme-dark.css'), build(true));
console.log('theme-light.css and theme-dark.css written');
