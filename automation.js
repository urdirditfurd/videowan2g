'use strict';

require('dotenv').config({ quiet: true });

const crypto = require('crypto');
const dns = require('dns');
const fs = require('fs');
const net = require('net');
const os = require('os');
const path = require('path');
const { execFile } = require('child_process');
const { promisify } = require('util');
const nodeFetch = require('node-fetch');
const { simpleGit } = require('simple-git');

const execFileAsync = promisify(execFile);

const API_BASE = 'https://apihub.agnes-ai.com/v1';
const POLL_BASE = 'https://apihub.agnes-ai.com/agnesapi';
const API_HOST = 'apihub.agnes-ai.com';
const MODEL_TEXT = 'agnes-2.5-flash';
const MODEL_IMAGE = 'agnes-image-2.5-flash';
const MODEL_VIDEO = 'agnes-video-2.5-flash';
const ALLOWED_REPO = 'https://github.com/urdirditfurd/videowan2g';
const ROOT = __dirname;
const VIDEO_DIR = path.join(ROOT, 'generated_videos');
const TIMING_PATH = path.join(VIDEO_DIR, 'timing-cache.json');
const MAX_VIDEO_BYTES = 100 * 1024 * 1024;
const DOWNLOAD_MAX_REDIRECTS = 2;
const QUEUE_BACKOFF_SECONDS = [60, 120, 300];
const MAX_QUEUE_WAIT_SEC = 360;
const DEFAULT_QUEUE_RETRIES = 12;
const IMAGE_INTERVAL_MS = 3400;
const IMAGE_INTERVAL_MIN = 3200;
const IMAGE_INTERVAL_MAX = 12000;
const VIDEO_INTERVAL_MIN = 62000;
const VIDEO_INTERVAL_MAX = 90000;
const VIDEO_INTERVAL_SAFE = 75000;
const VIDEO_FIRST_POLL_DELAY_SEC = 80;
const VIDEO_MAX_POLL_ATTEMPTS = 100;
const WRITE_FLAGS = fs.constants.O_CREAT | fs.constants.O_EXCL | fs.constants.O_WRONLY | fs.constants.O_NOFOLLOW;
const READ_FLAGS = fs.constants.O_RDONLY | fs.constants.O_NOFOLLOW;

const GIT_ENV_ALLOW = [
  'PATH',
  'HOME',
  'GIT_CONFIG_NOSYSTEM',
  'GIT_TERMINAL_PROMPT',
  'GIT_AUTHOR_NAME',
  'GIT_AUTHOR_EMAIL',
  'GIT_COMMITTER_NAME',
  'GIT_COMMITTER_EMAIL',
  'GIT_ASKPASS',
];

const GHIBLI_PALETTE = 'soft pastel palette inspired by Studio Ghibli watercolor backgrounds: sage green #8fad88, warm cream #f2efe9, dusky blue #7a9eb8, soft gold #c4a882, gentle forest greens, muted earth tones, warm amber highlights. Never grey, never desaturated, never neon. Rich but calm. Hand-painted watercolor feel with soft gradients.';

const MOODS = {
  rainy_forest: { label: 'Forêt sous la pluie', desc: 'a lush green forest under soft rain, misty atmosphere, warm lamplight glowing through wet leaves, wet earth and moss details' },
  golden_sunset: { label: 'Coucher de soleil doré', desc: 'a golden hour sunset with warm amber light, long soft shadows, glowing dust motes in the air, nostalgic warmth' },
  cozy_room: { label: 'Intérieur douillet', desc: 'a cozy wooden interior with warm candlelight, soft blankets, steaming tea, rain visible through a window, amber glow' },
  japanese_garden: { label: 'Jardin japonais', desc: 'a serene Japanese garden with stone lanterns, koi pond, cherry blossoms, moss-covered rocks, soft diffused daylight' },
  mountain_lake: { label: 'Lac de montagne', desc: 'a still mountain lake reflecting soft clouds, pine trees, gentle morning mist, cool blue and sage tones' },
  autumn_leaves: { label: "Feuilles d'automne", desc: 'an autumn forest with falling orange and gold leaves, warm late afternoon light, soft breeze, rustic wooden path' },
  night_stars: { label: 'Nuit étoilée', desc: 'a peaceful night with a starry sky, soft moonlight, silhouetted trees, gentle blue and violet tones, distant fireflies' },
  ocean_breeze: { label: 'Brise océane', desc: 'a calm ocean shore with gentle waves, soft sea breeze, warm sand, pastel sky, distant seagulls, serene horizon' },
};

const TIMES_OF_DAY = ['aube pâle', 'matin brumeux', 'fin de matinée claire', 'après-midi doré', 'heure bleue', 'crépuscule', 'nuit calme', 'petite heure avant l aube'];
const SEASONS = ['début du printemps', 'plein printemps', 'début d été', 'fin d été', 'début d automne', 'automne tardif', 'début d hiver', 'cœur de l hiver'];
const WEATHERS = ['pluie fine', 'bruine', 'ciel dégagé', 'nuages laiteux', 'neige légère', 'brume au sol', 'après l averse', 'vent très léger'];

const SYSTEM_PROMPT = 'You are a Studio Ghibli-inspired art director specialized in lofi ambience videos. You always return strict JSON matching the requested schema exactly. Every scene includes a cute calm capybara. Every scene is peaceful, detailed, watercolor, and loopable with slow gentle motion. No text, no watermark. Return only JSON.';

let imageIntervalMs = IMAGE_INTERVAL_MS;
let videoIntervalMs = VIDEO_INTERVAL_SAFE;
let imageAttemptsSinceLastError = 0;
let videoAttemptsSinceLastError = 0;

const state = { stopRequested: false };
const loggingEnabled = !process.argv.includes('--self-test');

function log(level, category, message) {
  if (!loggingEnabled) return;
  console.log(`${new Date().toISOString()} [${level}] [${category}] ${message}`);
}

function sleep(ms) {
  return new Promise((resolve, reject) => {
    const step = 1000;
    let left = ms;
    const tick = () => {
      if (state.stopRequested) {
        reject(new Error('Arrêt demandé'));
        return;
      }
      if (left <= 0) {
        resolve();
        return;
      }
      const wait = Math.min(step, left);
      left -= wait;
      setTimeout(tick, wait);
    };
    tick();
  });
}

function resetRateState() {
  imageIntervalMs = IMAGE_INTERVAL_MS;
  videoIntervalMs = VIDEO_INTERVAL_SAFE;
  imageAttemptsSinceLastError = 0;
  videoAttemptsSinceLastError = 0;
}

function pick(list, randomFn = Math.random) {
  const index = Math.floor(randomFn() * list.length);
  return list[Math.max(0, Math.min(list.length - 1, index))];
}

function randomVariety(randomFn = Math.random) {
  return {
    timeOfDay: pick(TIMES_OF_DAY, randomFn),
    season: pick(SEASONS, randomFn),
    weather: pick(WEATHERS, randomFn),
  };
}

function sanitizeTitle(title) {
  const base = String(title || 'ambiance lofi')
    .normalize('NFKD')
    .replace(/[\u0300-\u036f]/g, '')
    .replace(/[^a-zA-Z0-9]+/g, '_')
    .replace(/^_+|_+$/g, '')
    .slice(0, 60);
  return base || 'ambiance_lofi';
}

function videoOutputPath(title, now) {
  const name = `${sanitizeTitle(title)}_${now}.mp4`;
  if (name !== path.basename(name)) throw new Error('Nom de fichier invalide');
  const filePath = path.resolve(VIDEO_DIR, name);
  const relative = path.relative(VIDEO_DIR, filePath);
  if (relative.startsWith('..') || path.isAbsolute(relative)) throw new Error('Chemin vidéo hors dossier');
  return filePath;
}

function stripBrackets(hostname) {
  return String(hostname || '').replace(/^\[|\]$/g, '').toLowerCase();
}

function ipv4ToInt(ip) {
  const parts = ip.split('.');
  if (parts.length !== 4) return null;
  let value = 0;
  for (const part of parts) {
    if (!/^\d+$/.test(part)) return null;
    const n = Number(part);
    if (n > 255) return null;
    value = (value * 256) + n;
  }
  return value >>> 0;
}

function isPrivateIPv4(ip) {
  const n = ipv4ToInt(ip);
  if (n === null) return true;
  const inRange = (start, end) => n >= start && n <= end;
  if (inRange(0x00000000, 0x00ffffff)) return true;
  if (inRange(0x0a000000, 0x0affffff)) return true;
  if (inRange(0x7f000000, 0x7fffffff)) return true;
  if (inRange(0xa9fe0000, 0xa9feffff)) return true;
  if (inRange(0xac100000, 0xac1fffff)) return true;
  if (inRange(0xc0a80000, 0xc0a8ffff)) return true;
  if (inRange(0x64400000, 0x647fffff)) return true;
  if (inRange(0xe0000000, 0xffffffff)) return true;
  return false;
}

function mappedIPv4(host) {
  const dotted = host.match(/^::ffff:(\d{1,3}(?:\.\d{1,3}){3})$/i);
  if (dotted) return dotted[1];
  const hex = host.match(/^::ffff:([0-9a-f]{1,4}):([0-9a-f]{1,4})$/i);
  if (!hex) return null;
  const hi = parseInt(hex[1], 16);
  const lo = parseInt(hex[2], 16);
  return `${(hi >> 8) & 255}.${hi & 255}.${(lo >> 8) & 255}.${lo & 255}`;
}

function isPrivateIPv6(host) {
  const h = host.toLowerCase();
  if (h === '::1' || h === '::' || h === '0:0:0:0:0:0:0:1') return true;
  const mapped = mappedIPv4(h);
  if (mapped) return isPrivateIPv4(mapped);
  const first = h.split(':').find((part) => part.length > 0) || '';
  const value = parseInt(first, 16);
  if (Number.isNaN(value)) return true;
  if ((value & 0xfe00) === 0xfc00) return true;
  if ((value & 0xffc0) === 0xfe80) return true;
  if ((value & 0xff00) === 0xff00) return true;
  return false;
}

function isBlockedHostname(hostname) {
  const host = stripBrackets(hostname);
  if (!host) return true;
  if (host === 'localhost' || host.endsWith('.localhost') || host.endsWith('.local')) return true;
  if (host === 'metadata.google.internal' || host.endsWith('.internal')) return true;
  const family = net.isIP(host);
  if (family === 4) return isPrivateIPv4(host);
  if (family === 6) return isPrivateIPv6(host);
  return false;
}

async function defaultResolve(hostname) {
  const records = await dns.promises.lookup(hostname, { all: true, verbatim: true });
  return records.map((record) => record.address);
}

async function assertPublicHttpsUrl(rawUrl, deps = {}) {
  let url;
  try {
    url = new URL(rawUrl);
  } catch (err) {
    throw new Error('URL invalide');
  }
  if (url.protocol !== 'https:') throw new Error('Seul HTTPS est accepté');
  if (url.username || url.password) throw new Error("Identifiants interdits dans l'URL");
  const host = stripBrackets(url.hostname);
  if (isBlockedHostname(host)) throw new Error('Hôte non public refusé');
  if (net.isIP(host) === 0) {
    const resolve = deps.resolve || defaultResolve;
    const addresses = await resolve(host);
    if (!Array.isArray(addresses) || addresses.length === 0) throw new Error('Résolution DNS vide');
    for (const address of addresses) {
      if (isBlockedHostname(address)) throw new Error('Résolution vers une adresse privée refusée');
    }
  }
  return url;
}

function assertAgnesUrl(rawUrl) {
  let url;
  try {
    url = new URL(rawUrl);
  } catch (err) {
    throw new Error('URL API invalide');
  }
  if (url.protocol !== 'https:') throw new Error('API: HTTPS requis');
  if (url.username || url.password) throw new Error('API: identifiants interdits');
  if (stripBrackets(url.hostname) !== API_HOST) throw new Error('API: hôte non autorisé');
}

function assertAllowedRepo(rawUrl) {
  let url;
  try {
    url = new URL(rawUrl);
  } catch (err) {
    throw new Error('URL de dépôt invalide');
  }
  if (url.protocol !== 'https:') throw new Error('Le dépôt doit être en HTTPS');
  if (url.username || url.password) throw new Error("Identifiants interdits dans l'URL du dépôt");
  const pathname = url.pathname.replace(/\/+$/, '').replace(/\.git$/, '');
  if (`${url.protocol}//${url.hostname}${pathname}` !== ALLOWED_REPO) throw new Error('Dépôt non autorisé');
}

function escapeControlCharsInStrings(text) {
  let result = '';
  let inString = false;
  let escaped = false;
  for (let i = 0; i < text.length; i += 1) {
    const ch = text[i];
    if (escaped) {
      result += ch;
      escaped = false;
      continue;
    }
    if (ch === '\\') {
      result += ch;
      escaped = true;
      continue;
    }
    if (ch === '"') {
      inString = !inString;
      result += ch;
      continue;
    }
    if (inString) {
      if (ch === '\n') { result += '\\n'; continue; }
      if (ch === '\r') { result += '\\r'; continue; }
      if (ch === '\t') { result += '\\t'; continue; }
      const code = ch.charCodeAt(0);
      if (code < 32) {
        result += `\\u${code.toString(16).padStart(4, '0')}`;
        continue;
      }
    }
    result += ch;
  }
  return result;
}

function balanceJson(text) {
  let openBraces = 0;
  let openBrackets = 0;
  let inString = false;
  let escaped = false;
  for (let i = 0; i < text.length; i += 1) {
    const ch = text[i];
    if (escaped) { escaped = false; continue; }
    if (ch === '\\') { escaped = true; continue; }
    if (ch === '"') { inString = !inString; continue; }
    if (inString) continue;
    if (ch === '{') openBraces += 1;
    else if (ch === '}') openBraces -= 1;
    else if (ch === '[') openBrackets += 1;
    else if (ch === ']') openBrackets -= 1;
  }
  let result = text;
  if (inString) result += '"';
  while (openBrackets > 0) { result += ']'; openBrackets -= 1; }
  while (openBraces > 0) { result += '}'; openBraces -= 1; }
  return result;
}

function salvageExpansion(text) {
  const titleMatch = text.match(/"title"\s*:\s*"((?:[^"\\]|\\.)*)"/);
  const imgMatch = text.match(/"image_prompt"\s*:\s*"((?:[^"\\]|\\.)*)"/);
  const vidMatch = text.match(/"video_motion_prompt"\s*:\s*"((?:[^"\\]|\\.)*)"/);
  if (!imgMatch && !vidMatch) return null;
  const clean = (value) => value.replace(/\\n/g, ' ').replace(/\\"/g, '"');
  return {
    title: titleMatch ? clean(titleMatch[1]) : 'Ambiance Lofi',
    image_prompt: imgMatch ? clean(imgMatch[1]) : '',
    video_motion_prompt: vidMatch ? clean(vidMatch[1]) : '',
  };
}

function extractJson(text) {
  if (!text) return null;
  let source = String(text).trim();
  const fenced = source.match(/```(?:json)?\s*([\s\S]*?)```/);
  if (fenced) source = fenced[1].trim();
  const start = source.indexOf('{');
  if (start < 0) return null;
  source = source.slice(start);
  const attempts = [
    () => JSON.parse(source),
    () => JSON.parse(escapeControlCharsInStrings(source)),
    () => {
      const end = source.lastIndexOf('}');
      if (end <= 0) throw new Error('accolade manquante');
      return JSON.parse(escapeControlCharsInStrings(source.slice(0, end + 1)));
    },
    () => JSON.parse(balanceJson(escapeControlCharsInStrings(source))),
    () => {
      const salvaged = salvageExpansion(source);
      if (!salvaged || (!salvaged.image_prompt && !salvaged.video_motion_prompt)) throw new Error('salvage vide');
      return salvaged;
    },
  ];
  for (const attempt of attempts) {
    try {
      return attempt();
    } catch (err) {
      continue;
    }
  }
  return null;
}

function buildExpansionPrompt(userDescription, mood, variety) {
  const chosen = variety || randomVariety();
  const moodInfo = MOODS[mood] || MOODS.rainy_forest;
  return [
    'You are an art director for Studio Ghibli-inspired lofi ambience videos.',
    '',
    'USER SCENE IDEA (French):',
    userDescription,
    '',
    'AMBIENCE (French mood chosen):',
    `${moodInfo.label} — ${moodInfo.desc}`,
    '',
    'UNIQUE VARIATION (mandatory, make this video visually distinct):',
    `- Time of day: ${chosen.timeOfDay}`,
    `- Season: ${chosen.season}`,
    `- Weather: ${chosen.weather}`,
    'These three must be visibly reflected in both image_prompt and video_motion_prompt.',
    'Even if the base idea is familiar, the light, season and weather must make this frame unique.',
    '',
    'YOUR TASK:',
    'Expand the user idea into a rich, poetic, ultra-detailed visual scene for a single lofi ambience video.',
    'ALWAYS include a cute, calm, peaceful capybara (capybara mascot) somewhere visible in the scene, integrated naturally into the environment.',
    'The mood must be lofi: slow, nostalgic, peaceful, detailed, soothing, hand-painted watercolor, Studio Ghibli / Hayao Miyazaki inspired.',
    '',
    'HARD RULES:',
    '- The image must feel like a Ghibli watercolor background: soft gradients, painterly, detailed, calm, warm.',
    '- The scene must be peaceful and loopable in motion.',
    '- The capybara must appear cute but not intrusive — it can be resting, soaking in water, sitting on a wooden porch, next to a teacup, under a blanket, etc.',
    '- The video motion must be SUBTLE and SLOW: gentle breathing of the capybara, softly falling leaves, drifting mist, slow water ripples, faint swaying grass, floating dust motes.',
    '- No fast movement, no action, no characters running, no camera shake. Perfect seamless loop.',
    '- No readable text, no watermark, no logos.',
    `- Color palette: ${GHIBLI_PALETTE}`,
    '',
    'OUTPUT — strict JSON only, no markdown:',
    '{',
    '  "title": "A short poetic title in French (max 60 chars)",',
    '  "image_prompt": "Ultra-detailed English visual description for an image model. Include: setting, lighting, colors, atmosphere, the capybara (with its location and pose), the unique time/season/weather, Ghibli watercolor style, cinematic composition. 150-250 words.",',
    '  "video_motion_prompt": "English description of slow gentle motions for the video model: capybara breathing, leaves falling, mist drifting, water ripples, seamless loop. Reflect the weather. 60-120 words."',
    '}',
  ].join('\n');
}

function buildGhibliStyleBlock() {
  return [
    'VISUAL STYLE: Studio Ghibli / Hayao Miyazaki inspired hand-painted watercolor background. Soft painterly gradients, detailed natural textures, warm nostalgic atmosphere.',
    'COMPOSITION: cinematic, peaceful, balanced, with gentle depth. The capybara is clearly visible, cute, calm.',
    'LIGHTING: soft natural light, warm amber highlights, gentle shadow, no harsh contrast.',
    `COLOR RULE: ${GHIBLI_PALETTE}`,
    'MANDATORY: a cute peaceful capybara integrated naturally into the scene.',
    'NO readable text, NO watermark, NO logos.',
  ].join('\n');
}

function buildLofiImagePrompt(expansion) {
  return [
    'GHIBLI-STYLE LOFI AMBIENCE KEYFRAME.',
    expansion.image_prompt,
    buildGhibliStyleBlock(),
    'Ultra-detailed painterly image, calm and loopable composition.',
  ].join('\n');
}

function buildLofiVideoPrompt(expansion, durationSeconds) {
  return [
    `LOFI AMBIENCE VIDEO — ${durationSeconds} seconds — seamless loop.`,
    'Studio Ghibli inspired watercolor scene with a cute calm capybara.',
    `BASE SCENE: ${expansion.image_prompt}`,
    `GENTLE MOTION: ${expansion.video_motion_prompt}`,
    'ABSOLUTE MOTION RULE: SLOW, GENTLE, SUBTLE motion only. No fast movement, no camera shake, no action. The scene must feel peaceful and loopable.',
    'Perfect seamless loop, calm pacing.',
    'CAMERA: locked-off or extremely slow drift. No cuts.',
    'STYLE: hand-painted Ghibli watercolor, soft gradients, warm nostalgic light.',
    `PALETTE: ${GHIBLI_PALETTE}`,
    'SOUND: soft ambient nature sounds only — no music, no voice. Wind, birds, water, rustling leaves.',
    'NO readable text, NO watermark, NO logos.',
  ].join('\n');
}

function queueBackoffSeconds(queueAttempt, randomFn = Math.random) {
  const index = Math.min(queueAttempt, QUEUE_BACKOFF_SECONDS.length - 1);
  const factor = 0.8 + (randomFn() * 0.4);
  const seconds = Math.round(QUEUE_BACKOFF_SECONDS[index] * factor);
  return Math.min(MAX_QUEUE_WAIT_SEC, Math.max(1, seconds));
}

function queueWaitSeconds(queueAttempt, retryAfter, randomFn) {
  if (retryAfter > 0) return Math.min(MAX_QUEUE_WAIT_SEC, retryAfter);
  return queueBackoffSeconds(queueAttempt, randomFn);
}

function headerInt(res, name) {
  try {
    return parseInt(res.headers.get(name) || '0', 10) || 0;
  } catch (err) {
    return 0;
  }
}

function noteSuccess(kind) {
  if (kind === 'image') {
    imageAttemptsSinceLastError += 1;
    if (imageAttemptsSinceLastError >= 4 && imageIntervalMs > IMAGE_INTERVAL_MIN) {
      imageIntervalMs = Math.max(IMAGE_INTERVAL_MIN, imageIntervalMs - 250);
      imageAttemptsSinceLastError = 0;
    }
  }
  if (kind === 'video') {
    videoAttemptsSinceLastError += 1;
    if (videoAttemptsSinceLastError >= 2 && videoIntervalMs > VIDEO_INTERVAL_MIN) {
      videoIntervalMs = Math.max(VIDEO_INTERVAL_MIN, videoIntervalMs - 3000);
      videoAttemptsSinceLastError = 0;
    }
  }
}

function terminalError(message) {
  const error = new Error(message);
  error.terminal = true;
  return error;
}

async function readErrorText(res) {
  try {
    const source = typeof res.clone === 'function' ? res.clone() : res;
    return String(await source.text()).slice(0, 1200);
  } catch (err) {
    return '';
  }
}

async function apiFetch(url, options = {}, label = 'API', kind = 'default', deps = {}) {
  assertAgnesUrl(url);
  const sleepFn = deps.sleep || sleep;
  const fetchImpl = deps.fetch || nodeFetch;
  const randomFn = deps.random || Math.random;
  const queueMax = deps.queueMaxRetries || queueRetryLimit();
  const maxAttempts = kind === 'video' || kind === 'image' ? 10 : 8;
  let attempt = 0;
  let queueRetries = 0;

  while (attempt < maxAttempts) {
    if (state.stopRequested) throw terminalError('Arrêt demandé');
    let res;
    try {
      res = await fetchImpl(url, { ...options, redirect: 'error' });
    } catch (err) {
      if (err && err.message === 'Arrêt demandé') throw err;
      if (err && err.terminal) throw err;
      const wait = Math.min(kind === 'video' ? 120 : 45, kind === 'video' ? 10 + attempt * 8 : 3 + attempt * 4);
      log('warn', 'NETWORK', `${label} — ${err.message} — reprise dans ${wait}s`);
      await sleepFn(wait * 1000);
      attempt += 1;
      continue;
    }

    const retryAfter = headerInt(res, 'retry-after');
    let errorText = '';
    if (!res.ok) errorText = await readErrorText(res);
    if (res.ok) {
      noteSuccess(kind);
      return res;
    }

    const queueFull = /video_queue_full|queue is full/i.test(errorText);
    if (kind === 'video' && queueFull) {
      if (queueRetries >= queueMax) throw terminalError('File vidéo saturée — répétitions épuisées');
      videoAttemptsSinceLastError = 0;
      const wait = queueWaitSeconds(queueRetries, retryAfter, randomFn);
      queueRetries += 1;
      log('warn', 'VIDEO_QUEUE', `File vidéo saturée — pause ${wait}s (${queueRetries}/${queueMax})`);
      await sleepFn(wait * 1000);
      continue;
    }

    if ([400, 401, 404].includes(res.status) && kind !== 'video') {
      throw terminalError(`HTTP ${res.status} — ${errorText.slice(0, 300)}`);
    }
    if (res.status === 429) {
      const wait = retryAfter > 0
        ? Math.min(MAX_QUEUE_WAIT_SEC, retryAfter)
        : Math.min(kind === 'video' ? 180 : 120, kind === 'video' ? 60 + attempt * 8 : 10 + attempt * 12);
      if (kind === 'image') imageIntervalMs = Math.min(IMAGE_INTERVAL_MAX, imageIntervalMs + 1000);
      if (kind === 'video') videoIntervalMs = Math.min(VIDEO_INTERVAL_MAX, videoIntervalMs + 6000);
      log('warn', 'RATELIMIT', `${label} — 429 — pause ${wait}s`);
      await sleepFn(wait * 1000);
      attempt += 1;
      continue;
    }
    if ([500, 502, 503, 504, 520].includes(res.status)) {
      const wait = Math.min(kind === 'video' ? 180 : 60, kind === 'video' ? 20 + attempt * 10 : 5 + attempt * 5);
      log('warn', 'SERVICE', `${label} — HTTP ${res.status} — pause ${wait}s`);
      await sleepFn(wait * 1000);
      attempt += 1;
      continue;
    }
    return res;
  }
  throw new Error(`Tentatives épuisées pour ${label} après ${maxAttempts} essais`);
}

function queueRetryLimit() {
  const parsed = Number(process.env.VIDEO_QUEUE_MAX_RETRIES || DEFAULT_QUEUE_RETRIES);
  if (!Number.isInteger(parsed) || parsed < 1 || parsed > 100) return DEFAULT_QUEUE_RETRIES;
  return parsed;
}

async function retryTask(name, fn, maxAttempts = 3, deps = {}) {
  const sleepFn = deps.sleep || sleep;
  let lastError = null;
  for (let i = 1; i <= maxAttempts; i += 1) {
    if (state.stopRequested) throw terminalError('Arrêt demandé');
    try {
      log('info', 'RETRY', `${name} — essai ${i}/${maxAttempts}`);
      return await fn();
    } catch (err) {
      lastError = err;
      if (err && (err.terminal || err.message === 'Arrêt demandé')) throw err;
      log('warn', 'RETRY', `${name} échoué : ${err.message}`);
      if (i < maxAttempts) {
        const wait = Math.min(90, 8 * i * i);
        await sleepFn(wait * 1000);
      }
    }
  }
  throw lastError || new Error(`${name} échoué`);
}

function createRateLimitedQueue(maxConcurrent, minIntervalMs) {
  let running = 0;
  let lastStart = 0;
  let timer = null;
  const queue = [];
  let resolveIdle = null;
  const interval = () => (typeof minIntervalMs === 'function' ? minIntervalMs() : minIntervalMs);

  function schedule(delay) {
    if (timer) return;
    timer = setTimeout(() => {
      timer = null;
      tryNext();
    }, delay);
  }

  function tryNext() {
    if (running >= maxConcurrent) return;
    if (queue.length === 0) {
      if (running === 0 && resolveIdle) {
        const done = resolveIdle;
        resolveIdle = null;
        done();
      }
      return;
    }
    const wait = Math.max(0, lastStart + interval() - Date.now());
    if (wait > 0) {
      schedule(wait);
      return;
    }
    const task = queue.shift();
    running += 1;
    lastStart = Date.now();
    Promise.resolve()
      .then(task)
      .finally(() => {
        running -= 1;
        schedule(0);
      });
  }

  return {
    add(fn) {
      return new Promise((resolve, reject) => {
        queue.push(() => fn().then(resolve, reject));
        schedule(0);
      });
    },
    idle() {
      return new Promise((resolve) => {
        if (running === 0 && queue.length === 0 && !timer) {
          resolve();
          return;
        }
        resolveIdle = resolve;
      });
    },
  };
}

function authHeaders(apiKey) {
  return {
    Authorization: `Bearer ${apiKey}`,
    'Content-Type': 'application/json',
  };
}

async function generateExpansion(description, mood, variety, ctx) {
  const prompt = buildExpansionPrompt(description, mood, variety);
  const body = {
    model: MODEL_TEXT,
    messages: [
      { role: 'system', content: SYSTEM_PROMPT },
      { role: 'user', content: prompt },
    ],
    temperature: 0.85,
    max_tokens: 2000,
    response_format: { type: 'json_object' },
  };
  const res = await apiFetch(`${API_BASE}/chat/completions`, {
    method: 'POST',
    headers: authHeaders(ctx.apiKey),
    body: JSON.stringify(body),
  }, 'Expansion lofi', 'text', ctx.deps || {});
  if (!res.ok) {
    const errText = await res.text();
    throw new Error(`HTTP ${res.status} — ${errText.slice(0, 300)}`);
  }
  const data = await res.json();
  const content = data && data.choices && data.choices[0] && data.choices[0].message && data.choices[0].message.content;
  const expansion = extractJson(content);
  if (!expansion || !expansion.image_prompt) throw new Error('Expansion invalide');
  if (!expansion.title) expansion.title = 'Ambiance Lofi';
  if (!expansion.video_motion_prompt) {
    expansion.video_motion_prompt = 'Gentle slow motion: capybara breathing softly, leaves drifting down, faint mist, seamless loop.';
  }
  return expansion;
}

async function generateImage(prompt, format, ctx) {
  return retryTask('Image', async () => {
    const ratio = format === '9:16' ? '9:16' : '16:9';
    const body = { model: MODEL_IMAGE, prompt, size: '1K', ratio, n: 1 };
    const res = await apiFetch(`${API_BASE}/images/generations`, {
      method: 'POST',
      headers: authHeaders(ctx.apiKey),
      body: JSON.stringify(body),
    }, 'Image', 'image', ctx.deps || {});
    if (!res.ok) {
      const errText = await res.text();
      throw new Error(`HTTP ${res.status} — ${errText.slice(0, 300)}`);
    }
    const data = await res.json();
    const item = data && data.data && data.data[0];
    if (!item) throw new Error('Pas de donnée image');
    if (item.url) return item.url;
    if (item.b64_json) return `data:image/png;base64,${item.b64_json}`;
    throw new Error('Format image inconnu');
  }, 4, ctx.deps || {});
}

async function assertFrameReference(frame) {
  if (typeof frame !== 'string' || frame.length === 0) throw new Error('Image keyframe manquante');
  if (frame.startsWith('data:image/')) return;
  await assertPublicHttpsUrl(frame);
}

async function generateVideo(imageUrl, prompt, format, durationSeconds, ctx) {
  await assertFrameReference(imageUrl);
  return retryTask('Vidéo', async () => {
    const aspect = format === '9:16' ? '9:16' : '16:9';
    const body = {
      model: MODEL_VIDEO,
      prompt,
      seconds: String(durationSeconds),
      mode: 'keyframe',
      size: '720P',
      aspect_ratio: aspect,
      first_frame: imageUrl,
      last_frame: imageUrl,
    };
    const res = await apiFetch(`${API_BASE}/videos`, {
      method: 'POST',
      headers: authHeaders(ctx.apiKey),
      body: JSON.stringify(body),
    }, 'Vidéo', 'video', ctx.deps || {});
    if (!res.ok) {
      const errText = await res.text();
      throw new Error(`HTTP ${res.status} — ${errText.slice(0, 300)}`);
    }
    const data = await res.json();
    const videoId = data.video_id || data.id || data.task_id;
    if (!videoId) throw new Error('Pas de video_id');
    return videoId;
  }, 3, ctx.deps || {});
}

function loadTimingCache() {
  try {
    return JSON.parse(fs.readFileSync(TIMING_PATH, 'utf8'));
  } catch (err) {
    return {};
  }
}

function recordGenerationTime(ms) {
  const cache = loadTimingCache();
  const samples = Array.isArray(cache.video) ? cache.video : [];
  samples.push(ms);
  cache.video = samples.slice(-10);
  fs.mkdirSync(VIDEO_DIR, { recursive: true });
  fs.writeFileSync(TIMING_PATH, JSON.stringify(cache));
}

function estimateGenerationTime(fallbackMs) {
  const samples = loadTimingCache().video;
  if (!Array.isArray(samples) || samples.length === 0) return fallbackMs;
  const sorted = [...samples].sort((a, b) => a - b);
  const mid = Math.floor(sorted.length / 2);
  return sorted.length % 2 === 0 ? (sorted[mid - 1] + sorted[mid]) / 2 : sorted[mid];
}

async function pollVideo(videoId, durationSeconds, ctx) {
  const sleepFn = (ctx.deps && ctx.deps.sleep) || sleep;
  const fetchImpl = (ctx.deps && ctx.deps.apiFetch) || apiFetch;
  const estimatedMs = estimateGenerationTime(90000 + durationSeconds * 7000);
  const minWait = Math.max(25, Math.min(VIDEO_FIRST_POLL_DELAY_SEC, Math.floor(estimatedMs / 1000 * 0.35)));
  log('info', 'POLLING', `Attente initiale ${minWait}s`);
  await sleepFn(minWait * 1000);
  const url = `${POLL_BASE}?video_id=${encodeURIComponent(videoId)}&model_name=${encodeURIComponent(MODEL_VIDEO)}`;
  for (let attempt = 1; attempt <= VIDEO_MAX_POLL_ATTEMPTS; attempt += 1) {
    if (state.stopRequested) throw terminalError('Arrêt demandé');
    let res;
    try {
      res = await fetchImpl(url, { method: 'GET', headers: authHeaders(ctx.apiKey) }, 'Polling', 'video', ctx.deps || {});
    } catch (err) {
      if (err && err.terminal) throw err;
      await sleepFn(Math.min(90000, 15000 + attempt * 1500));
      continue;
    }
    let data;
    try {
      data = await res.json();
    } catch (err) {
      await sleepFn(Math.min(60000, 10000 + attempt * 1000));
      continue;
    }
    const status = String((data && data.status) || 'unknown').toLowerCase();
    const progress = Number((data && data.progress) || 0);
    log('info', 'POLLING', `Statut ${status} (${progress}%)`);
    if (status === 'completed' || status === 'succeeded' || status === 'done') {
      const videoUrl = data.url || (data.metadata && data.metadata.url) || (data.output && data.output.url);
      if (!videoUrl) throw new Error("Terminé mais pas d'URL");
      return videoUrl;
    }
    if (status === 'failed' || status === 'error' || status === 'cancelled') {
      throw terminalError(`Échec définitif (${status})`);
    }
    const wait = attempt < 5 ? 8 : Math.min(30, 10 + Math.floor(attempt / 5) * 3);
    await sleepFn(wait * 1000);
  }
  throw new Error('Délai maximal de polling dépassé');
}

function chunkToBuffer(chunk) {
  if (Buffer.isBuffer(chunk)) return chunk;
  if (typeof chunk === 'number' && Number.isInteger(chunk) && chunk >= 0 && chunk <= 255) {
    return Buffer.from([chunk]);
  }
  if (chunk instanceof Uint8Array) {
    return Buffer.from(chunk.buffer, chunk.byteOffset, chunk.byteLength);
  }
  if (typeof chunk === 'string') return Buffer.from(chunk);
  throw new Error('Fragment de corps illisible');
}

async function readBodyLimited(body, maxBytes) {
  if (Buffer.isBuffer(body)) {
    if (body.length > maxBytes) throw new Error('Vidéo trop volumineuse');
    return body;
  }
  if (!body || typeof body[Symbol.asyncIterator] !== 'function') throw new Error('Corps de réponse illisible');
  const parts = [];
  let total = 0;
  for await (const chunk of body) {
    const buf = chunkToBuffer(chunk);
    total += buf.length;
    if (total > maxBytes) throw new Error('Vidéo trop volumineuse');
    parts.push(buf);
  }
  return Buffer.concat(parts, total);
}

function isMediaHeader(buf) {
  if (!Buffer.isBuffer(buf) || buf.length < 12) return false;
  if (buf[0] === 0x1a && buf[1] === 0x45 && buf[2] === 0xdf && buf[3] === 0xa3) return true;
  return buf.slice(0, 32).toString('latin1').includes('ftyp');
}

function isRedirect(status) {
  return status === 301 || status === 302 || status === 303 || status === 307 || status === 308;
}

async function downloadVideo(videoUrl, deps = {}) {
  const fetchImpl = deps.fetch || nodeFetch;
  let current = videoUrl;
  let redirects = 0;
  while (redirects <= DOWNLOAD_MAX_REDIRECTS) {
    await assertPublicHttpsUrl(current, deps);
    const res = await fetchImpl(current, {
      method: 'GET',
      redirect: 'manual',
      size: MAX_VIDEO_BYTES,
      headers: { Accept: 'video/mp4,video/webm,*/*' },
    });
    if (isRedirect(res.status)) {
      redirects += 1;
      if (redirects > DOWNLOAD_MAX_REDIRECTS) throw new Error('Trop de redirections');
      const location = res.headers.get('location');
      if (!location) throw new Error('Redirection sans cible');
      current = new URL(location, current).toString();
      continue;
    }
    if (!res.ok) throw new Error(`Téléchargement HTTP ${res.status}`);
    const buf = await readBodyLimited(res.body, MAX_VIDEO_BYTES);
    if (!isMediaHeader(buf)) throw new Error('En-tête média invalide');
    return buf;
  }
  throw new Error('Trop de redirections');
}

function assertVideoStat(st) {
  if (!st || typeof st.isSymbolicLink !== 'function' || typeof st.isFile !== 'function') {
    throw new Error('Statut de fichier illisible');
  }
  if (st.isSymbolicLink()) throw new Error('Lien symbolique refusé');
  if (!st.isFile()) throw new Error('Pas un fichier régulier');
  if (st.size <= 0) throw new Error('Fichier vide');
  if (st.size >= MAX_VIDEO_BYTES) throw new Error('Vidéo trop volumineuse (≥ 100 Mo)');
}

async function assertLocalVideo(filePath) {
  const st = await fs.promises.lstat(filePath);
  assertVideoStat(st);
  const fh = await fs.promises.open(filePath, READ_FLAGS);
  try {
    const buf = Buffer.alloc(32);
    await fh.read(buf, 0, 32, 0);
    if (!isMediaHeader(buf)) throw new Error('En-tête média invalide');
  } finally {
    await fh.close();
  }
}

async function writeVideoExclusive(filePath, buffer) {
  await fs.promises.mkdir(path.dirname(filePath), { recursive: true });
  try {
    const existing = await fs.promises.lstat(filePath);
    if (existing.isSymbolicLink()) throw new Error('Lien symbolique refusé');
    throw new Error('Le fichier vidéo existe déjà');
  } catch (err) {
    if (err.code !== 'ENOENT') throw err;
  }
  const fh = await fs.promises.open(filePath, WRITE_FLAGS, 0o644);
  try {
    await fh.writeFile(buffer);
  } finally {
    await fh.close();
  }
}

function safeIdentity(value, fallback) {
  const text = String(value || '').replace(/[\r\n\0]/g, '').trim();
  if (!text) return fallback;
  return text.slice(0, 120);
}

function buildGitEnv(extra = {}, pathValue = process.env.PATH || '/usr/bin:/bin') {
  const name = safeIdentity(process.env.GIT_AUTHOR_NAME, 'Lofi Capybara Studio');
  const email = safeIdentity(process.env.GIT_AUTHOR_EMAIL, 'lofi-capybara@users.noreply.github.com');
  const env = {
    PATH: pathValue,
    HOME: '',
    GIT_CONFIG_NOSYSTEM: '1',
    GIT_TERMINAL_PROMPT: '0',
    GIT_AUTHOR_NAME: name,
    GIT_AUTHOR_EMAIL: email,
    GIT_COMMITTER_NAME: safeIdentity(process.env.GIT_COMMITTER_NAME, name),
    GIT_COMMITTER_EMAIL: safeIdentity(process.env.GIT_COMMITTER_EMAIL, email),
  };
  if (extra.GIT_ASKPASS) env.GIT_ASKPASS = extra.GIT_ASKPASS;
  return env;
}

function createGit(baseDir, env) {
  return simpleGit({
    baseDir,
    unsafe: {
      allowUnsafeAskPass: true,
      allowUnsafeCredentialHelper: true,
    },
    allowEnvironment: GIT_ENV_ALLOW,
  }).env(env);
}

async function writeAskpass(token) {
  if (/[\r\n\0]/.test(token)) throw new Error('GITHUB_TOKEN invalide');
  const filePath = path.join(os.tmpdir(), `lofi-askpass-${process.pid}-${crypto.randomBytes(8).toString('hex')}.cjs`);
  const script = `#!/usr/bin/env node
'use strict';
const prompt = String(process.argv[2] || '');
if (/username/i.test(prompt)) process.stdout.write('x-access-token\\n');
else process.stdout.write(${JSON.stringify(token)} + '\\n');
`;
  await fs.promises.writeFile(filePath, script, { mode: 0o700 });
  await fs.promises.chmod(filePath, 0o700);
  return filePath;
}

function commitMessage(title) {
  const clean = String(title || 'ambiance')
    .replace(/[\r\n\0]/g, ' ')
    .replace(/\s+/g, ' ')
    .trim()
    .slice(0, 120);
  return `feat: add new lofi capybara video: ${clean || 'ambiance'}`;
}

async function ensureOrigin(git, repoUrl) {
  assertAllowedRepo(repoUrl);
  let hasOrigin = true;
  try {
    await git.raw(['remote', 'get-url', 'origin']);
  } catch (err) {
    hasOrigin = false;
  }
  if (hasOrigin) await git.raw(['remote', 'set-url', 'origin', repoUrl]);
  else await git.raw(['remote', 'add', 'origin', repoUrl]);
  const origin = String(await git.raw(['remote', 'get-url', 'origin'])).trim();
  let allowed = false;
  try {
    assertAllowedRepo(origin);
    allowed = true;
  } catch (err) {
    allowed = false;
  }
  if (!allowed) throw new Error('Remote origin refusé (URL inattendue, non affichée)');
}

async function publishVideo(filePath, title, options = {}) {
  const repoUrl = options.repoUrl || process.env.GIT_REPO_URL;
  const token = options.token || process.env.GITHUB_TOKEN;
  const baseDir = options.baseDir || ROOT;
  if (!token) throw new Error('GITHUB_TOKEN manquant');
  assertAllowedRepo(repoUrl);
  await assertLocalVideo(filePath);
  const relative = path.relative(baseDir, filePath);
  if (relative.startsWith('..') || path.isAbsolute(relative) || !relative.endsWith('.mp4')) {
    throw new Error('Chemin vidéo refusé');
  }
  const git = createGit(baseDir, buildGitEnv());
  await ensureOrigin(git, repoUrl);
  await git.raw(['add', '-f', '--', relative]);
  await git.raw(['-c', 'commit.gpgsign=false', 'commit', '-m', commitMessage(title), '--', relative]);
  const askpass = await writeAskpass(token);
  try {
    const pushGit = createGit(baseDir, buildGitEnv({ GIT_ASKPASS: askpass }));
    await pushGit.raw(['-c', 'credential.helper=', 'push', 'origin', 'HEAD']);
  } finally {
    await fs.promises.unlink(askpass).catch(() => {});
  }
}

function normalizeScene(item, index) {
  const description = typeof item === 'string' ? item : item && item.description;
  const mood = item && typeof item === 'object' ? item.mood : null;
  const text = String(description || '').trim();
  if (text.length < 4) throw new Error(`Scène ${index + 1} trop courte`);
  if (mood && !MOODS[mood]) throw new Error(`Scène ${index + 1} : ambiance inconnue`);
  return { description: text, mood: mood || null };
}

function loadScenes(filePath) {
  const raw = JSON.parse(fs.readFileSync(filePath, 'utf8'));
  if (!Array.isArray(raw) || raw.length === 0) throw new Error('scenes.json doit contenir un tableau non vide');
  return raw.map(normalizeScene);
}

function readDuration() {
  const value = Number(process.env.VIDEO_DURATION_SECONDS || 5);
  return [5, 6, 8, 10].includes(value) ? value : 5;
}

function readFormat() {
  return process.env.VIDEO_FORMAT === '9:16' ? '9:16' : '16:9';
}

function publicUrlLabel(raw) {
  const value = String(raw || '');
  if (value.startsWith('data:')) return 'image locale';
  try {
    const url = new URL(value);
    return `${url.protocol}//${url.hostname}${url.pathname}`;
  } catch (err) {
    return 'url';
  }
}

async function assertSafeVideoDir() {
  try {
    const st = await fs.promises.lstat(VIDEO_DIR);
    if (st.isSymbolicLink() || !st.isDirectory()) throw new Error('Dossier vidéos invalide');
  } catch (err) {
    if (err.code !== 'ENOENT') throw err;
    await fs.promises.mkdir(VIDEO_DIR, { recursive: true });
  }
}

async function runScene(scene, ctx) {
  const mood = scene.mood || pick(Object.keys(MOODS));
  const variety = randomVariety();
  log('info', 'SCENE', `${scene.description} · ${MOODS[mood].label} · ${variety.timeOfDay}, ${variety.season}, ${variety.weather}`);
  const expansion = await retryTask('Expansion', () => generateExpansion(scene.description, mood, variety, ctx), 3, ctx.deps || {});
  log('info', 'EXPANSION', expansion.title);
  const imageUrl = await ctx.imageQueue.add(() => generateImage(buildLofiImagePrompt(expansion), ctx.format, ctx));
  log('info', 'IMAGE', publicUrlLabel(imageUrl));
  const videoPrompt = buildLofiVideoPrompt(expansion, ctx.durationSeconds);
  const videoId = await ctx.videoQueue.add(() => generateVideo(imageUrl, videoPrompt, ctx.format, ctx.durationSeconds, ctx));
  const started = Date.now();
  const videoUrl = await pollVideo(videoId, ctx.durationSeconds, ctx);
  recordGenerationTime(Date.now() - started);
  const media = await downloadVideo(videoUrl);
  const filePath = videoOutputPath(expansion.title, Date.now());
  await writeVideoExclusive(filePath, media);
  await assertLocalVideo(filePath);
  await publishVideo(filePath, expansion.title, { repoUrl: ctx.repoUrl, token: ctx.token });
  log('info', 'GIT', `publié : ${path.basename(filePath)}`);
  return filePath;
}

async function main() {
  const apiKey = process.env.AGNES_API_KEY;
  const token = process.env.GITHUB_TOKEN;
  const repoUrl = process.env.GIT_REPO_URL;
  if (!apiKey || !token || !repoUrl) {
    throw new Error('AGNES_API_KEY, GITHUB_TOKEN et GIT_REPO_URL sont requis dans .env');
  }
  assertAllowedRepo(repoUrl);
  await assertSafeVideoDir();
  const scenes = loadScenes(path.join(ROOT, 'scenes.json'));
  const ctx = {
    apiKey,
    token,
    repoUrl,
    format: readFormat(),
    durationSeconds: readDuration(),
    imageQueue: createRateLimitedQueue(1, () => imageIntervalMs),
    videoQueue: createRateLimitedQueue(1, () => videoIntervalMs),
  };
  let failed = 0;
  for (const scene of scenes) {
    if (state.stopRequested) break;
    try {
      await runScene(scene, ctx);
    } catch (err) {
      failed += 1;
      log('error', 'SCENE', String(err && err.message || err).slice(0, 500));
    }
  }
  if (failed > 0) process.exitCode = 1;
}

function sampleMp4() {
  const buf = Buffer.alloc(32, 0);
  buf.write('ftyp', 4, 'latin1');
  return buf;
}

function mockResponse(status, body, headers) {
  const buf = Buffer.isBuffer(body) ? body : Buffer.from(String(body));
  const lower = {};
  for (const [key, value] of Object.entries(headers || {})) lower[key.toLowerCase()] = String(value);
  function make() {
    let consumed = false;
    return {
      ok: status >= 200 && status < 300,
      status,
      headers: { get(name) { return lower[String(name).toLowerCase()] || null; } },
      body: buf,
      async text() {
        if (consumed) throw new Error('corps déjà consommé');
        consumed = true;
        return buf.toString('utf8');
      },
      async json() {
        return JSON.parse(await this.text());
      },
      clone() { return make(); },
    };
  }
  return make();
}

async function expectError(fn) {
  try {
    await fn();
  } catch (err) {
    return err;
  }
  throw new Error('Exception attendue absente');
}

async function runSelfTest() {
  resetRateState();
  let checks = 0;
  async function check(fn) {
    await fn();
    checks += 1;
  }

  const publicResolve = async () => ['8.8.8.8'];
  const blockedUrls = [
    'https://2130706433/x',
    'https://0x7f000001/x',
    'https://0177.0.0.1/x',
    'https://127.1/x',
    'https://[::ffff:127.0.0.1]/x',
    'https://[::ffff:10.1.2.3]/x',
    'https://169.254.169.254/latest',
    'https://10.1.2.3/x',
    'https://192.168.0.8/x',
    'https://172.16.5.5/x',
    'https://localhost/x',
    'https://metadata.google.internal/x',
    'http://cdn.example/v.mp4',
    'https://user:secret@cdn.example/v.mp4',
  ];
  await check(async () => {
    for (const raw of blockedUrls) {
      const err = await expectError(() => assertPublicHttpsUrl(raw, { resolve: publicResolve }));
      if (!err || /secret/.test(err.message)) throw new Error('Garde URL insuffisante');
    }
  });
  await check(async () => {
    const err = await expectError(() => assertPublicHttpsUrl('https://cdn.example/a.mp4', {
      resolve: async () => ['127.0.0.1'],
    }));
    if (!err) throw new Error('DNS privé accepté');
    await assertPublicHttpsUrl('https://cdn.example/a.mp4', { resolve: publicResolve });
  });
  await check(async () => {
    assertAllowedRepo(ALLOWED_REPO);
    assertAllowedRepo(`${ALLOWED_REPO}.git`);
    const rejected = [
      'http://github.com/urdirditfurd/videowan2g',
      'https://github.com/other/videowan2g',
      'https://user:token@github.com/urdirditfurd/videowan2g',
    ];
    for (const raw of rejected) {
      const err = await expectError(async () => assertAllowedRepo(raw));
      if (!err || err.message.includes('token')) throw new Error('Dépôt mal filtré');
    }
  });

  await check(async () => {
    const sleeps = [];
    let calls = 0;
    const err = await expectError(() => apiFetch(`${API_BASE}/videos`, { method: 'POST' }, 'Vidéo', 'video', {
      queueMaxRetries: 12,
      random: () => 0.5,
      sleep: async (ms) => { sleeps.push(ms); },
      fetch: async () => {
        calls += 1;
        return mockResponse(503, '{"error":"video_queue_full"}');
      },
    }));
    if (!err || !err.terminal || sleeps.length !== 12 || calls !== 13) throw new Error('Budget de file incorrect');
    if (sleeps[0] !== 60000 || sleeps[1] !== 120000 || sleeps[2] !== 300000 || sleeps[11] !== 300000) {
      throw new Error('Backoff de file incorrect');
    }
    if (/Tentatives épuisées/.test(err.message)) throw new Error('La file a consommé le budget normal');
  });
  await check(async () => {
    let calls = 0;
    const res = await apiFetch(`${API_BASE}/videos`, { method: 'POST' }, 'Vidéo', 'video', {
      queueMaxRetries: 12,
      random: () => 0.5,
      sleep: async () => {},
      fetch: async () => {
        calls += 1;
        if (calls <= 5) return mockResponse(429, 'video_queue_full', { 'retry-after': '15' });
        return mockResponse(200, '{"video_id":"v1"}');
      },
    });
    const text = await res.text();
    if (calls !== 6 || !text.includes('v1')) throw new Error('Reprise après file pleine incorrecte');
  });
  await check(async () => {
    let calls = 0;
    const err = await expectError(() => apiFetch(`${API_BASE}/chat/completions`, { method: 'POST' }, 'Expansion', 'text', {
      sleep: async () => { throw new Error('pause inattendue'); },
      fetch: async () => {
        calls += 1;
        return mockResponse(400, 'erreur-saisie-unique');
      },
    }));
    if (calls !== 1 || !err || !err.terminal || !err.message.includes('erreur-saisie-unique')) {
      throw new Error('Erreur HTTP terminale incorrecte');
    }
  });
  await check(async () => {
    let redirect = null;
    const res = await apiFetch(`${API_BASE}/videos`, { method: 'POST' }, 'Vidéo', 'video', {
      fetch: async (url, options) => {
        redirect = options.redirect;
        return mockResponse(400, 'DETAIL_BODY_TOKEN');
      },
    });
    const text = await res.text();
    if (redirect !== 'error' || text !== 'DETAIL_BODY_TOKEN') throw new Error('Corps vidéo non relu');
  });
  await check(async () => {
    let called = false;
    const err = await expectError(() => apiFetch('https://evil.example/steal', {}, 'X', 'text', {
      fetch: async () => { called = true; return mockResponse(200, '{}'); },
    }));
    if (called || !err) throw new Error('Hôte API non bloqué');
  });

  await check(async () => {
    const calls = [];
    const buf = await downloadVideo('https://cdn.example/start.mp4', {
      resolve: publicResolve,
      fetch: async (url, options) => {
        calls.push({ url, auth: options.headers && options.headers.Authorization });
        if (calls.length < 3) {
          return mockResponse(302, '', { location: `https://cdn.example/hop-${calls.length}.mp4` });
        }
        return mockResponse(200, sampleMp4());
      },
    });
    if (!isMediaHeader(buf) || calls.length !== 3 || calls.some((call) => call.auth)) {
      throw new Error('Téléchargement redirigé incorrect');
    }
  });
  await check(async () => {
    let calls = 0;
    const err = await expectError(() => downloadVideo('https://cdn.example/start.mp4', {
      resolve: publicResolve,
      fetch: async () => {
        calls += 1;
        return mockResponse(302, '', { location: 'https://cdn.example/again.mp4' });
      },
    }));
    if (!err || calls !== 3) throw new Error('Plafond de redirections incorrect');
  });
  await check(async () => {
    let calls = 0;
    const err = await expectError(() => downloadVideo('https://cdn.example/start.mp4', {
      resolve: publicResolve,
      fetch: async () => {
        calls += 1;
        return mockResponse(302, '', { location: 'https://127.0.0.1/secret.mp4' });
      },
    }));
    if (!err || calls !== 1 || /secret/.test(err.message)) throw new Error('Redirection privée acceptée');
  });
  await check(async () => {
    const err = await expectError(() => downloadVideo('https://cdn.example/plain.bin', {
      resolve: publicResolve,
      fetch: async () => mockResponse(200, Buffer.from('pas-un-media-valide!!')),
    }));
    if (!err) throw new Error('Média invalide accepté');
    const webm = Buffer.alloc(16, 0);
    webm[0] = 0x1a; webm[1] = 0x45; webm[2] = 0xdf; webm[3] = 0xa3;
    if (!isMediaHeader(webm) || !isMediaHeader(sampleMp4())) throw new Error('En-têtes média refusés');
  });

  await check(async () => {
    const src = Buffer.from([9, 8, 7]);
    if (chunkToBuffer(src) !== src) throw new Error('Buffer non préservé');
    const one = chunkToBuffer(7);
    if (one.length !== 1 || one[0] !== 7) throw new Error('Octet mal relu');
    const response = new nodeFetch.Response(Buffer.from('hello-ftyp'));
    const rebuilt = await readBodyLimited(response.body, 100);
    if (rebuilt.toString() !== 'hello-ftyp') throw new Error('Corps node-fetch mal réassemblé');
  });
  await check(async () => {
    assertVideoStat({ isSymbolicLink: () => false, isFile: () => true, size: MAX_VIDEO_BYTES - 1 });
    const err = await expectError(async () => assertVideoStat({
      isSymbolicLink: () => false,
      isFile: () => true,
      size: MAX_VIDEO_BYTES,
    }));
    if (!err) throw new Error('Plafond 100 Mo ignoré');
  });

  const temp = await fs.promises.mkdtemp(path.join(os.tmpdir(), 'lofi-self-'));
  try {
    await check(async () => {
      const target = path.join(temp, 'target.mp4');
      const link = path.join(temp, 'link.mp4');
      await fs.promises.writeFile(target, 'ORIGINAL');
      await fs.promises.symlink(target, link);
      const err = await expectError(() => writeVideoExclusive(link, sampleMp4()));
      const after = await fs.promises.readFile(target, 'utf8');
      if (!err || after !== 'ORIGINAL') throw new Error('Lien symbolique écrasé');
      const videoErr = await expectError(() => assertLocalVideo(link));
      if (!videoErr) throw new Error('assertLocalVideo suit un lien');
    });
    await check(async () => {
      const file = path.join(temp, 'once.mp4');
      await writeVideoExclusive(file, sampleMp4());
      const err = await expectError(() => writeVideoExclusive(file, Buffer.from('nouveau')));
      const after = await fs.promises.readFile(file);
      if (!err || !after.includes('ftyp')) throw new Error('Fichier existant écrasé');
    });

    const prompt = buildExpansionPrompt('maison sous la pluie', 'rainy_forest', {
      timeOfDay: 'heure bleue',
      season: 'automne tardif',
      weather: 'brume au sol',
    });
    await check(async () => {
      if (!prompt.includes('heure bleue') || !prompt.includes('automne tardif') || !prompt.includes('brume au sol') || !/capybara/i.test(prompt)) {
        throw new Error('Prompt d unicité incomplet');
      }
      const variety = randomVariety(() => 0);
      if (variety.timeOfDay !== TIMES_OF_DAY[0]) throw new Error('Variété aléatoire incorrecte');
    });
    await check(async () => {
      const parsed = extractJson('```json\n{"title":"Brume","image_prompt":"a capybara","video_motion_prompt":"slow mist"}\n```');
      const broken = extractJson('prefix {"title": "Soir", "image_prompt": "capybara on a porch", "video_motion_prompt": "breathing"');
      if (!parsed || parsed.title !== 'Brume' || !broken || broken.image_prompt !== 'capybara on a porch') {
        throw new Error('Parse JSON incorrect');
      }
    });
    await check(async () => {
      const scenes = loadScenes(path.join(ROOT, 'scenes.json'));
      if (scenes.length !== 5 || scenes.some((scene) => scene.description.length < 4)) throw new Error('scenes.json inattendu');
      const named = videoOutputPath('Café / pluie!', 1700000000000);
      if (!named.endsWith('Cafe_pluie_1700000000000.mp4')) throw new Error('Nom de fichier inattendu');
    });

    await check(async () => {
      const starts = [];
      const queue = createRateLimitedQueue(1, 40);
      await queue.add(async () => { starts.push(Date.now()); });
      await queue.add(async () => { starts.push(Date.now()); });
      if (starts.length !== 2 || starts[1] - starts[0] < 30) throw new Error('File locale non espacée');
    });

    const canaryKey = 'canary-agnes-secret-value';
    const canaryToken = 'canary-github-secret-value';
    process.env.AGNES_API_KEY = canaryKey;
    process.env.GITHUB_TOKEN = canaryToken;
    await check(async () => {
      const work = path.join(temp, 'repo');
      const bin = path.join(temp, 'bin');
      const logPath = path.join(temp, 'git-log.jsonl');
      await fs.promises.mkdir(path.join(work, 'generated_videos'), { recursive: true });
      await fs.promises.mkdir(bin, { recursive: true });
      const video = path.join(work, 'generated_videos', 'brume_1.mp4');
      await writeVideoExclusive(video, sampleMp4());
      const fakePath = path.join(temp, 'fake-git.cjs');
      await fs.promises.writeFile(fakePath, `'use strict';
const fs = require('fs');
const entry = { argv: process.argv.slice(2), env: process.env };
fs.appendFileSync(${JSON.stringify(logPath)}, JSON.stringify(entry) + '\\n');
if (process.argv.includes('get-url')) process.stdout.write(${JSON.stringify(`${ALLOWED_REPO}\n`)});
process.exit(0);
`);
      await fs.promises.writeFile(path.join(bin, 'git'), `#!/bin/sh\nexec /usr/bin/env -u PWD -u SHLVL -u _ node ${JSON.stringify(fakePath)} "$@"\n`, { mode: 0o755 });
      await fs.promises.chmod(path.join(bin, 'git'), 0o755);
      const previousPath = process.env.PATH;
      process.env.PATH = `${bin}:${previousPath}`;
      try {
        await publishVideo(video, 'Brume dorée', {
          repoUrl: ALLOWED_REPO,
          token: canaryToken,
          baseDir: work,
        });
      } finally {
        process.env.PATH = previousPath;
      }
      const entries = fs.readFileSync(logPath, 'utf8').trim().split('\n').map((line) => JSON.parse(line));
      const allowedKeys = new Set([...GIT_ENV_ALLOW, 'GIT_TEST_DISALLOW_ABBREVIATED_OPTIONS']);
      const unexpected = new Set();
      for (const entry of entries) {
        for (const key of Object.keys(entry.env)) {
          if (!allowedKeys.has(key)) unexpected.add(key);
        }
      }
      if (unexpected.size > 0) throw new Error(`Variable git inattendue: ${[...unexpected].join(',')}`);
      for (const entry of entries) {
        const blob = JSON.stringify(entry.env);
        if (blob.includes(canaryKey) || blob.includes(canaryToken) || blob.includes('AGNES_API_KEY') || blob.includes('GITHUB_TOKEN')) {
          throw new Error('Secret présent dans le processus git');
        }
        if (entry.env.HOME !== '' || entry.env.GIT_CONFIG_NOSYSTEM !== '1' || entry.env.GIT_TERMINAL_PROMPT !== '0') {
          throw new Error('Environnement git non isolé');
        }
      }
      const add = entries.find((entry) => entry.argv.includes('add'));
      const commit = entries.find((entry) => entry.argv.includes('commit'));
      const push = entries.find((entry) => entry.argv.includes('push'));
      if (!add || !add.argv.includes('-f') || !add.argv.includes('--')) throw new Error('git add -f absent');
      if (!commit || !commit.argv.includes('commit.gpgsign=false') || !commit.argv.includes('--')) throw new Error('commit non isolé');
      if (!push || !push.argv.includes('credential.helper=') || !push.env.GIT_ASKPASS) throw new Error('push non protégé');
      if (entries.some((entry) => !entry.argv.includes('push') && entry.env.GIT_ASKPASS)) throw new Error('askpass trop large');
      if (fs.existsSync(push.env.GIT_ASKPASS)) throw new Error('askpass non supprimé');
    });
    await check(async () => {
      const work = path.join(temp, 'poison-repo');
      const bin = path.join(temp, 'poison-bin');
      await fs.promises.mkdir(work, { recursive: true });
      await fs.promises.mkdir(bin, { recursive: true });
      const video = path.join(work, 'brume_2.mp4');
      await fs.promises.writeFile(video, sampleMp4());
      const fakePath = path.join(temp, 'poison-git.cjs');
      const poisoned = `https://x-access-token:${canaryToken}@github.com/urdirditfurd/videowan2g`;
      await fs.promises.writeFile(fakePath, `'use strict';
if (process.argv.includes('get-url')) process.stdout.write(${JSON.stringify(`${poisoned}\n`)});
process.exit(0);
`);
      await fs.promises.writeFile(path.join(bin, 'git'), `#!/bin/sh\nexec /usr/bin/env -u PWD -u SHLVL -u _ node ${JSON.stringify(fakePath)} "$@"\n`, { mode: 0o755 });
      await fs.promises.chmod(path.join(bin, 'git'), 0o755);
      const previousPath = process.env.PATH;
      process.env.PATH = `${bin}:${previousPath}`;
      let err;
      try {
        err = await expectError(() => publishVideo(video, 'Brume', {
          repoUrl: ALLOWED_REPO,
          token: canaryToken,
          baseDir: work,
        }));
      } finally {
        process.env.PATH = previousPath;
      }
      if (!err || err.message.includes(canaryToken) || err.message.includes('x-access-token')) {
        throw new Error('URL distante divulguée');
      }
    });
    await check(async () => {
      const dir = await fs.promises.mkdtemp(path.join(os.tmpdir(), 'lofi-real-git-'));
      try {
        const env = buildGitEnv({}, '/usr/bin:/bin');
        await execFileAsync('/usr/bin/git', ['init'], { cwd: dir, env });
        await execFileAsync('/usr/bin/git', ['remote', 'add', 'origin', ALLOWED_REPO], { cwd: dir, env });
        const { stdout } = await execFileAsync('/usr/bin/git', ['remote', 'get-url', 'origin'], { cwd: dir, env });
        if (stdout.trim() !== ALLOWED_REPO) throw new Error('Remote réécrit par la configuration git ambiante');
      } finally {
        await fs.promises.rm(dir, { recursive: true, force: true });
      }
    });
  } finally {
    await fs.promises.rm(temp, { recursive: true, force: true });
  }

  console.log(`SELF-TEST PASSED, ${checks} checks`);
}

if (require.main === module) {
  process.on('SIGINT', () => {
    state.stopRequested = true;
    log('warn', 'STOP', 'Arrêt demandé');
  });
  const pending = process.argv.includes('--self-test') ? runSelfTest() : main();
  pending.catch((err) => {
    console.error(String(err && err.message || err).slice(0, 500));
    process.exitCode = 1;
  });
}
