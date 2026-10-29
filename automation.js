import { createHash } from 'node:crypto';
import { spawnSync } from 'node:child_process';
import dns from 'node:dns/promises';
import { constants as fsConstants, readFileSync } from 'node:fs';
import fs from 'node:fs/promises';
import net from 'node:net';
import os from 'node:os';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import dotenv from 'dotenv';
import fetch from 'node-fetch';
import { simpleGit } from 'simple-git';

const ROOT = path.dirname(fileURLToPath(import.meta.url));
const API_BASE = 'https://apihub.agnes-ai.com/v1';
const POLL_BASE = 'https://apihub.agnes-ai.com/agnesapi';
const MODEL_TEXT = 'agnes-2.5-flash';
const MODEL_IMAGE = 'agnes-image-2.5-flash';
const MODEL_VIDEO = 'agnes-video-2.5-flash';
const ALLOWED_REPO = 'https://github.com/urdirditfurd/videowan2g';
const VIDEOS_DIR = path.join(ROOT, 'generated_videos');
const MAX_VIDEO_BYTES = 100 * 1024 * 1024;
const MAX_REDIRECTS = 2;
const VIDEO_QUEUE_MAX_RETRIES = 12;
const QUEUE_BACKOFF_SECONDS = [60, 120, 300];
const QUEUE_WAIT_CAP_SECONDS = 360;
const VIDEO_FIRST_POLL_DELAY_SEC = 80;
const VIDEO_MAX_POLL_ATTEMPTS = 100;
const IMAGE_PARALLEL = 1;
const IMAGE_INTERVAL_MS = 3400;
const IMAGE_INTERVAL_MIN = 3200;
const IMAGE_INTERVAL_MAX = 12000;
const VIDEO_INTERVAL_MIN = 62000;
const VIDEO_INTERVAL_MAX = 90000;
const VIDEO_INTERVAL_SAFE = 75000;
const OPEN_EXCLUSIVE = fsConstants.O_CREAT | fsConstants.O_EXCL | fsConstants.O_NOFOLLOW | fsConstants.O_WRONLY;
const OPEN_READ_NOFOLLOW = fsConstants.O_RDONLY | fsConstants.O_NOFOLLOW;

const GIT_ALLOWED_ENV = [
  'GIT_AUTHOR_NAME',
  'GIT_AUTHOR_EMAIL',
  'GIT_COMMITTER_NAME',
  'GIT_COMMITTER_EMAIL',
  'GIT_CONFIG_NOSYSTEM',
  'GIT_TERMINAL_PROMPT',
  'GIT_ASKPASS',
];

const FORBIDDEN_GIT_ENV = new Set(['AGNES_API_KEY', 'GITHUB_TOKEN', 'GH_TOKEN', 'GITHUB_PAT']);

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

const TIMES_OF_DAY = [
  'blue hour before sunrise',
  'misty early morning',
  'soft late morning',
  'hazy noon',
  'warm mid-afternoon',
  'golden hour',
  'dusk with the first stars',
  'quiet night',
];

const SEASONS = ['early spring', 'late spring', 'high summer', 'early autumn', 'late autumn', 'deep winter'];

const WEATHERS = [
  'fine rain',
  'drizzle on leaves',
  'light fog',
  'clear still air',
  'soft wind after rain',
  'gentle snowfall',
  'humid air after a warm shower',
  'thin high clouds',
];

const state = {
  stopRequested: false,
  durationSeconds: 5,
  format: '16:9',
  mood: 'rainy_forest',
};

let createImageIntervalMs = IMAGE_INTERVAL_MS;
let createVideoIntervalMs = VIDEO_INTERVAL_SAFE;
let imageAttemptsSinceLastError = 0;
let videoAttemptsSinceLastError = 0;
let fetchImpl = fetch;
let sleepImpl = (ms) => new Promise((resolve) => setTimeout(resolve, ms));
let rng = Math.random;
let quietLogs = false;
const timingSamples = [];

function redact(text) {
  return String(text)
    .replace(/sk-[A-Za-z0-9_-]{4,}/g, 'sk-[REDACTED]')
    .replace(/ghp_[A-Za-z0-9]{4,}/g, 'ghp_[REDACTED]')
    .replace(/github_pat_[A-Za-z0-9_]{4,}/g, 'github_pat_[REDACTED]')
    .replace(/x-access-token:[^@\s]+/gi, 'x-access-token:[REDACTED]')
    .replace(/Bearer\s+[A-Za-z0-9._-]+/gi, 'Bearer [REDACTED]');
}

function log(level, category, message) {
  if (quietLogs) return;
  const time = new Date().toISOString();
  console.log(`${time} [${level}] [${category}] ${redact(message)}`);
}

function sleep(ms) {
  return sleepImpl(ms);
}

function pick(list, randomFn = rng) {
  const index = Math.min(list.length - 1, Math.max(0, Math.floor(randomFn() * list.length)));
  return list[index];
}

function loadDotEnv() {
  const result = dotenv.config({ path: path.join(ROOT, '.env'), quiet: true });
  if (result.error && result.error.code !== 'ENOENT') throw result.error;
}

function sha256(value) {
  return createHash('sha256').update(value).digest('hex');
}

function stripIpBrackets(host) {
  const value = String(host || '').toLowerCase().replace(/\.$/, '');
  if (value.startsWith('[') && value.endsWith(']')) return value.slice(1, -1);
  return value;
}

function ipv4Parts(ip) {
  const match = /^(\d{1,3})\.(\d{1,3})\.(\d{1,3})\.(\d{1,3})$/.exec(ip);
  if (!match) return null;
  const parts = match.slice(1).map(Number);
  if (parts.some((part) => part > 255)) return null;
  return parts;
}

function isPrivateIPv4(parts) {
  const [a, b] = parts;
  if (a === 0 || a === 10 || a === 127) return true;
  if (a === 169 && b === 254) return true;
  if (a === 172 && b >= 16 && b <= 31) return true;
  if (a === 192 && b === 168) return true;
  if (a === 100 && b >= 64 && b <= 127) return true;
  if (a === 192 && b === 0) return true;
  if (a === 198 && (b === 18 || b === 19)) return true;
  if (a === 198 && b === 51) return true;
  if (a === 203 && b === 0) return true;
  if (a >= 224) return true;
  return false;
}

function isPrivateAddress(address) {
  const ip = stripIpBrackets(address);
  if (!ip || ip.includes('%')) return true;
  if (ip === '::1' || ip === '::' || ip === '0.0.0.0') return true;
  const dotted = ipv4Parts(ip);
  if (dotted) return isPrivateIPv4(dotted);
  const mappedDotted = /^::ffff:(\d{1,3}(?:\.\d{1,3}){3})$/i.exec(ip);
  if (mappedDotted) return isPrivateAddress(mappedDotted[1]);
  const mappedHex = /^::ffff:([0-9a-f]{1,4}):([0-9a-f]{1,4})$/i.exec(ip);
  if (mappedHex) {
    const hi = Number.parseInt(mappedHex[1], 16);
    const lo = Number.parseInt(mappedHex[2], 16);
    const rebuilt = `${(hi >> 8) & 255}.${hi & 255}.${(lo >> 8) & 255}.${lo & 255}`;
    return isPrivateAddress(rebuilt);
  }
  if (ip.includes(':')) {
    if (/^fe[89ab]/i.test(ip)) return true;
    if (/^f[cd]/i.test(ip)) return true;
    if (/^ff/i.test(ip)) return true;
    return false;
  }
  return false;
}

function assertUrlIsPublic(url) {
  if (url.protocol !== 'https:') throw new Error('https requis');
  if (url.username || url.password) throw new Error('Identifiants interdits dans l’URL');
  const host = stripIpBrackets(url.hostname);
  if (!host || host.includes('%')) throw new Error('Hôte interdit');
  const blocked = new Set([
    'localhost',
    'localhost.localdomain',
    'metadata',
    'metadata.google.internal',
    'metadata.google.com',
  ]);
  if (blocked.has(host) || host.endsWith('.localhost') || host.endsWith('.local') || host.endsWith('.internal')) {
    throw new Error('Hôte interdit');
  }
  if (net.isIP(host) || host.includes(':')) {
    if (isPrivateAddress(host)) throw new Error('Adresse non publique');
  }
}

async function assertResolvedPublic(hostname) {
  const host = stripIpBrackets(hostname);
  if (net.isIP(host) || host.includes(':')) {
    if (isPrivateAddress(host)) throw new Error('Adresse non publique');
    return;
  }
  let records;
  try {
    records = await dns.lookup(host, { all: true, verbatim: true });
  } catch {
    throw new Error('Résolution DNS impossible');
  }
  if (!records.length) throw new Error('Résolution DNS vide');
  for (const record of records) {
    if (isPrivateAddress(record.address)) throw new Error('Résolution vers une adresse non publique');
  }
}

async function assertPublicHttpsUrl(raw, options = {}) {
  const url = new URL(raw);
  assertUrlIsPublic(url);
  if (!options.skipDns) await assertResolvedPublic(url.hostname);
  return url;
}

function assertAgnesUrl(raw) {
  const url = new URL(raw);
  if (url.protocol !== 'https:' || url.hostname !== 'apihub.agnes-ai.com') {
    throw new Error('Hôte API inattendu');
  }
  if (url.username || url.password) throw new Error('Identifiants interdits dans l’URL API');
  return url;
}

function assertAllowedRepo(raw) {
  let url;
  try {
    url = new URL(String(raw || ''));
  } catch {
    throw new Error('URL de dépôt invalide');
  }
  if (url.username || url.password) throw new Error('URL de dépôt : identifiants interdits');
  if (url.protocol !== 'https:') throw new Error('URL de dépôt : https requis');
  if (url.hostname !== 'github.com') throw new Error('URL de dépôt : hôte non autorisé');
  if (url.search || url.hash) throw new Error('URL de dépôt : paramètres interdits');
  const pathname = url.pathname.replace(/\/+$/, '');
  if (pathname !== '/urdirditfurd/videowan2g' && pathname !== '/urdirditfurd/videowan2g.git') {
    throw new Error('URL de dépôt non autorisée');
  }
  return ALLOWED_REPO;
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
      if (ch === '\n') {
        result += '\\n';
        continue;
      }
      if (ch === '\r') {
        result += '\\r';
        continue;
      }
      if (ch === '\t') {
        result += '\\t';
        continue;
      }
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
    if (escaped) {
      escaped = false;
      continue;
    }
    if (ch === '\\') {
      escaped = true;
      continue;
    }
    if (ch === '"') {
      inString = !inString;
      continue;
    }
    if (inString) continue;
    if (ch === '{') openBraces += 1;
    else if (ch === '}') openBraces -= 1;
    else if (ch === '[') openBrackets += 1;
    else if (ch === ']') openBrackets -= 1;
  }
  let result = text;
  if (inString) result += '"';
  while (openBrackets > 0) {
    result += ']';
    openBrackets -= 1;
  }
  while (openBraces > 0) {
    result += '}';
    openBraces -= 1;
  }
  return result;
}

function unescapeJsonString(value) {
  return value.replace(/\\n/g, ' ').replace(/\\"/g, '"');
}

function salvageExpansion(text) {
  const titleMatch = text.match(/"title"\s*:\s*"((?:[^"\\]|\\.)*)"/);
  const imageMatch = text.match(/"image_prompt"\s*:\s*"((?:[^"\\]|\\.)*)"/);
  const videoMatch = text.match(/"video_motion_prompt"\s*:\s*"((?:[^"\\]|\\.)*)"/);
  if (!imageMatch && !videoMatch) return null;
  return {
    title: titleMatch ? unescapeJsonString(titleMatch[1]) : 'Ambiance Lofi',
    image_prompt: imageMatch ? unescapeJsonString(imageMatch[1]) : '',
    video_motion_prompt: videoMatch ? unescapeJsonString(videoMatch[1]) : '',
  };
}

function extractJson(text) {
  if (!text) return null;
  let current = text.trim();
  const fenced = current.match(/```(?:json)?\s*([\s\S]*?)```/);
  if (fenced) current = fenced[1].trim();
  const firstBrace = current.indexOf('{');
  if (firstBrace < 0) return null;
  current = current.slice(firstBrace);
  try {
    return JSON.parse(current);
  } catch {
    // stratégie suivante
  }
  try {
    return JSON.parse(escapeControlCharsInStrings(current));
  } catch {
    // stratégie suivante
  }
  try {
    const lastBrace = current.lastIndexOf('}');
    if (lastBrace > 0) return JSON.parse(escapeControlCharsInStrings(current.slice(0, lastBrace + 1)));
  } catch {
    // stratégie suivante
  }
  try {
    return JSON.parse(balanceJson(escapeControlCharsInStrings(current)));
  } catch {
    // stratégie suivante
  }
  const salvaged = salvageExpansion(current);
  if (salvaged && (salvaged.image_prompt || salvaged.video_motion_prompt)) return salvaged;
  return null;
}

function buildExpansionPrompt(userDescription, mood, randomFn = rng) {
  const selected = MOODS[mood] || MOODS.rainy_forest;
  const timeOfDay = pick(TIMES_OF_DAY, randomFn);
  const season = pick(SEASONS, randomFn);
  const weather = pick(WEATHERS, randomFn);
  return [
    'You are an art director for Studio Ghibli-inspired lofi ambience videos.',
    '',
    'USER SCENE IDEA (French):',
    userDescription,
    '',
    'AMBIENCE (French mood chosen):',
    selected.label,
    '—',
    selected.desc,
    '',
    'UNIQUENESS SEEDS — these must visibly change the picture, even if the user idea is similar to another scene:',
    `TIME OF DAY: ${timeOfDay}`,
    `SEASON: ${season}`,
    `WEATHER: ${weather}`,
    'Do not ignore these seeds. They are mandatory and must appear in the image and in the motion.',
    '',
    'YOUR TASK:',
    'Expand the user idea into a rich, poetic, ultra-detailed visual scene for a single lofi ambience video.',
    'ALWAYS include a cute, calm, peaceful capybara (capybara mascot) somewhere visible in the scene, integrated naturally into the environment.',
    'The mood must be lofi: slow, nostalgic, peaceful, detailed, soothing, hand-painted watercolor, Studio Ghibli / Hayao Miyazaki inspired.',
    'Every generation must be visually unique. Vary the capybara pose, the props, the camera distance and the small natural details.',
    '',
    'HARD RULES:',
    '- The image must feel like a Ghibli watercolor background: soft gradients, painterly, detailed, calm, warm.',
    '- The scene must be peaceful and loopable in motion.',
    '- The capybara must appear cute but not intrusive — it can be resting, soaking in water, sitting on a wooden porch, next to a teacup, under a blanket, etc.',
    '- The video motion must be SUBTLE and SLOW: gentle breathing of the capybara, softly falling leaves, drifting mist, slow water ripples, faint swaying grass, floating dust motes.',
    '- No fast movement, no action, no characters running, no camera shake. Perfect seamless loop.',
    '- No readable text, no watermark, no logos.',
    '- Color palette:',
    GHIBLI_PALETTE,
    '',
    'OUTPUT — strict JSON only, no markdown:',
    '{',
    '  "title": "A short poetic title in French (max 60 chars)",',
    '  "image_prompt": "Ultra-detailed English visual description for an image model. Include: setting, lighting, colors, atmosphere, the capybara (with its location and pose), the time of day, the season, the weather, Ghibli watercolor style, cinematic composition. 150-250 words.",',
    '  "video_motion_prompt": "English description of slow gentle motions for the video model: capybara breathing, leaves falling, mist drifting, water ripples, weather motion, seamless loop. 60-120 words."',
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

function buildLofiVideoPrompt(expansion) {
  return [
    `LOFI AMBIENCE VIDEO — ${state.durationSeconds} seconds — seamless loop.`,
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

function queueFullDelaySeconds(index, retryAfterHeader) {
  const scheduleIndex = Math.min(Math.max(index, 0), QUEUE_BACKOFF_SECONDS.length - 1);
  const base = QUEUE_BACKOFF_SECONDS[scheduleIndex];
  const retryAfter = Number(retryAfterHeader) || 0;
  let wait;
  if (retryAfter > 0) {
    wait = retryAfter;
  } else {
    const span = Math.floor(base * 0.15);
    const jitter = span > 0 ? Math.floor(rng() * span) : 0;
    wait = base + jitter;
  }
  return Math.min(QUEUE_WAIT_CAP_SECONDS, Math.max(1, wait));
}

function createRateLimitedQueue(maxConcurrent, getMinIntervalMs) {
  let running = 0;
  let lastStart = 0;
  const queue = [];
  let resolveIdle = null;
  let timer = null;

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
    const wait = Math.max(0, lastStart + getMinIntervalMs() - Date.now());
    if (wait > 0) {
      if (timer) return;
      timer = setTimeout(() => {
        timer = null;
        tryNext();
      }, wait);
      return;
    }
    const task = queue.shift();
    running += 1;
    lastStart = Date.now();
    Promise.resolve()
      .then(task)
      .finally(() => {
        running -= 1;
        setTimeout(tryNext, 0);
      });
  }

  return {
    add(fn) {
      return new Promise((resolve, reject) => {
        queue.push(() => fn().then(resolve, reject));
        setTimeout(tryNext, 0);
      });
    },
    idle() {
      return new Promise((resolve) => {
        if (running === 0 && queue.length === 0) {
          resolve();
          return;
        }
        resolveIdle = resolve;
      });
    },
  };
}

const imageQueue = createRateLimitedQueue(IMAGE_PARALLEL, () => createImageIntervalMs);
const videoQueue = createRateLimitedQueue(1, () => createVideoIntervalMs);

async function apiFetch(url, options = {}, label = 'API', kind = 'default') {
  assertAgnesUrl(url);
  const maxAttempts = kind === 'video' ? 10 : kind === 'image' ? 10 : 8;
  let attempt = 0;
  let queueWaits = 0;
  while (attempt < maxAttempts) {
    if (state.stopRequested) throw new Error('Arrêt demandé');
    const method = options.method || 'GET';
    log('api', `REQ ${label}`, `${method} ${url} (tentative ${attempt + 1}/${maxAttempts})`);
    const request = {
      ...options,
      signal: options.signal || AbortSignal.timeout(kind === 'video' ? 180000 : 120000),
    };
    let res;
    try {
      res = await fetchImpl(url, request);
    } catch (error) {
      if (error.message === 'Arrêt demandé') throw error;
      const wait = Math.min(kind === 'video' ? 120 : 45, kind === 'video' ? 10 + attempt * 8 : 3 + attempt * 4);
      log('warn', 'NETWORK', `${error.message} — retry dans ${wait}s`);
      await sleep(wait * 1000);
      attempt += 1;
      continue;
    }

    let retryAfter = 0;
    try {
      retryAfter = Number.parseInt(res.headers.get('Retry-After') || '0', 10) || 0;
    } catch {
      retryAfter = 0;
    }
    let errorText = '';
    if (!res.ok && ([400, 401, 404, 429, 500, 502, 503, 504, 520].includes(res.status) || kind === 'video')) {
      try {
        errorText = (await res.clone().text()).slice(0, 1200);
      } catch {
        errorText = '';
      }
    }
    log('response', `RÉP ${label}`, `HTTP ${res.status}`);
    if (res.ok) {
      if (kind === 'image') {
        imageAttemptsSinceLastError += 1;
        if (imageAttemptsSinceLastError >= 4 && createImageIntervalMs > IMAGE_INTERVAL_MIN) {
          createImageIntervalMs = Math.max(IMAGE_INTERVAL_MIN, createImageIntervalMs - 250);
          imageAttemptsSinceLastError = 0;
        }
      }
      if (kind === 'video') {
        videoAttemptsSinceLastError += 1;
        if (videoAttemptsSinceLastError >= 2 && createVideoIntervalMs > VIDEO_INTERVAL_MIN) {
          createVideoIntervalMs = Math.max(VIDEO_INTERVAL_MIN, createVideoIntervalMs - 3000);
          videoAttemptsSinceLastError = 0;
        }
      }
      return res;
    }
    if ([400, 401, 404].includes(res.status) && kind !== 'video') {
      throw new Error(`HTTP ${res.status} — ${errorText.slice(0, 300)}`);
    }
    const queueFull = /video_queue_full|queue is full/i.test(errorText);
    if (kind === 'video' && queueFull) {
      if (queueWaits >= VIDEO_QUEUE_MAX_RETRIES) {
        throw new Error(`File vidéo saturée après ${VIDEO_QUEUE_MAX_RETRIES} attentes`);
      }
      const wait = queueFullDelaySeconds(queueWaits, retryAfter);
      queueWaits += 1;
      videoAttemptsSinceLastError = 0;
      log('warn', 'VIDEO_QUEUE', `Queue vidéo saturée — nouvelle tentative dans ${wait}s`);
      await sleep(wait * 1000);
      continue;
    }
    if (res.status === 429) {
      const wait = retryAfter > 0
        ? retryAfter
        : Math.min(kind === 'video' ? 180 : 120, kind === 'video' ? 60 + attempt * 8 : 10 + attempt * 12);
      if (kind === 'image') createImageIntervalMs = Math.min(IMAGE_INTERVAL_MAX, createImageIntervalMs + 1000);
      if (kind === 'video') createVideoIntervalMs = Math.min(VIDEO_INTERVAL_MAX, createVideoIntervalMs + 6000);
      log('warn', 'RATELIMIT', `429 — attente ${wait}s`);
      await sleep(wait * 1000);
      attempt += 1;
      continue;
    }
    if ([500, 502, 503, 504, 520].includes(res.status)) {
      const wait = Math.min(kind === 'video' ? 180 : 60, kind === 'video' ? 20 + attempt * 10 : 5 + attempt * 5);
      log('warn', 'SERVICE', `HTTP ${res.status} — attente ${wait}s`);
      await sleep(wait * 1000);
      attempt += 1;
      continue;
    }
    return res;
  }
  throw new Error(`Tentatives épuisées pour ${label} après ${maxAttempts} essais`);
}

async function retryTask(name, fn, maxAttempts = 3) {
  let lastError = null;
  for (let i = 1; i <= maxAttempts; i += 1) {
    if (state.stopRequested) throw new Error('Arrêt demandé');
    try {
      log('info', 'RETRY', `${name} — essai ${i}/${maxAttempts}`);
      return await fn();
    } catch (error) {
      lastError = error;
      if (error.message === 'Arrêt demandé') throw error;
      log('warn', 'RETRY', `${name} échoué : ${error.message}`);
      if (i < maxAttempts) {
        const wait = Math.min(90, 8 * i * i);
        await sleep(wait * 1000);
      }
    }
  }
  throw lastError || new Error(`${name} échoué`);
}

async function generateExpansion(userDescription, mood) {
  const prompt = buildExpansionPrompt(userDescription, mood);
  const body = {
    model: MODEL_TEXT,
    messages: [
      {
        role: 'system',
        content: 'You are a Studio Ghibli-inspired art director specialized in lofi ambience videos. You always return strict JSON matching the requested schema exactly. Every scene includes a cute calm capybara. Every scene is peaceful, detailed, watercolor, and loopable with slow gentle motion. Each scene must be visually unique. No text, no watermark. Return only JSON.',
      },
      { role: 'user', content: prompt },
    ],
    temperature: 0.85,
    max_tokens: 2000,
    response_format: { type: 'json_object' },
  };
  const res = await apiFetch(`${API_BASE}/chat/completions`, {
    method: 'POST',
    headers: {
      Authorization: `Bearer ${process.env.AGNES_API_KEY}`,
      'Content-Type': 'application/json',
    },
    body: JSON.stringify(body),
  }, 'Expansion lofi', 'text');
  if (!res.ok) {
    const err = await res.text();
    throw new Error(`HTTP ${res.status} — ${err.slice(0, 300)}`);
  }
  const data = await res.json();
  const content = data.choices && data.choices[0] && data.choices[0].message && data.choices[0].message.content;
  const expansion = extractJson(content);
  if (!expansion || !expansion.image_prompt) throw new Error('Expansion invalide');
  if (!expansion.title) expansion.title = 'Ambiance Lofi';
  if (!expansion.video_motion_prompt) {
    expansion.video_motion_prompt = 'Gentle slow motion: capybara breathing softly, leaves drifting down, faint mist, seamless loop.';
  }
  log('success', 'EXPANSION', `Titre : ${expansion.title}`);
  return expansion;
}

async function generateImage(prompt, format) {
  return imageQueue.add(() => retryTask('Image', async () => {
    const ratio = format === '9:16' ? '9:16' : '16:9';
    const body = { model: MODEL_IMAGE, prompt, size: '1K', ratio, n: 1 };
    const res = await apiFetch(`${API_BASE}/images/generations`, {
      method: 'POST',
      headers: {
        Authorization: `Bearer ${process.env.AGNES_API_KEY}`,
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(body),
    }, 'Image', 'image');
    if (!res.ok) {
      const err = await res.text();
      throw new Error(`HTTP ${res.status} — ${err.slice(0, 500)}`);
    }
    const data = await res.json();
    const item = data.data && data.data[0];
    if (!item) throw new Error('Pas de donnée image');
    if (item.url) return item.url;
    if (item.b64_json) return `data:image/png;base64,${item.b64_json}`;
    throw new Error('Format image inconnu');
  }, 4));
}

async function generateVideo(imageUrl, prompt, format) {
  return videoQueue.add(() => retryTask('Vidéo', async () => {
    const aspect = format === '9:16' ? '9:16' : '16:9';
    const body = {
      model: MODEL_VIDEO,
      prompt,
      seconds: String(state.durationSeconds),
      mode: 'keyframe',
      size: '720P',
      aspect_ratio: aspect,
      first_frame: imageUrl,
      last_frame: imageUrl,
    };
    const res = await apiFetch(`${API_BASE}/videos`, {
      method: 'POST',
      headers: {
        Authorization: `Bearer ${process.env.AGNES_API_KEY}`,
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(body),
    }, 'Vidéo', 'video');
    if (!res.ok) {
      const err = await res.text();
      throw new Error(`HTTP ${res.status} — ${err.slice(0, 500)}`);
    }
    const data = await res.json();
    const videoId = data.video_id || data.id || data.task_id;
    if (!videoId) throw new Error('Pas de video_id');
    return String(videoId);
  }, 3));
}

function estimateGenerationTime(fallbackMs) {
  if (timingSamples.length === 0) return fallbackMs;
  const sorted = [...timingSamples].sort((a, b) => a - b);
  const mid = Math.floor(sorted.length / 2);
  return sorted.length % 2 === 0 ? (sorted[mid - 1] + sorted[mid]) / 2 : sorted[mid];
}

function pollUrl(videoId) {
  if (!/^[\w.-]+$/.test(videoId)) throw new Error('video_id inattendu');
  const url = new URL(POLL_BASE);
  url.searchParams.set('video_id', videoId);
  url.searchParams.set('model_name', MODEL_VIDEO);
  return url.href;
}

function videoStatusKind(status) {
  switch (status) {
    case 'completed':
    case 'succeeded':
    case 'done':
      return 'success';
    case 'failed':
    case 'error':
    case 'cancelled':
      return 'failure';
    default:
      return 'pending';
  }
}

async function pollVideo(videoId, onProgress) {
  const estimatedMs = estimateGenerationTime(90000 + state.durationSeconds * 7000);
  const minWait = Math.max(25, Math.min(VIDEO_FIRST_POLL_DELAY_SEC, Math.floor(estimatedMs / 1000 * 0.35)));
  log('info', 'POLLING', `Attente initiale ${minWait}s pour ${videoId}`);
  for (let remaining = minWait; remaining > 0; remaining -= 1) {
    if (state.stopRequested) throw new Error('Arrêt demandé');
    onProgress(`File vidéo — ${remaining}s`);
    await sleep(1000);
  }
  for (let attempt = 1; attempt <= VIDEO_MAX_POLL_ATTEMPTS; attempt += 1) {
    if (state.stopRequested) throw new Error('Arrêt demandé');
    let res;
    try {
      res = await apiFetch(pollUrl(videoId), {
        method: 'GET',
        headers: { Authorization: `Bearer ${process.env.AGNES_API_KEY}` },
      }, 'Polling', 'video');
    } catch (error) {
      log('warn', 'POLLING', `Erreur transitoire : ${error.message}`);
      await sleep(Math.min(90000, 15000 + attempt * 1500));
      continue;
    }
    let payload;
    try {
      payload = await res.json();
    } catch {
      log('warn', 'POLLING', 'Réponse non JSON, reprise');
      await sleep(Math.min(60000, 10000 + attempt * 1000));
      continue;
    }
    const status = String(payload.status || 'unknown').toLowerCase();
    const progress = Number(payload.progress || 0);
    onProgress(`Création — ${progress}%`);
    log('response', `POLL #${attempt}`, `Statut : ${status} (${progress}%)`);
    const kind = videoStatusKind(status);
    if (kind === 'success') {
      const videoUrl = payload.url || (payload.metadata && payload.metadata.url) || (payload.output && payload.output.url);
      if (!videoUrl) throw new Error('Terminé mais pas d’URL');
      return videoUrl;
    }
    if (kind === 'failure') throw new Error(`Échec définitif (${status})`);
    const wait = attempt < 5 ? 8 : Math.min(30, 10 + Math.floor(attempt / 5) * 3);
    onProgress(`File vidéo — nouvelle vérification dans ${wait}s`);
    await sleep(wait * 1000);
  }
  throw new Error('Délai maximal de polling dépassé');
}

function isVideoMagic(buffer) {
  if (buffer.length >= 4 && buffer[0] === 0x1a && buffer[1] === 0x45 && buffer[2] === 0xdf && buffer[3] === 0xa3) return true;
  const head = buffer.subarray(0, Math.min(buffer.length, 32)).toString('latin1');
  return head.includes('ftyp');
}

async function writeExclusiveFile(filePath, buffer) {
  await fs.mkdir(path.dirname(filePath), { recursive: true });
  let existing = null;
  try {
    existing = await fs.lstat(filePath);
  } catch (error) {
    if (error.code !== 'ENOENT') throw error;
  }
  if (existing) {
    if (existing.isSymbolicLink()) throw new Error('Lien symbolique refusé');
    throw new Error('Le fichier existe déjà');
  }
  let handle;
  try {
    handle = await fs.open(filePath, OPEN_EXCLUSIVE, 0o644);
  } catch (error) {
    if (error.code === 'ELOOP' || error.code === 'EEXIST') throw new Error('Lien symbolique refusé');
    throw error;
  }
  try {
    await handle.writeFile(buffer);
  } catch (error) {
    await handle.close();
    await fs.unlink(filePath).catch(() => {});
    throw error;
  }
  await handle.close();
}

async function assertLocalVideo(filePath) {
  const info = await fs.lstat(filePath);
  if (info.isSymbolicLink()) throw new Error('Lien symbolique refusé');
  if (!info.isFile()) throw new Error('Pas un fichier régulier');
  if (info.size <= 0) throw new Error('Fichier vide');
  if (info.size >= MAX_VIDEO_BYTES) throw new Error('Vidéo trop volumineuse');
  const handle = await fs.open(filePath, OPEN_READ_NOFOLLOW);
  try {
    const buffer = Buffer.alloc(32);
    await handle.read(buffer, 0, 32, 0);
    if (!isVideoMagic(buffer)) throw new Error('Signature vidéo invalide');
  } finally {
    await handle.close();
  }
}

function videoDestination(title) {
  const safe = sanitizeTitle(title);
  const stamp = new Date().toISOString().replace(/[-:]/g, '').replace(/\.\d{3}Z$/, 'Z');
  const name = `${safe}_${stamp}.mp4`;
  if (name !== path.basename(name) || !name.endsWith('.mp4')) throw new Error('Nom de fichier refusé');
  const dest = path.resolve(VIDEOS_DIR, name);
  const root = path.resolve(VIDEOS_DIR);
  if (dest !== path.join(root, name)) throw new Error('Chemin vidéo refusé');
  return dest;
}

function sanitizeTitle(title) {
  const cleaned = String(title || '')
    .normalize('NFKD')
    .replace(/[\u0300-\u036f]/g, '')
    .replace(/[^A-Za-z0-9._-]+/g, '_')
    .replace(/_+/g, '_')
    .replace(/^[._-]+|[._-]+$/g, '')
    .slice(0, 60);
  return cleaned || 'ambiance_lofi';
}

function commitSubject(title) {
  const clean = String(title || 'sans titre').replace(/[\r\n\0]/g, ' ').trim().slice(0, 120);
  return `feat: add new lofi capybara video: ${clean}`;
}

async function downloadVideo(videoUrl, destPath, options = {}) {
  const doFetch = options.fetchImpl || fetchImpl;
  let current = videoUrl;
  for (let redirectCount = 0; ; redirectCount += 1) {
    if (state.stopRequested) throw new Error('Arrêt demandé');
    const safe = await assertPublicHttpsUrl(current, { skipDns: options.skipDns === true });
    const res = await doFetch(safe.href, {
      redirect: 'manual',
      signal: AbortSignal.timeout(300000),
    });
    if (res.status >= 300 && res.status < 400) {
      if (redirectCount >= MAX_REDIRECTS) throw new Error('Trop de redirections');
      const location = res.headers.get('location');
      if (!location) throw new Error('Redirection sans cible');
      current = new URL(location, safe).href;
      continue;
    }
    if (!res.ok) throw new Error(`Téléchargement HTTP ${res.status}`);
    const declared = Number(res.headers.get('content-length') || 0);
    if (declared >= MAX_VIDEO_BYTES) throw new Error('Vidéo trop volumineuse');
    const buffer = Buffer.from(await res.arrayBuffer());
    if (buffer.length <= 0) throw new Error('Fichier vide');
    if (buffer.length >= MAX_VIDEO_BYTES) throw new Error('Vidéo trop volumineuse');
    if (!isVideoMagic(buffer.subarray(0, 32))) throw new Error('Signature vidéo invalide');
    await writeExclusiveFile(destPath, buffer);
    return destPath;
  }
}

function buildGitEnv(extraEnv) {
  const env = {
    PATH: process.env.PATH || '/usr/bin:/bin',
    HOME: '',
    GIT_CONFIG_NOSYSTEM: '1',
    GIT_TERMINAL_PROMPT: '0',
    GIT_AUTHOR_NAME: 'Lofi Capybara Bot',
    GIT_AUTHOR_EMAIL: 'lofi-capybara@users.noreply.github.com',
    GIT_COMMITTER_NAME: 'Lofi Capybara Bot',
    GIT_COMMITTER_EMAIL: 'lofi-capybara@users.noreply.github.com',
  };
  if (extraEnv) {
    for (const [key, value] of Object.entries(extraEnv)) {
      if (FORBIDDEN_GIT_ENV.has(key)) continue;
      env[key] = String(value);
    }
  }
  return env;
}

function createGit(options = {}) {
  const git = simpleGit({
    baseDir: options.baseDir || ROOT,
    binary: options.binary || 'git',
    allowEnvironment: GIT_ALLOWED_ENV,
    // askpass et credential.helper= sont des garde-fous voulus ; simple-git les bloque sinon.
    unsafe: { allowUnsafeAskPass: true, allowUnsafeCredentialHelper: true },
  });
  git.env(buildGitEnv(options.extraEnv));
  return git;
}

function askpassSource(token) {
  return `#!/usr/bin/env node
const token = ${JSON.stringify(token)};
const prompt = process.argv[2] || '';
if (/password/i.test(prompt)) process.stdout.write(token + '\\n');
else process.stdout.write('x-access-token\\n');
`;
}

async function writeAskpass(token) {
  const dir = await fs.mkdtemp(path.join(os.tmpdir(), 'lofi-askpass-'));
  await fs.chmod(dir, 0o700);
  const askpassPath = path.join(dir, 'askpass.cjs');
  await fs.writeFile(askpassPath, askpassSource(token), { mode: 0o700 });
  await fs.chmod(askpassPath, 0o700);
  return askpassPath;
}

async function stageAndCommit(git, filePath, message, baseDir) {
  const relative = path.relative(baseDir, filePath);
  if (!relative || relative.startsWith('..') || path.isAbsolute(relative)) throw new Error('Chemin vidéo hors du dépôt');
  await git.raw(['add', '--', relative]);
  await git.raw(['-c', 'commit.gpgsign=false', 'commit', '-m', message, '--', relative]);
}

async function ensureOrigin(git, repoUrl) {
  try {
    await git.raw(['remote', 'set-url', 'origin', repoUrl]);
  } catch (error) {
    const message = redact(error.message || '');
    if (/no such remote|could not set url/i.test(message)) {
      await git.raw(['remote', 'add', 'origin', repoUrl]);
      return;
    }
    throw new Error(message);
  }
}

async function publishVideo(filePath, title, hooks = {}) {
  const repoUrl = assertAllowedRepo(process.env.GITHUB_REPO_URL || '');
  const token = process.env.GITHUB_TOKEN || '';
  if (!token.trim()) throw new Error('GITHUB_TOKEN manquant');
  if (/[\r\n\0]/.test(token)) throw new Error('GITHUB_TOKEN invalide');
  await assertLocalVideo(filePath);
  const baseDir = hooks.baseDir || ROOT;
  const askpassPath = await writeAskpass(token);
  try {
    const git = createGit({
      binary: hooks.binary,
      baseDir,
      extraEnv: { ...(hooks.extraEnv || {}), GIT_ASKPASS: askpassPath },
    });
    await ensureOrigin(git, repoUrl);
    await stageAndCommit(git, filePath, commitSubject(title), baseDir);
    await git.raw(['-c', 'credential.helper=', 'push', 'origin', 'HEAD']);
  } catch (error) {
    throw new Error(redact(error.message || 'Échec git'));
  } finally {
    await fs.rm(path.dirname(askpassPath), { recursive: true, force: true });
  }
}

function readScenes(filePath) {
  const raw = JSON.parse(readFileSync(filePath, 'utf8'));
  if (!Array.isArray(raw) || raw.length === 0) throw new Error('scenes.json doit être un tableau non vide');
  return raw.map((item, index) => {
    const position = index + 1;
    if (typeof item === 'string') {
      const description = item.trim();
      if (description.length < 4) throw new Error(`Scène ${position} trop courte`);
      return { description, mood: state.mood };
    }
    if (item && typeof item === 'object' && typeof item.description === 'string') {
      const description = item.description.trim();
      if (description.length < 4) throw new Error(`Scène ${position} trop courte`);
      const mood = typeof item.mood === 'string' && MOODS[item.mood] ? item.mood : state.mood;
      return { description, mood };
    }
    throw new Error(`Scène ${position} invalide`);
  });
}

function loadRuntimeConfig() {
  const duration = Number(process.env.VIDEO_DURATION_SECONDS || 5);
  if (![5, 6, 8, 10].includes(duration)) throw new Error('VIDEO_DURATION_SECONDS invalide');
  const format = process.env.VIDEO_FORMAT || '16:9';
  if (format !== '16:9' && format !== '9:16') throw new Error('VIDEO_FORMAT invalide');
  const mood = process.env.VIDEO_MOOD || 'rainy_forest';
  if (!MOODS[mood]) throw new Error('VIDEO_MOOD invalide');
  state.durationSeconds = duration;
  state.format = format;
  state.mood = mood;
  if (!process.env.AGNES_API_KEY) throw new Error('AGNES_API_KEY manquante');
  if (!process.env.GITHUB_TOKEN) throw new Error('GITHUB_TOKEN manquant');
  assertAllowedRepo(process.env.GITHUB_REPO_URL || '');
}

async function runPipeline(scene) {
  log('info', 'SCENE', scene.description);
  const expansion = await generateExpansion(scene.description, scene.mood);
  const imageUrl = await generateImage(buildLofiImagePrompt(expansion), state.format);
  log('success', 'IMAGE', 'Image prête');
  createVideoIntervalMs = VIDEO_INTERVAL_SAFE;
  videoAttemptsSinceLastError = 0;
  const started = Date.now();
  const videoId = await generateVideo(imageUrl, buildLofiVideoPrompt(expansion), state.format);
  log('success', 'VIDÉO', `Tâche : ${videoId}`);
  const videoUrl = await pollVideo(videoId, (progress) => log('info', 'POLL', progress));
  timingSamples.push(Date.now() - started);
  if (timingSamples.length > 10) timingSamples.shift();
  const dest = videoDestination(expansion.title);
  await downloadVideo(videoUrl, dest);
  await assertLocalVideo(dest);
  await publishVideo(dest, expansion.title);
  log('success', 'GIT', `Vidéo publiée : ${path.basename(dest)}`);
  return dest;
}

function mockResponse(status, body, headers = {}) {
  const payload = typeof body === 'string' ? body : JSON.stringify(body);
  return {
    ok: status >= 200 && status < 300,
    status,
    headers: {
      get(name) {
        const key = Object.keys(headers).find((item) => item.toLowerCase() === String(name).toLowerCase());
        return key ? String(headers[key]) : null;
      },
    },
    clone() {
      return mockResponse(status, payload, headers);
    },
    async text() {
      return payload;
    },
    async json() {
      return JSON.parse(payload);
    },
    async arrayBuffer() {
      return Uint8Array.from(Buffer.from(payload)).buffer;
    },
  };
}

function assert(condition, message) {
  if (!condition) throw new Error(message);
}

async function expectThrow(fn, pattern) {
  try {
    await fn();
  } catch (error) {
    if (pattern && !pattern.test(error.message)) throw new Error(`Message inattendu : ${error.message}`);
    return;
  }
  throw new Error('Une erreur était attendue');
}

async function runSelfTest() {
  const previous = {
    fetchImpl,
    sleepImpl,
    rng,
    quietLogs,
    imageInterval: createImageIntervalMs,
    videoInterval: createVideoIntervalMs,
    agnes: process.env.AGNES_API_KEY,
    token: process.env.GITHUB_TOKEN,
    repo: process.env.GITHUB_REPO_URL,
    stop: state.stopRequested,
  };
  quietLogs = true;
  const sleeps = [];
  sleepImpl = async (ms) => {
    sleeps.push(ms);
  };
  rng = () => 0;
  const failures = [];
  async function check(name, fn) {
    try {
      await fn();
      console.log(`ok ${name}`);
    } catch (error) {
      failures.push(`${name}: ${error.message}`);
      console.log(`échec ${name}: ${error.message}`);
    }
  }

  await check('dépôt autorisé', async () => {
    assert(assertAllowedRepo(ALLOWED_REPO) === ALLOWED_REPO, 'URL canonique');
    assert(assertAllowedRepo(`${ALLOWED_REPO}.git`) === ALLOWED_REPO, 'suffixe .git');
    await expectThrow(() => assertAllowedRepo('http://github.com/urdirditfurd/videowan2g'), /https/);
    await expectThrow(() => assertAllowedRepo('https://user:token@github.com/urdirditfurd/videowan2g'), /identifiants/);
    await expectThrow(() => assertAllowedRepo('https://github.com/urdirditfurd/videowan2g.evil'), /non autorisée/);
    await expectThrow(() => assertAllowedRepo('https://evil.com/urdirditfurd/videowan2g'), /hôte/);
  });

  await check('adresses obfusquées refusées', async () => {
    const blocked = [
      'https://2130706433/video.mp4',
      'https://0x7f000001/video.mp4',
      'https://0177.0.0.1/video.mp4',
      'https://127.0.0.1/video.mp4',
      'https://[::ffff:127.0.0.1]/video.mp4',
      'https://[::1]/video.mp4',
      'https://169.254.169.254/latest/meta-data',
      'https://localhost/video.mp4',
      'https://10.1.2.3/video.mp4',
      'http://example.com/video.mp4',
      'https://user:pass@example.com/video.mp4',
    ];
    for (const sample of blocked) {
      let threw = false;
      try {
        assertUrlIsPublic(new URL(sample));
      } catch {
        threw = true;
      }
      assert(threw, `aurait dû refuser ${sample}`);
    }
    assertUrlIsPublic(new URL('https://cdn.example/video.mp4'));
    assert(isPrivateAddress('::ffff:7f00:1'), 'IPv4 mappée loopback');
  });

  await check('prompt unique', async () => {
    const prompt = buildExpansionPrompt('maison japonaise sous la pluie', 'rainy_forest', () => 0);
    assert(prompt.includes('TIME OF DAY: blue hour before sunrise'), 'heure');
    assert(prompt.includes('SEASON: early spring'), 'saison');
    assert(prompt.includes('WEATHER: fine rain'), 'météo');
    assert(prompt.includes('capybara'), 'capybara');
    assert(prompt.includes('Studio Ghibli'), 'ghibli');
    const other = buildExpansionPrompt('maison japonaise sous la pluie', 'rainy_forest', () => 0.99);
    assert(other !== prompt, 'les graines doivent varier');
  });

  await check('extraction JSON', async () => {
    const fenced = '```json\n{"title":"Pluie","image_prompt":"a capybara","video_motion_prompt":"slow"}\n```';
    const parsed = extractJson(fenced);
    assert(parsed.title === 'Pluie' && parsed.image_prompt === 'a capybara', 'json clôturé');
    const broken = '{"title":"Brume","image_prompt":"soft rain","video_motion_prompt":"drift"';
    const salvaged = extractJson(broken);
    assert(salvaged && salvaged.image_prompt === 'soft rain', 'récupération');
  });

  await check('backoff de file sans consommer les essais', async () => {
    const calls = [];
    fetchImpl = async () => {
      calls.push(1);
      if (calls.length < 3) return mockResponse(503, '{"error":"video_queue_full"}');
      return mockResponse(200, '{"ok":true}');
    };
    sleeps.length = 0;
    rng = () => 0;
    const res = await apiFetch('https://apihub.agnes-ai.com/v1/videos', { method: 'POST' }, 'Vidéo', 'video');
    assert(res.ok, 'réponse finale');
    assert(calls.length === 3, 'trois appels');
    assert(sleeps[0] === 60000 && sleeps[1] === 120000, `attentes ${sleeps.join(',')}`);
    assert(queueFullDelaySeconds(5, 0) === 300, 'palier 300');
    assert(queueFullDelaySeconds(0, 999) === 360, 'plafond 360');
  });

  await check('erreur HTTP terminale hors du catch réseau', async () => {
    let calls = 0;
    fetchImpl = async () => {
      calls += 1;
      return mockResponse(401, 'clé refusée');
    };
    await expectThrow(() => apiFetch('https://apihub.agnes-ai.com/v1/chat/completions', { method: 'POST' }, 'Expansion', 'text'), /HTTP 401/);
    assert(calls === 1, 'pas de retry sur 401 texte');
    let reads = 0;
    fetchImpl = async () => ({
      ok: false,
      status: 400,
      headers: { get: () => null },
      clone() {
        return { async text() { return 'détail vidéo'; } };
      },
      async text() {
        reads += 1;
        return 'détail vidéo';
      },
    });
    const returned = await apiFetch('https://apihub.agnes-ai.com/v1/videos', { method: 'POST' }, 'Vidéo', 'video');
    assert((await returned.text()) === 'détail vidéo', 'corps encore lisible');
    assert(reads === 1, 'le corps original n’est lu qu’une fois');
  });

  await check('file limitée', async () => {
    const queue = createRateLimitedQueue(1, () => 30);
    const order = [];
    const started = Date.now();
    await Promise.all([
      queue.add(async () => { order.push('a'); }),
      queue.add(async () => { order.push('b'); }),
    ]);
    assert(order.join('') === 'ab', 'ordre');
    assert(Date.now() - started >= 25, 'intervalle');
  });

  await check('écriture sans suivre un lien', async () => {
    const dir = await fs.mkdtemp(path.join(os.tmpdir(), 'lofi-link-'));
    try {
      const target = path.join(dir, 'cible.txt');
      const link = path.join(dir, 'lien.mp4');
      await fs.writeFile(target, 'original');
      await fs.symlink(target, link);
      await expectThrow(() => writeExclusiveFile(link, Buffer.from('nouveau')), /Lien symbolique/);
      assert(await fs.readFile(target, 'utf8') === 'original', 'cible intacte');
      await expectThrow(() => assertLocalVideo(link), /Lien symbolique/);
      const real = path.join(dir, 'ok.mp4');
      const payload = Buffer.from('0000ftypisom0000');
      await writeExclusiveFile(real, payload);
      await assertLocalVideo(real);
      await expectThrow(() => writeExclusiveFile(real, payload), /existe déjà/);
    } finally {
      await fs.rm(dir, { recursive: true, force: true });
    }
  });

  await check('téléchargement borné', async () => {
    const dir = await fs.mkdtemp(path.join(os.tmpdir(), 'lofi-dl-'));
    const dest = path.join(dir, 'video.mp4');
    try {
      await expectThrow(() => downloadVideo('https://127.0.0.1/a.mp4', dest, { skipDns: true }), /non publique|interdit|https/);
      await expectThrow(() => downloadVideo('https://2130706433/a.mp4', dest, { skipDns: true }), /non publique/);
      await expectThrow(() => downloadVideo('https://[::ffff:127.0.0.1]/a.mp4', dest, { skipDns: true }), /non publique/);
      let hops = 0;
      await expectThrow(() => downloadVideo('https://cdn.example/start.mp4', dest, {
        skipDns: true,
        fetchImpl: async () => {
          hops += 1;
          return mockResponse(302, '', { location: `https://cdn.example/next-${hops}` });
        },
      }), /Trop de redirections/);
      assert(hops === 3, 'deux redirections puis arrêt');
      await expectThrow(() => downloadVideo('https://cdn.example/start.mp4', dest, {
        skipDns: true,
        fetchImpl: async () => mockResponse(302, '', { location: 'https://2130706433/secret' }),
      }), /non publique/);
      await expectThrow(() => downloadVideo('https://cdn.example/start.mp4', dest, {
        skipDns: true,
        fetchImpl: async () => mockResponse(302, '', { location: 'https://[::ffff:127.0.0.1]/secret' }),
      }), /non publique/);
      const good = Buffer.concat([Buffer.from('0000ftypisom'), Buffer.alloc(8)]);
      await downloadVideo('https://cdn.example/ok.mp4', dest, {
        skipDns: true,
        fetchImpl: async () => mockResponse(200, good.toString('latin1'), { 'content-length': String(good.length) }),
      });
      await assertLocalVideo(dest);
    } finally {
      await fs.rm(dir, { recursive: true, force: true });
    }
  });

  await check('git sans secrets', async () => {
    const agnesSentinel = 'capybara-sentinel-agnes-7f3a9c';
    const githubSentinel = 'capybara-sentinel-github-7f3a9c';
    process.env.AGNES_API_KEY = agnesSentinel;
    process.env.GITHUB_TOKEN = githubSentinel;
    process.env.GITHUB_REPO_URL = ALLOWED_REPO;
    const dir = await fs.mkdtemp(path.join(os.tmpdir(), 'lofi-git-'));
    const dump = path.join(dir, 'dump.json');
    const fake = path.join(dir, 'fake-git.cjs');
    const video = path.join(dir, 'generated_videos', 'pluie_test.mp4');
    await fs.mkdir(path.dirname(video), { recursive: true });
    await fs.writeFile(fake, `#!/usr/bin/env node\nconst fs = require('fs');\nconst out = process.env.FAKE_GIT_DUMP;\nconst entry = { argv: process.argv.slice(2), env: process.env };\nlet prev = [];\ntry { prev = JSON.parse(fs.readFileSync(out, 'utf8')); } catch (e) { prev = []; }\nif (!Array.isArray(prev)) prev = [];\nprev.push(entry);\nfs.writeFileSync(out, JSON.stringify(prev));\nprocess.exit(0);\n`);
    await fs.chmod(fake, 0o755);
    await fs.writeFile(dump, '[]');
    const payload = Buffer.from('0000ftypisom0000');
    await fs.writeFile(video, payload);
    try {
      const git = createGit({ binary: fake, baseDir: dir, extraEnv: { FAKE_GIT_DUMP: dump } });
      await git.raw(['status']);
      await publishVideo(video, 'Pluie douce', { binary: fake, baseDir: dir, extraEnv: { FAKE_GIT_DUMP: dump } });
      const entries = JSON.parse(await fs.readFile(dump, 'utf8'));
      const blob = JSON.stringify(entries);
      assert(!blob.includes('AGNES_API_KEY'), 'nom AGNES_API_KEY présent');
      assert(!blob.includes('GITHUB_TOKEN'), 'nom GITHUB_TOKEN présent');
      assert(!blob.includes(agnesSentinel), 'valeur Agnes présente');
      assert(!blob.includes(githubSentinel), 'valeur GitHub présente');
      assert(entries[0].env.HOME === '', 'HOME vide');
      assert(entries[0].env.GIT_CONFIG_NOSYSTEM === '1', 'config système ignorée');
      const joined = entries.map((entry) => entry.argv.join(' ')).join('\n');
      assert(joined.includes('--'), 'pathspec');
      assert(joined.includes('commit.gpgsign=false'), 'signature désactivée');
      assert(joined.includes('push origin HEAD'), 'push');
      assert(joined.includes('credential.helper='), 'helper désactivé');
      assert(joined.includes('feat: add new lofi capybara video: Pluie douce'), 'message de commit');
      assert(joined.includes(ALLOWED_REPO), 'remote propre');
      assert(!joined.includes(githubSentinel), 'jeton dans les arguments');
      const askpass = entries.map((entry) => entry.env.GIT_ASKPASS).find(Boolean);
      assert(askpass, 'askpass posé');
      assert(!(await fs.stat(askpass).then(() => true, () => false)), 'askpass supprimé');
    } finally {
      await fs.rm(dir, { recursive: true, force: true });
    }
  });

  await check('askpass', async () => {
    const token = 'capybara-sentinel-github-askpass';
    const askpassPath = await writeAskpass(token);
    try {
      const password = spawnSync(process.execPath, [askpassPath, 'Password for https://github.com:'], { encoding: 'utf8' });
      const username = spawnSync(process.execPath, [askpassPath, 'Username for https://github.com:'], { encoding: 'utf8' });
      assert(password.status === 0 && sha256(password.stdout) === sha256(`${token}\n`), 'mot de passe askpass');
      assert(username.stdout === 'x-access-token\n', 'identifiant askpass');
    } finally {
      await fs.rm(path.dirname(askpassPath), { recursive: true, force: true });
    }
  });

  await check('scènes exemple', async () => {
    state.mood = 'rainy_forest';
    const scenes = readScenes(path.join(ROOT, 'scenes.json'));
    assert(scenes.length === 5, 'cinq scènes');
    assert(scenes[0].description.includes('pluie'), 'première scène');
  });

  fetchImpl = previous.fetchImpl;
  sleepImpl = previous.sleepImpl;
  rng = previous.rng;
  quietLogs = previous.quietLogs;
  createImageIntervalMs = previous.imageInterval;
  createVideoIntervalMs = previous.videoInterval;
  state.stopRequested = previous.stop;
  if (previous.agnes === undefined) delete process.env.AGNES_API_KEY;
  else process.env.AGNES_API_KEY = previous.agnes;
  if (previous.token === undefined) delete process.env.GITHUB_TOKEN;
  else process.env.GITHUB_TOKEN = previous.token;
  if (previous.repo === undefined) delete process.env.GITHUB_REPO_URL;
  else process.env.GITHUB_REPO_URL = previous.repo;
  if (failures.length) {
    console.log(`${failures.length} test(s) en échec`);
    return false;
  }
  console.log('auto-test OK');
  return true;
}

async function main() {
  loadDotEnv();
  if (process.argv.includes('--self-test')) {
    const ok = await runSelfTest();
    process.exitCode = ok ? 0 : 1;
    return;
  }
  loadRuntimeConfig();
  const scenes = readScenes(path.join(ROOT, 'scenes.json'));
  let failed = 0;
  for (const scene of scenes) {
    if (state.stopRequested) break;
    try {
      await runPipeline(scene);
    } catch (error) {
      failed += 1;
      log('error', 'SCENE', redact(error.message || 'échec'));
    }
  }
  if (state.stopRequested || failed > 0) process.exitCode = 1;
}

process.on('SIGINT', () => {
  state.stopRequested = true;
  log('warn', 'STOP', 'Arrêt demandé');
});

const isDirectRun = process.argv[1] && path.resolve(process.argv[1]) === fileURLToPath(import.meta.url);
if (isDirectRun) {
  main().catch((error) => {
    log('error', 'GLOBAL', redact(error.message || 'échec'));
    process.exitCode = 1;
  });
}
