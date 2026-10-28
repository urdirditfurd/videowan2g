/**
 * Pipeline headless Lofi Capybara — Agnes AI, puis publication sur videowan2g.
 *
 * npm install
 * cp .env.example .env
 * node automation.js --self-test
 * node automation.js
 */

import { execFileSync } from 'node:child_process';
import dns from 'node:dns/promises';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { Readable, Transform } from 'node:stream';
import { pipeline } from 'node:stream/promises';
import { fileURLToPath } from 'node:url';
import dotenv from 'dotenv';
import fetch from 'node-fetch';
import { simpleGit } from 'simple-git';

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)));
const VIDEOS_DIR = path.join(ROOT, 'generated_videos');
const API_BASE = 'https://apihub.agnes-ai.com/v1';
const POLL_BASE = 'https://apihub.agnes-ai.com/agnesapi';
const MODEL_TEXT = 'agnes-2.5-flash';
const MODEL_IMAGE = 'agnes-image-2.5-flash';
const MODEL_VIDEO = 'agnes-video-2.5-flash';
const ALLOWED_SLUG = 'urdirditfurd/videowan2g';
const CANONICAL_REMOTE = 'https://github.com/urdirditfurd/videowan2g.git';
const VIDEO_NAME_RE = /^[a-z0-9_]+_\d+(?:_\d+)?\.mp4$/;
const MAX_REDIRECTS = 2;
const VIDEO_QUEUE_BACKOFF_SEC = [60, 120, 300];
const VIDEO_QUEUE_MAX_WAIT_SEC = 360;
const VIDEO_QUEUE_MAX_RETRIES = 12;
const IMAGE_INTERVAL_MIN = 3200;
const IMAGE_INTERVAL_MAX = 12000;
const VIDEO_INTERVAL_MIN = 62000;
const VIDEO_INTERVAL_MAX = 90000;
const VIDEO_INTERVAL_SAFE = 75000;
const VIDEO_FIRST_POLL_DELAY_SEC = 80;
const VIDEO_MAX_POLL_ATTEMPTS = 100;
const DEFAULT_MAX_VIDEO_BYTES = 100 * 1024 * 1024;
const GIT_ENV_ALLOW = [
  'GIT_AUTHOR_NAME',
  'GIT_AUTHOR_EMAIL',
  'GIT_COMMITTER_NAME',
  'GIT_COMMITTER_EMAIL',
  'GIT_CONFIG_NOSYSTEM',
  'GIT_TERMINAL_PROMPT',
  'GIT_ASKPASS',
];

const GHIBLI_PALETTE = 'soft pastel palette inspired by Studio Ghibli watercolor backgrounds: sage green #8fad88, warm cream #f2efe9, dusky blue #7a9eb8, soft gold #c4a882, gentle forest greens, muted earth tones, warm amber highlights. Never grey, never desaturated, never neon. Rich but calm. Hand-painted watercolor feel with soft gradients.';

const MOODS = {
  rainy_forest: {
    label: 'Forêt sous la pluie',
    desc: 'a lush green forest under soft rain, misty atmosphere, warm lamplight glowing through wet leaves, wet earth and moss details',
  },
  golden_sunset: {
    label: 'Coucher de soleil doré',
    desc: 'a golden hour sunset with warm amber light, long soft shadows, glowing dust motes in the air, nostalgic warmth',
  },
  cozy_room: {
    label: 'Intérieur douillet',
    desc: 'a cozy wooden interior with warm candlelight, soft blankets, steaming tea, rain visible through a window, amber glow',
  },
  japanese_garden: {
    label: 'Jardin japonais',
    desc: 'a serene Japanese garden with stone lanterns, koi pond, cherry blossoms, moss-covered rocks, soft diffused daylight',
  },
  mountain_lake: {
    label: 'Lac de montagne',
    desc: 'a still mountain lake reflecting soft clouds, pine trees, gentle morning mist, cool blue and sage tones',
  },
  autumn_leaves: {
    label: "Feuilles d'automne",
    desc: 'an autumn forest with falling orange and gold leaves, warm late afternoon light, soft breeze, rustic wooden path',
  },
  night_stars: {
    label: 'Nuit étoilée',
    desc: 'a peaceful night with a starry sky, soft moonlight, silhouetted trees, gentle blue and violet tones, distant fireflies',
  },
  ocean_breeze: {
    label: 'Brise océane',
    desc: 'a calm ocean shore with gentle waves, soft sea breeze, warm sand, pastel sky, distant seagulls, serene horizon',
  },
};

const TIMES_OF_DAY = [
  'aube brumeuse',
  'matin clair',
  'midi doux',
  'fin d’après-midi dorée',
  'heure bleue',
  'nuit calme',
];
const SEASONS = ['début du printemps', 'plein été', 'automne avancé', 'hiver doux'];
const WEATHERS = ['bruine fine', 'ciel dégagé', 'brume légère', 'neige fondue', 'vent léger', 'après la pluie'];
const BLOCKED_HOSTS = new Set([
  'localhost',
  'localhost.localdomain',
  'metadata.google.internal',
  'metadata.google.com',
]);
const REDIRECT_STATUSES = new Set([301, 302, 303, 307, 308]);
const RETRYABLE_STATUSES = new Set([500, 502, 503, 504, 520]);
const ALLOWED_DURATIONS = new Set([5, 6, 8, 10]);
const ALLOWED_FORMATS = new Set(['16:9', '9:16']);

const deps = {
  fetch,
  sleep(ms) {
    return new Promise((resolve) => {
      setTimeout(resolve, ms);
    });
  },
  lookup(hostname, options) {
    return dns.lookup(hostname, options);
  },
  rng: Math.random,
  maxVideoBytes: DEFAULT_MAX_VIDEO_BYTES,
  quiet: false,
};

const state = {
  stopRequested: false,
  durationSeconds: 5,
  format: '16:9',
  mood: 'rainy_forest',
};

let createImageIntervalMs = 4000;
let createVideoIntervalMs = VIDEO_INTERVAL_SAFE;
let imageAttemptsSinceLastError = 0;
let videoAttemptsSinceLastError = 0;
const timingSamples = { video: [] };

dotenv.config({ path: path.join(ROOT, '.env'), quiet: true });

function log(level, category, message) {
  if (deps.quiet) return;
  const line = `${new Date().toISOString()} [${category}] ${message}`;
  if (level === 'error' || level === 'warn') console.error(line);
  else console.log(line);
}

function resetRuntime() {
  createImageIntervalMs = 4000;
  createVideoIntervalMs = VIDEO_INTERVAL_SAFE;
  imageAttemptsSinceLastError = 0;
  videoAttemptsSinceLastError = 0;
  state.stopRequested = false;
  state.durationSeconds = 5;
  state.format = '16:9';
  state.mood = 'rainy_forest';
  timingSamples.video = [];
  deps.rng = Math.random;
  deps.maxVideoBytes = DEFAULT_MAX_VIDEO_BYTES;
  deps.quiet = true;
  deps.fetch = fetch;
  deps.sleep = (ms) => new Promise((resolve) => {
    setTimeout(resolve, ms);
  });
  deps.lookup = (hostname, options) => dns.lookup(hostname, options);
}

function pick(list, rng = deps.rng) {
  const index = Math.min(list.length - 1, Math.max(0, Math.floor(rng() * list.length)));
  return list[index];
}

function pickVariation(rng = deps.rng) {
  return {
    timeOfDay: pick(TIMES_OF_DAY, rng),
    season: pick(SEASONS, rng),
    weather: pick(WEATHERS, rng),
  };
}

function getApiKey() {
  return String(process.env.AGNES_API_KEY || '').trim();
}

function authHeaders(json = false) {
  const headers = { Authorization: `Bearer ${getApiKey()}` };
  if (json) headers['Content-Type'] = 'application/json';
  return headers;
}

function parseDuration(value) {
  const seconds = Number.parseInt(String(value), 10);
  if (!ALLOWED_DURATIONS.has(seconds)) {
    throw new Error('VIDEO_SECONDS doit valoir 5, 6, 8 ou 10');
  }
  return seconds;
}

function parseFormat(value) {
  const format = String(value || '').trim();
  if (!ALLOWED_FORMATS.has(format)) throw new Error('VIDEO_FORMAT doit valoir 16:9 ou 9:16');
  return format;
}

function resolveScenesPath() {
  const requested = process.env.SCENES_FILE || 'scenes.json';
  const resolved = path.resolve(ROOT, requested);
  const prefix = ROOT.endsWith(path.sep) ? ROOT : `${ROOT}${path.sep}`;
  if (resolved !== ROOT && !resolved.startsWith(prefix)) {
    throw new Error('SCENES_FILE hors du projet');
  }
  if (path.basename(resolved) === '.env' || resolved.endsWith('.env')) {
    throw new Error('SCENES_FILE ne peut pas être un fichier d’environnement');
  }
  return resolved;
}

function loadScenes(filePath = resolveScenesPath()) {
  const data = JSON.parse(fs.readFileSync(filePath, 'utf8'));
  if (!Array.isArray(data)) throw new Error('scenes.json doit être un tableau');
  return data.map((item, index) => {
    if (typeof item === 'string') return { description: item.trim() };
    if (item && typeof item.description === 'string') {
      return {
        description: item.description.trim(),
        mood: item.mood,
        durationSeconds: item.durationSeconds,
        format: item.format,
      };
    }
    throw new Error(`Scène ${index + 1} invalide`);
  });
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
  let raw = String(text).trim();
  const fenced = raw.match(/```(?:json)?\s*([\s\S]*?)```/);
  if (fenced) raw = fenced[1].trim();
  const start = raw.indexOf('{');
  if (start < 0) return null;
  raw = raw.slice(start);
  const attempts = [
    () => JSON.parse(raw),
    () => JSON.parse(escapeControlCharsInStrings(raw)),
    () => {
      const end = raw.lastIndexOf('}');
      if (end <= 0) throw new Error('accolade manquante');
      return JSON.parse(escapeControlCharsInStrings(raw.slice(0, end + 1)));
    },
    () => JSON.parse(balanceJson(escapeControlCharsInStrings(raw))),
    () => {
      const salvaged = salvageExpansion(raw);
      if (!salvaged || (!salvaged.image_prompt && !salvaged.video_motion_prompt)) {
        throw new Error('salvage vide');
      }
      return salvaged;
    },
  ];
  for (const attempt of attempts) {
    try {
      return attempt();
    } catch {
      // stratégie suivante
    }
  }
  return null;
}

function buildExpansionPrompt(userDescription, mood, variation) {
  const ambience = MOODS[mood] || MOODS.rainy_forest;
  return [
    'You are an art director for Studio Ghibli-inspired lofi ambience videos.',
    '',
    'USER SCENE IDEA (French):',
    userDescription,
    '',
    'AMBIENCE (French mood chosen):',
    ambience.label,
    '—',
    ambience.desc,
    '',
    'UNIQUE VARIATION (mandatory — this exact combination must be visible, so the video stays unique):',
    `Time of day: ${variation.timeOfDay}.`,
    `Season: ${variation.season}.`,
    `Weather: ${variation.weather}.`,
    'Show these three details in the light, the vegetation, and the air. Do not ignore them, even if the base idea is similar to another scene.',
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
    '- Color palette:',
    GHIBLI_PALETTE,
    '',
    'OUTPUT — strict JSON only, no markdown:',
    '{',
    '  "title": "A short poetic title in French (max 60 chars)",',
    '  "image_prompt": "Ultra-detailed English visual description for an image model. Include: setting, lighting, colors, atmosphere, the capybara (with its location and pose), the mandatory time of day, season and weather, Ghibli watercolor style, cinematic composition. 150-250 words.",',
    '  "video_motion_prompt": "English description of slow gentle motions for the video model: capybara breathing, leaves falling, mist drifting, water ripples, seamless loop. 60-120 words."',
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

function createRateLimitedQueue(maxConcurrent, minIntervalMs) {
  let running = 0;
  let lastStart = 0;
  let wake = null;
  const queue = [];
  let resolveIdle = null;
  const interval = () => (typeof minIntervalMs === 'function' ? minIntervalMs() : minIntervalMs);

  function schedule(ms) {
    if (wake !== null) return;
    wake = setTimeout(() => {
      wake = null;
      tryNext();
    }, ms);
  }

  function tryNext() {
    if (running >= maxConcurrent) return;
    if (queue.length === 0) {
      if (running === 0 && resolveIdle) {
        const finish = resolveIdle;
        resolveIdle = null;
        finish();
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
        if (running === 0 && queue.length === 0) {
          resolve();
          return;
        }
        resolveIdle = resolve;
      });
    },
  };
}

const imageQueue = createRateLimitedQueue(1, () => createImageIntervalMs);
const videoQueue = createRateLimitedQueue(1, () => createVideoIntervalMs);

function computeQueueWaitSeconds(queueAttempt, retryAfterSec = 0, rng = deps.rng) {
  const header = Number(retryAfterSec);
  if (Number.isFinite(header) && header > 0) {
    return Math.min(VIDEO_QUEUE_MAX_WAIT_SEC, Math.round(header));
  }
  const index = Math.min(Math.max(0, queueAttempt), VIDEO_QUEUE_BACKOFF_SEC.length - 1);
  const base = VIDEO_QUEUE_BACKOFF_SEC[index];
  const roll = Math.min(1, Math.max(0, Number(rng()) || 0));
  const factor = 0.85 + roll * 0.3;
  return Math.min(VIDEO_QUEUE_MAX_WAIT_SEC, Math.max(1, Math.round(base * factor)));
}

function networkWaitSeconds(kind, attempt) {
  return Math.min(kind === 'video' ? 120 : 45, kind === 'video' ? 10 + attempt * 8 : 3 + attempt * 4);
}

function rateLimitWaitSeconds(kind, attempt, retryAfter) {
  const cap = kind === 'video' ? 180 : 120;
  if (retryAfter > 0) return Math.min(cap, retryAfter);
  return Math.min(cap, kind === 'video' ? 60 + attempt * 8 : 10 + attempt * 12);
}

function serverWaitSeconds(kind, attempt) {
  return Math.min(kind === 'video' ? 180 : 60, kind === 'video' ? 20 + attempt * 10 : 5 + attempt * 5);
}

function parseRetryAfter(res) {
  try {
    const raw = res.headers?.get?.('retry-after') || '0';
    const value = Number.parseInt(raw, 10);
    return Number.isFinite(value) && value > 0 ? value : 0;
  } catch {
    return 0;
  }
}

async function readBodyPreview(res) {
  try {
    const clone = typeof res.clone === 'function' ? res.clone() : res;
    return String(await clone.text() || '').slice(0, 1200);
  } catch {
    return '';
  }
}

function noteSuccess(kind) {
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
}

async function apiFetch(url, options = {}, label = 'API', kind = 'default') {
  const maxAttempts = kind === 'video' ? 10 : kind === 'image' ? 10 : 8;
  let attempt = 0;
  let queueRetries = 0;

  while (attempt < maxAttempts) {
    if (state.stopRequested) throw new Error('Arrêt demandé');
    let res;
    try {
      log('api', `REQ ${label}`, `${options.method || 'GET'} ${url} (tentative ${attempt + 1}/${maxAttempts})`);
      res = await deps.fetch(url, options);
    } catch (error) {
      if (error && error.message === 'Arrêt demandé') throw error;
      const wait = networkWaitSeconds(kind, attempt);
      log('warn', 'NETWORK', `${error.message} — retry dans ${wait}s`);
      attempt += 1;
      if (attempt >= maxAttempts) break;
      await deps.sleep(wait * 1000);
      continue;
    }

    if (res.ok) {
      noteSuccess(kind);
      return res;
    }

    const errorText = await readBodyPreview(res);
    const retryAfter = parseRetryAfter(res);
    log('response', `RÉP ${label}`, `HTTP ${res.status}`);
    const queueFull = kind === 'video' && /video_queue_full|queue is full/i.test(errorText);

    if (queueFull) {
      if (queueRetries >= VIDEO_QUEUE_MAX_RETRIES) {
        throw new Error(`File vidéo saturée — reprises de file épuisées pour ${label}`);
      }
      const wait = computeQueueWaitSeconds(queueRetries, retryAfter);
      queueRetries += 1;
      videoAttemptsSinceLastError = 0;
      log('warn', 'VIDEO_QUEUE', `Queue vidéo saturée — nouvelle tentative dans ${wait}s`);
      await deps.sleep(wait * 1000);
      continue;
    }

    if ([400, 401, 404].includes(res.status) && kind !== 'video') {
      throw new Error(`HTTP ${res.status} — ${errorText.slice(0, 300)}`);
    }

    if (res.status === 429) {
      if (kind === 'image') createImageIntervalMs = Math.min(IMAGE_INTERVAL_MAX, createImageIntervalMs + 1000);
      if (kind === 'video') createVideoIntervalMs = Math.min(VIDEO_INTERVAL_MAX, createVideoIntervalMs + 6000);
      const wait = rateLimitWaitSeconds(kind, attempt, retryAfter);
      log('warn', 'RATELIMIT', `429 — attente ${wait}s`);
      attempt += 1;
      if (attempt >= maxAttempts) break;
      await deps.sleep(wait * 1000);
      continue;
    }

    if (RETRYABLE_STATUSES.has(res.status)) {
      const wait = serverWaitSeconds(kind, attempt);
      log('warn', 'SERVICE', `HTTP ${res.status} — attente ${wait}s`);
      attempt += 1;
      if (attempt >= maxAttempts) break;
      await deps.sleep(wait * 1000);
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
        await deps.sleep(wait * 1000);
      }
    }
  }
  throw lastError || new Error(`${name} échoué`);
}

async function generateExpansion(userDescription, mood, variation) {
  const prompt = buildExpansionPrompt(userDescription, mood, variation);
  const body = {
    model: MODEL_TEXT,
    messages: [
      {
        role: 'system',
        content: 'You are a Studio Ghibli-inspired art director specialized in lofi ambience videos. You always return strict JSON matching the requested schema exactly. Every scene includes a cute calm capybara. Every scene is peaceful, detailed, watercolor, and loopable with slow gentle motion. Each scene must visibly reflect the requested time of day, season and weather so it stays unique. No text, no watermark. Return only JSON.',
      },
      { role: 'user', content: prompt },
    ],
    temperature: 0.85,
    max_tokens: 2000,
    response_format: { type: 'json_object' },
  };
  const res = await apiFetch(`${API_BASE}/chat/completions`, {
    method: 'POST',
    headers: authHeaders(true),
    body: JSON.stringify(body),
  }, 'Expansion lofi', 'text');
  if (!res.ok) {
    const err = await res.text();
    throw new Error(`HTTP ${res.status} — ${String(err).slice(0, 300)}`);
  }
  const data = await res.json();
  const content = data.choices && data.choices[0] && data.choices[0].message && data.choices[0].message.content;
  const expansion = extractJson(content);
  if (!expansion || !expansion.image_prompt) throw new Error('Expansion invalide');
  if (!expansion.title) expansion.title = 'Ambiance Lofi';
  if (!expansion.video_motion_prompt) {
    expansion.video_motion_prompt = 'Gentle slow motion: capybara breathing softly, leaves drifting down, faint mist, seamless loop.';
  }
  log('info', 'EXPANSION', `Titre : ${expansion.title}`);
  return expansion;
}

function assertImageReference(imageUrl) {
  if (String(imageUrl).startsWith('data:image/')) return;
  let url;
  try {
    url = new URL(imageUrl);
  } catch {
    throw new Error('URL image invalide');
  }
  if (url.protocol !== 'https:' || url.username || url.password) {
    throw new Error('URL image refusée');
  }
}

async function generateImage(prompt, format) {
  return retryTask('Image', async () => {
    const ratio = format === '9:16' ? '9:16' : '16:9';
    const body = { model: MODEL_IMAGE, prompt, size: '1K', ratio, n: 1 };
    const res = await apiFetch(`${API_BASE}/images/generations`, {
      method: 'POST',
      headers: authHeaders(true),
      body: JSON.stringify(body),
    }, 'Image', 'image');
    if (!res.ok) {
      const err = await res.text();
      throw new Error(`HTTP ${res.status} — ${String(err).slice(0, 500)}`);
    }
    const data = await res.json();
    const item = data.data && data.data[0];
    if (!item) throw new Error('Pas de donnée image');
    if (item.url) {
      assertImageReference(item.url);
      return item.url;
    }
    if (item.b64_json) return `data:image/png;base64,${item.b64_json}`;
    throw new Error('Format image inconnu');
  }, 4);
}

async function generateVideo(imageUrl, prompt) {
  return retryTask('Vidéo', async () => {
    assertImageReference(imageUrl);
    const aspect = state.format === '9:16' ? '9:16' : '16:9';
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
      headers: authHeaders(true),
      body: JSON.stringify(body),
    }, 'Vidéo', 'video');
    if (!res.ok) {
      const err = await res.text();
      throw new Error(`HTTP ${res.status} — ${String(err).slice(0, 500)}`);
    }
    const data = await res.json();
    const videoId = data.video_id || data.id || data.task_id;
    if (!videoId) throw new Error('Pas de video_id');
    return videoId;
  }, 3);
}

function estimateGenerationTime(key, fallbackMs) {
  const samples = timingSamples[key];
  if (!samples || samples.length === 0) return fallbackMs;
  const sorted = [...samples].sort((a, b) => a - b);
  const mid = Math.floor(sorted.length / 2);
  return sorted.length % 2 === 0 ? (sorted[mid - 1] + sorted[mid]) / 2 : sorted[mid];
}

function recordGenerationTime(key, ms) {
  if (!timingSamples[key]) timingSamples[key] = [];
  timingSamples[key].push(ms);
  if (timingSamples[key].length > 10) timingSamples[key] = timingSamples[key].slice(-10);
}

async function pollVideo(videoId, onProgress = () => {}) {
  const estimatedMs = estimateGenerationTime('video', 90000 + state.durationSeconds * 7000);
  const minWait = Math.max(25, Math.min(VIDEO_FIRST_POLL_DELAY_SEC, Math.floor((estimatedMs / 1000) * 0.35)));
  let remaining = minWait;
  while (remaining > 0) {
    if (state.stopRequested) throw new Error('Arrêt demandé');
    onProgress(`File vidéo — ${remaining}s`);
    await deps.sleep(1000);
    remaining -= 1;
  }
  let attempt = 0;
  while (attempt < VIDEO_MAX_POLL_ATTEMPTS) {
    if (state.stopRequested) throw new Error('Arrêt demandé');
    attempt += 1;
    const url = `${POLL_BASE}?video_id=${encodeURIComponent(videoId)}&model_name=${encodeURIComponent(MODEL_VIDEO)}`;
    let res;
    try {
      res = await apiFetch(url, { method: 'GET', headers: authHeaders(false) }, 'Polling', 'video');
    } catch (error) {
      if (error.message === 'Arrêt demandé') throw error;
      log('warn', 'POLLING', error.message);
      await deps.sleep(Math.min(90000, 15000 + attempt * 1500));
      continue;
    }
    let data;
    try {
      data = await res.json();
    } catch {
      await deps.sleep(Math.min(60000, 10000 + attempt * 1000));
      continue;
    }
    const status = String(data.status || 'unknown').toLowerCase();
    const progress = Number(data.progress || 0);
    onProgress(`Création — ${progress}%`);
    if (status === 'completed' || status === 'succeeded' || status === 'done') {
      const videoUrl = data.url || (data.metadata && data.metadata.url) || (data.output && data.output.url);
      if (!videoUrl) throw new Error('Terminé mais pas d’URL');
      return videoUrl;
    }
    if (status === 'failed' || status === 'error' || status === 'cancelled') {
      throw new Error(`Échec définitif (${status})`);
    }
    const wait = attempt < 5 ? 8 : Math.min(30, 10 + Math.floor(attempt / 5) * 3);
    onProgress(`File vidéo — nouvelle vérification dans ${wait}s`);
    await deps.sleep(wait * 1000);
  }
  throw new Error('Délai maximal de polling dépassé');
}

function stripHost(hostname) {
  return String(hostname || '').toLowerCase().replace(/^\[|\]$/g, '').split('%')[0];
}

function parseIpv4(ip) {
  const parts = String(ip).split('.');
  if (parts.length !== 4) return null;
  const nums = [];
  for (const part of parts) {
    if (!/^\d{1,3}$/.test(part)) return null;
    const value = Number(part);
    if (value > 255) return null;
    nums.push(value);
  }
  return nums;
}

function isBlockedIpv4(ip) {
  const parts = parseIpv4(ip);
  if (!parts) return true;
  const [a, b] = parts;
  if (a === 0 || a === 10 || a === 127) return true;
  if (a === 169 && b === 254) return true;
  if (a === 172 && b >= 16 && b <= 31) return true;
  if (a === 192 && b === 168) return true;
  if (a === 100 && b >= 64 && b <= 127) return true;
  if (a === 192 && b === 0 && (parts[2] === 0 || parts[2] === 2)) return true;
  if (a === 198 && (b === 18 || b === 19)) return true;
  if (a >= 224) return true;
  return false;
}

function expandIpv6(ip) {
  const value = stripHost(ip);
  if (!value || value.includes('.')) return null;
  const halves = value.split('::');
  if (halves.length > 2) return null;
  const pad = (group) => {
    if (!/^[0-9a-f]{1,4}$/.test(group)) throw new Error('groupe ipv6');
    return group.padStart(4, '0');
  };
  if (halves.length === 1) {
    const groups = halves[0].split(':');
    if (groups.length !== 8) return null;
    return groups.map(pad).join(':');
  }
  const left = halves[0] ? halves[0].split(':').map(pad) : [];
  const right = halves[1] ? halves[1].split(':').map(pad) : [];
  const missing = 8 - left.length - right.length;
  if (missing < 0) return null;
  return [...left, ...Array(missing).fill('0000'), ...right].join(':');
}

function ipv4FromExpanded(full) {
  const parts = full.split(':');
  if (parts.length !== 8) return null;
  if (!parts.slice(0, 5).every((part) => part === '0000') || parts[5] !== 'ffff') return null;
  const hi = Number.parseInt(parts[6], 16);
  const lo = Number.parseInt(parts[7], 16);
  const n = ((hi << 16) + lo) >>> 0;
  return `${(n >>> 24) & 255}.${(n >>> 16) & 255}.${(n >>> 8) & 255}.${n & 255}`;
}

function isBlockedIpv6(full) {
  const parts = full.split(':');
  if (parts.length !== 8) return true;
  if (parts.every((part) => part === '0000')) return true;
  if (parts.slice(0, 7).every((part) => part === '0000') && parts[7] === '0001') return true;
  const mapped = ipv4FromExpanded(full);
  if (mapped) return isBlockedIpv4(mapped);
  const first = Number.parseInt(parts[0], 16);
  if ((first & 0xfe00) === 0xfc00) return true;
  if ((first & 0xffc0) === 0xfe80) return true;
  if ((first & 0xffc0) === 0xfec0) return true;
  if ((first & 0xff00) === 0xff00) return true;
  return false;
}

function isBlockedIp(raw) {
  const ip = stripHost(raw);
  if (!ip) return true;
  const mappedEarly = ip.match(/^::ffff:(\d{1,3}(?:\.\d{1,3}){3})$/);
  if (mappedEarly) return isBlockedIpv4(mappedEarly[1]);
  const mappedHex = ip.match(/^::ffff:([0-9a-f]{1,4}):([0-9a-f]{1,4})$/);
  if (mappedHex) {
    const n = (Number.parseInt(mappedHex[1], 16) << 16) + Number.parseInt(mappedHex[2], 16);
    return isBlockedIpv4(`${(n >>> 24) & 255}.${(n >>> 16) & 255}.${(n >>> 8) & 255}.${n & 255}`);
  }
  if (parseIpv4(ip)) return isBlockedIpv4(ip);
  if (ip.includes(':')) {
    try {
      const full = expandIpv6(ip);
      if (!full) return true;
      return isBlockedIpv6(full);
    } catch {
      return true;
    }
  }
  return true;
}

function hostnameBlocked(hostname) {
  const host = stripHost(hostname);
  if (!host) return true;
  if (BLOCKED_HOSTS.has(host)) return true;
  if (host.endsWith('.localhost') || host.endsWith('.local') || host.endsWith('.internal')) return true;
  return false;
}

function isIpLiteral(hostname) {
  const host = stripHost(hostname);
  if (parseIpv4(host)) return true;
  if (host.includes(':')) return true;
  return false;
}

async function assertSafeVideoUrl(raw) {
  let url;
  try {
    url = new URL(String(raw));
  } catch {
    throw new Error('URL vidéo invalide');
  }
  if (url.protocol !== 'https:') throw new Error('URL vidéo : https requis');
  if (url.username || url.password) throw new Error('URL vidéo : identifiants interdits');
  if (hostnameBlocked(url.hostname)) throw new Error('Hôte vidéo interdit');
  if (isIpLiteral(url.hostname)) {
    if (isBlockedIp(url.hostname)) throw new Error('Adresse vidéo interdite');
    return url;
  }
  if (/^\d+$/.test(url.hostname)) throw new Error('Adresse vidéo interdite');
  let records;
  try {
    records = await deps.lookup(url.hostname, { all: true, verbatim: true });
  } catch {
    throw new Error('Résolution DNS impossible');
  }
  const list = Array.isArray(records) ? records : [records];
  if (list.length === 0) throw new Error('Résolution DNS impossible');
  for (const record of list) {
    const address = typeof record === 'string' ? record : record.address;
    if (isBlockedIp(address)) throw new Error('Adresse vidéo interdite');
  }
  return url;
}

function hasVideoMagic(buf) {
  if (!Buffer.isBuffer(buf) || buf.length < 4) return false;
  if (buf[0] === 0x1a && buf[1] === 0x45 && buf[2] === 0xdf && buf[3] === 0xa3) return true;
  return buf.length >= 8 && buf.subarray(4, 8).toString('ascii') === 'ftyp';
}

function assertVideoContentType(header) {
  if (!header) return;
  const value = String(header).split(';')[0].trim().toLowerCase();
  if (value.startsWith('video/')) return;
  if (value === 'application/octet-stream' || value === 'binary/octet-stream' || value === 'application/mp4') return;
  throw new Error('Type de contenu vidéo refusé');
}

function releaseBody(res) {
  const body = res && res.body;
  if (body && typeof body.destroy === 'function') body.destroy();
}

function videoWriteFlags() {
  const flags = fs.constants.O_WRONLY | fs.constants.O_CREAT | fs.constants.O_EXCL;
  if (typeof fs.constants.O_NOFOLLOW !== 'number') {
    throw new Error('Ouverture sans suivi de lien indisponible');
  }
  return flags | fs.constants.O_NOFOLLOW;
}

async function writeBody(res, destPath) {
  await fs.promises.mkdir(path.dirname(destPath), { recursive: true });
  let existing = null;
  try {
    existing = await fs.promises.lstat(destPath);
  } catch (error) {
    if (!error || error.code !== 'ENOENT') throw error;
  }
  if (existing) {
    throw new Error(existing.isSymbolicLink() ? 'Lien symbolique refusé' : 'Fichier vidéo déjà présent');
  }
  const magic = { buf: Buffer.alloc(0), ok: false };
  let total = 0;
  const limiter = new Transform({
    transform(chunk, _encoding, callback) {
      const buf = Buffer.isBuffer(chunk) ? chunk : Buffer.from(chunk);
      total += buf.length;
      if (total >= deps.maxVideoBytes) {
        callback(new Error('Vidéo trop volumineuse'));
        return;
      }
      if (!magic.ok) {
        magic.buf = Buffer.concat([magic.buf, buf]).subarray(0, 32);
        if (magic.buf.length >= 12) {
          if (!hasVideoMagic(magic.buf)) {
            callback(new Error('En-tête vidéo refusé'));
            return;
          }
          magic.ok = true;
        }
      }
      callback(null, buf);
    },
    flush(callback) {
      if (!magic.ok && !hasVideoMagic(magic.buf)) {
        callback(new Error('En-tête vidéo refusé'));
        return;
      }
      callback();
    },
  });
  let handle;
  try {
    handle = await fs.promises.open(destPath, videoWriteFlags(), 0o644);
  } catch (error) {
    if (error && error.code === 'ELOOP') throw new Error('Lien symbolique refusé');
    if (error && error.code === 'EEXIST') throw new Error('Fichier vidéo déjà présent');
    throw error;
  }
  try {
    await pipeline(res.body, limiter, handle.createWriteStream());
  } catch (error) {
    await fs.promises.unlink(destPath).catch(() => {});
    throw error;
  }
}

async function assertLocalVideo(filePath) {
  const stat = await fs.promises.lstat(filePath);
  if (stat.isSymbolicLink()) throw new Error('Lien symbolique refusé');
  if (!stat.isFile() || stat.size <= 0 || stat.size >= deps.maxVideoBytes) {
    throw new Error('Taille de vidéo refusée');
  }
  const handle = await fs.promises.open(filePath, 'r');
  try {
    const buf = Buffer.alloc(32);
    await handle.read(buf, 0, 32, 0);
    if (!hasVideoMagic(buf)) throw new Error('En-tête vidéo refusé');
  } finally {
    await handle.close();
  }
}

async function downloadVideo(url, destPath) {
  let current = String(url);
  for (let hop = 0; hop <= MAX_REDIRECTS; hop += 1) {
    if (state.stopRequested) throw new Error('Arrêt demandé');
    await assertSafeVideoUrl(current);
    const res = await deps.fetch(current, {
      method: 'GET',
      redirect: 'manual',
      headers: { Accept: 'video/mp4,video/webm,application/octet-stream' },
    });
    if (REDIRECT_STATUSES.has(res.status)) {
      const location = res.headers.get('location');
      releaseBody(res);
      if (!location) throw new Error('Redirection sans cible');
      if (hop === MAX_REDIRECTS) throw new Error('Trop de redirections');
      current = new URL(location, current).href;
      continue;
    }
    if (!res.ok) {
      releaseBody(res);
      throw new Error(`Téléchargement HTTP ${res.status}`);
    }
    assertVideoContentType(res.headers.get('content-type'));
    const declared = Number(res.headers.get('content-length') || '0');
    if (Number.isFinite(declared) && declared >= deps.maxVideoBytes) {
      releaseBody(res);
      throw new Error('Vidéo trop volumineuse');
    }
    await writeBody(res, destPath);
    try {
      await assertLocalVideo(destPath);
    } catch (error) {
      await fs.promises.unlink(destPath).catch(() => {});
      throw error;
    }
    return destPath;
  }
  throw new Error('Trop de redirections');
}

function sanitizeTitle(title) {
  const slug = String(title || 'ambiance_lofi')
    .normalize('NFKD')
    .replace(/[\u0300-\u036f]/g, '')
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, '_')
    .replace(/^_+|_+$/g, '')
    .slice(0, 80);
  return slug || 'ambiance_lofi';
}

function sanitizeCommitTitle(title) {
  const clean = String(title || 'Ambiance Lofi')
    .replace(/[\r\n\0]/g, ' ')
    .replace(/\s+/g, ' ')
    .trim()
    .replace(/^[-–—]+/, '')
    .slice(0, 120);
  return clean || 'Ambiance Lofi';
}

function allocateVideoPath(title, now = Date.now()) {
  const base = sanitizeTitle(title);
  let name = `${base}_${now}.mp4`;
  let full = path.join(VIDEOS_DIR, name);
  let extra = 2;
  while (fs.existsSync(full)) {
    name = `${base}_${now}_${extra}.mp4`;
    full = path.join(VIDEOS_DIR, name);
    extra += 1;
    if (extra > 100) throw new Error('Nom de fichier indisponible');
  }
  if (!VIDEO_NAME_RE.test(path.basename(full))) throw new Error('Nom de fichier refusé');
  return full;
}

function oneLine(value, fallback) {
  const clean = String(value || '').replace(/[\r\n\0]/g, '').trim();
  return clean || fallback;
}

function buildGitProcessEnv(options = {}) {
  const authorName = oneLine(options.authorName || process.env.GIT_AUTHOR_NAME, 'Lofi Capybara Studio');
  const authorEmail = oneLine(options.authorEmail || process.env.GIT_AUTHOR_EMAIL, 'lofi-capybara@users.noreply.github.com');
  const env = {
    PATH: process.env.PATH || '/usr/bin:/bin',
    HOME: '',
    GIT_CONFIG_NOSYSTEM: '1',
    GIT_AUTHOR_NAME: authorName,
    GIT_AUTHOR_EMAIL: authorEmail,
    GIT_COMMITTER_NAME: oneLine(options.committerName || process.env.GIT_COMMITTER_NAME, authorName),
    GIT_COMMITTER_EMAIL: oneLine(options.committerEmail || process.env.GIT_COMMITTER_EMAIL, authorEmail),
    GIT_TERMINAL_PROMPT: '0',
  };
  if (options.askpassPath) env.GIT_ASKPASS = options.askpassPath;
  return env;
}

function createGit(options = {}) {
  const git = simpleGit({
    baseDir: options.baseDir || ROOT,
    binary: options.binary || 'git',
    allowEnvironment: GIT_ENV_ALLOW,
  });
  git.env(buildGitProcessEnv(options));
  return git;
}

function repoSlug(raw) {
  const text = String(raw || '').trim();
  const scp = text.match(/^git@github\.com:([^/\s]+)\/([^/\s]+?)(?:\.git)?$/i);
  if (scp) return `${scp[1]}/${scp[2]}`.toLowerCase();
  try {
    const url = new URL(text);
    if (url.protocol !== 'https:' || url.hostname !== 'github.com' || url.port) return null;
    const parts = url.pathname.replace(/\.git$/, '').replace(/^\/|\/$/g, '').split('/');
    if (parts.length !== 2 || !parts[0] || !parts[1]) return null;
    return `${parts[0]}/${parts[1]}`.toLowerCase();
  } catch {
    return null;
  }
}

function assertAllowedRemote(raw) {
  const text = String(raw || '').trim();
  let url;
  try {
    url = new URL(text);
  } catch {
    throw new Error('URL du dépôt invalide');
  }
  if (url.username || url.password) throw new Error('URL du dépôt : identifiants interdits');
  if (url.search || url.hash) throw new Error('URL du dépôt inattendue');
  if (repoSlug(text) !== ALLOWED_SLUG) throw new Error('Dépôt non autorisé');
  return CANONICAL_REMOTE;
}

async function ensureOrigin(git, cleanUrl) {
  let current = '';
  try {
    current = String(await git.raw(['remote', 'get-url', 'origin'])).trim();
  } catch {
    current = '';
  }
  if (!current) {
    await git.raw(['remote', 'add', 'origin', cleanUrl]);
    return;
  }
  if (repoSlug(current) !== ALLOWED_SLUG) {
    throw new Error('Le remote origin ne pointe pas vers videowan2g');
  }
  if (current !== cleanUrl) await git.raw(['remote', 'set-url', 'origin', cleanUrl]);
}

function writeAskpass(token) {
  const clean = String(token || '');
  if (!clean || /[\r\n\0]/.test(clean) || clean.length > 2048) {
    throw new Error('GITHUB_TOKEN invalide');
  }
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'lofi-askpass-'));
  const file = path.join(dir, 'askpass.cjs');
  const source = `#!/usr/bin/env node
const prompt = String(process.argv[2] || '');
if (/username/i.test(prompt)) process.stdout.write('x-access-token\\n');
else process.stdout.write(${JSON.stringify(clean)} + '\\n');
`;
  fs.writeFileSync(file, source, { mode: 0o700 });
  return { dir, file };
}

function videoPathForGit(filePath) {
  const root = path.resolve(VIDEOS_DIR);
  const resolved = path.resolve(filePath);
  if (path.dirname(resolved) !== root) throw new Error('Chemin vidéo hors dossier');
  if (!VIDEO_NAME_RE.test(path.basename(resolved))) throw new Error('Nom de fichier refusé');
  const relative = path.relative(ROOT, resolved);
  if (relative.startsWith('..') || path.isAbsolute(relative)) throw new Error('Chemin vidéo refusé');
  return relative;
}

async function stageAndCommit(git, relative, title) {
  const message = `feat: add new lofi capybara video: ${sanitizeCommitTitle(title)}`;
  await git.raw(['add', '--', relative]);
  await git.raw(['-c', 'commit.gpgsign=false', 'commit', '-m', message, '--', relative]);
}

async function publishVideo(filePath, title) {
  await assertLocalVideo(filePath);
  const relative = videoPathForGit(filePath);
  const remote = assertAllowedRemote(process.env.GIT_REMOTE_URL || '');
  const token = String(process.env.GITHUB_TOKEN || '').trim();
  if (!token) throw new Error('GITHUB_TOKEN manquant');
  const askpass = writeAskpass(token);
  try {
    const git = createGit({ baseDir: ROOT, askpassPath: askpass.file });
    await ensureOrigin(git, remote);
    await stageAndCommit(git, relative, title);
    await git.raw(['-c', 'credential.helper=', 'push', 'origin', 'HEAD']);
  } finally {
    fs.rmSync(askpass.dir, { recursive: true, force: true });
  }
}

async function runScene(scene, config, hooks = {}) {
  const description = String(scene.description || '').trim();
  if (description.length < 4) throw new Error('Description trop courte');
  if (scene.mood && !MOODS[scene.mood]) throw new Error(`Ambiance inconnue : ${scene.mood}`);
  state.mood = scene.mood || pick(Object.keys(MOODS));
  state.durationSeconds = scene.durationSeconds ? parseDuration(scene.durationSeconds) : config.durationSeconds;
  state.format = scene.format ? parseFormat(scene.format) : config.format;
  const variation = pickVariation();
  log('info', 'SCENE', `${description} · ${MOODS[state.mood].label} · ${variation.timeOfDay}, ${variation.season}, ${variation.weather}`);
  const expansion = await generateExpansion(description, state.mood, variation);
  const imageUrl = await imageQueue.add(() => generateImage(buildLofiImagePrompt(expansion), state.format));
  log('info', 'IMAGE', 'Image keyframe prête');
  const started = Date.now();
  const videoId = await videoQueue.add(() => generateVideo(imageUrl, buildLofiVideoPrompt(expansion)));
  const videoUrl = await pollVideo(videoId, (progress) => log('info', 'VIDEO', progress));
  recordGenerationTime('video', Date.now() - started);
  await fs.promises.mkdir(VIDEOS_DIR, { recursive: true });
  const dest = hooks.destPath || allocateVideoPath(expansion.title);
  await downloadVideo(videoUrl, dest);
  if (hooks.publish) await hooks.publish(dest, expansion.title);
  else await publishVideo(dest, expansion.title);
  log('info', 'DONE', `${expansion.title} → ${dest}`);
  return { title: expansion.title, filePath: dest, variation };
}

function requireRuntimeConfig() {
  const apiKey = getApiKey();
  if (!apiKey) throw new Error('AGNES_API_KEY manquante dans .env');
  if (!apiKey.startsWith('sk-')) log('warn', 'API', 'La clé Agnes ne commence pas par sk-');
  if (!String(process.env.GITHUB_TOKEN || '').trim()) throw new Error('GITHUB_TOKEN manquant dans .env');
  const remote = assertAllowedRemote(process.env.GIT_REMOTE_URL || '');
  return {
    remote,
    durationSeconds: parseDuration(process.env.VIDEO_SECONDS || '5'),
    format: parseFormat(process.env.VIDEO_FORMAT || '16:9'),
  };
}

async function main() {
  const config = requireRuntimeConfig();
  const scenes = loadScenes();
  if (scenes.length === 0) {
    log('warn', 'SCENES', 'Aucune scène dans scenes.json');
    return;
  }
  let failed = 0;
  for (const scene of scenes) {
    if (state.stopRequested) break;
    try {
      await runScene(scene, config);
    } catch (error) {
      failed += 1;
      log('error', 'SCENE', error.message);
      if (error.message === 'Arrêt demandé') break;
    }
  }
  if (failed > 0) process.exitCode = 1;
}

function mockResponse(status, body, headers = {}) {
  const buf = Buffer.isBuffer(body) ? body : Buffer.from(String(body ?? ''));
  const headerMap = new Map(Object.entries(headers).map(([key, value]) => [key.toLowerCase(), String(value)]));
  function create() {
    let consumed = false;
    return {
      ok: status >= 200 && status < 300,
      status,
      headers: { get: (name) => headerMap.get(String(name).toLowerCase()) ?? null },
      body: ReadableFrom(buf),
      clone() {
        return create();
      },
      text() {
        if (consumed) return Promise.reject(new Error('body used'));
        consumed = true;
        return Promise.resolve(buf.toString('utf8'));
      },
      json() {
        return this.text().then((text) => JSON.parse(text));
      },
    };
  }
  return create();
}

function ReadableFrom(buf) {
  return Readable.from([buf]);
}

function ftypBuffer() {
  return Buffer.concat([
    Buffer.from([0x00, 0x00, 0x00, 0x18]),
    Buffer.from('ftypisom'),
    Buffer.from('00000000', 'hex'),
  ]);
}

function assert(condition, message) {
  if (!condition) throw new Error(message);
}

async function testQueueWaits() {
  const fixed = () => 0.5;
  assert(computeQueueWaitSeconds(0, 0, fixed) === 60, 'attente 60');
  assert(computeQueueWaitSeconds(1, 0, fixed) === 120, 'attente 120');
  assert(computeQueueWaitSeconds(2, 0, fixed) === 300, 'attente 300');
  assert(computeQueueWaitSeconds(8, 0, fixed) === 300, 'plafond de palier');
  assert(computeQueueWaitSeconds(0, 15, fixed) === 15, 'Retry-After respecté');
  assert(computeQueueWaitSeconds(0, 9999, fixed) === 360, 'plafond 360');
  const low = computeQueueWaitSeconds(0, 0, () => 0);
  const high = computeQueueWaitSeconds(2, 0, () => 1);
  assert(low >= 51 && low <= 69, 'jitter 60');
  assert(high <= 360 && high >= 255, 'jitter 300');
}

async function testBlockedAddresses() {
  const blocked = [
    'https://127.0.0.1/v.mp4',
    'https://2130706433/v.mp4',
    'https://0x7f000001/v.mp4',
    'https://0177.0.0.1/v.mp4',
    'https://10.1.2.3/v.mp4',
    'https://192.168.1.8/v.mp4',
    'https://169.254.169.254/latest/meta-data',
    'https://[::1]/v.mp4',
    'https://[::ffff:7f00:1]/v.mp4',
    'https://[::ffff:127.0.0.1]/v.mp4',
    'http://cdn.example.com/v.mp4',
    'https://user:pass@cdn.example.com/v.mp4',
    'https://localhost/v.mp4',
    'https://metadata.google.internal/v.mp4',
  ];
  for (const url of blocked) {
    let threw = false;
    try {
      await assertSafeVideoUrl(url);
    } catch {
      threw = true;
    }
    assert(threw, `devrait refuser ${url}`);
  }
  deps.lookup = async () => [{ address: '127.0.0.1', family: 4 }];
  let dnsBlocked = false;
  try {
    await assertSafeVideoUrl('https://cdn.example.com/v.mp4');
  } catch {
    dnsBlocked = true;
  }
  assert(dnsBlocked, 'DNS privé refusé');
  deps.lookup = async () => [{ address: '1.1.1.1', family: 4 }];
  await assertSafeVideoUrl('https://cdn.example.com/v.mp4');
  await assertSafeVideoUrl('https://8.8.8.8/v.mp4');
  await assertSafeVideoUrl('https://[2606:4700:4700::1111]/v.mp4');
  assert(assertAllowedRemote('https://github.com/urdirditfurd/videowan2g') === CANONICAL_REMOTE, 'remote canonique');
  assert(assertAllowedRemote('https://github.com/urdirditfurd/videowan2g.git') === CANONICAL_REMOTE, 'remote .git');
  let remoteRejected = false;
  try {
    assertAllowedRemote('https://x-access-token:secret@github.com/urdirditfurd/videowan2g.git');
  } catch {
    remoteRejected = true;
  }
  assert(remoteRejected, 'identifiants dans l’URL refusés');
  assert(repoSlug('git@github.com:urdirditfurd/videowan2g.git') === ALLOWED_SLUG, 'slug ssh');
  assert(repoSlug('https://github.com/autre/depot.git') !== ALLOWED_SLUG, 'autre dépôt');
}

async function testJsonAndPrompt() {
  const fenced = '```json\n{"title":"Brume","image_prompt":"house","video_motion_prompt":"slow"}\n```';
  const parsed = extractJson(fenced);
  assert(parsed.title === 'Brume' && parsed.image_prompt === 'house', 'json balisé');
  const broken = '{"title":"Bonjour","image_prompt":"ligne1\nligne2","video_motion_prompt":"lent"}';
  const repaired = extractJson(broken);
  assert(repaired && repaired.image_prompt.includes('ligne1'), 'json réparé');
  const salvaged = extractJson('préambule {"image_prompt":"capybara on a porch","video_motion_prompt":"breathing"');
  assert(salvaged && salvaged.image_prompt.includes('capybara'), 'salvage');
  const variation = { timeOfDay: 'heure bleue', season: 'automne avancé', weather: 'bruine fine' };
  const prompt = buildExpansionPrompt('maison japonaise sous la pluie', 'rainy_forest', variation);
  assert(prompt.includes('heure bleue') && prompt.includes('automne avancé') && prompt.includes('bruine fine'), 'variation');
  assert(prompt.includes('capybara') && prompt.includes('maison japonaise sous la pluie'), 'scène et mascotte');
  assert(sanitizeTitle('../../Maison japonaise !') === 'maison_japonaise', 'nom sûr');
  assert(hasVideoMagic(ftypBuffer()), 'ftyp');
  assert(hasVideoMagic(Buffer.from([0x1a, 0x45, 0xdf, 0xa3, 0x00])), 'webm');
  assert(!hasVideoMagic(Buffer.from('<html>')), 'html refusé');
}

async function testApiFetchQueue() {
  const sleeps = [];
  deps.rng = () => 0.5;
  deps.sleep = async (ms) => {
    sleeps.push(ms);
  };
  let calls = 0;
  deps.fetch = async () => {
    calls += 1;
    if (calls <= 11) return mockResponse(503, '{"error":"video_queue_full"}');
    return mockResponse(200, '{"ok":true}');
  };
  const res = await apiFetch('https://apihub.agnes-ai.com/v1/videos', { method: 'POST' }, 'Vidéo', 'video');
  assert(res.ok, 'la file finit par accepter');
  assert(calls === 12, 'douze appels malgré un budget de 10');
  assert(sleeps.length === 11, 'onze attentes de file');
  assert(sleeps[0] === 60000 && sleeps[1] === 120000 && sleeps[2] === 300000 && sleeps[10] === 300000, 'paliers 60/120/300');

  sleeps.length = 0;
  calls = 0;
  deps.fetch = async () => {
    calls += 1;
    return mockResponse(429, 'queue is full');
  };
  let message = '';
  try {
    await apiFetch('https://apihub.agnes-ai.com/v1/videos', { method: 'POST' }, 'Vidéo', 'video');
  } catch (error) {
    message = error.message;
  }
  assert(message.includes('saturée'), message);
  assert(calls === VIDEO_QUEUE_MAX_RETRIES + 1, 'budget de file indépendant');
  assert(sleeps.length === VIDEO_QUEUE_MAX_RETRIES, 'une attente par reprise de file');
}

async function testTerminalHttpAndClone() {
  let calls = 0;
  deps.fetch = async () => {
    calls += 1;
    return mockResponse(401, 'cle-refusee-detail');
  };
  let message = '';
  try {
    await apiFetch('https://apihub.agnes-ai.com/v1/chat/completions', { method: 'POST' }, 'Expansion', 'text');
  } catch (error) {
    message = error.message;
  }
  assert(calls === 1, '401 texte non rejoué');
  assert(message.includes('401') && message.includes('cle-refusee-detail'), message);

  deps.fetch = async () => mockResponse(400, 'prompt-trop-long');
  const res = await apiFetch('https://apihub.agnes-ai.com/v1/videos', { method: 'POST' }, 'Vidéo', 'video');
  assert((await res.text()) === 'prompt-trop-long', 'corps vidéo encore lisible');
}

async function testDownload() {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'videowan2g-dl-'));
  const dest = path.join(dir, 'clip.mp4');
  const urls = [
    'https://cdn.example.com/start.mp4',
    'https://cdn.example.com/mid.mp4',
    'https://cdn.example.com/final.mp4',
    'https://cdn.example.com/forbidden.mp4',
  ];
  deps.lookup = async () => [{ address: '1.1.1.1', family: 4 }];
  const seen = [];
  deps.fetch = async (url) => {
    seen.push(String(url));
    if (url === urls[0]) return mockResponse(302, '', { location: urls[1] });
    if (url === urls[1]) return mockResponse(302, '', { location: urls[2] });
    if (url === urls[2]) return mockResponse(200, ftypBuffer(), { 'content-type': 'video/mp4' });
    throw new Error(`fetch inattendu ${url}`);
  };
  await downloadVideo(urls[0], dest);
  assert(seen.length === 3 && !seen.includes(urls[3]), 'deux redirections');
  await assertLocalVideo(dest);

  seen.length = 0;
  deps.fetch = async (url) => {
    seen.push(String(url));
    const next = [urls[1], urls[2], urls[3]][seen.length - 1];
    return mockResponse(302, '', { location: next });
  };
  let redirected = false;
  try {
    await downloadVideo(urls[0], path.join(dir, 'too-many.mp4'));
  } catch (error) {
    redirected = error.message.includes('Trop de redirections');
  }
  assert(redirected && seen.length === 3 && !seen.includes(urls[3]), 'troisième redirection non suivie');

  seen.length = 0;
  deps.fetch = async (url) => {
    seen.push(String(url));
    return mockResponse(302, '', { location: 'https://127.0.0.1/secret.mp4' });
  };
  let privateHop = false;
  try {
    await downloadVideo(urls[0], path.join(dir, 'private.mp4'));
  } catch (error) {
    privateHop = /interdite|interdit/.test(error.message);
  }
  assert(privateHop && seen.length === 1, 'pas de fetch vers l’adresse privée');

  deps.maxVideoBytes = 16;
  deps.fetch = async () => mockResponse(200, Buffer.alloc(20, 1), { 'content-type': 'application/octet-stream', 'content-length': '20' });
  let tooBig = false;
  try {
    await downloadVideo('https://8.8.8.8/big.mp4', path.join(dir, 'big.mp4'));
  } catch (error) {
    tooBig = error.message.includes('volumineuse');
  }
  assert(tooBig && !fs.existsSync(path.join(dir, 'big.mp4')), 'taille refusée');
  deps.maxVideoBytes = DEFAULT_MAX_VIDEO_BYTES;

  const target = path.join(dir, 'cible.txt');
  const planted = path.join(dir, 'plante.mp4');
  fs.writeFileSync(target, 'intact');
  fs.symlinkSync(target, planted);
  deps.fetch = async () => mockResponse(200, ftypBuffer(), { 'content-type': 'video/mp4' });
  let symlinkWrite = false;
  try {
    await downloadVideo('https://cdn.example.com/final.mp4', planted);
  } catch (error) {
    symlinkWrite = error.message.includes('Lien symbolique');
  }
  assert(symlinkWrite, 'écriture à travers un lien refusée');
  assert(fs.readFileSync(target, 'utf8') === 'intact', 'cible du lien intacte');
  assert(fs.lstatSync(planted).isSymbolicLink(), 'le lien n’a pas été remplacé');

  const link = path.join(dir, 'lien.mp4');
  fs.symlinkSync(dest, link);
  let symlinkRead = false;
  try {
    await assertLocalVideo(link);
  } catch (error) {
    symlinkRead = error.message.includes('Lien symbolique');
  }
  assert(symlinkRead, 'lecture d’un lien refusée');
}

async function testRetryAndQueueSpacing() {
  const sleeps = [];
  deps.sleep = async (ms) => {
    sleeps.push(ms);
  };
  let calls = 0;
  const value = await retryTask('Image', async () => {
    calls += 1;
    if (calls < 3) throw new Error('temporaire');
    return 'ok';
  }, 3);
  assert(value === 'ok' && sleeps[0] === 8000 && sleeps[1] === 32000, 'retryTask quadratique');

  const gaps = [];
  let last = 0;
  const queue = createRateLimitedQueue(1, 40);
  await queue.add(async () => {
    last = Date.now();
  });
  await queue.add(async () => {
    gaps.push(Date.now() - last);
  });
  assert(gaps[0] >= 40, 'intervalle de file');
}

async function testGitIsolation() {
  process.env.AGNES_API_KEY = 'SENTINEL_AGNES_VALUE';
  process.env.GITHUB_TOKEN = 'SENTINEL_GITHUB_VALUE';
  process.env.GIT_AUTHOR_NAME = 'Self Test';
  process.env.GIT_AUTHOR_EMAIL = 'self-test@example.com';
  process.env.GIT_COMMITTER_NAME = 'Self Test';
  process.env.GIT_COMMITTER_EMAIL = 'self-test@example.com';

  const repo = fs.mkdtempSync(path.join(os.tmpdir(), 'videowan2g-repo-'));
  execFileSync('git', ['init', '-q'], { cwd: repo });
  execFileSync('git', ['config', 'user.name', 'Wrong Name'], { cwd: repo });
  execFileSync('git', ['config', 'user.email', 'wrong@example.com'], { cwd: repo });
  execFileSync('git', ['remote', 'add', 'origin', 'https://x-access-token:SENTINEL_GITHUB_VALUE@github.com/urdirditfurd/videowan2g.git'], { cwd: repo });
  fs.writeFileSync(path.join(repo, 'note.txt'), 'bonjour\n');
  const git = createGit({ baseDir: repo });
  await ensureOrigin(git, CANONICAL_REMOTE);
  const origin = execFileSync('git', ['-C', repo, 'remote', 'get-url', 'origin'], {
    encoding: 'utf8',
    env: { PATH: process.env.PATH, HOME: '', GIT_CONFIG_NOSYSTEM: '1' },
  }).trim();
  const config = fs.readFileSync(path.join(repo, '.git', 'config'), 'utf8');
  assert(origin === CANONICAL_REMOTE, 'remote non nettoyé');
  assert(!config.includes('SENTINEL_GITHUB_VALUE') && !config.includes('x-access-token'), 'jeton retiré du remote');
  await git.raw(['add', '--', 'note.txt']);
  await git.raw(['commit', '-m', 'test identity']);
  const author = execFileSync('git', ['-C', repo, 'log', '-1', '--format=%an <%ae>'], { encoding: 'utf8' }).trim();
  assert(author === 'Self Test <self-test@example.com>', author);

  const gitEnv = { PATH: process.env.PATH, HOME: '', GIT_CONFIG_NOSYSTEM: '1' };
  fs.writeFileSync(path.join(repo, 'secret.txt'), 'ne-pas-publier\n');
  fs.writeFileSync(path.join(repo, 'brume_1.mp4'), ftypBuffer());
  await git.raw(['add', '--', 'secret.txt']);
  await stageAndCommit(git, 'brume_1.mp4', 'Brume');
  const committed = execFileSync('git', ['-C', repo, 'show', '--name-only', '--format=', 'HEAD'], {
    encoding: 'utf8',
    env: gitEnv,
  }).trim();
  assert(committed === 'brume_1.mp4', committed);
  const stillStaged = execFileSync('git', ['-C', repo, 'diff', '--cached', '--name-only'], {
    encoding: 'utf8',
    env: gitEnv,
  });
  assert(stillStaged.includes('secret.txt'), 'fichier hors vidéo exclu du commit');
  const subject = execFileSync('git', ['-C', repo, 'log', '-1', '--format=%s'], {
    encoding: 'utf8',
    env: gitEnv,
  }).trim();
  assert(subject === 'feat: add new lofi capybara video: Brume', subject);

  execFileSync('git', ['remote', 'set-url', 'origin', 'https://github.com/autre/depot.git'], { cwd: repo });
  let refused = false;
  const readOrigin = () => execFileSync('git', ['-C', repo, 'remote', 'get-url', 'origin'], {
    encoding: 'utf8',
    env: { PATH: process.env.PATH, HOME: '', GIT_CONFIG_NOSYSTEM: '1' },
  }).trim();
  const before = readOrigin();
  try {
    await ensureOrigin(createGit({ baseDir: repo }), CANONICAL_REMOTE);
  } catch (error) {
    refused = error.message.includes('ne pointe pas');
  }
  const after = readOrigin();
  assert(refused && after === before, 'autre dépôt inchangé');

  const fakeDir = fs.mkdtempSync(path.join(os.tmpdir(), 'videowan2g-fake-'));
  const dumpPath = path.join(fakeDir, 'env.json');
  const fakePath = path.join(fakeDir, 'fake-git.cjs');
  fs.writeFileSync(fakePath, `#!/usr/bin/env node
const fs = require('fs');
fs.writeFileSync(${JSON.stringify(dumpPath)}, JSON.stringify(process.env));
process.exit(0);
`, { mode: 0o755 });
  fs.rmSync(dumpPath, { force: true });
  const fakeRepo = fs.mkdtempSync(path.join(os.tmpdir(), 'videowan2g-fake-repo-'));
  const fakeGit = createGit({ baseDir: fakeRepo, binary: fakePath });
  await fakeGit.raw(['status']);
  const dumped = JSON.parse(fs.readFileSync(dumpPath, 'utf8'));
  const blob = JSON.stringify(dumped);
  assert(!Object.prototype.hasOwnProperty.call(dumped, 'AGNES_API_KEY'), 'nom AGNES absent');
  assert(!Object.prototype.hasOwnProperty.call(dumped, 'GITHUB_TOKEN'), 'nom GITHUB absent');
  assert(!blob.includes('SENTINEL_AGNES_VALUE') && !blob.includes('SENTINEL_GITHUB_VALUE'), 'valeurs absentes');
  assert(dumped.HOME === '', 'HOME vide');
  assert(dumped.GIT_CONFIG_NOSYSTEM === '1', 'config système ignorée');
  assert(dumped.GIT_AUTHOR_NAME === 'Self Test', 'identité transmise');
  assert(!dumped.GIT_ASKPASS, 'pas de askpass dans ce test');
}

async function testPipelineOffline() {
  deps.rng = () => 0;
  deps.sleep = async () => {};
  deps.lookup = async () => [{ address: '1.1.1.1', family: 4 }];
  process.env.AGNES_API_KEY = 'sk-test';
  const expansion = {
    title: 'Pluie douce',
    image_prompt: 'A cedar house in soft rain with a calm capybara on the porch.',
    video_motion_prompt: 'Slow mist and gentle breathing.',
  };
  let prompt = '';
  deps.fetch = async (url, options = {}) => {
    const target = String(url);
    if (target.endsWith('/chat/completions')) {
      prompt = JSON.parse(options.body).messages[1].content;
      return mockResponse(200, JSON.stringify({ choices: [{ message: { content: JSON.stringify(expansion) } }] }));
    }
    if (target.endsWith('/images/generations')) {
      return mockResponse(200, JSON.stringify({ data: [{ url: 'https://cdn.example.com/frame.png' }] }));
    }
    if (target.endsWith('/videos')) {
      const body = JSON.parse(options.body);
      assert(body.mode === 'keyframe' && body.aspect_ratio === '16:9', 'vidéo keyframe 16:9');
      assert(body.first_frame === 'https://cdn.example.com/frame.png', 'image de départ');
      return mockResponse(200, JSON.stringify({ video_id: 'vid_123' }));
    }
    if (target.includes('video_id=vid_123')) {
      return mockResponse(200, JSON.stringify({ status: 'completed', progress: 100, url: 'https://cdn.example.com/out.mp4' }));
    }
    if (target === 'https://cdn.example.com/out.mp4') {
      return mockResponse(200, ftypBuffer(), { 'content-type': 'video/mp4' });
    }
    throw new Error(`URL inattendue ${target}`);
  };
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'videowan2g-out-'));
  const dest = path.join(dir, 'pluie_douce_1.mp4');
  const result = await runScene(
    { description: 'maison japonaise sous la pluie', mood: 'rainy_forest' },
    { durationSeconds: 5, format: '16:9' },
    {
      destPath: dest,
      async publish(filePath) {
        await assertLocalVideo(filePath);
      },
    },
  );
  assert(result.title === 'Pluie douce', result.title);
  assert(prompt.includes(TIMES_OF_DAY[0]) && prompt.includes(SEASONS[0]) && prompt.includes(WEATHERS[0]), 'variation envoyée');
  assert(prompt.includes('capybara'), 'mascotte exigée');
  await assertLocalVideo(dest);
}

async function runSelfTest() {
  const steps = [
    ['attentes de file', testQueueWaits],
    ['adresses et dépôt', testBlockedAddresses],
    ['json et prompt', testJsonAndPrompt],
    ['file saturée', testApiFetchQueue],
    ['erreurs HTTP', testTerminalHttpAndClone],
    ['téléchargement', testDownload],
    ['retry et cadence', testRetryAndQueueSpacing],
    ['git isolé', testGitIsolation],
    ['pipeline hors réseau', testPipelineOffline],
  ];
  for (const [name, fn] of steps) {
    resetRuntime();
    deps.quiet = true;
    await fn();
    console.log(`ok ${name}`);
  }
  console.log('Auto-test terminé');
  return 0;
}

const isSelfTest = process.argv.includes('--self-test');
if (isSelfTest) {
  runSelfTest().then((code) => {
    process.exitCode = code;
  }).catch((error) => {
    console.error(error && error.stack ? error.stack : error);
    process.exitCode = 1;
  });
} else {
  process.on('SIGINT', () => {
    state.stopRequested = true;
    log('warn', 'STOP', 'Arrêt demandé');
  });
  main().catch((error) => {
    log('error', 'GLOBAL', error.message || String(error));
    process.exitCode = 1;
  });
}
