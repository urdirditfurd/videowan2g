#!/usr/bin/env node
/**
 * Pipeline headless Lofi Capybara.
 * Description → expansion Agnes → image 16:9 → vidéo keyframe → mp4 → commit/push.
 *
 *   npm ci
 *   cp .env.example .env
 *   node automation.js --self-test
 *   node automation.js
 */

import { lookup } from 'node:dns/promises';
import { createWriteStream } from 'node:fs';
import fs from 'node:fs/promises';
import http from 'node:http';
import net from 'node:net';
import os from 'node:os';
import path from 'node:path';
import { Readable, Transform } from 'node:stream';
import { pipeline } from 'node:stream/promises';
import { fileURLToPath } from 'node:url';
import dotenv from 'dotenv';
import fetch from 'node-fetch';
import { simpleGit } from 'simple-git';

const ROOT = path.dirname(fileURLToPath(import.meta.url));
dotenv.config({ path: path.join(ROOT, '.env') });

const API_BASE = 'https://apihub.agnes-ai.com/v1';
const POLL_BASE = 'https://apihub.agnes-ai.com/agnesapi';
const MODEL_TEXT = 'agnes-2.5-flash';
const MODEL_IMAGE = 'agnes-image-2.5-flash';
const MODEL_VIDEO = 'agnes-video-2.5-flash';

const ALLOWED_REPO = 'https://github.com/urdirditfurd/videowan2g.git';
const VIDEO_DIR = path.join(ROOT, 'generated_videos');
const CACHE_DIR = path.join(ROOT, '.cache');
const TIMING_PATH = path.join(CACHE_DIR, 'lofi-timing.json');
const EMPTY_HOME = path.join(CACHE_DIR, 'empty-git-home');

const MAX_VIDEO_BYTES = 100 * 1024 * 1024;
const MAX_REDIRECTS = 2;
const VIDEO_QUEUE_BACKOFF_SEC = [60, 120, 300];
const VIDEO_QUEUE_MAX_WAIT_SEC = 360;
const DEFAULT_QUEUE_RETRIES = 40;

const IMAGE_INTERVAL_MIN = 3200;
const IMAGE_INTERVAL_MAX = 12000;
const VIDEO_INTERVAL_MIN = 62000;
const VIDEO_INTERVAL_MAX = 90000;
const VIDEO_INTERVAL_SAFE = 75000;
const VIDEO_FIRST_POLL_DELAY_SEC = 80;
const VIDEO_MAX_POLL_ATTEMPTS = 100;

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
  ['aube pâle', 'pale dawn with a cool lavender sky'],
  ['matin brumeux', 'misty morning, low soft sun'],
  ['fin de matinée claire', 'late clear morning, gentle brightness'],
  ['après-midi doré', 'golden afternoon'],
  ['heure bleue', 'blue hour just after sunset'],
  ['crépuscule', 'dusk with warm windows lighting up'],
  ['nuit calme', 'calm night with a visible moon'],
  ['nuit profonde étoilée', 'deep starry night'],
];

const SEASONS = [
  ['début du printemps', 'early spring, young leaves and blossoms'],
  ['plein été', 'full summer, lush greens and warm air'],
  ['automne avancé', 'late autumn, amber and rust foliage'],
  ['hiver doux', 'mild winter, frost or light snow, still cozy'],
];

const WEATHERS = [
  ['brume légère', 'light mist'],
  ['pluie fine', 'fine rain'],
  ['ciel dégagé', 'clear calm sky'],
  ['nuages lents', 'slow drifting clouds'],
  ['après une averse', 'just after a shower, wet surfaces'],
  ['neige légère', 'light snowflakes'],
  ['vent très doux', 'a very soft breeze'],
  ['orage lointain', 'a distant storm, the scene itself remaining peaceful'],
];

const SENSORY_DETAILS = [
  ['vapeur d’un thé', 'steam rising from a teacup'],
  ['lanterne en papier', 'a glowing paper lantern'],
  ['lucioles', 'a few slow fireflies'],
  ['carillons en bois', 'wooden wind chimes barely moving'],
  ['pile de livres', 'a small stack of old books'],
  ['linge qui sèche', 'laundry drying on a line'],
  ['barque en bois', 'a still wooden rowboat'],
  ['lanterne de pierre', 'a mossy stone lantern'],
];

const AMBIENT_ENV_ALLOWLIST = ['PATH', 'PATHEXT', 'SYSTEMROOT', 'LANG', 'LC_ALL', 'LC_CTYPE', 'TZ', 'TMPDIR', 'TMP', 'TEMP'];
const GIT_AUTHOR_ENV_KEYS = [
  'GIT_AUTHOR_NAME',
  'GIT_AUTHOR_EMAIL',
  'GIT_COMMITTER_NAME',
  'GIT_COMMITTER_EMAIL',
  'GIT_CONFIG_NOSYSTEM',
];

const pacing = {
  createImageIntervalMs: 4000,
  createVideoIntervalMs: VIDEO_INTERVAL_SAFE,
  imageAttemptsSinceLastError: 0,
  videoAttemptsSinceLastError: 0,
};

const state = {
  stopRequested: false,
  durationSeconds: 5,
  format: '16:9',
};

class QueueSaturatedError extends Error {
  constructor() {
    super('File vidéo saturée : nombre maximal d’attentes atteint');
    this.name = 'QueueSaturatedError';
  }
}

function log(level, category, message, data) {
  const time = new Date().toISOString().slice(11, 19);
  const line = `${time} [${level}] [${category}] ${redact(message)}`;
  if (level === 'error') console.error(line);
  else console.log(line);
  if (data !== undefined && data !== null) {
    let rendered;
    if (typeof data === 'string') rendered = data;
    else {
      try {
        rendered = JSON.stringify(data, null, 2);
      } catch {
        rendered = String(data);
      }
    }
    const clipped = rendered.length > 2000 ? `${rendered.slice(0, 2000)}\n…[tronqué]` : rendered;
    console.log(redact(clipped));
  }
}

function redact(value, extraSecret) {
  let text = String(value ?? '');
  const secrets = [process.env.AGNES_API_KEY, process.env.GITHUB_TOKEN, extraSecret];
  for (const secret of secrets) {
    if (secret && secret.length > 4) text = text.split(secret).join('[REDACTED]');
  }
  return text;
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

function pick(list, random = Math.random) {
  const index = Math.min(list.length - 1, Math.floor(random() * list.length));
  return list[index];
}

function queueRetryLimit() {
  const raw = process.env.VIDEO_QUEUE_MAX_RETRIES;
  if (raw === undefined || raw === '') return DEFAULT_QUEUE_RETRIES;
  const parsed = Number(raw);
  if (!Number.isInteger(parsed) || parsed < 1 || parsed > 500) return DEFAULT_QUEUE_RETRIES;
  return parsed;
}

function queueFullDelaySeconds(attempt, random = Math.random) {
  const index = Math.min(Math.max(0, attempt), VIDEO_QUEUE_BACKOFF_SEC.length - 1);
  const base = VIDEO_QUEUE_BACKOFF_SEC[index];
  const jittered = base * (0.85 + random() * 0.3);
  return Math.min(VIDEO_QUEUE_MAX_WAIT_SEC, Math.max(1, Math.round(jittered)));
}

function limitsFor(kind) {
  switch (kind) {
    case 'text':
      return { maxAttempts: 8, rateCap: 120, rateBase: 10, rateStep: 12, serverCap: 60, serverBase: 5, serverStep: 5, netCap: 45, netBase: 3, netStep: 4 };
    case 'image':
      return { maxAttempts: 10, rateCap: 120, rateBase: 10, rateStep: 12, serverCap: 60, serverBase: 5, serverStep: 5, netCap: 45, netBase: 3, netStep: 4 };
    case 'video':
      return { maxAttempts: 10, rateCap: 180, rateBase: 60, rateStep: 8, serverCap: 180, serverBase: 20, serverStep: 10, netCap: 120, netBase: 10, netStep: 8 };
    default: {
      const unexpected = kind;
      throw new Error(`Type d’appel inconnu : ${String(unexpected)}`);
    }
  }
}

function planApiRetry({ kind, status, errorText, retryAfter = 0, attempt, queueRetries, random = Math.random }) {
  const queueFull = kind === 'video' && /video_queue_full|queue is full/i.test(errorText || '');
  if (queueFull) {
    if (queueRetries >= queueRetryLimit()) return { action: 'fail-queue', countsAsAttempt: false };
    const scheduled = queueFullDelaySeconds(queueRetries, random);
    const hinted = retryAfter > 0 ? Math.min(VIDEO_QUEUE_MAX_WAIT_SEC, retryAfter) : 0;
    return {
      action: 'queue-wait',
      countsAsAttempt: false,
      waitSeconds: Math.min(VIDEO_QUEUE_MAX_WAIT_SEC, Math.max(scheduled, hinted)),
    };
  }
  const limits = limitsFor(kind);
  if ([400, 401, 404].includes(status) && kind !== 'video') {
    return { action: 'fail-client', countsAsAttempt: false };
  }
  if (status === 429) {
    const wait = retryAfter > 0 ? retryAfter : Math.min(limits.rateCap, limits.rateBase + attempt * limits.rateStep);
    return { action: 'wait', countsAsAttempt: true, waitSeconds: Math.min(limits.rateCap, wait), growInterval: true };
  }
  if ([500, 502, 503, 504, 520].includes(status)) {
    return {
      action: 'wait',
      countsAsAttempt: true,
      waitSeconds: Math.min(limits.serverCap, limits.serverBase + attempt * limits.serverStep),
    };
  }
  if (status === 0) {
    return {
      action: 'wait',
      countsAsAttempt: true,
      waitSeconds: Math.min(limits.netCap, limits.netBase + attempt * limits.netStep),
    };
  }
  return { action: 'return', countsAsAttempt: false };
}

function noteSuccess(kind) {
  if (kind === 'image') {
    pacing.imageAttemptsSinceLastError += 1;
    if (pacing.imageAttemptsSinceLastError >= 4 && pacing.createImageIntervalMs > IMAGE_INTERVAL_MIN) {
      pacing.createImageIntervalMs = Math.max(IMAGE_INTERVAL_MIN, pacing.createImageIntervalMs - 250);
      pacing.imageAttemptsSinceLastError = 0;
    }
    return;
  }
  if (kind === 'video') {
    pacing.videoAttemptsSinceLastError += 1;
    if (pacing.videoAttemptsSinceLastError >= 2 && pacing.createVideoIntervalMs > VIDEO_INTERVAL_MIN) {
      pacing.createVideoIntervalMs = Math.max(VIDEO_INTERVAL_MIN, pacing.createVideoIntervalMs - 3000);
      pacing.videoAttemptsSinceLastError = 0;
    }
  }
}

function growInterval(kind) {
  if (kind === 'image') {
    pacing.createImageIntervalMs = Math.min(IMAGE_INTERVAL_MAX, pacing.createImageIntervalMs + 1000);
    pacing.imageAttemptsSinceLastError = 0;
  }
  if (kind === 'video') {
    pacing.createVideoIntervalMs = Math.min(VIDEO_INTERVAL_MAX, pacing.createVideoIntervalMs + 6000);
    pacing.videoAttemptsSinceLastError = 0;
  }
}

function previewPayload(body) {
  if (!body) return undefined;
  try {
    const parsed = JSON.parse(body);
    const copy = JSON.parse(JSON.stringify(parsed));
    for (const key of ['image', 'first_frame', 'last_frame']) {
      if (typeof copy[key] === 'string' && copy[key].length > 80) {
        copy[key] = `${copy[key].slice(0, 80)}…[${copy[key].length}]`;
      }
    }
    return copy;
  } catch {
    return String(body).slice(0, 500);
  }
}

function parseRetryAfter(res) {
  const raw = res.headers.get('retry-after');
  const seconds = Number.parseInt(raw || '0', 10);
  return Number.isFinite(seconds) && seconds > 0 ? seconds : 0;
}

async function readErrorText(res, kind) {
  const interesting = [400, 401, 404, 429, 500, 502, 503, 504, 520].includes(res.status) || kind === 'video';
  if (!interesting) return '';
  try {
    const source = typeof res.clone === 'function' ? res.clone() : res;
    return (await source.text()).slice(0, 1200);
  } catch {
    return '';
  }
}

async function apiFetch(url, options = {}, label = 'API', kind = 'text') {
  const maxAttempts = limitsFor(kind).maxAttempts;
  let attempt = 0;
  let queueRetries = 0;
  const method = options.method || 'GET';
  while (attempt < maxAttempts) {
    if (state.stopRequested) throw new Error('Arrêt demandé');
    log('api', `REQ ${label}`, `${method} ${url} (tentative ${attempt + 1}/${maxAttempts})`, previewPayload(options.body));
    let res;
    const started = Date.now();
    try {
      res = await fetch(url, {
        ...options,
        signal: AbortSignal.timeout(kind === 'video' ? 180000 : 120000),
      });
    } catch (error) {
      if (error.message === 'Arrêt demandé' || state.stopRequested) throw new Error('Arrêt demandé');
      const plan = planApiRetry({ kind, status: 0, errorText: '', attempt, queueRetries });
      log('warn', 'NETWORK', `${error.message} — reprise dans ${plan.waitSeconds}s`);
      attempt += 1;
      if (attempt >= maxAttempts) break;
      await sleep(plan.waitSeconds * 1000);
      continue;
    }
    const elapsed = Date.now() - started;
    if (res.ok) {
      log('response', `RÉP ${label}`, `HTTP ${res.status} en ${elapsed} ms`);
      noteSuccess(kind);
      return res;
    }
    const errorText = await readErrorText(res, kind);
    log('response', `RÉP ${label}`, `HTTP ${res.status} en ${elapsed} ms`, errorText || undefined);
    const plan = planApiRetry({
      kind,
      status: res.status,
      errorText,
      retryAfter: parseRetryAfter(res),
      attempt,
      queueRetries,
    });
    if (plan.action === 'fail-client') {
      throw new Error(`HTTP ${res.status} — ${errorText.slice(0, 300)}`);
    }
    if (plan.action === 'fail-queue') throw new QueueSaturatedError();
    if (plan.action === 'queue-wait') {
      queueRetries += 1;
      log('warn', 'VIDEO_QUEUE', `File vidéo saturée — nouvelle tentative dans ${plan.waitSeconds}s (attente ${queueRetries})`);
      await sleep(plan.waitSeconds * 1000);
      continue;
    }
    if (plan.action === 'wait') {
      if (plan.growInterval) growInterval(kind);
      log('warn', 'RETRY', `HTTP ${res.status} — nouvelle tentative dans ${plan.waitSeconds}s`);
      attempt += 1;
      if (attempt >= maxAttempts) break;
      await sleep(plan.waitSeconds * 1000);
      continue;
    }
    return res;
  }
  throw new Error(`Tentatives épuisées pour ${label} après ${maxAttempts} essais`);
}

function createRateLimitedQueue(maxConcurrent, minIntervalMs) {
  let running = 0;
  let lastStart = 0;
  const queue = [];
  let resolveIdle = null;
  const interval = () => (typeof minIntervalMs === 'function' ? minIntervalMs() : minIntervalMs);
  function tryNext() {
    if (running >= maxConcurrent) return;
    if (queue.length === 0) {
      if (running === 0 && resolveIdle) {
        resolveIdle();
        resolveIdle = null;
      }
      return;
    }
    const wait = Math.max(0, lastStart + interval() - Date.now());
    if (wait > 0) {
      setTimeout(tryNext, wait);
      return;
    }
    const task = queue.shift();
    running += 1;
    lastStart = Date.now();
    task().finally(() => {
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

const imageQueue = createRateLimitedQueue(1, () => pacing.createImageIntervalMs);
const videoQueue = createRateLimitedQueue(1, () => pacing.createVideoIntervalMs);

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
  const unescape = (value) => value.replace(/\\n/g, ' ').replace(/\\"/g, '"');
  return {
    title: titleMatch ? unescape(titleMatch[1]) : 'Ambiance Lofi',
    image_prompt: imgMatch ? unescape(imgMatch[1]) : '',
    video_motion_prompt: vidMatch ? unescape(vidMatch[1]) : '',
  };
}

function extractJson(text) {
  if (!text) return null;
  let source = text.trim();
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

function buildExpansionPrompt(userDescription, mood, random = Math.random) {
  const selected = MOODS[mood] || MOODS.rainy_forest;
  const timeOfDay = pick(TIMES_OF_DAY, random);
  const season = pick(SEASONS, random);
  const weather = pick(WEATHERS, random);
  const detail = pick(SENSORY_DETAILS, random);
  const variation = {
    timeOfDay: timeOfDay[0],
    timeOfDayEn: timeOfDay[1],
    season: season[0],
    seasonEn: season[1],
    weather: weather[0],
    weatherEn: weather[1],
    detail: detail[0],
    detailEn: detail[1],
  };
  const prompt = [
    'You are an art director for Studio Ghibli-inspired lofi ambience videos.',
    '',
    'USER SCENE IDEA (French):',
    userDescription,
    '',
    'AMBIENCE (French mood chosen):',
    `${selected.label} — ${selected.desc}`,
    '',
    'MANDATORY UNIQUE VARIATION — make this scene visually distinct even if the idea is familiar:',
    `- Time of day: ${variation.timeOfDay} / ${variation.timeOfDayEn}`,
    `- Season: ${variation.season} / ${variation.seasonEn}`,
    `- Weather: ${variation.weather} / ${variation.weatherEn}`,
    `- Sensory detail to integrate naturally: ${variation.detail} / ${variation.detailEn}`,
    'These four axes must be visible in the image. Do not collapse them into a generic version of the idea.',
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
    '  "image_prompt": "Ultra-detailed English visual description for an image model. Include: setting, lighting, colors, atmosphere, the unique time/season/weather/detail, the capybara (with its location and pose), Ghibli watercolor style, cinematic composition. 150-250 words.",',
    '  "video_motion_prompt": "English description of slow gentle motions for the video model: capybara breathing, leaves falling, mist drifting, water ripples, seamless loop. 60-120 words."',
    '}',
  ].join('\n');
  return { prompt, variation, mood: selected };
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

function messageText(content) {
  if (typeof content === 'string') return content;
  if (Array.isArray(content)) {
    return content.map((part) => (typeof part === 'string' ? part : part?.text || '')).join('\n');
  }
  return '';
}

async function generateExpansion(userDescription, mood) {
  const { prompt, variation } = buildExpansionPrompt(userDescription, mood);
  log('ai', 'VARIATION', `${variation.timeOfDay} · ${variation.season} · ${variation.weather} · ${variation.detail}`);
  const body = {
    model: MODEL_TEXT,
    messages: [
      {
        role: 'system',
        content: 'You are a Studio Ghibli-inspired art director specialized in lofi ambience videos. You always return strict JSON matching the requested schema exactly. Every scene includes a cute calm capybara. Every scene is peaceful, detailed, watercolor, and loopable with slow gentle motion. Honor the mandatory unique time of day, season, weather and sensory detail. No text, no watermark. Return only JSON.',
      },
      { role: 'user', content: prompt },
    ],
    temperature: 0.85,
    max_tokens: 2000,
    response_format: { type: 'json_object' },
  };
  const res = await apiFetch(
    `${API_BASE}/chat/completions`,
    {
      method: 'POST',
      headers: authHeaders(),
      body: JSON.stringify(body),
    },
    'Expansion lofi',
    'text',
  );
  if (!res.ok) {
    const err = await res.text();
    throw new Error(`HTTP ${res.status} — ${err.slice(0, 300)}`);
  }
  const data = await res.json();
  const content = messageText(data?.choices?.[0]?.message?.content);
  const expansion = extractJson(content);
  if (!expansion || !expansion.image_prompt) throw new Error('Expansion invalide');
  if (!expansion.title) expansion.title = 'Ambiance Lofi';
  if (!expansion.video_motion_prompt) {
    expansion.video_motion_prompt = 'Gentle slow motion: capybara breathing softly, leaves drifting down, faint mist, seamless loop.';
  }
  log('success', 'EXPANSION', `Titre : ${expansion.title}`);
  return expansion;
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
      if (error.message === 'Arrêt demandé' || error instanceof QueueSaturatedError) throw error;
      log('warn', 'RETRY', `${name} échoué : ${error.message}`);
      if (i < maxAttempts) {
        const wait = Math.min(90, 8 * i * i);
        log('warn', 'RETRY', `${name} — reprise automatique dans ${wait}s`);
        await sleep(wait * 1000);
      }
    }
  }
  throw lastError || new Error(`${name} échoué`);
}

function aspectRatio() {
  return state.format === '9:16' ? '9:16' : '16:9';
}

async function generateImage(prompt) {
  return retryTask('Image', () => imageQueue.add(async () => {
    const body = {
      model: MODEL_IMAGE,
      prompt,
      size: '1K',
      ratio: aspectRatio(),
      n: 1,
    };
    const res = await apiFetch(
      `${API_BASE}/images/generations`,
      { method: 'POST', headers: authHeaders(), body: JSON.stringify(body) },
      'Image',
      'image',
    );
    if (!res.ok) {
      const err = await res.text();
      throw new Error(`HTTP ${res.status} — ${err.slice(0, 500)}`);
    }
    const data = await res.json();
    const item = data?.data?.[0];
    if (!item) throw new Error('Pas de donnée image');
    if (item.url) return item.url;
    if (item.b64_json) return `data:image/png;base64,${item.b64_json}`;
    throw new Error('Format image inconnu');
  }), 4);
}

async function generateVideo(imageUrl, prompt) {
  return retryTask('Vidéo', () => videoQueue.add(async () => {
    const body = {
      model: MODEL_VIDEO,
      prompt,
      seconds: String(state.durationSeconds),
      mode: 'keyframe',
      size: '720P',
      aspect_ratio: aspectRatio(),
      first_frame: imageUrl,
      last_frame: imageUrl,
    };
    const res = await apiFetch(
      `${API_BASE}/videos`,
      { method: 'POST', headers: authHeaders(), body: JSON.stringify(body) },
      'Vidéo',
      'video',
    );
    if (!res.ok) {
      const err = await res.text();
      throw new Error(`HTTP ${res.status} — ${err.slice(0, 500)}`);
    }
    const data = await res.json();
    const videoId = data.video_id || data.id || data.task_id;
    if (!videoId) throw new Error('Pas de video_id');
    return videoId;
  }), 3);
}

async function loadTimingCache() {
  try {
    return JSON.parse(await fs.readFile(TIMING_PATH, 'utf8'));
  } catch {
    return {};
  }
}

async function recordGenerationTime(key, ms) {
  const cache = await loadTimingCache();
  const samples = Array.isArray(cache[key]) ? cache[key] : [];
  samples.push(ms);
  cache[key] = samples.slice(-10);
  await fs.mkdir(CACHE_DIR, { recursive: true });
  await fs.writeFile(TIMING_PATH, JSON.stringify(cache));
}

function estimateGenerationTime(samples, fallbackMs) {
  if (!samples || samples.length === 0) return fallbackMs;
  const sorted = [...samples].sort((a, b) => a - b);
  const mid = Math.floor(sorted.length / 2);
  return sorted.length % 2 === 0 ? (sorted[mid - 1] + sorted[mid]) / 2 : sorted[mid];
}

async function pollVideo(videoId) {
  const cache = await loadTimingCache();
  const estimatedMs = estimateGenerationTime(cache.video, 90000 + state.durationSeconds * 7000);
  let remaining = Math.max(25, Math.min(VIDEO_FIRST_POLL_DELAY_SEC, Math.floor(estimatedMs / 1000 * 0.35)));
  log('info', 'POLLING', `Attente initiale ${remaining}s pour ${videoId}`);
  while (remaining > 0) {
    if (state.stopRequested) throw new Error('Arrêt demandé');
    await sleep(1000);
    remaining -= 1;
  }
  let attempt = 0;
  while (attempt < VIDEO_MAX_POLL_ATTEMPTS) {
    if (state.stopRequested) throw new Error('Arrêt demandé');
    attempt += 1;
    const url = `${POLL_BASE}?video_id=${encodeURIComponent(videoId)}&model_name=${encodeURIComponent(MODEL_VIDEO)}`;
    let res;
    try {
      res = await apiFetch(url, { method: 'GET', headers: { Authorization: `Bearer ${apiKey()}` } }, 'Polling', 'video');
    } catch (error) {
      if (error.message === 'Arrêt demandé' || error instanceof QueueSaturatedError) throw error;
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
    log('response', `POLL #${attempt}`, `Statut : ${status} (${progress}%)`);
    if (status === 'completed' || status === 'succeeded' || status === 'done') {
      const videoUrl = payload.url || payload.metadata?.url || payload.output?.url;
      if (!videoUrl) throw new Error('Terminé mais pas d’URL');
      return videoUrl;
    }
    if (status === 'failed' || status === 'error' || status === 'cancelled') {
      throw new Error(`Échec définitif (${status})`);
    }
    const wait = attempt < 5 ? 8 : Math.min(30, 10 + Math.floor(attempt / 5) * 3);
    await sleep(wait * 1000);
  }
  throw new Error('Délai maximal de polling dépassé');
}

function authHeaders() {
  return {
    Authorization: `Bearer ${apiKey()}`,
    'Content-Type': 'application/json',
  };
}

function apiKey() {
  const key = (process.env.AGNES_API_KEY || '').trim();
  if (!key) throw new Error('AGNES_API_KEY manquant dans .env');
  return key;
}

function assertAllowedRepoUrl(raw) {
  if (typeof raw !== 'string' || !raw.trim()) throw new Error('GIT_REPO_URL manquant');
  let url;
  try {
    url = new URL(raw.trim());
  } catch {
    throw new Error('GIT_REPO_URL invalide');
  }
  if (url.protocol !== 'https:') throw new Error('GIT_REPO_URL doit utiliser https');
  if (url.username || url.password) throw new Error('GIT_REPO_URL ne doit pas contenir d’identifiants');
  if (url.hostname !== 'github.com') throw new Error('Hôte Git non autorisé');
  if (url.search || url.hash) throw new Error('GIT_REPO_URL inattendu');
  const pathname = url.pathname.replace(/\.git$/, '').replace(/\/$/, '');
  if (pathname !== '/urdirditfurd/videowan2g') {
    throw new Error('Dépôt refusé : seul https://github.com/urdirditfurd/videowan2g est autorisé');
  }
}

function sanitizeBranch(name) {
  const branch = String(name || 'main').trim();
  if (!/^[A-Za-z0-9._/-]+$/.test(branch) || branch.startsWith('-') || branch.includes('..') || branch.endsWith('/')) {
    throw new Error('GIT_BRANCH invalide');
  }
  return branch;
}

function buildSanitizedGitEnv(source = process.env) {
  const env = {};
  for (const key of AMBIENT_ENV_ALLOWLIST) {
    if (source[key]) env[key] = source[key];
  }
  env.GIT_AUTHOR_NAME = source.GIT_AUTHOR_NAME || 'Lofi Capybara Bot';
  env.GIT_AUTHOR_EMAIL = source.GIT_AUTHOR_EMAIL || 'lofi-capybara@users.noreply.github.com';
  env.GIT_COMMITTER_NAME = source.GIT_COMMITTER_NAME || env.GIT_AUTHOR_NAME;
  env.GIT_COMMITTER_EMAIL = source.GIT_COMMITTER_EMAIL || env.GIT_AUTHOR_EMAIL;
  env.GIT_CONFIG_NOSYSTEM = '1';
  env.HOME = EMPTY_HOME;
  delete env.AGNES_API_KEY;
  delete env.GITHUB_TOKEN;
  return env;
}

function createGit({ baseDir = ROOT, binary = 'git' } = {}) {
  const git = simpleGit({
    baseDir,
    binary,
    allowEnvironment: GIT_AUTHOR_ENV_KEYS,
    unsafe: { allowUnsafeCustomBinary: true },
  });
  git.env(buildSanitizedGitEnv());
  return git;
}

function buildAuthenticatedRemote(token, repoUrl = process.env.GIT_REPO_URL) {
  assertAllowedRepoUrl(repoUrl);
  if (!token || /\s/.test(token)) throw new Error('GITHUB_TOKEN invalide');
  const url = new URL(ALLOWED_REPO);
  url.username = 'x-access-token';
  url.password = token;
  return url.href;
}

function sanitizeTitle(title) {
  const cleaned = String(title ?? '')
    .normalize('NFKD')
    .replace(/[\u0300-\u036f]/g, '')
    .replace(/[^A-Za-z0-9]+/g, '_')
    .replace(/^_+|_+$/g, '')
    .slice(0, 80);
  return cleaned || 'lofi_capybara';
}

function videoFileName(title, timestamp = Date.now()) {
  return `${sanitizeTitle(title)}_${timestamp}.mp4`;
}

function commitSubject(title) {
  const oneLine = String(title || 'Ambiance Lofi').replace(/[\r\n]+/g, ' ').replace(/\s+/g, ' ').trim().slice(0, 120);
  return redact(`feat: add new lofi capybara video: ${oneLine || 'Ambiance Lofi'}`);
}

function mappedIpv4(value) {
  const dotted = value.match(/^::ffff:(\d{1,3}(?:\.\d{1,3}){3})$/);
  if (dotted) return dotted[1];
  const hex = value.match(/^::ffff:([0-9a-f]{1,4}):([0-9a-f]{1,4})$/);
  if (!hex) return null;
  const hi = Number.parseInt(hex[1], 16);
  const lo = Number.parseInt(hex[2], 16);
  return `${(hi >> 8) & 255}.${hi & 255}.${(lo >> 8) & 255}.${lo & 255}`;
}

function isPrivateIp(ip) {
  if (net.isIPv4(ip)) {
    const parts = ip.split('.').map(Number);
    const [a, b] = parts;
    if (a === 0 || a === 10 || a === 127) return true;
    if (a === 169 && b === 254) return true;
    if (a === 172 && b >= 16 && b <= 31) return true;
    if (a === 192 && b === 168) return true;
    if (a === 192 && b === 0) return true;
    if (a === 100 && b >= 64 && b <= 127) return true;
    if (a === 198 && (b === 18 || b === 19)) return true;
    if (a >= 224) return true;
    return false;
  }
  if (net.isIPv6(ip)) {
    const value = ip.toLowerCase();
    if (value === '::1' || value === '::') return true;
    if (value.startsWith('fe8') || value.startsWith('fe9') || value.startsWith('fea') || value.startsWith('feb')) return true;
    if (value.startsWith('fc') || value.startsWith('fd')) return true;
    if (value.startsWith('ff')) return true;
    const mapped = mappedIpv4(value);
    if (mapped) return isPrivateIp(mapped);
    return false;
  }
  return true;
}

function isBlockedHost(hostname) {
  const host = hostname.toLowerCase().replace(/^\[|\]$/g, '').replace(/\.$/, '');
  if (!host || host.includes('%')) return true;
  if (host === 'localhost' || host.endsWith('.localhost') || host.endsWith('.local') || host.endsWith('.localdomain')) return true;
  if (host.endsWith('.internal') || host === 'metadata.google.internal' || host === 'metadata.goog') return true;
  if (/^\d+$/.test(host)) return true;
  if (net.isIP(host)) return isPrivateIp(host);
  if (/^\d{1,3}(\.\d{1,3}){3}$/.test(host)) {
    const parts = host.split('.');
    if (parts.some((part) => part.length > 1 && part.startsWith('0'))) return true;
    const numbers = parts.map(Number);
    if (numbers.some((n) => n > 255)) return true;
    return isPrivateIp(numbers.join('.'));
  }
  return false;
}

function inspectVideoUrl(raw) {
  let url;
  try {
    url = new URL(raw);
  } catch {
    throw new Error('URL vidéo invalide');
  }
  if (url.protocol !== 'https:') throw new Error('URL vidéo : https requis');
  if (url.username || url.password) throw new Error('URL vidéo : identifiants refusés');
  if (isBlockedHost(url.hostname)) throw new Error('URL vidéo : hôte local, privé ou métadonnées refusé');
  return url;
}

async function assertPublicDns(hostname) {
  const host = hostname.replace(/^\[|\]$/g, '');
  if (net.isIP(host)) {
    if (isPrivateIp(host)) throw new Error('Adresse privée refusée');
    return;
  }
  let records;
  try {
    records = await lookup(host, { all: true, verbatim: true });
  } catch {
    throw new Error('Résolution DNS impossible');
  }
  if (!records.length) throw new Error('DNS vide');
  for (const record of records) {
    if (isPrivateIp(record.address)) throw new Error('Résolution vers une adresse privée refusée');
  }
}

function exceedsGithubLimit(size) {
  return size >= MAX_VIDEO_BYTES;
}

function detectContainer(buffer) {
  if (!buffer || buffer.length < 12) return null;
  if (buffer[0] === 0x1a && buffer[1] === 0x45 && buffer[2] === 0xdf && buffer[3] === 0xa3) return 'webm';
  const head = buffer.subarray(0, Math.min(buffer.length, 32)).toString('latin1');
  if (head.includes('ftyp')) return 'mp4';
  return null;
}

function asNodeStream(body) {
  if (!body) throw new Error('Corps vidéo vide');
  if (typeof body.pipe === 'function') return body;
  if (typeof Readable.fromWeb === 'function' && typeof body.getReader === 'function') {
    return Readable.fromWeb(body);
  }
  throw new Error('Flux vidéo illisible');
}

async function drain(res) {
  const body = res.body;
  if (!body) return;
  if (typeof body.cancel === 'function') {
    await body.cancel().catch(() => {});
    return;
  }
  if (typeof body.destroy === 'function') body.destroy();
}

async function downloadVideo(videoUrl, destPath) {
  let current = videoUrl;
  let redirects = 0;
  while (redirects <= MAX_REDIRECTS) {
    const url = inspectVideoUrl(current);
    await assertPublicDns(url.hostname);
    const res = await fetch(url, {
      redirect: 'manual',
      headers: { Accept: 'video/mp4,video/webm,*/*', 'User-Agent': 'videowan2g-lofi/1.0' },
      signal: AbortSignal.timeout(180000),
    });
    if ([301, 302, 303, 307, 308].includes(res.status)) {
      const location = res.headers.get('location');
      await drain(res);
      if (!location) throw new Error('Redirection sans Location');
      redirects += 1;
      if (redirects > MAX_REDIRECTS) throw new Error('Trop de redirections');
      current = new URL(location, url).href;
      continue;
    }
    if (!res.ok) {
      await drain(res);
      throw new Error(`Téléchargement HTTP ${res.status}`);
    }
    const declared = Number(res.headers.get('content-length') || 0);
    if (Number.isFinite(declared) && exceedsGithubLimit(declared)) {
      await drain(res);
      throw new Error('Vidéo trop volumineuse pour un commit GitHub');
    }
    const partial = `${destPath}.partial`;
    let received = 0;
    const limiter = new Transform({
      transform(chunk, _encoding, callback) {
        received += chunk.length;
        if (exceedsGithubLimit(received)) {
          callback(new Error('Vidéo trop volumineuse pour un commit GitHub'));
          return;
        }
        callback(null, chunk);
      },
    });
    try {
      await pipeline(asNodeStream(res.body), limiter, createWriteStream(partial));
      const header = Buffer.alloc(64);
      const file = await fs.open(partial, 'r');
      try {
        await file.read(header, 0, 64, 0);
      } finally {
        await file.close();
      }
      const container = detectContainer(header);
      if (!container) throw new Error('En-tête MP4 ou WebM absent');
      const finalPath = container === 'webm' ? destPath.replace(/\.mp4$/, '.webm') : destPath;
      const info = await fs.stat(partial);
      if (info.size < 32 || exceedsGithubLimit(info.size)) throw new Error('Taille de vidéo refusée');
      await fs.rename(partial, finalPath);
      return finalPath;
    } catch (error) {
      await fs.rm(partial, { force: true });
      throw error;
    }
  }
  throw new Error('Trop de redirections');
}

async function publishVideo(absPath, title) {
  assertAllowedRepoUrl(process.env.GIT_REPO_URL);
  const token = (process.env.GITHUB_TOKEN || '').trim();
  if (!token) throw new Error('GITHUB_TOKEN manquant dans .env');
  const relative = path.relative(ROOT, absPath);
  if (relative.startsWith('..') || !relative.startsWith(`generated_videos${path.sep}`)) {
    throw new Error('Chemin vidéo inattendu');
  }
  const info = await fs.stat(absPath);
  if (exceedsGithubLimit(info.size)) throw new Error('Fichier ≥ 100 Mo, commit annulé');
  const header = Buffer.alloc(64);
  const file = await fs.open(absPath, 'r');
  try {
    await file.read(header, 0, 64, 0);
  } finally {
    await file.close();
  }
  if (!detectContainer(header)) throw new Error('Signature absente, commit annulé');
  await fs.mkdir(EMPTY_HOME, { recursive: true });
  const branch = sanitizeBranch(process.env.GIT_BRANCH || 'main');
  const git = createGit({ baseDir: ROOT });
  await git.add(['--', relative]);
  await git.commit(commitSubject(title));
  const remote = buildAuthenticatedRemote(token);
  try {
    await git.push(remote, `HEAD:refs/heads/${branch}`);
  } catch (error) {
    throw new Error(redact(error?.message || String(error), token));
  }
  log('success', 'GIT', `Poussé sur ${branch} : ${relative}`);
}

function readSettings() {
  const duration = Number(process.env.VIDEO_DURATION_SECONDS || 5);
  state.durationSeconds = [5, 6, 8, 10].includes(duration) ? duration : 5;
  state.format = process.env.VIDEO_FORMAT === '9:16' ? '9:16' : '16:9';
}

async function loadScenes() {
  const raw = JSON.parse(await fs.readFile(path.join(ROOT, 'scenes.json'), 'utf8'));
  if (!Array.isArray(raw) || raw.length === 0) throw new Error('scenes.json doit être un tableau non vide');
  return raw.map((entry, index) => {
    if (typeof entry === 'string') {
      const description = entry.trim();
      if (description.length < 4) throw new Error(`Scène ${index + 1} trop courte`);
      return { description, mood: pick(Object.keys(MOODS)) };
    }
    if (entry && typeof entry.description === 'string') {
      const description = entry.description.trim();
      if (description.length < 4) throw new Error(`Scène ${index + 1} trop courte`);
      const mood = MOODS[entry.mood] ? entry.mood : pick(Object.keys(MOODS));
      return { description, mood };
    }
    throw new Error(`Scène ${index + 1} illisible`);
  });
}

async function runScene(scene, index, total) {
  log('state', 'SCENE', `(${index + 1}/${total}) ${scene.description} · ${MOODS[scene.mood].label}`);
  const started = Date.now();
  const expansion = await generateExpansion(scene.description, scene.mood);
  const imageUrl = await generateImage(buildLofiImagePrompt(expansion));
  log('success', 'IMAGE', 'Image keyframe prête');
  pacing.createVideoIntervalMs = VIDEO_INTERVAL_SAFE;
  pacing.videoAttemptsSinceLastError = 0;
  const videoId = await generateVideo(imageUrl, buildLofiVideoPrompt(expansion));
  log('success', 'VIDÉO', `Tâche ${videoId}`);
  const videoUrl = await pollVideo(videoId);
  await fs.mkdir(VIDEO_DIR, { recursive: true });
  const target = path.join(VIDEO_DIR, videoFileName(expansion.title));
  const saved = await downloadVideo(videoUrl, target);
  await recordGenerationTime('video', Date.now() - started);
  log('success', 'FICHIER', saved);
  await publishVideo(saved, expansion.title);
  return saved;
}

async function main() {
  readSettings();
  assertAllowedRepoUrl(process.env.GIT_REPO_URL);
  apiKey();
  if (!(process.env.GITHUB_TOKEN || '').trim()) throw new Error('GITHUB_TOKEN manquant dans .env');
  const scenes = await loadScenes();
  log('info', 'START', `${scenes.length} scène(s) · ${MODEL_TEXT} / ${MODEL_IMAGE} / ${MODEL_VIDEO}`);
  log('info', 'CONFIG', `${state.durationSeconds}s · ${state.format} · dépôt autorisé videowan2g`);
  let failures = 0;
  for (let i = 0; i < scenes.length; i += 1) {
    if (state.stopRequested) break;
    try {
      await runScene(scenes[i], i, scenes.length);
    } catch (error) {
      failures += 1;
      log('error', 'SCENE', error.message);
      if (error.message === 'Arrêt demandé') break;
    }
  }
  if (state.stopRequested) throw new Error('Arrêt demandé');
  if (failures > 0) throw new Error(`${failures} scène(s) en échec`);
  log('success', 'FIN', 'Bibliothèque mise à jour');
}

function assert(condition, message) {
  if (!condition) throw new Error(message);
}

function listenLocal(handler) {
  const server = http.createServer((req, res) => {
    req.resume();
    handler(req, res);
  });
  return new Promise((resolve, reject) => {
    server.on('error', reject);
    server.listen(0, '127.0.0.1', () => {
      const { port } = server.address();
      resolve({
        port,
        close() {
          return new Promise((done) => {
            server.close(() => done());
          });
        },
      });
    });
  });
}

async function runSelfTest() {
  const zero = queueFullDelaySeconds(0, () => 0);
  const mid = queueFullDelaySeconds(1, () => 0.5);
  const late = queueFullDelaySeconds(4, () => 1);
  assert(zero === 51, `backoff 0 inattendu : ${zero}`);
  assert(mid === 120, `backoff 1 inattendu : ${mid}`);
  assert(late === 345, `backoff cap inattendu : ${late}`);
  assert(late <= VIDEO_QUEUE_MAX_WAIT_SEC, 'plafond 360 dépassé');

  const queued = planApiRetry({
    kind: 'video',
    status: 429,
    errorText: '{"error":"video_queue_full"}',
    attempt: 9,
    queueRetries: 0,
    random: () => 0.5,
  });
  assert(queued.action === 'queue-wait' && queued.countsAsAttempt === false, 'la file saturée ne doit pas consommer une tentative');
  assert(queued.waitSeconds === 60, `attente file inattendue : ${queued.waitSeconds}`);
  const exhausted = planApiRetry({
    kind: 'video',
    status: 503,
    errorText: 'queue is full',
    attempt: 0,
    queueRetries: queueRetryLimit(),
  });
  assert(exhausted.action === 'fail-queue', 'le plafond de file doit arrêter la tâche');
  const client = planApiRetry({ kind: 'text', status: 401, errorText: 'nope', attempt: 0, queueRetries: 0 });
  assert(client.action === 'fail-client', '401 texte doit être terminal');

  const local = await listenLocal((req, res) => {
    if (req.url === '/bad') {
      res.writeHead(400, { 'Content-Type': 'application/json' });
      res.end('{"error":"frame-rejected"}');
      return;
    }
    if (req.url === '/auth') {
      res.writeHead(401, { 'Content-Type': 'application/json' });
      res.end('{"error":"invalid-key"}');
      return;
    }
    res.writeHead(404);
    res.end('missing');
  });
  try {
    const videoRes = await apiFetch(
      `http://127.0.0.1:${local.port}/bad`,
      { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: '{}' },
      'Vidéo test',
      'video',
    );
    assert(videoRes.status === 400, 'la vidéo 400 doit rester lisible par l’appelant');
    const videoBody = await videoRes.text();
    assert(videoBody.includes('frame-rejected'), 'détail HTTP vidéo perdu');
    let authMessage = '';
    try {
      await apiFetch(`http://127.0.0.1:${local.port}/auth`, { method: 'GET' }, 'Expansion test', 'text');
    } catch (error) {
      authMessage = error.message;
    }
    assert(authMessage.includes('invalid-key'), `401 texte sans détail : ${authMessage}`);
  } finally {
    await local.close();
  }

  const broken = 'prefix {"title": "Lac\\ncalme", "image_prompt": "capybara by the lake", "video_motion_prompt": "slow mist"';
  const parsed = extractJson(broken);
  assert(parsed && parsed.image_prompt.includes('capybara'), 'extractJson n’a pas récupéré le prompt');

  const { prompt, variation } = buildExpansionPrompt('maison japonaise sous la pluie', 'rainy_forest', () => 0);
  assert(prompt.includes('capybara'), 'le prompt doit exiger le capybara');
  assert(prompt.includes(variation.timeOfDay) && prompt.includes(variation.season) && prompt.includes(variation.weather), 'variation absente du prompt');
  assert(variation.timeOfDay === TIMES_OF_DAY[0][0], 'tirage déterministe incorrect');

  assert(sanitizeTitle('Café / pluie !!') === 'Cafe_pluie', 'sanitation du titre');
  assert(!videoFileName('a/b', 1).includes('/'), 'le nom de fichier doit rester plat');
  assert(videoFileName('Lac', 42) === 'Lac_42.mp4', 'nom de fichier inattendu');

  assertAllowedRepoUrl(ALLOWED_REPO);
  assertAllowedRepoUrl('https://github.com/urdirditfurd/videowan2g');
  for (const bad of [
    'http://github.com/urdirditfurd/videowan2g.git',
    'https://user:token@github.com/urdirditfurd/videowan2g.git',
    'https://github.com/other/videowan2g.git',
    'https://github.com.evil/urdirditfurd/videowan2g.git',
    'https://github.com/urdirditfurd/videowan2g.git?token=1',
  ]) {
    let rejected = false;
    try {
      assertAllowedRepoUrl(bad);
    } catch {
      rejected = true;
    }
    assert(rejected, `URL acceptée à tort : ${bad}`);
  }

  const previousRepo = process.env.GIT_REPO_URL;
  process.env.GIT_REPO_URL = ALLOWED_REPO;
  try {
    const remote = buildAuthenticatedRemote('token-canary');
    assert(!redact(remote, 'token-canary').includes('token-canary'), 'le jeton reste visible après redaction');
    assert(remote.startsWith('https://x-access-token:'), 'remote authentifiée inattendue');
    assert(remote.includes('github.com/urdirditfurd/videowan2g.git'), 'dépôt cible incorrect');
  } finally {
    if (previousRepo === undefined) delete process.env.GIT_REPO_URL;
    else process.env.GIT_REPO_URL = previousRepo;
  }

  assert(isBlockedHost('127.0.0.1'), 'loopback non bloqué');
  assert(isBlockedHost('169.254.169.254'), 'métadonnées non bloquées');
  assert(isBlockedHost('10.1.2.3'), 'réseau privé non bloqué');
  assert(isBlockedHost('localhost'), 'localhost non bloqué');
  assert(isBlockedHost('metadata.google.internal'), 'hôte metadata non bloqué');
  assert(!isBlockedHost('cdn.example.com'), 'hôte public refusé à tort');
  let privateUrlRejected = false;
  try {
    inspectVideoUrl('https://127.0.0.1/video.mp4');
  } catch {
    privateUrlRejected = true;
  }
  assert(privateUrlRejected, 'URL privée acceptée');
  for (const blocked of ['https://[::1]/video.mp4', 'https://[::ffff:127.0.0.1]/video.mp4', 'https://[fd00::1]/video.mp4']) {
    let rejected = false;
    try {
      inspectVideoUrl(blocked);
    } catch {
      rejected = true;
    }
    assert(rejected, `URL IPv6 privée acceptée : ${blocked}`);
  }
  let httpRejected = false;
  try {
    inspectVideoUrl('http://cdn.example.com/v.mp4');
  } catch {
    httpRejected = true;
  }
  assert(httpRejected, 'http accepté');

  const mp4 = Buffer.alloc(16);
  mp4.write('ftyp', 4, 'latin1');
  assert(detectContainer(mp4) === 'mp4', 'ftyp non reconnu');
  assert(detectContainer(Buffer.from([0x1a, 0x45, 0xdf, 0xa3, 0, 0, 0, 0, 0, 0, 0, 0])) === 'webm', 'webm non reconnu');
  assert(detectContainer(Buffer.from('pas une video!!')) === null, 'fichier arbitraire accepté');
  assert(exceedsGithubLimit(MAX_VIDEO_BYTES) && !exceedsGithubLimit(MAX_VIDEO_BYTES - 1), 'seuil 100 Mo incorrect');

  const poisoned = { ...process.env, AGNES_API_KEY: 'sk-canary', GITHUB_TOKEN: 'ghp-canary' };
  const sanitized = buildSanitizedGitEnv(poisoned);
  assert(!Object.prototype.hasOwnProperty.call(sanitized, 'AGNES_API_KEY'), 'AGNES_API_KEY dans l’environnement git');
  assert(!Object.prototype.hasOwnProperty.call(sanitized, 'GITHUB_TOKEN'), 'GITHUB_TOKEN dans l’environnement git');
  assert(sanitized.GIT_CONFIG_NOSYSTEM === '1', 'GIT_CONFIG_NOSYSTEM absent');

  const temp = await fs.mkdtemp(path.join(os.tmpdir(), 'videowan2g-'));
  const dump = path.join(temp, 'keys.txt');
  const fakeGit = path.join(temp, 'fake-git.cjs');
  const nodeBin = process.execPath;
  await fs.writeFile(
    fakeGit,
    `#!${nodeBin}\nconst fs = require('fs');\nconst payload = { keys: Object.keys(process.env).sort(), argv: process.argv.slice(2) };\nfs.writeFileSync(${JSON.stringify(dump)}, JSON.stringify(payload));\nprocess.exit(0);\n`,
    { mode: 0o755 },
  );
  const previousKey = process.env.AGNES_API_KEY;
  const previousToken = process.env.GITHUB_TOKEN;
  process.env.AGNES_API_KEY = 'sk-canary-should-not-leak';
  process.env.GITHUB_TOKEN = 'ghp-canary-should-not-leak';
  try {
    await fs.mkdir(EMPTY_HOME, { recursive: true });
    const git = createGit({ baseDir: temp, binary: fakeGit });
    await git.raw(['version']);
    const first = JSON.parse(await fs.readFile(dump, 'utf8'));
    const expected = new Set([...Object.keys(buildSanitizedGitEnv()), 'GIT_TEST_DISALLOW_ABBREVIATED_OPTIONS']);
    for (const name of first.keys) {
      assert(expected.has(name), `variable inattendue dans le processus git : ${name}`);
    }
    assert(!first.keys.includes('AGNES_API_KEY') && !first.keys.includes('GITHUB_TOKEN'), 'secret transmis au processus git');
    assert(first.keys.includes('GIT_AUTHOR_NAME') && first.keys.includes('PATH'), 'environnement git incomplet');
    assert(!first.keys.join('\n').includes('canary'), 'valeur secrète présente dans le dump');
    const previousRepoForPush = process.env.GIT_REPO_URL;
    process.env.GIT_REPO_URL = ALLOWED_REPO;
    try {
      const remote = buildAuthenticatedRemote('push-canary-token');
      await git.push(remote, 'HEAD:refs/heads/main');
    } finally {
      if (previousRepoForPush === undefined) delete process.env.GIT_REPO_URL;
      else process.env.GIT_REPO_URL = previousRepoForPush;
    }
    const pushed = JSON.parse(await fs.readFile(dump, 'utf8'));
    for (const name of pushed.keys) {
      assert(expected.has(name), `variable inattendue pendant le push : ${name}`);
    }
    assert(pushed.argv[0] === 'push', 'git push non invoqué');
    assert(pushed.argv.some((arg) => arg.includes('push-canary-token')), 'URL de push sans jeton');
    assert(!pushed.keys.includes('GITHUB_TOKEN') && !pushed.keys.includes('AGNES_API_KEY'), 'secret transmis pendant le push');
  } finally {
    if (previousKey === undefined) delete process.env.AGNES_API_KEY;
    else process.env.AGNES_API_KEY = previousKey;
    if (previousToken === undefined) delete process.env.GITHUB_TOKEN;
    else process.env.GITHUB_TOKEN = previousToken;
    await fs.rm(temp, { recursive: true, force: true });
  }

  const repo = await fs.mkdtemp(path.join(os.tmpdir(), 'videowan2g-repo-'));
  try {
    await fs.mkdir(EMPTY_HOME, { recursive: true });
    const git = createGit({ baseDir: repo });
    await git.init();
    await fs.mkdir(path.join(repo, 'generated_videos'));
    const sample = Buffer.alloc(16);
    sample.write('ftyp', 4, 'latin1');
    await fs.writeFile(path.join(repo, 'generated_videos', 'Demo_1.mp4'), sample);
    await git.add(['--', 'generated_videos/Demo_1.mp4']);
    await git.commit('feat: add new lofi capybara video: Demo');
    const author = (await git.raw(['log', '-1', '--format=%an|%ae'])).trim();
    const identity = buildSanitizedGitEnv();
    assert(author === `${identity.GIT_AUTHOR_NAME}|${identity.GIT_AUTHOR_EMAIL}`, 'identité de commit inattendue');
  } finally {
    await fs.rm(repo, { recursive: true, force: true });
  }

  const starts = [];
  const queue = createRateLimitedQueue(1, 40);
  await Promise.all([
    queue.add(async () => {
      starts.push(Date.now());
    }),
    queue.add(async () => {
      starts.push(Date.now());
    }),
  ]);
  await queue.idle();
  assert(starts.length === 2 && starts[1] - starts[0] >= 30, 'la file n’espace pas les démarrages');

  log('success', 'SELFTEST', 'Auto-test OK');
}

const invokedDirectly = process.argv[1] && path.resolve(process.argv[1]) === fileURLToPath(import.meta.url);
if (invokedDirectly) {
  process.on('SIGINT', () => {
    state.stopRequested = true;
    log('warn', 'STOP', 'Arrêt demandé');
  });
  process.on('SIGTERM', () => {
    state.stopRequested = true;
    log('warn', 'STOP', 'Arrêt demandé');
  });
  const runner = process.argv.includes('--self-test') ? runSelfTest() : main();
  runner.catch((error) => {
    log('error', 'GLOBAL', error.message);
    process.exitCode = 1;
  });
}
