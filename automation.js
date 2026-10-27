#!/usr/bin/env node
/**
 * Génération headless de vidéos lofi capybara (Agnes AI) et publication GitHub.
 *
 * Installation :
 *   npm install
 *   cp .env.example .env
 *
 * Lancement :
 *   node automation.js
 *
 * Auto-test sans réseau :
 *   node automation.js --self-test
 */

import dotenv from 'dotenv';
import { constants as fsConstants, createWriteStream } from 'node:fs';
import fs from 'node:fs/promises';
import os from 'node:os';
import path from 'node:path';
import { Readable } from 'node:stream';
import { pipeline } from 'node:stream/promises';
import { fileURLToPath } from 'node:url';
import assert from 'node:assert/strict';
import fetch from 'node-fetch';
import { simpleGit } from 'simple-git';

const ROOT = path.dirname(fileURLToPath(import.meta.url));
dotenv.config({ path: path.join(ROOT, '.env') });

const API_BASE = 'https://apihub.agnes-ai.com/v1';
const POLL_BASE = 'https://apihub.agnes-ai.com/agnesapi';
const MODEL_TEXT = 'agnes-2.5-flash';
const MODEL_IMAGE = 'agnes-image-2.5-flash';
const MODEL_VIDEO = 'agnes-video-2.5-flash';

const IMAGE_PARALLEL = 1;
const IMAGE_INTERVAL_MIN = 3200;
const IMAGE_INTERVAL_MAX = 12000;
const VIDEO_INTERVAL_MIN = 62000;
const VIDEO_INTERVAL_MAX = 90000;
const VIDEO_INTERVAL_SAFE = 75000;
const VIDEO_FIRST_POLL_DELAY_SEC = 80;
const VIDEO_MAX_POLL_ATTEMPTS = 100;
const ALLOWED_DURATIONS = new Set([5, 6, 8, 10]);
const ALLOWED_FORMATS = new Set(['16:9', '9:16']);
const QUEUE_BACKOFF_BASE_SECONDS = [60, 120, 300];
const PLACEHOLDERS = new Set([
  'sk-remplacez-moi',
  'ghp_remplacez-moi',
  'changeme',
  'your-token',
]);

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

const GHIBLI_PALETTE = 'soft pastel palette inspired by Studio Ghibli watercolor backgrounds: sage green #8fad88, warm cream #f2efe9, dusky blue #7a9eb8, soft gold #c4a882, gentle forest greens, muted earth tones, warm amber highlights. Never grey, never desaturated, never neon. Rich but calm. Hand-painted watercolor feel with soft gradients.';

const TIME_OF_DAY = [
  'aube nacrée, juste avant le lever du soleil',
  'matin brumeux',
  'fin de matinée claire',
  'début d’après-midi calme',
  'lumière dorée de fin d’après-midi',
  'heure bleue',
  'crépuscule',
  'nuit douce sous un ciel étoilé',
];

const SEASONS = [
  'début du printemps',
  'plein été',
  'fin d’été',
  'automne avancé',
  'début d’hiver',
  'hiver doux',
];

const WEATHER = [
  'bruine fine',
  'pluie lente et régulière',
  'ciel dégagé',
  'brume basse',
  'après l’orage, sol encore mouillé',
  'neige légère qui tombe',
  'vent très léger dans les feuillages',
  'air immobile et humide',
];

const UNIQUE_DETAILS = [
  'une lanterne en papier allumée',
  'une tasse de thé qui fume',
  'un parapluie en papier huilé posé contre le bois',
  'des carpes qui affleurent sous la surface',
  'un carillon à vent presque immobile',
  'un sentier de pierres moussues',
  'un bateau de bois amarré',
  'une statue de pierre couverte de mousse',
  'un plaid en laine plié',
  'des lucioles lointaines',
  'un poêle en fonte qui rougeoie doucement',
  'des pétales qui dérivent sur l’eau',
];

const SYSTEM_PROMPT = [
  'You are a Studio Ghibli-inspired art director specialized in lofi ambience videos.',
  'You always return strict JSON matching the requested schema exactly.',
  'Every scene includes a cute calm capybara.',
  'Every scene is peaceful, detailed, watercolor, and loopable with slow gentle motion.',
  'When the user prompt specifies a time of day, a season, and a weather, reflect all three clearly so the image is visually unique.',
  'No text, no watermark. Return only JSON.',
].join(' ');

const pacing = {
  createImageIntervalMs: 4000,
  createVideoIntervalMs: VIDEO_INTERVAL_SAFE,
  imageAttemptsSinceLastError: 0,
  videoAttemptsSinceLastError: 0,
};

const state = {
  stopRequested: false,
};

const imageQueue = createRateLimitedQueue(IMAGE_PARALLEL, () => pacing.createImageIntervalMs);
const videoQueue = createRateLimitedQueue(1, () => pacing.createVideoIntervalMs);

let logSink = (level, category, message) => {
  const line = `${new Date().toISOString()} [${level}] [${category}] ${message}`;
  if (level === 'error') console.error(line);
  else console.log(line);
};

class FatalApiError extends Error {
  constructor(message) {
    super(message);
    this.name = 'FatalApiError';
  }
}

function log(level, category, message, data) {
  const extra = data === undefined ? '' : ` ${truncate(safeJson(data), verboseLimit())}`;
  logSink(level, category, redactSecrets(`${message}${extra}`));
}

function verboseLimit() {
  return process.env.VERBOSE === '1' ? 2500 : 500;
}

function truncate(value, max) {
  const text = String(value ?? '');
  if (text.length <= max) return text;
  return `${text.slice(0, max)}…`;
}

function sleep(ms) {
  return new Promise((resolve) => {
    setTimeout(resolve, ms);
  });
}

function choice(list, random) {
  return list[Math.floor(random() * list.length)];
}

function pickVariation(random = Math.random) {
  return {
    timeOfDay: choice(TIME_OF_DAY, random),
    season: choice(SEASONS, random),
    weather: choice(WEATHER, random),
    detail: choice(UNIQUE_DETAILS, random),
  };
}

function intEnv(name, fallback, min, max) {
  const raw = process.env[name];
  if (raw === undefined || raw.trim() === '') return fallback;
  const value = Number.parseInt(raw, 10);
  if (!Number.isFinite(value) || value < min || value > max) {
    throw new Error(`${name} doit être un entier entre ${min} et ${max}`);
  }
  return value;
}

function requireEnv(name) {
  const value = (process.env[name] || '').trim();
  if (!value) {
    throw new Error(`Variable manquante : ${name}. Copiez .env.example vers .env puis renseignez-la.`);
  }
  if (PLACEHOLDERS.has(value)) {
    throw new Error(`${name} contient encore la valeur d'exemple`);
  }
  return value;
}

function redactSecrets(value) {
  let text = String(value ?? '');
  for (const secret of [process.env.GITHUB_TOKEN, process.env.AGNES_API_KEY]) {
    if (secret && secret.length > 6) text = text.split(secret).join('***');
  }
  text = text.replace(/x-access-token:[^@\s'"]+/gi, 'x-access-token:***');
  text = text.replace(/ghp_[A-Za-z0-9]+/g, 'ghp_***');
  text = text.replace(/github_pat_[A-Za-z0-9_]+/g, 'github_pat_***');
  text = text.replace(/sk-[A-Za-z0-9_-]{8,}/g, 'sk-***');
  return text;
}

function safeJson(value) {
  try {
    return JSON.stringify(redactLongStrings(value));
  } catch {
    return String(value);
  }
}

function redactLongStrings(value) {
  if (typeof value === 'string') {
    if (value.length > 180) return `${value.slice(0, 80)}…[${value.length}]`;
    return value;
  }
  if (Array.isArray(value)) return value.map(redactLongStrings);
  if (value && typeof value === 'object') {
    return Object.fromEntries(Object.entries(value).map(([key, item]) => [key, redactLongStrings(item)]));
  }
  return value;
}

function previewBody(body) {
  if (!body) return undefined;
  try {
    return redactLongStrings(JSON.parse(body));
  } catch {
    return truncate(String(body), 500);
  }
}

function headerValue(res, name) {
  const headers = res.headers;
  if (!headers || typeof headers.get !== 'function') return '';
  return headers.get(name) || headers.get(name.toLowerCase()) || '';
}

function parseRetryAfter(header) {
  if (!header) return 0;
  const trimmed = header.trim();
  if (/^\d+$/.test(trimmed)) return Number.parseInt(trimmed, 10);
  const when = Date.parse(trimmed);
  if (Number.isFinite(when)) return Math.max(0, Math.round((when - Date.now()) / 1000));
  return 0;
}

function queueBackoffSeconds(queueHitIndex, random = Math.random) {
  const base = QUEUE_BACKOFF_BASE_SECONDS[Math.min(queueHitIndex, QUEUE_BACKOFF_BASE_SECONDS.length - 1)];
  const jitter = 0.85 + random() * 0.3;
  return Math.max(1, Math.round(base * jitter));
}

function snapshotPacing() {
  return { ...pacing };
}

function restorePacing(saved) {
  Object.assign(pacing, saved);
}

function relaxPacing(kind) {
  if (kind === 'image') {
    pacing.imageAttemptsSinceLastError += 1;
    if (pacing.imageAttemptsSinceLastError >= 4 && pacing.createImageIntervalMs > IMAGE_INTERVAL_MIN) {
      pacing.createImageIntervalMs = Math.max(IMAGE_INTERVAL_MIN, pacing.createImageIntervalMs - 250);
      pacing.imageAttemptsSinceLastError = 0;
    }
  }
  if (kind === 'video') {
    pacing.videoAttemptsSinceLastError += 1;
    if (pacing.videoAttemptsSinceLastError >= 2 && pacing.createVideoIntervalMs > VIDEO_INTERVAL_MIN) {
      pacing.createVideoIntervalMs = Math.max(VIDEO_INTERVAL_MIN, pacing.createVideoIntervalMs - 3000);
      pacing.videoAttemptsSinceLastError = 0;
    }
  }
}

function tightenPacing(kind) {
  if (kind === 'image') {
    pacing.createImageIntervalMs = Math.min(IMAGE_INTERVAL_MAX, pacing.createImageIntervalMs + 1000);
  }
  if (kind === 'video') {
    pacing.createVideoIntervalMs = Math.min(VIDEO_INTERVAL_MAX, pacing.createVideoIntervalMs + 6000);
  }
}

function createRateLimitedQueue(maxConcurrent, minIntervalMs) {
  let running = 0;
  let lastStart = 0;
  const queue = [];
  let resolveIdle = null;
  let scheduled = false;

  const interval = () => (typeof minIntervalMs === 'function' ? minIntervalMs() : minIntervalMs);

  function schedule(ms) {
    if (scheduled) return;
    scheduled = true;
    setTimeout(() => {
      scheduled = false;
      tryNext();
    }, ms);
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
        if (running === 0 && queue.length === 0) {
          resolve();
          return;
        }
        resolveIdle = resolve;
      });
    },
  };
}

function escapeControlCharsInStrings(text) {
  let result = '';
  let inString = false;
  let escape = false;
  for (let i = 0; i < text.length; i += 1) {
    const ch = text[i];
    if (escape) {
      result += ch;
      escape = false;
      continue;
    }
    if (ch === '\\') {
      result += ch;
      escape = true;
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
  let escape = false;
  for (let i = 0; i < text.length; i += 1) {
    const ch = text[i];
    if (escape) {
      escape = false;
      continue;
    }
    if (ch === '\\') {
      escape = true;
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
  const imgMatch = text.match(/"image_prompt"\s*:\s*"((?:[^"\\]|\\.)*)"/);
  const vidMatch = text.match(/"video_motion_prompt"\s*:\s*"((?:[^"\\]|\\.)*)"/);
  if (!imgMatch && !vidMatch) return null;
  return {
    title: titleMatch ? unescapeJsonString(titleMatch[1]) : 'Ambiance Lofi',
    image_prompt: imgMatch ? unescapeJsonString(imgMatch[1]) : '',
    video_motion_prompt: vidMatch ? unescapeJsonString(vidMatch[1]) : '',
  };
}

function extractJson(text) {
  if (!text) return null;
  if (typeof text === 'object') return text;
  let raw = String(text).trim();
  const fenced = raw.match(/```(?:json)?\s*([\s\S]*?)```/);
  if (fenced) raw = fenced[1].trim();
  const firstBrace = raw.indexOf('{');
  if (firstBrace < 0) {
    log('error', 'PARSE', 'Aucun { trouvé');
    return null;
  }
  raw = raw.slice(firstBrace);
  try {
    return JSON.parse(raw);
  } catch {
    log('warn', 'PARSE', 'Stratégie 1 échouée');
  }
  try {
    return JSON.parse(escapeControlCharsInStrings(raw));
  } catch {
    log('warn', 'PARSE', 'Stratégie 2 échouée');
  }
  try {
    const lastBrace = raw.lastIndexOf('}');
    if (lastBrace > 0) return JSON.parse(escapeControlCharsInStrings(raw.slice(0, lastBrace + 1)));
  } catch {
    log('warn', 'PARSE', 'Stratégie 3 échouée');
  }
  try {
    return JSON.parse(balanceJson(escapeControlCharsInStrings(raw)));
  } catch {
    log('warn', 'PARSE', 'Stratégie 4 échouée');
  }
  const salvaged = salvageExpansion(raw);
  if (salvaged && (salvaged.image_prompt || salvaged.video_motion_prompt)) return salvaged;
  log('error', 'PARSE', 'Aucune stratégie n’a fonctionné');
  return null;
}

function buildExpansionPrompt(userDescription, mood, variation = pickVariation()) {
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
    'RANDOM VARIATION — mandatory, this makes the frame unique even if the base idea is familiar:',
    `- Time of day: ${variation.timeOfDay}`,
    `- Season: ${variation.season}`,
    `- Weather: ${variation.weather}`,
    `- Unique prop or detail, naturally integrated: ${variation.detail}`,
    'The time of day, season, and weather must be visible in the lighting, vegetation, and atmosphere.',
    'Do not reuse a generic rainy-forest look unless that is what this variation actually asks for.',
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
    '  "image_prompt": "Ultra-detailed English visual description for an image model. Include: setting, lighting, colors, atmosphere, the capybara (with its location and pose), the required time of day, season, weather, and unique detail, Ghibli watercolor style, cinematic composition. 150-250 words.",',
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

function timeoutFor(kind, method) {
  if (kind === 'video' && method !== 'GET') return 180000;
  if (kind === 'video') return 60000;
  return 120000;
}

async function readErrorText(res) {
  try {
    if (typeof res.clone === 'function') return (await res.clone().text()).slice(0, 1200);
    if (typeof res.text === 'function') return (await res.text()).slice(0, 1200);
  } catch {
    return '';
  }
  return '';
}

async function apiFetch(url, options = {}, label = 'API', kind = 'default', hooks = {}) {
  const doFetch = hooks.fetchImpl || fetch;
  const sleepFn = hooks.sleep || sleep;
  const backoffSeconds = hooks.queueBackoffSeconds || ((index) => queueBackoffSeconds(index));
  const maxAttempts = hooks.maxAttempts ?? (kind === 'video' ? 10 : kind === 'image' ? 10 : 8);
  const queueFullMax = hooks.queueFullMax ?? intEnv('VIDEO_QUEUE_MAX_RETRIES', 36, 1, 200);
  const retryStatuses = new Set([400, 401, 404, 429, 500, 502, 503, 504, 520]);

  let attempt = 0;
  let queueFullHits = 0;

  while (attempt < maxAttempts) {
    if (state.stopRequested) throw new FatalApiError('Arrêt demandé');
    const method = options.method || 'GET';
    log('api', `REQ ${label}`, `${method} ${url} (tentative ${attempt + 1}/${maxAttempts})`, previewBody(options.body));
    try {
      const started = Date.now();
      const res = await doFetch(url, {
        ...options,
        signal: options.signal || AbortSignal.timeout(timeoutFor(kind, method)),
      });
      const retryAfter = parseRetryAfter(headerValue(res, 'Retry-After'));
      let errorText = '';
      if (!res.ok && (retryStatuses.has(res.status) || kind === 'video')) {
        errorText = await readErrorText(res);
      }
      log('response', `RÉP ${label}`, `HTTP ${res.status} en ${Date.now() - started}ms`, errorText || undefined);

      if (res.ok) {
        relaxPacing(kind);
        return res;
      }
      const queueFull = kind === 'video' && /video_queue_full|queue is full/i.test(errorText);
      if (queueFull) {
        queueFullHits += 1;
        pacing.videoAttemptsSinceLastError = 0;
        if (queueFullHits > queueFullMax) {
          throw new FatalApiError(`File vidéo saturée après ${queueFullMax} attentes`);
        }
        const scheduled = backoffSeconds(queueFullHits - 1);
        const hinted = retryAfter > 0 ? retryAfter : scheduled;
        const wait = Math.min(360, Math.max(scheduled, hinted));
        log('warn', 'VIDEO_QUEUE', `Queue vidéo saturée — nouvelle tentative dans ${wait}s`);
        await sleepFn(wait * 1000);
        continue;
      }

      if (res.status === 401 || (kind !== 'video' && [400, 404].includes(res.status))) {
        throw new FatalApiError(`HTTP ${res.status} — ${errorText.slice(0, 300)}`);
      }

      if (res.status === 429) {
        tightenPacing(kind);
        const fallback = kind === 'video'
          ? Math.min(180, 60 + attempt * 8)
          : Math.min(120, 10 + attempt * 12);
        const wait = retryAfter > 0 ? retryAfter : fallback;
        log('warn', 'RATELIMIT', `429 — attente ${wait}s`);
        await sleepFn(wait * 1000);
        attempt += 1;
        continue;
      }

      if ([500, 502, 503, 504, 520].includes(res.status)) {
        const wait = Math.min(kind === 'video' ? 180 : 60, kind === 'video' ? 20 + attempt * 10 : 5 + attempt * 5);
        if (res.status === 503 && kind === 'video') {
          log('warn', 'SERVICE', '503 — service vidéo occupé, conservation du processus');
        } else {
          log('warn', 'SERVICE', `HTTP ${res.status} — attente ${wait}s`);
        }
        await sleepFn(wait * 1000);
        attempt += 1;
        continue;
      }

      return res;
    } catch (error) {
      if (error instanceof FatalApiError) throw error;
      const wait = Math.min(kind === 'video' ? 120 : 45, kind === 'video' ? 10 + attempt * 8 : 3 + attempt * 4);
      log('warn', 'NETWORK', `${error.message} — retry dans ${wait}s`);
      await sleepFn(wait * 1000);
      attempt += 1;
    }
  }

  throw new Error(`Tentatives épuisées pour ${label} après ${maxAttempts} essais`);
}

async function retryTask(name, fn, maxAttempts = 3, hooks = {}) {
  const sleepFn = hooks.sleep || sleep;
  let lastError = null;
  for (let attempt = 1; attempt <= maxAttempts; attempt += 1) {
    if (state.stopRequested) throw new FatalApiError('Arrêt demandé');
    try {
      log('info', 'RETRY', `${name} — essai ${attempt}/${maxAttempts}`);
      return await fn();
    } catch (error) {
      lastError = error;
      if (error instanceof FatalApiError) throw error;
      log('warn', 'RETRY', `${name} échoué : ${error.message}`);
      if (attempt < maxAttempts) {
        const wait = Math.min(90, 8 * attempt * attempt);
        log('warn', 'RETRY', `${name} — reprise automatique dans ${wait}s`);
        await sleepFn(wait * 1000);
      }
    }
  }
  throw lastError || new Error(`${name} échoué`);
}

function agnesHeaders() {
  return {
    Authorization: `Bearer ${requireEnv('AGNES_API_KEY')}`,
    'Content-Type': 'application/json',
  };
}

async function generateExpansion(scene, variation) {
  const prompt = buildExpansionPrompt(scene.description, scene.mood, variation);
  log('ai', 'PROMPT LOFI', 'Prompt envoyé', truncate(prompt, verboseLimit()));
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
  const res = await apiFetch(
    `${API_BASE}/chat/completions`,
    { method: 'POST', headers: agnesHeaders(), body: JSON.stringify(body) },
    'Expansion lofi',
    'text',
  );
  if (!res.ok) {
    const err = await res.text();
    throw new Error(`HTTP ${res.status} — ${err.slice(0, 300)}`);
  }
  const data = await res.json();
  const content = data.choices && data.choices[0] && data.choices[0].message && data.choices[0].message.content;
  const expansion = extractJson(content);
  if (!expansion || !expansion.image_prompt || String(expansion.image_prompt).trim().length < 40) {
    throw new Error('Expansion invalide');
  }
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
    const res = await apiFetch(
      `${API_BASE}/images/generations`,
      { method: 'POST', headers: agnesHeaders(), body: JSON.stringify(body) },
      'Image',
      'image',
    );
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

async function generateVideo(imageUrl, prompt, scene) {
  return videoQueue.add(() => retryTask('Vidéo', async () => {
    const aspect = scene.format === '9:16' ? '9:16' : '16:9';
    const body = {
      model: MODEL_VIDEO,
      prompt,
      seconds: String(scene.durationSeconds),
      mode: 'keyframe',
      size: '720P',
      aspect_ratio: aspect,
      first_frame: imageUrl,
      last_frame: imageUrl,
    };
    log('ai', 'PROMPT VIDÉO', 'Envoyé', truncate(prompt, verboseLimit()));
    const res = await apiFetch(
      `${API_BASE}/videos`,
      { method: 'POST', headers: agnesHeaders(), body: JSON.stringify(body) },
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
  }, 3));
}

const TIMING_CACHE_PATH = path.join(ROOT, '.timing-cache.json');

async function loadTimingCache() {
  try {
    return JSON.parse(await fs.readFile(TIMING_CACHE_PATH, 'utf8'));
  } catch {
    return {};
  }
}

async function saveTimingCache(cache) {
  await fs.writeFile(TIMING_CACHE_PATH, JSON.stringify(cache));
}

async function recordGenerationTime(key, elapsedMs) {
  const cache = await loadTimingCache();
  if (!Array.isArray(cache[key])) cache[key] = [];
  cache[key].push(elapsedMs);
  if (cache[key].length > 10) cache[key] = cache[key].slice(-10);
  await saveTimingCache(cache);
}

async function estimateGenerationTime(key, fallbackMs) {
  const cache = await loadTimingCache();
  const samples = cache[key];
  if (!samples || samples.length === 0) return fallbackMs;
  const sorted = [...samples].sort((a, b) => a - b);
  const mid = Math.floor(sorted.length / 2);
  if (sorted.length % 2 === 0) return (sorted[mid - 1] + sorted[mid]) / 2;
  return sorted[mid];
}

async function pollVideo(videoId, durationSeconds) {
  const estimatedMs = await estimateGenerationTime('video', 90000 + durationSeconds * 7000);
  const firstPollCap = intEnv('VIDEO_FIRST_POLL_DELAY_SEC', VIDEO_FIRST_POLL_DELAY_SEC, 0, 600);
  const maxPolls = intEnv('VIDEO_MAX_POLL_ATTEMPTS', VIDEO_MAX_POLL_ATTEMPTS, 1, 500);
  const minWait = Math.max(25, Math.min(firstPollCap, Math.floor((estimatedMs / 1000) * 0.35)));
  log('info', 'POLLING', `Attente initiale ${minWait}s pour ${videoId}`);
  for (let remaining = minWait; remaining > 0; remaining -= 1) {
    if (state.stopRequested) throw new FatalApiError('Arrêt demandé');
    if (remaining === minWait || remaining % 15 === 0) {
      log('info', 'POLLING', `File vidéo — ${remaining}s`);
    }
    await sleep(1000);
  }

  for (let attempt = 1; attempt <= maxPolls; attempt += 1) {
    if (state.stopRequested) throw new FatalApiError('Arrêt demandé');
    const url = `${POLL_BASE}?video_id=${encodeURIComponent(videoId)}&model_name=${encodeURIComponent(MODEL_VIDEO)}`;
    let res;
    try {
      res = await apiFetch(url, { method: 'GET', headers: { Authorization: `Bearer ${requireEnv('AGNES_API_KEY')}` } }, 'Polling', 'video');
    } catch (error) {
      if (error instanceof FatalApiError) throw error;
      log('warn', 'POLLING', `Erreur transitoire : ${error.message}`);
      await sleep(Math.min(90000, 15000 + attempt * 1500));
      continue;
    }
    if (!res.ok) {
      log('warn', 'POLLING', `HTTP ${res.status}`);
      await sleep(Math.min(60000, 10000 + attempt * 1000));
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
      const videoUrl = payload.url || (payload.metadata && payload.metadata.url) || (payload.output && payload.output.url);
      if (!videoUrl) throw new Error('Terminé mais pas d’URL');
      log('success', 'POLLING', `Terminé après ${attempt} polls`);
      return videoUrl;
    }
    if (status === 'failed' || status === 'error' || status === 'cancelled') {
      throw new FatalApiError(`Échec définitif (${status})`);
    }
    const wait = attempt < 5 ? 8 : Math.min(30, 10 + Math.floor(attempt / 5) * 3);
    log('info', 'POLLING', `Nouvelle vérification dans ${wait}s`);
    await sleep(wait * 1000);
  }
  throw new Error('Délai maximal de polling dépassé');
}

function sanitizeTitle(title) {
  const cleaned = String(title || 'Ambiance Lofi')
    .normalize('NFD')
    .replace(/[\u0300-\u036f]/g, '')
    .replace(/[^a-zA-Z0-9]+/g, '_')
    .replace(/^_+|_+$/g, '')
    .slice(0, 60);
  return cleaned || 'Ambiance_Lofi';
}

function videoFileName(title, timestamp = Date.now()) {
  return `${sanitizeTitle(title)}_${timestamp}.mp4`;
}

function toNodeStream(body) {
  if (!body) throw new Error('Corps de réponse vide');
  if (typeof body.pipe === 'function') return body;
  if (typeof Readable.fromWeb === 'function') return Readable.fromWeb(body);
  throw new Error('Flux vidéo illisible');
}

function assertVideoContentType(type) {
  const value = String(type || '').toLowerCase();
  if (!value) return;
  if (value.includes('text/html') || value.includes('application/json') || value.startsWith('text/')) {
    throw new Error(`URL vidéo inattendue (${type})`);
  }
}

function assertHttpUrl(rawUrl) {
  let parsed;
  try {
    parsed = new URL(rawUrl);
  } catch {
    throw new Error('URL vidéo invalide');
  }
  if (parsed.protocol !== 'https:' && parsed.protocol !== 'http:') {
    throw new Error('URL vidéo refusée');
  }
  return parsed;
}

function assertSafeDownloadUrl(rawUrl) {
  const parsed = assertHttpUrl(rawUrl);
  if (parsed.username || parsed.password) throw new Error('URL vidéo refusée');
  const host = parsed.hostname.replace(/^\[|\]$/g, '').toLowerCase();
  if (
    host === 'localhost'
    || host.endsWith('.localhost')
    || host.endsWith('.local')
    || host === 'metadata.google.internal'
    || host === 'metadata.internal'
  ) {
    throw new Error('URL vidéo refusée');
  }
  if (host.includes(':')) {
    const normalized = host.split('%')[0];
    const mapped = normalized.match(/^::ffff:(\d{1,3}(?:\.\d{1,3}){3})$/i);
    if (mapped) {
      assertSafeDownloadUrl(`http://${mapped[1]}/`);
      return parsed;
    }
    if (normalized === '::1' || normalized === '::' || normalized.startsWith('fe80:') || normalized.startsWith('fc') || normalized.startsWith('fd')) {
      throw new Error('URL vidéo refusée');
    }
  }
  const ipv4 = host.match(/^(\d{1,3})\.(\d{1,3})\.(\d{1,3})\.(\d{1,3})$/);
  if (!ipv4) return parsed;
  const parts = ipv4.slice(1).map((part) => Number.parseInt(part, 10));
  if (parts.some((part) => part > 255)) throw new Error('URL vidéo refusée');
  const [a, b] = parts;
  const blocked = a === 0
    || a === 10
    || a === 127
    || (a === 100 && b >= 64 && b <= 127)
    || (a === 169 && b === 254)
    || (a === 172 && b >= 16 && b <= 31)
    || (a === 192 && b === 168)
    || (a === 198 && (b === 18 || b === 19));
  if (blocked) throw new Error('URL vidéo refusée');
  return parsed;
}

function assertVideoMagic(header) {
  if (!header || header.length < 12) throw new Error('URL vidéo inattendue (contenu non vidéo)');
  const ftyp = header.subarray(4, 8).toString('ascii') === 'ftyp';
  const webm = header[0] === 0x1a && header[1] === 0x45 && header[2] === 0xdf && header[3] === 0xa3;
  if (!ftyp && !webm) throw new Error('URL vidéo inattendue (contenu non vidéo)');
}

function videoDestination(fileName) {
  const videosDir = path.resolve(ROOT, 'generated_videos');
  const destPath = path.resolve(videosDir, fileName);
  const relative = path.relative(videosDir, destPath);
  if (relative.startsWith('..') || path.isAbsolute(relative) || relative.includes('..')) {
    throw new Error('Nom de fichier vidéo refusé');
  }
  return { videosDir, destPath };
}

async function fetchVideoResponse(url, hooks = {}) {
  const doFetch = hooks.fetchImpl || fetch;
  const timeoutMs = hooks.timeoutMs || 180000;
  let current = url;
  for (let hop = 0; hop < 3; hop += 1) {
    assertSafeDownloadUrl(current);
    const res = await doFetch(current, {
      redirect: 'manual',
      signal: AbortSignal.timeout(timeoutMs),
    });
    if (res.status >= 300 && res.status < 400) {
      const location = headerValue(res, 'location');
      if (res.body && typeof res.body.cancel === 'function') {
        await res.body.cancel().catch(() => {});
      } else if (res.body && typeof res.body.destroy === 'function') {
        res.body.destroy();
      }
      if (!location) throw new Error('URL vidéo refusée');
      current = new URL(location, current).toString();
      continue;
    }
    if (res.url) assertSafeDownloadUrl(res.url);
    return res;
  }
  throw new Error('URL vidéo refusée');
}

async function downloadToFile(url, destPath, hooks = {}) {
  const res = await fetchVideoResponse(url, hooks);
  if (!res.ok) throw new Error(`Téléchargement HTTP ${res.status}`);
  assertVideoContentType(headerValue(res, 'content-type'));
  await fs.mkdir(path.dirname(destPath), { recursive: true });
  try {
    await pipeline(toNodeStream(res.body), createWriteStream(destPath, { flags: 'wx' }));
    const handle = await fs.open(destPath, 'r');
    try {
      const header = Buffer.alloc(12);
      await handle.read(header, 0, 12, 0);
      assertVideoMagic(header);
    } finally {
      await handle.close();
    }
  } catch (error) {
    await fs.rm(destPath, { force: true });
    throw error;
  }
  const stat = await fs.stat(destPath);
  if (stat.size < 1024) {
    await fs.rm(destPath, { force: true });
    throw new Error('Fichier vidéo trop petit, téléchargement suspect');
  }
  return destPath;
}

async function downloadVideo(url, title) {
  const { videosDir, destPath } = videoDestination(videoFileName(title));
  await fs.mkdir(videosDir, { recursive: true });
  log('info', 'DOWNLOAD', path.basename(destPath));
  let lastError = null;
  for (let attempt = 1; attempt <= 3; attempt += 1) {
    try {
      await downloadToFile(url, destPath);
      const stat = await fs.stat(destPath);
      assertGitHubFileSize(stat.size);
      log('success', 'DOWNLOAD', destPath);
      return destPath;
    } catch (error) {
      lastError = error;
      await fs.rm(destPath, { force: true });
      const permanent = /trop lourde|URL vidéo refusée|URL vidéo invalide|URL vidéo inattendue|contenu non vidéo/.test(error.message || '');
      if (attempt === 3 || permanent) break;
      const wait = 5 * attempt;
      log('warn', 'DOWNLOAD', `${error.message} — nouvel essai dans ${wait}s`);
      await sleep(wait * 1000);
    }
  }
  throw lastError || new Error('Téléchargement échoué');
}

const ALLOWED_REPO = 'urdirditfurd/videowan2g';
const GITHUB_MAX_FILE_BYTES = 100 * 1024 * 1024;

function buildAuthedRemote(repoUrl, token) {
  let url;
  try {
    url = new URL(repoUrl);
  } catch {
    throw new Error('GITHUB_REPO_URL invalide');
  }
  if (url.protocol !== 'https:') throw new Error('GITHUB_REPO_URL doit utiliser https');
  if (url.hostname !== 'github.com') throw new Error('GITHUB_REPO_URL doit pointer vers github.com');
  if (url.username || url.password || url.search || url.hash) {
    throw new Error('GITHUB_REPO_URL ne doit contenir ni identifiant ni paramètre');
  }
  const parts = url.pathname.replace(/\.git$/i, '').split('/').filter(Boolean);
  if (parts.length !== 2 || `${parts[0]}/${parts[1]}` !== ALLOWED_REPO) {
    throw new Error(`GITHUB_REPO_URL doit cibler https://github.com/${ALLOWED_REPO}`);
  }
  url.username = 'x-access-token';
  url.password = token;
  url.hash = '';
  url.search = '';
  return url.toString();
}

function assertPublishableVideo(relativePath) {
  if (!/^generated_videos\/[A-Za-z0-9][A-Za-z0-9_-]{0,80}\.mp4$/.test(relativePath)) {
    throw new Error('Publication refusée : seul un mp4 de generated_videos/ peut être commité');
  }
}

function assertGitHubFileSize(size) {
  if (!Number.isFinite(size) || size < 1024) {
    throw new Error('Fichier vidéo trop petit pour être publié');
  }
  if (size >= GITHUB_MAX_FILE_BYTES) {
    throw new Error(`Vidéo trop lourde pour GitHub (${size} octets, limite 100 Mo)`);
  }
}

const GIT_PASSTHROUGH_KEYS = [
  'PATH',
  'HOME',
  'USER',
  'LOGNAME',
  'LANG',
  'LC_ALL',
  'LC_CTYPE',
  'LC_MESSAGES',
  'TMPDIR',
  'TMP',
  'TEMP',
  'SYSTEMROOT',
  'COMSPEC',
  'PATHEXT',
  'HTTP_PROXY',
  'HTTPS_PROXY',
  'http_proxy',
  'https_proxy',
  'NO_PROXY',
  'no_proxy',
  'ALL_PROXY',
  'all_proxy',
];

const GIT_IDENTITY_ENV = [
  'GIT_AUTHOR_NAME',
  'GIT_AUTHOR_EMAIL',
  'GIT_COMMITTER_NAME',
  'GIT_COMMITTER_EMAIL',
];

function containsSecret(value, secrets) {
  return secrets.some((secret) => value.includes(secret));
}

function gitProcessEnv(extra = {}) {
  const secrets = [process.env.AGNES_API_KEY, process.env.GITHUB_TOKEN]
    .map((value) => (value || '').trim())
    .filter((value) => value.length > 6 && !PLACEHOLDERS.has(value));
  const env = {};
  for (const key of GIT_PASSTHROUGH_KEYS) {
    const value = process.env[key];
    if (typeof value !== 'string' || value === '') continue;
    if (containsSecret(value, secrets)) continue;
    env[key] = value;
  }
  for (const key of GIT_IDENTITY_ENV) {
    const value = extra[key];
    if (typeof value !== 'string' || value.trim() === '') continue;
    if (containsSecret(value, secrets)) continue;
    env[key] = value;
  }
  return env;
}

function createGit(baseDir, identity = {}) {
  const git = simpleGit({
    baseDir,
    allowEnvironment: GIT_IDENTITY_ENV,
  });
  git.env(gitProcessEnv(identity));
  return git;
}

function assertRuntimeNumbers() {
  intEnv('VIDEO_QUEUE_MAX_RETRIES', 36, 1, 200);
  intEnv('VIDEO_FIRST_POLL_DELAY_SEC', VIDEO_FIRST_POLL_DELAY_SEC, 0, 600);
  intEnv('VIDEO_MAX_POLL_ATTEMPTS', VIDEO_MAX_POLL_ATTEMPTS, 1, 500);
}

async function publishVideo(destPath, title) {
  if (process.env.SKIP_GIT === '1') {
    log('info', 'GIT', 'SKIP_GIT=1 — publication ignorée');
    return;
  }
  const token = requireEnv('GITHUB_TOKEN');
  const repoUrl = requireEnv('GITHUB_REPO_URL');
  const authedRemote = buildAuthedRemote(repoUrl, token);
  const gitName = (process.env.GIT_USER_NAME || 'Lofi Capybara Bot').trim();
  const gitEmail = (process.env.GIT_USER_EMAIL || 'lofi-capybara-bot@users.noreply.github.com').trim();
  const git = createGit(ROOT, {
    GIT_AUTHOR_NAME: gitName,
    GIT_AUTHOR_EMAIL: gitEmail,
    GIT_COMMITTER_NAME: gitName,
    GIT_COMMITTER_EMAIL: gitEmail,
  });
  const current = (await git.revparse(['--abbrev-ref', 'HEAD'])).trim();
  if (!current || current === 'HEAD') {
    throw new Error('HEAD détaché : extrayez une branche avant de publier');
  }
  const requested = (process.env.GIT_BRANCH || current).trim();
  if (requested !== current) {
    throw new Error(`GIT_BRANCH=${requested} mais la branche courante est ${current}. Extrayez cette branche avant de lancer le script.`);
  }
  const relativePath = path.relative(ROOT, destPath).split(path.sep).join('/');
  assertPublishableVideo(relativePath);
  const stat = await fs.stat(destPath);
  assertGitHubFileSize(stat.size);
  if (stat.size >= 50 * 1024 * 1024) {
    log('warn', 'GIT', `Fichier volumineux (${Math.round(stat.size / (1024 * 1024))} Mo)`);
  }
  const safeTitle = String(title).replace(/[\r\n]+/g, ' ').trim();
  const message = `feat: add new lofi capybara video: ${safeTitle}`;

  try {
    await git.add([relativePath]);
    const status = await git.status();
    const listed = status.staged.some((file) => file.replaceAll('\\', '/') === relativePath);
    if (!listed) throw new Error(`Rien à valider pour ${relativePath}`);
    await git.raw(['commit', '-m', message, '--', relativePath]);
    await git.raw(['push', authedRemote, `HEAD:refs/heads/${current}`]);
    log('success', 'GIT', `Publié : ${message}`);
  } catch (error) {
    throw new Error(redactSecrets(error && error.message ? error.message : error));
  }
}

function generationDefaults() {
  const mood = (process.env.DEFAULT_MOOD || 'rainy_forest').trim();
  if (!MOODS[mood]) throw new Error(`DEFAULT_MOOD inconnu : ${mood}`);
  const durationSeconds = intEnv('VIDEO_DURATION_SECONDS', 5, 5, 10);
  if (!ALLOWED_DURATIONS.has(durationSeconds)) {
    throw new Error('VIDEO_DURATION_SECONDS doit valoir 5, 6, 8 ou 10');
  }
  const format = (process.env.VIDEO_FORMAT || '16:9').trim();
  if (!ALLOWED_FORMATS.has(format)) throw new Error('VIDEO_FORMAT doit valoir 16:9 ou 9:16');
  return { mood, durationSeconds, format };
}

function normalizeScene(entry, index, defaults) {
  const source = typeof entry === 'string' ? { description: entry } : entry;
  if (!source || typeof source !== 'object' || typeof source.description !== 'string') {
    throw new Error(`Scène ${index + 1} : description manquante`);
  }
  const description = source.description.trim();
  if (description.length < 4) throw new Error(`Scène ${index + 1} : décrivez au moins 4 caractères`);
  const mood = source.mood || defaults.mood;
  if (!MOODS[mood]) throw new Error(`Scène ${index + 1} : ambiance inconnue « ${mood} »`);
  const durationSeconds = Number(source.durationSeconds ?? defaults.durationSeconds);
  if (!ALLOWED_DURATIONS.has(durationSeconds)) throw new Error(`Scène ${index + 1} : durée non supportée`);
  const format = source.format || defaults.format;
  if (!ALLOWED_FORMATS.has(format)) throw new Error(`Scène ${index + 1} : format non supporté`);
  return { description, mood, durationSeconds, format };
}

async function loadScenes(filePath, defaults) {
  const raw = await fs.readFile(filePath, 'utf8');
  const parsed = JSON.parse(raw);
  if (!Array.isArray(parsed) || parsed.length === 0) {
    throw new Error('scenes.json doit contenir un tableau non vide');
  }
  return parsed.map((entry, index) => normalizeScene(entry, index, defaults));
}

async function runScene(scene, index, total) {
  log('info', 'SCENE', `(${index + 1}/${total}) ${scene.description}`);
  const variation = pickVariation();
  log('info', 'VARIATION', `${variation.season} · ${variation.timeOfDay} · ${variation.weather} · ${variation.detail}`);
  const expansion = await generateExpansion(scene, variation);
  const imageUrl = await generateImage(buildLofiImagePrompt(expansion), scene.format);
  log('success', 'IMAGE', truncate(imageUrl, 180));
  pacing.createVideoIntervalMs = VIDEO_INTERVAL_SAFE;
  pacing.videoAttemptsSinceLastError = 0;
  const videoStarted = Date.now();
  const videoId = await generateVideo(imageUrl, buildLofiVideoPrompt(expansion, scene.durationSeconds), scene);
  log('success', 'VIDÉO', `Tâche ${videoId}`);
  const videoUrl = await pollVideo(videoId, scene.durationSeconds);
  await recordGenerationTime('video', Date.now() - videoStarted);
  const destPath = await downloadVideo(videoUrl, expansion.title);
  await publishVideo(destPath, expansion.title);
  return { title: expansion.title, file: destPath };
}

function installSignalHandlers() {
  for (const signal of ['SIGINT', 'SIGTERM']) {
    process.on(signal, () => {
      state.stopRequested = true;
      log('warn', 'STOP', 'Arrêt demandé — la tâche en cours se termine sans en lancer une autre');
    });
  }
}

async function main() {
  if (process.argv.includes('--self-test')) {
    await runSelfTest();
    return;
  }
  installSignalHandlers();
  assertRuntimeNumbers();
  const defaults = generationDefaults();
  const scenes = await loadScenes(path.join(ROOT, 'scenes.json'), defaults);
  requireEnv('AGNES_API_KEY');
  if (process.env.SKIP_GIT !== '1') {
    requireEnv('GITHUB_TOKEN');
    buildAuthedRemote(requireEnv('GITHUB_REPO_URL'), process.env.GITHUB_TOKEN);
  }
  log('info', 'START', `${scenes.length} scène(s) · ${defaults.mood} · ${defaults.durationSeconds}s · ${defaults.format}`);
  const successes = [];
  const failures = [];
  for (let index = 0; index < scenes.length; index += 1) {
    if (state.stopRequested) break;
    try {
      successes.push(await runScene(scenes[index], index, scenes.length));
    } catch (error) {
      const message = redactSecrets(error.message);
      failures.push({ description: scenes[index].description, message });
      log('error', 'SCENE', `Échec scène ${index + 1} : ${message}`);
    }
  }
  log('info', 'BILAN', `${successes.length} vidéo(s) publiée(s), ${failures.length} échec(s)`);
  if (state.stopRequested || failures.length > 0) process.exitCode = 1;
}

function jsonResponse({ status = 200, ok = status >= 200 && status < 300, body = '', headers = {} }) {
  const text = typeof body === 'string' ? body : JSON.stringify(body);
  return {
    ok,
    status,
    headers: {
      get(name) {
        const found = Object.entries(headers).find(([key]) => key.toLowerCase() === name.toLowerCase());
        return found ? found[1] : null;
      },
    },
    clone() {
      return jsonResponse({ status, ok, body: text, headers });
    },
    async text() {
      return text;
    },
    async json() {
      return JSON.parse(text);
    },
  };
}

async function runSelfTest() {
  const previousSink = logSink;
  logSink = () => {};
  const savedPacing = snapshotPacing();
  const savedStop = state.stopRequested;
  try {
    const fixed = {
      timeOfDay: TIME_OF_DAY[4],
      season: SEASONS[3],
      weather: WEATHER[1],
      detail: UNIQUE_DETAILS[1],
    };
    const prompt = buildExpansionPrompt('maison japonaise sous la pluie', 'rainy_forest', fixed);
    assert.match(prompt, /capybara/i);
    assert.match(prompt, /lumière dorée de fin d’après-midi/);
    assert.match(prompt, /automne avancé/);
    assert.match(prompt, /pluie lente et régulière/);
    assert.match(prompt, /tasse de thé qui fume/);
    assert.match(prompt, /image_prompt/);
    assert.match(prompt, /video_motion_prompt/);

    const fenced = extractJson('```json\n{"title":"Brume","image_prompt":"A calm capybara under rain in a painted forest with warm lamps and wet moss.","video_motion_prompt":"slow breathing"}\n```');
    assert.equal(fenced.title, 'Brume');

    const broken = 'préambule {"title": "Soir", "image_prompt": "capybara on a porch with tea steam and golden rain light across wooden steps", "video_motion_prompt": "mist drifts';
    const salvaged = extractJson(broken);
    assert.equal(salvaged.title, 'Soir');
    assert.match(salvaged.image_prompt, /capybara on a porch/);

    assert.equal(queueBackoffSeconds(0, () => 0.5), 60);
    assert.equal(queueBackoffSeconds(1, () => 0.5), 120);
    assert.equal(queueBackoffSeconds(2, () => 0.5), 300);
    assert.equal(queueBackoffSeconds(6, () => 0.5), 300);
    assert.equal(queueBackoffSeconds(0, () => 0), Math.round(60 * 0.85));

    assert.equal(sanitizeTitle('Maison japonaise — pluie !'), 'Maison_japonaise_pluie');
    assert.match(videoFileName('Thé doré', 1700000000000), /^The_dore_1700000000000\.mp4$/);

    const scene = normalizeScene(
      { description: '  quai sous la brume  ', mood: 'night_stars', durationSeconds: 8, format: '9:16' },
      0,
      { mood: 'rainy_forest', durationSeconds: 5, format: '16:9' },
    );
    assert.deepEqual(scene, {
      description: 'quai sous la brume',
      mood: 'night_stars',
      durationSeconds: 8,
      format: '9:16',
    });

    const scenes = await loadScenes(path.join(ROOT, 'scenes.json'), {
      mood: 'rainy_forest',
      durationSeconds: 5,
      format: '16:9',
    });
    assert.equal(scenes.length, 5);
    assert.ok(scenes.every((item) => item.description.length >= 4));

    const sampleToken = 'ghp_testtoken';
    const remote = buildAuthedRemote('https://github.com/urdirditfurd/videowan2g.git', sampleToken);
    assert.match(remote, /^https:\/\/x-access-token:/);
    assert.equal(redactSecrets(remote).includes(sampleToken), false);
    assert.throws(() => buildAuthedRemote('https://github.com/urdirditfurd/autre.git', sampleToken));
    assert.throws(() => buildAuthedRemote('https://github.com/autre/videowan2g.git', sampleToken));
    assert.throws(() => buildAuthedRemote('https://example.com/urdirditfurd/videowan2g.git', sampleToken));
    assert.throws(() => buildAuthedRemote('https://github.com/urdirditfurd/videowan2g.git?token=secret', sampleToken));
    assert.equal(new URL(remote).search, '');

    assert.doesNotThrow(() => assertSafeDownloadUrl('https://cdn.example.com/clip.mp4'));
    assert.throws(() => assertSafeDownloadUrl('http://127.0.0.1/clip.mp4'));
    assert.throws(() => assertSafeDownloadUrl('https://169.254.169.254/latest/meta-data'));
    assert.throws(() => assertSafeDownloadUrl('https://localhost/clip.mp4'));
    assert.throws(() => assertSafeDownloadUrl('http://192.168.1.20/clip.mp4'));
    assert.throws(() => assertVideoMagic(Buffer.alloc(12, 1)));
    const ftypHeader = Buffer.alloc(12);
    ftypHeader.write('ftyp', 4, 'ascii');
    assert.doesNotThrow(() => assertVideoMagic(ftypHeader));

    assert.doesNotThrow(() => assertPublishableVideo('generated_videos/Maison_japonaise_1700000000000.mp4'));
    assert.throws(() => assertPublishableVideo('.env'));
    assert.throws(() => assertPublishableVideo('generated_videos/../.env'));
    assert.throws(() => assertPublishableVideo('generated_videos/clip.txt'));
    assert.throws(() => assertGitHubFileSize(100));
    assert.throws(() => assertGitHubFileSize(GITHUB_MAX_FILE_BYTES));
    assert.doesNotThrow(() => assertGitHubFileSize(2048));

    const previousAgnes = process.env.AGNES_API_KEY;
    const previousGithub = process.env.GITHUB_TOKEN;
    process.env.AGNES_API_KEY = 'sk-selftestkeyvalue';
    process.env.GITHUB_TOKEN = 'ghp_selftesttokenvalue';
    const previousSsh = process.env.GIT_SSH_COMMAND;
    process.env.GIT_SSH_COMMAND = 'touch /tmp/lofi-should-not-run';
    try {
      const cleaned = gitProcessEnv({
        GIT_AUTHOR_NAME: 'Lofi Capybara Bot',
        LEAKED: 'prefix ghp_selftesttokenvalue suffix',
      });
      assert.equal(cleaned.AGNES_API_KEY, undefined);
      assert.equal(cleaned.GITHUB_TOKEN, undefined);
      assert.equal(cleaned.GIT_SSH_COMMAND, undefined);
      assert.equal(cleaned.LEAKED, undefined);
      assert.equal(cleaned.GIT_AUTHOR_NAME, 'Lofi Capybara Bot');
      assert.equal(typeof cleaned.PATH, 'string');
      assert.equal(Object.values(cleaned).some((value) => String(value).includes('sk-selftestkeyvalue')), false);
    } finally {
      if (previousSsh === undefined) delete process.env.GIT_SSH_COMMAND;
      else process.env.GIT_SSH_COMMAND = previousSsh;
      if (previousAgnes === undefined) delete process.env.AGNES_API_KEY;
      else process.env.AGNES_API_KEY = previousAgnes;
      if (previousGithub === undefined) delete process.env.GITHUB_TOKEN;
      else process.env.GITHUB_TOKEN = previousGithub;
    }

    const starts = [];
    let active = 0;
    let maxActive = 0;
    const queue = createRateLimitedQueue(1, 40);
    await Promise.all([0, 1, 2].map(() => queue.add(async () => {
      active += 1;
      maxActive = Math.max(maxActive, active);
      starts.push(Date.now());
      await sleep(10);
      active -= 1;
    })));
    assert.equal(maxActive, 1);
    assert.ok(starts[1] - starts[0] >= 35);
    assert.ok(starts[2] - starts[1] >= 35);

    const queueWaits = [];
    let queueCalls = 0;
    const queued = await apiFetch('https://example.test/videos', { method: 'POST', body: '{}' }, 'Vidéo', 'video', {
      maxAttempts: 2,
      queueFullMax: 5,
      sleep: async (ms) => {
        queueWaits.push(ms);
      },
      queueBackoffSeconds: (index) => [60, 120, 300][index],
      fetchImpl: async () => {
        queueCalls += 1;
        if (queueCalls < 4) {
          return jsonResponse({ status: 503, body: { error: { code: 'video_queue_full', message: 'queue is full' } } });
        }
        return jsonResponse({ status: 200, body: { video_id: 'vid_ok' } });
      },
    });
    assert.equal(queued.ok, true);
    assert.deepEqual(queueWaits, [60000, 120000, 300000]);

    const rateWaits = [];
    let rateCalls = 0;
    await apiFetch('https://example.test/videos', { method: 'POST', body: '{}' }, 'Vidéo', 'video', {
      maxAttempts: 4,
      sleep: async (ms) => {
        rateWaits.push(ms);
      },
      fetchImpl: async () => {
        rateCalls += 1;
        if (rateCalls < 3) return jsonResponse({ status: 429, body: 'rate limit' });
        return jsonResponse({ status: 200, body: { ok: true } });
      },
    });
    assert.deepEqual(rateWaits, [60000, 68000]);

    const mixedWaits = [];
    let mixedCalls = 0;
    await apiFetch('https://example.test/videos', { method: 'POST', body: '{}' }, 'Vidéo', 'video', {
      maxAttempts: 2,
      queueFullMax: 5,
      sleep: async (ms) => {
        mixedWaits.push(ms);
      },
      queueBackoffSeconds: (index) => [60, 120, 300][index],
      fetchImpl: async () => {
        mixedCalls += 1;
        if (mixedCalls < 3) return jsonResponse({ status: 429, body: { error: 'video_queue_full' } });
        return jsonResponse({ status: 200, body: { ok: true } });
      },
    });
    assert.deepEqual(mixedWaits, [60000, 120000]);

    await assert.rejects(
      () => apiFetch('https://example.test/chat', { method: 'POST' }, 'Expansion', 'text', {
        sleep: async () => {
          throw new Error('ne doit pas attendre');
        },
        fetchImpl: async () => jsonResponse({ status: 401, body: 'invalid key' }),
      }),
      /HTTP 401/,
    );

    let attempts = 0;
    const retried = await retryTask('Image', async () => {
      attempts += 1;
      if (attempts < 3) throw new Error('temporaire');
      return 'ok';
    }, 3, { sleep: async () => {} });
    assert.equal(retried, 'ok');
    await assert.rejects(
      () => retryTask('Image', async () => {
        throw new FatalApiError('HTTP 401 — invalid key');
      }, 3, {
        sleep: async () => {
          throw new Error('ne doit pas reprendre');
        },
      }),
      /HTTP 401/,
    );

    const tempVideo = path.join(os.tmpdir(), `lofi-self-test-${Date.now()}.mp4`);
    const fakeMp4 = Buffer.alloc(2048, 1);
    fakeMp4.writeUInt32BE(32, 0);
    fakeMp4.write('ftypisom', 4, 'ascii');
    await downloadToFile('https://example.test/video.mp4', tempVideo, {
      fetchImpl: async () => ({
        ok: true,
        status: 200,
        headers: { get: (name) => (name.toLowerCase() === 'content-type' ? 'video/mp4' : null) },
        body: Readable.from([fakeMp4]),
      }),
    });
    const stat = await fs.stat(tempVideo);
    assert.ok(stat.size >= 1024);
    await fs.rm(tempVideo, { force: true });

    const redirectCalls = [];
    const redirected = path.join(os.tmpdir(), `lofi-self-test-redirect-${Date.now()}.mp4`);
    await downloadToFile('https://example.test/start', redirected, {
      fetchImpl: async (target) => {
        redirectCalls.push(target);
        if (target.endsWith('/start')) {
          return {
            ok: false,
            status: 302,
            url: target,
            headers: { get: (name) => (name.toLowerCase() === 'location' ? 'https://cdn.example.com/final.mp4' : null) },
            body: Readable.from([]),
          };
        }
        return {
          ok: true,
          status: 200,
          url: target,
          headers: { get: (name) => (name.toLowerCase() === 'content-type' ? 'video/mp4' : null) },
          body: Readable.from([fakeMp4]),
        };
      },
    });
    assert.deepEqual(redirectCalls, ['https://example.test/start', 'https://cdn.example.com/final.mp4']);
    await fs.rm(redirected, { force: true });
    await assert.rejects(
      () => downloadToFile('https://example.test/start', path.join(os.tmpdir(), 'lofi-should-not-exist.mp4'), {
        fetchImpl: async () => ({
          ok: false,
          status: 302,
          headers: { get: (name) => (name.toLowerCase() === 'location' ? 'http://127.0.0.1/secret.mp4' : null) },
          body: Readable.from([]),
        }),
      }),
      /URL vidéo refusée/,
    );

    const smokeDir = await fs.mkdtemp(path.join(os.tmpdir(), 'lofi-git-'));
    const bareDir = await fs.mkdtemp(path.join(os.tmpdir(), 'lofi-git-bare-'));
    const binDir = await fs.mkdtemp(path.join(os.tmpdir(), 'lofi-git-bin-'));
    const dumpFile = path.join(os.tmpdir(), `lofi-git-env-${Date.now()}.txt`);
    const realGit = await resolveGitBinary(process.env.PATH || '');
    const wrapper = `#!/usr/bin/env node
const fs = require('node:fs');
const { spawnSync } = require('node:child_process');
const present = ['AGNES_API_KEY', 'GITHUB_TOKEN'].filter((key) => process.env[key]);
fs.writeFileSync(${JSON.stringify(dumpFile)}, present.join('\\n'));
const result = spawnSync(${JSON.stringify(realGit)}, process.argv.slice(2), { stdio: 'inherit', env: process.env });
process.exit(result.status === null ? 1 : result.status);
`;
    await fs.writeFile(path.join(binDir, 'git'), wrapper, { mode: 0o755 });
    const previousPath = process.env.PATH;
    const leakAgnes = process.env.AGNES_API_KEY;
    const leakGithub = process.env.GITHUB_TOKEN;
    process.env.PATH = `${binDir}${path.delimiter}${previousPath || ''}`;
    process.env.AGNES_API_KEY = 'sk-selftestkeyvalue';
    process.env.GITHUB_TOKEN = 'ghp_selftesttokenvalue';
    try {
      assert.equal(process.env.AGNES_API_KEY, 'sk-selftestkeyvalue');
      assert.equal(process.env.GITHUB_TOKEN, 'ghp_selftesttokenvalue');
      const guarded = createGit(smokeDir, {
        GIT_AUTHOR_NAME: 'Lofi Capybara Bot',
        GIT_AUTHOR_EMAIL: 'lofi-capybara-bot@users.noreply.github.com',
        GIT_COMMITTER_NAME: 'Lofi Capybara Bot',
        GIT_COMMITTER_EMAIL: 'lofi-capybara-bot@users.noreply.github.com',
      });
      await guarded.init();
      const leaked = await fs.readFile(dumpFile, 'utf8');
      assert.equal(
        leaked,
        '',
        'le processus git a reçu AGNES_API_KEY ou GITHUB_TOKEN',
      );
    } finally {
      if (previousPath === undefined) delete process.env.PATH;
      else process.env.PATH = previousPath;
      if (leakAgnes === undefined) delete process.env.AGNES_API_KEY;
      else process.env.AGNES_API_KEY = leakAgnes;
      if (leakGithub === undefined) delete process.env.GITHUB_TOKEN;
      else process.env.GITHUB_TOKEN = leakGithub;
      await fs.rm(binDir, { recursive: true, force: true });
      await fs.rm(dumpFile, { force: true });
    }
    try {
      const git = createGit(smokeDir, {
        GIT_AUTHOR_NAME: 'Lofi Capybara Bot',
        GIT_AUTHOR_EMAIL: 'lofi-capybara-bot@users.noreply.github.com',
        GIT_COMMITTER_NAME: 'Lofi Capybara Bot',
        GIT_COMMITTER_EMAIL: 'lofi-capybara-bot@users.noreply.github.com',
      });
      await git.init();
      await fs.writeFile(path.join(smokeDir, 'note.txt'), 'ok\n');
      await git.add(['note.txt']);
      await git.raw(['commit', '-m', 'feat: add new lofi capybara video: test', '--', 'note.txt']);
      const history = await git.log();
      assert.equal(history.latest.message, 'feat: add new lofi capybara video: test');
      assert.equal(history.latest.author_name, 'Lofi Capybara Bot');
      const bare = createGit(bareDir);
      await bare.init(['--bare']);
      await git.raw(['push', bareDir, 'HEAD:refs/heads/main']);
    } finally {
      await fs.rm(smokeDir, { recursive: true, force: true });
      await fs.rm(bareDir, { recursive: true, force: true });
    }

    console.log('Auto-test réussi.');
  } finally {
    logSink = previousSink;
    restorePacing(savedPacing);
    state.stopRequested = savedStop;
  }
}

async function resolveGitBinary(searchPath) {
  for (const dir of searchPath.split(path.delimiter).filter(Boolean)) {
    const candidate = path.join(dir, 'git');
    try {
      await fs.access(candidate, fsConstants.X_OK);
      return candidate;
    } catch {
      // essai du dossier suivant
    }
  }
  throw new Error('binaire git introuvable');
}

function isDirectExecution() {
  const entry = process.argv[1];
  if (!entry) return false;
  return path.resolve(entry) === fileURLToPath(import.meta.url);
}

if (isDirectExecution()) {
  main().catch((error) => {
    console.error(redactSecrets(error.stack || error.message));
    process.exitCode = 1;
  });
}
