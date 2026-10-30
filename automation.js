'use strict';

const crypto = require('crypto');
const dns = require('dns').promises;
const fs = require('fs');
const net = require('net');
const os = require('os');
const path = require('path');
const { spawnSync } = require('child_process');
const fetch = require('node-fetch');
const dotenv = require('dotenv');
const { simpleGit } = require('simple-git');

const API_BASE = 'https://apihub.agnes-ai.com/v1';
const POLL_BASE = 'https://apihub.agnes-ai.com/agnesapi';
const MODEL_TEXT = 'agnes-2.5-flash';
const MODEL_IMAGE = 'agnes-image-2.5-flash';
const MODEL_VIDEO = 'agnes-video-2.5-flash';
const ALLOWED_REPO = 'https://github.com/urdirditfurd/videowan2g';
const REPO_ROOT = __dirname;
const MAX_VIDEO_BYTES = 100 * 1024 * 1024;
const VIDEO_QUEUE_MAX_RETRIES = 12;
const VIDEO_QUEUE_BACKOFF_SECONDS = [60, 120, 300];
const VIDEO_QUEUE_WAIT_CAP_SECONDS = 360;
const IMAGE_PARALLEL = 1;
const IMAGE_INTERVAL_MS = 3400;
const IMAGE_INTERVAL_MIN = 3200;
const IMAGE_INTERVAL_MAX = 12000;
const VIDEO_INTERVAL_MIN = 62000;
const VIDEO_INTERVAL_MAX = 90000;
const VIDEO_INTERVAL_SAFE = 75000;
const VIDEO_FIRST_POLL_DELAY_SEC = 80;
const VIDEO_MAX_POLL_ATTEMPTS = 100;
const ALLOWED_GIT_ENV = [
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
  'pale dawn',
  'misty morning',
  'soft noon',
  'golden late afternoon',
  'blue hour',
  'quiet moonlit night',
];
const SEASONS = ['early spring', 'humid summer', 'late autumn', 'gentle winter'];
const WEATHERS = [
  'fine drizzle',
  'clear still air',
  'slow moving clouds',
  'light snow',
  'the moment just after rain',
  'a barely-there breeze',
];

const BLOCKED_HOSTS = new Set([
  'localhost',
  'localhost.localdomain',
  'metadata',
  'metadata.google',
  'metadata.google.com',
  'metadata.google.internal',
]);

const state = {
  stopRequested: false,
  durationSeconds: 5,
  format: '16:9',
  mood: 'rainy_forest',
};

const timingSamples = { video: [] };
let createImageIntervalMs = IMAGE_INTERVAL_MS;
let createVideoIntervalMs = VIDEO_INTERVAL_SAFE;
let imageAttemptsSinceLastError = 0;
let videoAttemptsSinceLastError = 0;
let quietLogs = false;
let outputDir = path.join(REPO_ROOT, 'generated_videos');
let fetchImpl = fetch;
let randomImpl = Math.random;

function sleep(ms) {
  return new Promise((resolve) => {
    setTimeout(resolve, ms);
  });
}

let sleepImpl = sleep;

async function defaultDnsLookup(hostname) {
  return dns.lookup(hostname, { all: true, verbatim: true });
}

let dnsLookupImpl = defaultDnsLookup;

class TerminalHttpError extends Error {
  constructor(message) {
    super(message);
    this.name = 'TerminalHttpError';
  }
}

class RejectedDownloadError extends Error {
  constructor(message) {
    super(message);
    this.name = 'RejectedDownloadError';
  }
}

function log(level, category, message) {
  if (quietLogs) return;
  const stamp = new Date().toISOString();
  console.log(stamp + ' [' + level + '] [' + category + '] ' + message);
}

function safeErrorMessage(err) {
  const raw = err && err.message ? String(err.message) : 'erreur inconnue';
  return raw
    .replace(/ghp_[A-Za-z0-9]+/g, '[jeton]')
    .replace(/github_pat_[A-Za-z0-9_]+/g, '[jeton]')
    .replace(/sk-[A-Za-z0-9_-]+/g, '[cle]')
    .slice(0, 500);
}

function pick(list, random) {
  const roll = random();
  const index = Math.min(list.length - 1, Math.max(0, Math.floor(roll * list.length)));
  return list[index];
}

function sanitizeTitle(title) {
  const cleaned = String(title || '')
    .normalize('NFKD')
    .replace(/[\u0300-\u036f]/g, '')
    .replace(/[^a-zA-Z0-9]+/g, '_')
    .replace(/^_+|_+$/g, '')
    .slice(0, 60);
  return cleaned || 'ambiance_lofi';
}

function commitMessage(title) {
  const clean = String(title || 'Ambiance Lofi')
    .replace(/[\u0000-\u001f\u007f]/g, ' ')
    .replace(/\s+/g, ' ')
    .trim()
    .slice(0, 120);
  return 'feat: add new lofi capybara video: ' + (clean || 'Ambiance Lofi');
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
        result += '\\u' + code.toString(16).padStart(4, '0');
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
  return {
    title: titleMatch ? titleMatch[1].replace(/\\n/g, ' ').replace(/\\"/g, '"') : 'Ambiance Lofi',
    image_prompt: imgMatch ? imgMatch[1].replace(/\\n/g, ' ').replace(/\\"/g, '"') : '',
    video_motion_prompt: vidMatch ? vidMatch[1].replace(/\\n/g, ' ').replace(/\\"/g, '"') : '',
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
  try {
    return JSON.parse(raw);
  } catch (err) {
    void err;
  }
  try {
    return JSON.parse(escapeControlCharsInStrings(raw));
  } catch (err) {
    void err;
  }
  try {
    const end = raw.lastIndexOf('}');
    if (end > 0) return JSON.parse(escapeControlCharsInStrings(raw.slice(0, end + 1)));
  } catch (err) {
    void err;
  }
  try {
    return JSON.parse(balanceJson(escapeControlCharsInStrings(raw)));
  } catch (err) {
    void err;
  }
  const salvaged = salvageExpansion(raw);
  if (salvaged && (salvaged.image_prompt || salvaged.video_motion_prompt)) return salvaged;
  return null;
}

function buildExpansionPrompt(userDescription, mood, random) {
  const roll = typeof random === 'function' ? random : randomImpl;
  const moodInfo = MOODS[mood] || MOODS.rainy_forest;
  const timeOfDay = pick(TIMES_OF_DAY, roll);
  const season = pick(SEASONS, roll);
  const weather = pick(WEATHERS, roll);
  return [
    'You are an art director for Studio Ghibli-inspired lofi ambience videos.',
    '',
    'USER SCENE IDEA (French):',
    userDescription,
    '',
    'AMBIENCE (French mood chosen):',
    moodInfo.label + ' — ' + moodInfo.desc,
    '',
    'UNIQUE VARIATION — mandatory, even when the user idea resembles another scene:',
    '- Time of day (heure de la journée): ' + timeOfDay,
    '- Season (saison): ' + season,
    '- Weather (type de météo): ' + weather,
    'This variation must be visually obvious in light, sky, vegetation, and precipitation.',
    'Never repeat a generic default look. Each scene must feel like a distinct painting.',
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
    '- Color palette: ' + GHIBLI_PALETTE,
    '',
    'OUTPUT — strict JSON only, no markdown:',
    '{',
    '  "title": "A short poetic title in French (max 60 chars)",',
    '  "image_prompt": "Ultra-detailed English visual description for an image model. Include: setting, lighting, colors, atmosphere, the capybara (with its location and pose), the unique time of day, season and weather, Ghibli watercolor style, cinematic composition. 150-250 words.",',
    '  "video_motion_prompt": "English description of slow gentle motions for the video model: capybara breathing, leaves falling, mist drifting, water ripples, seamless loop. 60-120 words."',
    '}',
  ].join('\n');
}

function buildGhibliStyleBlock() {
  return [
    'VISUAL STYLE: Studio Ghibli / Hayao Miyazaki inspired hand-painted watercolor background. Soft painterly gradients, detailed natural textures, warm nostalgic atmosphere.',
    'COMPOSITION: cinematic, peaceful, balanced, with gentle depth. The capybara is clearly visible, cute, calm.',
    'LIGHTING: soft natural light, warm amber highlights, gentle shadow, no harsh contrast.',
    'COLOR RULE: ' + GHIBLI_PALETTE,
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
    'LOFI AMBIENCE VIDEO — ' + state.durationSeconds + ' seconds — seamless loop.',
    'Studio Ghibli inspired watercolor scene with a cute calm capybara.',
    'BASE SCENE: ' + expansion.image_prompt,
    'GENTLE MOTION: ' + expansion.video_motion_prompt,
    'ABSOLUTE MOTION RULE: SLOW, GENTLE, SUBTLE motion only. No fast movement, no camera shake, no action. The scene must feel peaceful and loopable.',
    'Perfect seamless loop, calm pacing.',
    'CAMERA: locked-off or extremely slow drift. No cuts.',
    'STYLE: hand-painted Ghibli watercolor, soft gradients, warm nostalgic light.',
    'PALETTE: ' + GHIBLI_PALETTE,
    'SOUND: soft ambient nature sounds only — no music, no voice. Wind, birds, water, rustling leaves.',
    'NO readable text, NO watermark, NO logos.',
  ].join('\n');
}

function messageText(content) {
  if (typeof content === 'string') return content;
  if (!Array.isArray(content)) return '';
  return content.map((part) => {
    if (typeof part === 'string') return part;
    if (part && typeof part.text === 'string') return part.text;
    return '';
  }).join('\n');
}

function getApiKey() {
  const key = String(process.env.AGNES_API_KEY || '').trim();
  if (!key || /[\r\n\0]/.test(key)) throw new Error('AGNES_API_KEY manquante');
  return key;
}

function authHeaders(json) {
  const headers = { Authorization: 'Bearer ' + getApiKey() };
  if (json) headers['Content-Type'] = 'application/json';
  return headers;
}

function maxAttemptsFor(kind) {
  switch (kind) {
    case 'video':
      return 10;
    case 'image':
      return 10;
    case 'text':
      return 8;
    default: {
      const unexpected = kind;
      throw new Error('Type de requête inconnu: ' + String(unexpected));
    }
  }
}

function queueFullWaitSeconds(index) {
  const slot = Math.min(Math.max(index, 0), VIDEO_QUEUE_BACKOFF_SECONDS.length - 1);
  const base = VIDEO_QUEUE_BACKOFF_SECONDS[slot];
  const jitterSpan = Math.floor(base * 0.1);
  const jitter = jitterSpan > 0 ? Math.floor(randomImpl() * jitterSpan) : 0;
  return Math.min(VIDEO_QUEUE_WAIT_CAP_SECONDS, base + jitter);
}

function noteSuccess(kind) {
  if (kind === 'image') {
    imageAttemptsSinceLastError += 1;
    if (imageAttemptsSinceLastError >= 4 && createImageIntervalMs > IMAGE_INTERVAL_MIN) {
      createImageIntervalMs = Math.max(IMAGE_INTERVAL_MIN, createImageIntervalMs - 250);
      imageAttemptsSinceLastError = 0;
    }
    return;
  }
  if (kind === 'video') {
    videoAttemptsSinceLastError += 1;
    if (videoAttemptsSinceLastError >= 2 && createVideoIntervalMs > VIDEO_INTERVAL_MIN) {
      createVideoIntervalMs = Math.max(VIDEO_INTERVAL_MIN, createVideoIntervalMs - 3000);
      videoAttemptsSinceLastError = 0;
    }
  }
}

function rateLimitWait(kind, attempt, retryAfter) {
  if (retryAfter > 0) return retryAfter;
  if (kind === 'video') return Math.min(180, 60 + attempt * 8);
  if (kind === 'image') return Math.min(120, 10 + attempt * 12);
  return Math.min(120, 10 + attempt * 12);
}

function serverWait(kind, attempt) {
  if (kind === 'video') return Math.min(180, 20 + attempt * 10);
  return Math.min(60, 5 + attempt * 5);
}

function networkWait(kind, attempt) {
  if (kind === 'video') return Math.min(120, 10 + attempt * 8);
  return Math.min(45, 3 + attempt * 4);
}

function isRedirectRefusal(err) {
  const msg = err && err.message ? String(err.message) : '';
  return (err && err.type === 'no-redirect') || /redirect mode is set to error/i.test(msg);
}

async function readErrorText(res, kind) {
  const watched = new Set([400, 401, 404, 429, 500, 502, 503, 504, 520]);
  if (res.ok || (!watched.has(res.status) && kind !== 'video')) return '';
  try {
    return (await res.clone().text()).slice(0, 1200);
  } catch (err) {
    void err;
    return '';
  }
}

async function attemptOnce(url, options, kind, attempt) {
  try {
    const res = await fetchImpl(url, {
      method: options.method || 'GET',
      headers: options.headers || {},
      body: options.body,
      redirect: 'error',
      timeout: options.timeout || 120000,
    });
    const errorText = await readErrorText(res, kind);
    if (res.ok) {
      noteSuccess(kind);
      return { type: 'return', res: res };
    }
    const queueFull = kind === 'video' && /video_queue_full|queue is full/i.test(errorText);
    if (queueFull) {
      videoAttemptsSinceLastError = 0;
      return { type: 'queue' };
    }
    if ([400, 401, 404].includes(res.status) && kind !== 'video') {
      return {
        type: 'terminal',
        error: new TerminalHttpError('HTTP ' + res.status + ' — ' + errorText.slice(0, 300)),
      };
    }
    if (res.status === 429) {
      let retryAfter = 0;
      try {
        retryAfter = parseInt(res.headers.get('retry-after') || '0', 10) || 0;
      } catch (err) {
        void err;
      }
      if (kind === 'image') {
        createImageIntervalMs = Math.min(IMAGE_INTERVAL_MAX, createImageIntervalMs + 1000);
      }
      if (kind === 'video') {
        createVideoIntervalMs = Math.min(VIDEO_INTERVAL_MAX, createVideoIntervalMs + 6000);
      }
      return {
        type: 'retry',
        category: 'RATELIMIT',
        message: '429',
        waitSec: rateLimitWait(kind, attempt, retryAfter),
      };
    }
    if ([500, 502, 503, 504, 520].includes(res.status)) {
      return {
        type: 'retry',
        category: 'SERVICE',
        message: 'HTTP ' + res.status,
        waitSec: serverWait(kind, attempt),
      };
    }
    return { type: 'return', res: res };
  } catch (err) {
    if (err && err.message === 'Arrêt demandé') {
      return { type: 'terminal', error: err };
    }
    if (isRedirectRefusal(err)) {
      return { type: 'terminal', error: new TerminalHttpError('Redirection API refusée') };
    }
    return {
      type: 'retry',
      category: 'NETWORK',
      message: safeErrorMessage(err),
      waitSec: networkWait(kind, attempt),
    };
  }
}

async function apiFetch(url, options, label, kind) {
  const maxAttempts = maxAttemptsFor(kind);
  let attempt = 0;
  let queueRetries = 0;
  while (attempt < maxAttempts) {
    if (state.stopRequested) throw new Error('Arrêt demandé');
    log('api', label, (options.method || 'GET') + ' ' + url + ' (tentative ' + (attempt + 1) + '/' + maxAttempts + ')');
    const outcome = await attemptOnce(url, options || {}, kind, attempt);
    if (outcome.type === 'return') return outcome.res;
    if (outcome.type === 'terminal') throw outcome.error;
    if (outcome.type === 'queue') {
      queueRetries += 1;
      if (queueRetries > VIDEO_QUEUE_MAX_RETRIES) {
        throw new TerminalHttpError('File vidéo saturée après ' + VIDEO_QUEUE_MAX_RETRIES + ' attentes');
      }
      const waitSec = queueFullWaitSeconds(queueRetries - 1);
      log('warn', 'VIDEO_QUEUE', 'File saturée — pause ' + waitSec + 's (' + queueRetries + '/' + VIDEO_QUEUE_MAX_RETRIES + ', hors budget)');
      await sleepImpl(waitSec * 1000);
      continue;
    }
    if (outcome.type === 'retry') {
      attempt += 1;
      log('warn', outcome.category, outcome.message + ' — reprise dans ' + outcome.waitSec + 's');
      if (attempt >= maxAttempts) break;
      await sleepImpl(outcome.waitSec * 1000);
      continue;
    }
    const unexpected = outcome.type;
    throw new Error('Issue apiFetch inattendue: ' + String(unexpected));
  }
  throw new Error('Tentatives épuisées pour ' + label + ' après ' + maxAttempts + ' essais');
}

async function retryTask(name, fn, maxAttempts) {
  let lastError = null;
  for (let i = 1; i <= maxAttempts; i += 1) {
    if (state.stopRequested) throw new Error('Arrêt demandé');
    try {
      log('info', 'RETRY', name + ' — essai ' + i + '/' + maxAttempts);
      return await fn();
    } catch (err) {
      lastError = err;
      if (err && err.message === 'Arrêt demandé') throw err;
      if (err instanceof RejectedDownloadError) throw err;
      log('warn', 'RETRY', name + ' échoué : ' + safeErrorMessage(err));
      if (i < maxAttempts) {
        const waitSec = Math.min(90, 8 * i * i);
        await sleepImpl(waitSec * 1000);
      }
    }
  }
  throw lastError || new Error(name + ' échoué');
}

function createRateLimitedQueue(maxConcurrent, minIntervalMs) {
  const intervalOf = typeof minIntervalMs === 'function' ? minIntervalMs : function interval() {
    return minIntervalMs;
  };
  let running = 0;
  let lastStart = 0;
  const queue = [];
  let resolveIdle = null;
  let timer = null;

  function tryNext() {
    if (timer) {
      clearTimeout(timer);
      timer = null;
    }
    if (running >= maxConcurrent) return;
    if (queue.length === 0) {
      if (running === 0 && resolveIdle) {
        const done = resolveIdle;
        resolveIdle = null;
        done();
      }
      return;
    }
    const now = Date.now();
    const wait = Math.max(0, lastStart + intervalOf() - now);
    if (wait > 0) {
      timer = setTimeout(tryNext, wait);
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

const imageQueue = createRateLimitedQueue(IMAGE_PARALLEL, () => createImageIntervalMs);
const videoQueue = createRateLimitedQueue(1, () => createVideoIntervalMs);

async function generateExpansion(userDescription, mood) {
  const prompt = buildExpansionPrompt(userDescription, mood);
  const body = {
    model: MODEL_TEXT,
    messages: [
      {
        role: 'system',
        content: 'You are a Studio Ghibli-inspired art director specialized in lofi ambience videos. You always return strict JSON matching the requested schema exactly. Every scene includes a cute calm capybara. Every scene is peaceful, detailed, watercolor, and loopable with slow gentle motion. Every scene is visually unique. No text, no watermark. Return only JSON.',
      },
      { role: 'user', content: prompt },
    ],
    temperature: 0.85,
    max_tokens: 2000,
    response_format: { type: 'json_object' },
  };
  const res = await apiFetch(API_BASE + '/chat/completions', {
    method: 'POST',
    headers: authHeaders(true),
    body: JSON.stringify(body),
  }, 'Expansion lofi', 'text');
  if (!res.ok) {
    const errText = await res.text();
    throw new Error('HTTP ' + res.status + ' — ' + errText.slice(0, 300));
  }
  const data = await res.json();
  const content = messageText(data.choices && data.choices[0] && data.choices[0].message && data.choices[0].message.content);
  const expansion = extractJson(content);
  if (!expansion || !expansion.image_prompt) throw new Error('Expansion invalide');
  if (!expansion.title) expansion.title = 'Ambiance Lofi';
  if (!expansion.video_motion_prompt) {
    expansion.video_motion_prompt = 'Gentle slow motion: capybara breathing softly, leaves drifting down, faint mist, seamless loop.';
  }
  log('success', 'EXPANSION', 'Titre : ' + String(expansion.title).slice(0, 80));
  return expansion;
}

async function generateImage(prompt, format) {
  return retryTask('Image', async () => {
    const ratio = format === '9:16' ? '9:16' : '16:9';
    const body = { model: MODEL_IMAGE, prompt: prompt, size: '1K', ratio: ratio, n: 1 };
    const res = await apiFetch(API_BASE + '/images/generations', {
      method: 'POST',
      headers: authHeaders(true),
      body: JSON.stringify(body),
    }, 'Image', 'image');
    if (!res.ok) {
      const errText = await res.text();
      throw new Error('HTTP ' + res.status + ' — ' + errText.slice(0, 300));
    }
    const data = await res.json();
    const item = data.data && data.data[0];
    if (!item) throw new Error('Pas de donnée image');
    if (item.url) return item.url;
    if (item.b64_json) return 'data:image/png;base64,' + item.b64_json;
    throw new Error('Format image inconnu');
  }, 4);
}

async function generateVideo(imageUrl, prompt) {
  return retryTask('Vidéo', async () => {
    const aspect = state.format === '9:16' ? '9:16' : '16:9';
    const body = {
      model: MODEL_VIDEO,
      prompt: prompt,
      seconds: String(state.durationSeconds),
      mode: 'keyframe',
      size: '720P',
      aspect_ratio: aspect,
      first_frame: imageUrl,
      last_frame: imageUrl,
    };
    const res = await apiFetch(API_BASE + '/videos', {
      method: 'POST',
      headers: authHeaders(true),
      body: JSON.stringify(body),
    }, 'Vidéo', 'video');
    if (!res.ok) {
      const errText = await res.text();
      throw new Error('HTTP ' + res.status + ' — ' + errText.slice(0, 300));
    }
    const data = await res.json();
    const videoId = data.video_id || data.id || data.task_id;
    if (!videoId) throw new Error('Pas de video_id');
    return String(videoId);
  }, 3);
}

function estimateGenerationTime(key, fallbackMs) {
  const samples = timingSamples[key];
  if (!samples || samples.length === 0) return fallbackMs;
  const sorted = samples.slice().sort((a, b) => a - b);
  const mid = Math.floor(sorted.length / 2);
  if (sorted.length % 2 === 0) return (sorted[mid - 1] + sorted[mid]) / 2;
  return sorted[mid];
}

function recordGenerationTime(key, ms) {
  if (!timingSamples[key]) timingSamples[key] = [];
  timingSamples[key].push(ms);
  if (timingSamples[key].length > 10) timingSamples[key] = timingSamples[key].slice(-10);
}

async function pollVideo(videoId, onProgress) {
  if (!/^[A-Za-z0-9._:-]{1,128}$/.test(String(videoId || ''))) {
    throw new Error('Identifiant vidéo invalide');
  }
  const estimatedMs = estimateGenerationTime('video', 90000 + state.durationSeconds * 7000);
  const minWait = Math.max(25, Math.min(VIDEO_FIRST_POLL_DELAY_SEC, Math.floor(estimatedMs / 1000 * 0.35)));
  let remaining = minWait;
  log('info', 'POLLING', 'Attente initiale ' + remaining + 's pour ' + videoId);
  while (remaining > 0) {
    if (state.stopRequested) throw new Error('Arrêt demandé');
    onProgress('File vidéo — ' + remaining + 's');
    await sleepImpl(1000);
    remaining -= 1;
  }
  let attempt = 0;
  while (attempt < VIDEO_MAX_POLL_ATTEMPTS) {
    if (state.stopRequested) throw new Error('Arrêt demandé');
    attempt += 1;
    const url = POLL_BASE + '?video_id=' + encodeURIComponent(videoId) + '&model_name=' + encodeURIComponent(MODEL_VIDEO);
    let res;
    try {
      res = await apiFetch(url, { method: 'GET', headers: authHeaders(false) }, 'Polling', 'video');
    } catch (err) {
      if (err && err.message === 'Arrêt demandé') throw err;
      log('warn', 'POLLING', safeErrorMessage(err));
      await sleepImpl(Math.min(90000, 15000 + attempt * 1500));
      continue;
    }
    let data;
    try {
      data = await res.json();
    } catch (err) {
      void err;
      await sleepImpl(Math.min(60000, 10000 + attempt * 1000));
      continue;
    }
    const status = String(data.status || 'unknown').toLowerCase();
    const progress = Number(data.progress || 0);
    onProgress('Création — ' + progress + '%');
    if (status === 'completed' || status === 'succeeded' || status === 'done') {
      const videoUrl = data.url || (data.metadata && data.metadata.url) || (data.output && data.output.url);
      if (!videoUrl) throw new Error("Terminé mais pas d'URL");
      return String(videoUrl);
    }
    if (status === 'failed' || status === 'error' || status === 'cancelled') {
      throw new Error('Échec définitif (' + status + ')');
    }
    const waitSec = attempt < 5 ? 8 : Math.min(30, 10 + Math.floor(attempt / 5) * 3);
    onProgress('File vidéo — nouvelle vérification dans ' + waitSec + 's');
    await sleepImpl(waitSec * 1000);
  }
  throw new Error('Délai maximal de polling dépassé');
}

function normalizeHostname(hostname) {
  let host = String(hostname || '').trim().toLowerCase();
  if (host.startsWith('[') && host.endsWith(']')) host = host.slice(1, -1);
  if (host.endsWith('.')) host = host.slice(0, -1);
  return host;
}

function extractMappedIpv4(ipv6) {
  const host = ipv6.toLowerCase();
  const dotted = host.match(/^::ffff:(\d{1,3}(?:\.\d{1,3}){3})$/);
  if (dotted) return dotted[1];
  const hex = host.match(/^::ffff:([0-9a-f]{1,4}):([0-9a-f]{1,4})$/);
  if (!hex) return null;
  const hi = parseInt(hex[1], 16);
  const lo = parseInt(hex[2], 16);
  if (!Number.isFinite(hi) || !Number.isFinite(lo)) return null;
  return [(hi >> 8) & 255, hi & 255, (lo >> 8) & 255, lo & 255].join('.');
}

function isPrivateIpv4(host) {
  const parts = host.split('.').map((part) => Number(part));
  if (parts.length !== 4 || parts.some((part) => !Number.isInteger(part) || part < 0 || part > 255)) return false;
  const a = parts[0];
  const b = parts[1];
  const c = parts[2];
  if (a === 0 || a === 10 || a === 127) return true;
  if (a === 169 && b === 254) return true;
  if (a === 172 && b >= 16 && b <= 31) return true;
  if (a === 192 && b === 168) return true;
  if (a === 100 && b >= 64 && b <= 127) return true;
  if (a === 192 && b === 0 && (c === 0 || c === 2)) return true;
  if (a === 198 && (b === 18 || b === 19)) return true;
  if (a === 198 && b === 51 && c === 100) return true;
  if (a === 203 && b === 0 && c === 113) return true;
  if (a >= 224) return true;
  return false;
}

function isBlockedIp(host) {
  const normalized = normalizeHostname(host);
  const kind = net.isIP(normalized);
  if (kind === 4) return isPrivateIpv4(normalized);
  if (kind !== 6) return false;
  const mapped = extractMappedIpv4(normalized);
  if (mapped) return isPrivateIpv4(mapped);
  if (normalized === '::1' || normalized === '::') return true;
  if (normalized.startsWith('fc') || normalized.startsWith('fd')) return true;
  if (
    normalized.startsWith('fe8')
    || normalized.startsWith('fe9')
    || normalized.startsWith('fea')
    || normalized.startsWith('feb')
  ) {
    return true;
  }
  return false;
}

function isBlockedHostname(hostname) {
  const host = normalizeHostname(hostname);
  if (!host) return true;
  if (BLOCKED_HOSTS.has(host)) return true;
  if (host.endsWith('.localhost') || host.endsWith('.local') || host.endsWith('.internal')) return true;
  if (host.includes('%')) return true;
  return false;
}

function assertPublicHttpsUrl(raw) {
  let url;
  try {
    url = new URL(String(raw));
  } catch (err) {
    void err;
    throw new RejectedDownloadError('URL vidéo refusée');
  }
  if (url.protocol !== 'https:' || url.username || url.password) {
    throw new RejectedDownloadError('URL vidéo refusée');
  }
  const host = normalizeHostname(url.hostname);
  if (isBlockedHostname(host) || isBlockedIp(host)) {
    throw new RejectedDownloadError('URL vidéo refusée');
  }
  return url;
}

async function assertHostnameResolvesPublic(url) {
  const host = normalizeHostname(url.hostname);
  if (net.isIP(host) !== 0) {
    if (isBlockedIp(host)) throw new RejectedDownloadError('URL vidéo refusée');
    return;
  }
  let records;
  try {
    records = await dnsLookupImpl(host);
  } catch (err) {
    void err;
    throw new RejectedDownloadError('Résolution DNS refusée');
  }
  if (!records || records.length === 0) throw new RejectedDownloadError('Résolution DNS refusée');
  for (let i = 0; i < records.length; i += 1) {
    if (isBlockedIp(records[i].address)) throw new RejectedDownloadError('Hôte résolu vers une adresse privée');
  }
}

function destroyBody(res) {
  if (res && res.body && typeof res.body.destroy === 'function') res.body.destroy();
}

function chunkToBuffer(chunk) {
  if (Buffer.isBuffer(chunk)) return chunk;
  if (typeof chunk === 'number') return Buffer.from([chunk]);
  if (chunk instanceof Uint8Array) return Buffer.from(chunk);
  throw new RejectedDownloadError('Corps de réponse illisible');
}

async function readResponseBuffer(res) {
  if (!res || res.body == null) throw new RejectedDownloadError('Réponse vide');
  if (Buffer.isBuffer(res.body)) {
    if (res.body.length >= MAX_VIDEO_BYTES) throw new RejectedDownloadError('Vidéo trop volumineuse');
    return res.body;
  }
  const chunks = [];
  let written = 0;
  for await (const chunk of res.body) {
    const buf = chunkToBuffer(chunk);
    written += buf.length;
    if (written >= MAX_VIDEO_BYTES) {
      destroyBody(res);
      throw new RejectedDownloadError('Vidéo trop volumineuse');
    }
    chunks.push(buf);
  }
  return Buffer.concat(chunks);
}

async function fetchPublic(rawUrl, hopsLeft) {
  const url = assertPublicHttpsUrl(rawUrl);
  await assertHostnameResolvesPublic(url);
  const res = await fetchImpl(url.href, {
    method: 'GET',
    redirect: 'manual',
    timeout: 180000,
    headers: { 'User-Agent': 'videowan2g-lofi/1.0' },
  });
  if ([301, 302, 303, 307, 308].includes(res.status)) {
    const location = res.headers.get('location');
    destroyBody(res);
    if (!location) throw new RejectedDownloadError('Redirection sans cible');
    if (hopsLeft <= 0) throw new RejectedDownloadError('Trop de redirections');
    const next = new URL(location, url).href;
    return fetchPublic(next, hopsLeft - 1);
  }
  return res;
}

async function ensureOutputDir(dir) {
  try {
    const st = await fs.promises.lstat(dir);
    if (st.isSymbolicLink() || !st.isDirectory()) {
      throw new RejectedDownloadError('Dossier de sortie invalide');
    }
  } catch (err) {
    if (err instanceof RejectedDownloadError) throw err;
    if (err.code !== 'ENOENT') throw err;
    await fs.promises.mkdir(dir);
  }
}

async function openExclusiveNewFile(filePath) {
  const dirStat = await fs.promises.lstat(path.dirname(filePath));
  if (dirStat.isSymbolicLink() || !dirStat.isDirectory()) {
    throw new RejectedDownloadError('Dossier de sortie invalide');
  }
  try {
    const existing = await fs.promises.lstat(filePath);
    if (existing.isSymbolicLink()) throw new RejectedDownloadError('Refus d\'écrire à travers un lien symbolique');
    throw new RejectedDownloadError('Le fichier de sortie existe déjà');
  } catch (err) {
    if (err instanceof RejectedDownloadError) throw err;
    if (err.code !== 'ENOENT') throw err;
  }
  const flags = fs.constants.O_CREAT | fs.constants.O_EXCL | fs.constants.O_NOFOLLOW | fs.constants.O_WRONLY;
  try {
    return await fs.promises.open(filePath, flags, 0o644);
  } catch (err) {
    if (err.code === 'ELOOP' || err.code === 'EEXIST') {
      throw new RejectedDownloadError('Refus d\'écraser la cible du fichier de sortie');
    }
    throw err;
  }
}

function detectContainer(buf) {
  if (buf.length >= 12 && buf.slice(4, 8).toString('ascii') === 'ftyp') return 'mp4';
  if (buf.length >= 4 && buf[0] === 0x1a && buf[1] === 0x45 && buf[2] === 0xdf && buf[3] === 0xa3) return 'webm';
  return null;
}

async function assertLocalVideo(filePath) {
  const st = await fs.promises.lstat(filePath);
  if (st.isSymbolicLink()) throw new RejectedDownloadError('Lien symbolique refusé');
  if (!st.isFile()) throw new RejectedDownloadError('Pas un fichier régulier');
  if (st.size >= MAX_VIDEO_BYTES) throw new RejectedDownloadError('Vidéo trop volumineuse');
  if (st.size < 12) throw new RejectedDownloadError('Fichier vidéo trop court');
  const fh = await fs.promises.open(filePath, fs.constants.O_RDONLY | fs.constants.O_NOFOLLOW);
  try {
    const buf = Buffer.alloc(32);
    const read = await fh.read(buf, 0, 32, 0);
    const kind = detectContainer(buf.subarray(0, read.bytesRead));
    if (!kind) throw new RejectedDownloadError('En-tête vidéo invalide');
    return { size: st.size, kind: kind };
  } finally {
    await fh.close();
  }
}

async function writeBodyToNewFile(res, filePath) {
  const declared = Number(res.headers.get('content-length'));
  if (Number.isFinite(declared) && declared >= MAX_VIDEO_BYTES) {
    destroyBody(res);
    throw new RejectedDownloadError('Vidéo trop volumineuse');
  }
  const fh = await openExclusiveNewFile(filePath);
  let payload;
  try {
    payload = await readResponseBuffer(res);
    if (payload.length >= MAX_VIDEO_BYTES) throw new RejectedDownloadError('Vidéo trop volumineuse');
    if (payload.length > 0) await fh.write(payload, 0, payload.length, 0);
  } catch (err) {
    await fh.close().catch(() => {});
    await fs.promises.unlink(filePath).catch(() => {});
    throw err;
  }
  await fh.close();
  if (payload.length < 12) {
    await fs.promises.unlink(filePath).catch(() => {});
    throw new RejectedDownloadError('Fichier vidéo trop court');
  }
}

async function renameExclusive(fromPath, toPath) {
  try {
    const existing = await fs.promises.lstat(toPath);
    if (existing.isSymbolicLink()) throw new RejectedDownloadError('Cible lien symbolique');
    throw new RejectedDownloadError('Fichier déjà présent');
  } catch (err) {
    if (err instanceof RejectedDownloadError) throw err;
    if (err.code !== 'ENOENT') throw err;
  }
  const resolved = path.resolve(toPath);
  const root = path.resolve(outputDir);
  if (resolved !== root && !resolved.startsWith(root + path.sep)) {
    throw new RejectedDownloadError('Chemin de sortie refusé');
  }
  await fs.promises.rename(fromPath, toPath);
}

function videoFileName(title, ext) {
  return sanitizeTitle(title) + '_' + Date.now() + ext;
}

async function downloadVideo(rawUrl, title) {
  await ensureOutputDir(outputDir);
  const res = await fetchPublic(rawUrl, 2);
  if (!res.ok) {
    const status = res.status;
    destroyBody(res);
    throw new Error('Téléchargement HTTP ' + status);
  }
  const partial = path.join(outputDir, '.partial-' + crypto.randomBytes(8).toString('hex'));
  try {
    await writeBodyToNewFile(res, partial);
    const info = await assertLocalVideo(partial);
    const ext = info.kind === 'webm' ? '.webm' : '.mp4';
    const finalPath = path.join(outputDir, videoFileName(title, ext));
    await renameExclusive(partial, finalPath);
    return finalPath;
  } catch (err) {
    await fs.promises.unlink(partial).catch(() => {});
    throw err;
  }
}

function assertAllowedRepoUrl(raw) {
  let url;
  try {
    url = new URL(String(raw || '').trim());
  } catch (err) {
    void err;
    throw new Error('URL de dépôt refusée');
  }
  if (url.protocol !== 'https:' || url.username || url.password || url.hostname !== 'github.com') {
    throw new Error('URL de dépôt refusée');
  }
  const pathname = url.pathname.replace(/\/+$/, '').replace(/\.git$/, '');
  if (pathname !== '/urdirditfurd/videowan2g' || url.search || url.hash) {
    throw new Error('URL de dépôt refusée');
  }
  return ALLOWED_REPO;
}

function readGitIdentity() {
  const name = String(process.env.GIT_USER_NAME || 'videowan2g').trim();
  const email = String(process.env.GIT_USER_EMAIL || 'videowan2g@users.noreply.github.com').trim();
  if (!name || !email || /[\r\n\0]/.test(name) || /[\r\n\0]/.test(email)) {
    throw new Error('Identité git invalide');
  }
  return { name: name, email: email };
}

function buildGitEnv(extras) {
  const identity = readGitIdentity();
  const env = {
    PATH: process.env.PATH || '/usr/bin:/bin',
    HOME: '',
    GIT_CONFIG_NOSYSTEM: '1',
    GIT_TERMINAL_PROMPT: '0',
    GIT_AUTHOR_NAME: identity.name,
    GIT_AUTHOR_EMAIL: identity.email,
    GIT_COMMITTER_NAME: identity.name,
    GIT_COMMITTER_EMAIL: identity.email,
  };
  const extra = extras || {};
  const extraKeys = Object.keys(extra);
  for (let i = 0; i < extraKeys.length; i += 1) {
    const key = extraKeys[i];
    if (!ALLOWED_GIT_ENV.includes(key)) throw new Error('Variable git non autorisée');
    env[key] = String(extra[key]);
  }
  const secretValues = [process.env.AGNES_API_KEY, process.env.GITHUB_TOKEN]
    .map((value) => String(value || ''))
    .filter((value) => value.length >= 8);
  const names = Object.keys(env);
  for (let i = 0; i < names.length; i += 1) {
    const key = names[i];
    if (key === 'AGNES_API_KEY' || key === 'GITHUB_TOKEN') {
      throw new Error('Environnement git refusé');
    }
    const value = String(env[key]);
    if (secretValues.some((secret) => value.includes(secret))) {
      throw new Error('Environnement git refusé');
    }
  }
  return env;
}

function createGit(options) {
  const settings = options || {};
  const git = simpleGit({
    baseDir: settings.baseDir || REPO_ROOT,
    binary: settings.binary || 'git',
    unsafe: {
      allowUnsafeAskPass: true,
      allowUnsafeCredentialHelper: true,
    },
    allowEnvironment: ALLOWED_GIT_ENV,
  });
  git.env(buildGitEnv(settings.envExtras));
  return git;
}

function repoRelative(repoRoot, filePath) {
  const rel = path.relative(repoRoot, filePath);
  if (!rel || rel.startsWith('..') || path.isAbsolute(rel)) throw new Error('Chemin vidéo hors dépôt');
  const parts = rel.split(path.sep);
  if (parts.includes('..')) throw new Error('Chemin vidéo hors dépôt');
  return parts.join('/');
}

async function withAskpass(token, fn) {
  if (typeof token !== 'string' || token.length < 8 || /[\r\n\0]/.test(token)) {
    throw new Error('Jeton GitHub invalide');
  }
  const dir = await fs.promises.mkdtemp(path.join(os.tmpdir(), 'videowan2g-askpass-'));
  const scriptPath = path.join(dir, 'askpass.cjs');
  const body = [
    '#!/usr/bin/env node',
    '"use strict";',
    'const token = ' + JSON.stringify(token) + ';',
    'const prompt = String(process.argv[2] || "");',
    'if (/username/i.test(prompt)) process.stdout.write("x-access-token\\n");',
    'else process.stdout.write(String(token) + "\\n");',
    '',
  ].join('\n');
  await fs.promises.writeFile(scriptPath, body, { mode: 0o700 });
  await fs.promises.chmod(scriptPath, 0o700);
  try {
    return await fn(scriptPath);
  } finally {
    await fs.promises.rm(dir, { recursive: true, force: true });
  }
}

async function publishVideo(filePath, title, options) {
  const settings = options || {};
  const repoRoot = settings.repoRoot || REPO_ROOT;
  await assertLocalVideo(filePath);
  const rel = repoRelative(repoRoot, filePath);
  const token = String(process.env.GITHUB_TOKEN || '').trim();
  const expectedRepo = assertAllowedRepoUrl(process.env.GITHUB_REPO_URL || '');
  const git = createGit({ baseDir: repoRoot, binary: settings.binary });
  const remoteRaw = await git.raw(['remote', 'get-url', 'origin']);
  const actual = assertAllowedRepoUrl(String(remoteRaw || ''));
  if (actual !== expectedRepo) throw new Error('URL de dépôt refusée');
  const message = commitMessage(title);
  await git.raw(['add', '-f', '--', rel]);
  await git.raw(['-c', 'commit.gpgsign=false', 'commit', '-m', message, '--', rel]);
  await withAskpass(token, async (askpass) => {
    git.env(buildGitEnv({ GIT_ASKPASS: askpass }));
    await git.raw(['-c', 'credential.helper=', 'push', 'origin', 'HEAD']);
  });
  git.env(buildGitEnv());
  log('success', 'GIT', 'Publié : ' + rel);
  return rel;
}

function resolveDuration(scene) {
  const raw = scene.durationSeconds != null ? scene.durationSeconds : process.env.VIDEO_DURATION_SECONDS;
  const parsed = parseInt(raw == null || raw === '' ? '5' : String(raw), 10);
  if (parsed === 5 || parsed === 6 || parsed === 8 || parsed === 10) return parsed;
  return 5;
}

function resolveFormat(scene) {
  const raw = scene.format || process.env.VIDEO_FORMAT || '16:9';
  return raw === '9:16' ? '9:16' : '16:9';
}

function resolveMood(scene) {
  if (scene.mood && MOODS[scene.mood]) return scene.mood;
  if (process.env.VIDEO_MOOD && MOODS[process.env.VIDEO_MOOD]) return process.env.VIDEO_MOOD;
  const keys = Object.keys(MOODS);
  return pick(keys, randomImpl);
}

function normalizeScene(entry, index) {
  if (typeof entry === 'string') {
    return { description: entry.trim(), mood: null, durationSeconds: null, format: null };
  }
  if (entry && typeof entry === 'object' && typeof entry.description === 'string') {
    return {
      description: entry.description.trim(),
      mood: typeof entry.mood === 'string' ? entry.mood : null,
      durationSeconds: entry.durationSeconds != null ? entry.durationSeconds : null,
      format: typeof entry.format === 'string' ? entry.format : null,
    };
  }
  throw new Error('Scène invalide à l\'index ' + index);
}

function loadScenes(filePath) {
  const parsed = JSON.parse(fs.readFileSync(filePath, 'utf8'));
  if (!Array.isArray(parsed) || parsed.length === 0) throw new Error('scenes.json doit être un tableau non vide');
  return parsed.map(normalizeScene).map((scene) => {
    if (scene.description.length < 4) throw new Error('Description trop courte');
    if (scene.description.length > 4000) scene.description = scene.description.slice(0, 4000);
    return scene;
  });
}

async function processScene(scene) {
  state.durationSeconds = resolveDuration(scene);
  state.format = resolveFormat(scene);
  state.mood = resolveMood(scene);
  const started = Date.now();
  log('info', 'SCENE', scene.description.slice(0, 120));
  const expansion = await generateExpansion(scene.description, state.mood);
  const imagePrompt = buildLofiImagePrompt(expansion);
  const imageUrl = await imageQueue.add(() => generateImage(imagePrompt, state.format));
  log('success', 'IMAGE', 'Image prête');
  const videoPrompt = buildLofiVideoPrompt(expansion);
  const videoId = await videoQueue.add(() => generateVideo(imageUrl, videoPrompt));
  log('success', 'VIDÉO', 'Tâche ' + videoId);
  const videoUrl = await pollVideo(videoId, (progress) => {
    log('info', 'POLL', progress);
  });
  const filePath = await retryTask('Téléchargement', () => downloadVideo(videoUrl, expansion.title), 3);
  recordGenerationTime('video', Date.now() - started);
  await publishVideo(filePath, expansion.title);
  return filePath;
}

async function runBatch() {
  const repo = assertAllowedRepoUrl(process.env.GITHUB_REPO_URL || '');
  if (repo !== ALLOWED_REPO) throw new Error('URL de dépôt refusée');
  getApiKey();
  const token = String(process.env.GITHUB_TOKEN || '').trim();
  if (!token) throw new Error('GITHUB_TOKEN manquant');
  const scenes = loadScenes(path.join(REPO_ROOT, 'scenes.json'));
  let ok = 0;
  let failed = 0;
  for (let i = 0; i < scenes.length; i += 1) {
    if (state.stopRequested) break;
    try {
      await processScene(scenes[i]);
      ok += 1;
    } catch (err) {
      failed += 1;
      log('error', 'SCENE', safeErrorMessage(err));
    }
  }
  log('info', 'FIN', ok + ' réussite(s), ' + failed + ' échec(s)');
  if (failed > 0 || ok === 0) process.exitCode = 1;
}

function resetRuntime() {
  createImageIntervalMs = IMAGE_INTERVAL_MS;
  createVideoIntervalMs = VIDEO_INTERVAL_SAFE;
  imageAttemptsSinceLastError = 0;
  videoAttemptsSinceLastError = 0;
  state.stopRequested = false;
  state.durationSeconds = 5;
  state.format = '16:9';
  state.mood = 'rainy_forest';
  timingSamples.video = [];
  fetchImpl = fetch;
  sleepImpl = sleep;
  dnsLookupImpl = defaultDnsLookup;
  randomImpl = Math.random;
  outputDir = path.join(REPO_ROOT, 'generated_videos');
}

function jsonResponse(status, body, headers) {
  return new fetch.Response(body, { status: status, headers: headers || { 'Content-Type': 'application/json' } });
}

function mp4Buffer() {
  const buf = Buffer.alloc(32);
  buf.writeUInt32BE(24, 0);
  buf.write('ftyp', 4, 4, 'ascii');
  buf.write('isom', 8, 4, 'ascii');
  return buf;
}

function webmBuffer() {
  const buf = Buffer.alloc(16);
  buf[0] = 0x1a;
  buf[1] = 0x45;
  buf[2] = 0xdf;
  buf[3] = 0xa3;
  return buf;
}

async function runSelfTest() {
  quietLogs = true;
  const failures = [];
  let checks = 0;

  async function check(name, fn) {
    checks += 1;
    resetRuntime();
    try {
      await fn();
    } catch (err) {
      failures.push(name + ' — ' + safeErrorMessage(err));
    }
  }

  await check('extractJson', async () => {
    const plain = extractJson('{"title":"Lac","image_prompt":"lake","video_motion_prompt":"mist"}');
    if (!plain || plain.title !== 'Lac') throw new Error('stratégie 1');
    const fenced = extractJson('```json\n{"title":"Brume","image_prompt":"fog","video_motion_prompt":"drift"}\n```');
    if (!fenced || fenced.image_prompt !== 'fog') throw new Error('clôture');
    const broken = extractJson('{"title":"Soir","image_prompt":"line\nbreak","video_motion_prompt":"slow"}');
    if (!broken || !broken.image_prompt.includes('line')) throw new Error('contrôles');
    const partial = extractJson('prefix {"title":"A","image_prompt":"img","video_motion_prompt":"mov"');
    if (!partial || partial.image_prompt !== 'img') throw new Error('équilibre');
    const salvaged = extractJson('oops { "image_prompt": "only image", trailing junk');
    if (!salvaged || salvaged.image_prompt !== 'only image') throw new Error('regex');
  });

  await check('prompt unique', async () => {
    const dawn = buildExpansionPrompt('maison sous la pluie', 'rainy_forest', () => 0);
    if (!dawn.includes('capybara') || !dawn.includes('Studio Ghibli')) throw new Error('capybara');
    if (!dawn.includes(TIMES_OF_DAY[0]) || !dawn.includes(SEASONS[0]) || !dawn.includes(WEATHERS[0])) {
      throw new Error('variation initiale');
    }
    if (!dawn.includes('heure de la journée') || !dawn.includes('saison') || !dawn.includes('type de météo')) {
      throw new Error('axes');
    }
    const night = buildExpansionPrompt('maison sous la pluie', 'cozy_room', () => 0.999);
    if (!night.includes(TIMES_OF_DAY[TIMES_OF_DAY.length - 1])) throw new Error('autre heure');
    if (night === dawn) throw new Error('prompts identiques');
    const title = sanitizeTitle('Café / pluie !!!');
    if (title !== 'Cafe_pluie') throw new Error('sanitize ' + title);
    if (commitMessage('Lac\nbleu').indexOf('\n') !== -1) throw new Error('message');
  });

  await check('file saturée hors budget', async () => {
    const waits = [];
    randomImpl = () => 0;
    sleepImpl = async (ms) => {
      waits.push(ms);
    };
    let calls = 0;
    fetchImpl = async () => {
      calls += 1;
      return jsonResponse(503, '{"error":"video_queue_full"}');
    };
    let message = '';
    try {
      await apiFetch('https://apihub.agnes-ai.com/v1/videos', { method: 'POST', headers: {} }, 'Vidéo', 'video');
    } catch (err) {
      message = err.message;
    }
    if (!/12 attentes/.test(message)) throw new Error('pas de plafond');
    if (calls !== VIDEO_QUEUE_MAX_RETRIES + 1) throw new Error('appels ' + calls);
    if (waits.length !== VIDEO_QUEUE_MAX_RETRIES) throw new Error('pauses ' + waits.length);
    if (waits[0] !== 60000 || waits[1] !== 120000 || waits[2] !== 300000 || waits[11] !== 300000) {
      throw new Error('backoff');
    }
  });

  await check('429 consomme le budget', async () => {
    sleepImpl = async () => {};
    let calls = 0;
    fetchImpl = async () => {
      calls += 1;
      return jsonResponse(429, 'slow down', { 'retry-after': '0' });
    };
    let message = '';
    try {
      await apiFetch('https://apihub.agnes-ai.com/v1/videos', { method: 'POST' }, 'Vidéo', 'video');
    } catch (err) {
      message = err.message;
    }
    if (calls !== 10 || !/Tentatives épuisées/.test(message)) throw new Error('budget 429');
  });

  await check('erreur terminale et corps cloné', async () => {
    let calls = 0;
    fetchImpl = async () => {
      calls += 1;
      return jsonResponse(401, 'nope-body-marker');
    };
    let message = '';
    try {
      await apiFetch('https://apihub.agnes-ai.com/v1/chat/completions', { method: 'POST' }, 'Expansion', 'text');
    } catch (err) {
      message = err.message;
    }
    if (calls !== 1 || !message.includes('nope-body-marker')) throw new Error('401 non cloné');
    fetchImpl = async () => jsonResponse(400, 'video-body-still-here');
    const res = await apiFetch('https://apihub.agnes-ai.com/v1/videos', { method: 'POST' }, 'Vidéo', 'video');
    const body = await res.text();
    if (body !== 'video-body-still-here') throw new Error('corps vidéo consommé');
    fetchImpl = async () => {
      const err = new Error('redirect mode is set to error');
      err.type = 'no-redirect';
      throw err;
    };
    calls = 0;
    fetchImpl = async () => {
      calls += 1;
      const err = new Error('redirect mode is set to error: https://evil.example');
      err.type = 'no-redirect';
      throw err;
    };
    try {
      await apiFetch('https://apihub.agnes-ai.com/v1/videos', { method: 'POST' }, 'Vidéo', 'video');
      throw new Error('redirection acceptée');
    } catch (err) {
      if (calls !== 1 || !/Redirection API refusée/.test(err.message)) throw err;
    }
  });

  await check('retryTask', async () => {
    const waits = [];
    sleepImpl = async (ms) => {
      waits.push(ms);
    };
    let tries = 0;
    const value = await retryTask('Image', async () => {
      tries += 1;
      if (tries < 3) throw new Error('transient');
      return 'ok';
    }, 3);
    if (value !== 'ok' || waits[0] !== 8000 || waits[1] !== 32000) throw new Error('attentes retry');
    try {
      await retryTask('Image', async () => {
        throw new RejectedDownloadError('bloqué');
      }, 3);
      throw new Error('sécurité retentée');
    } catch (err) {
      if (!(err instanceof RejectedDownloadError)) throw err;
    }
  });

  await check('file locale', async () => {
    const queue = createRateLimitedQueue(1, 0);
    let running = 0;
    let maxRunning = 0;
    const tasks = [1, 2, 3].map(() => queue.add(async () => {
      running += 1;
      maxRunning = Math.max(maxRunning, running);
      await sleep(5);
      running -= 1;
    }));
    await Promise.all(tasks);
    if (maxRunning !== 1) throw new Error('concurrence ' + maxRunning);
  });

  await check('urls refusées', async () => {
    const blocked = [
      'http://cdn.example/a.mp4',
      'https://127.0.0.1/a.mp4',
      'https://2130706433/a.mp4',
      'https://0x7f000001/a.mp4',
      'https://0177.0.0.1/a.mp4',
      'https://127.1/a.mp4',
      'https://[::1]/a.mp4',
      'https://[::ffff:7f00:1]/a.mp4',
      'https://[::ffff:127.0.0.1]/a.mp4',
      'https://169.254.169.254/latest/meta-data',
      'https://metadata.google.internal/computeMetadata/v1/',
      'https://user:secret@cdn.example/a.mp4',
      'https://10.1.2.3/a.mp4',
    ];
    for (let i = 0; i < blocked.length; i += 1) {
      let refused = false;
      try {
        assertPublicHttpsUrl(blocked[i]);
      } catch (err) {
        refused = err instanceof RejectedDownloadError && !err.message.includes('secret');
      }
      if (!refused) throw new Error('acceptée ' + i);
    }
    const ok = assertPublicHttpsUrl('https://cdn.example/a.mp4');
    if (ok.hostname !== 'cdn.example') throw new Error('hôte public');
    dnsLookupImpl = async () => [{ address: '10.0.0.8', family: 4 }];
    let privateResolved = false;
    try {
      await assertHostnameResolvesPublic(ok);
    } catch (err) {
      privateResolved = err instanceof RejectedDownloadError;
    }
    if (!privateResolved) throw new Error('dns privé');
    dnsLookupImpl = async () => [{ address: '::ffff:127.0.0.1', family: 6 }];
    let mapped = false;
    try {
      await assertHostnameResolvesPublic(ok);
    } catch (err) {
      mapped = err instanceof RejectedDownloadError;
    }
    if (!mapped) throw new Error('ipv4 mappée');
  });

  await check('téléchargement', async () => {
    const tmp = await fs.promises.mkdtemp(path.join(os.tmpdir(), 'videowan2g-dl-'));
    outputDir = path.join(tmp, 'generated_videos');
    dnsLookupImpl = async () => [{ address: '93.184.216.34', family: 4 }];
    const seenAuth = [];
    let step = 0;
    fetchImpl = async (url, options) => {
      seenAuth.push(options && options.headers && options.headers.Authorization ? 'yes' : 'no');
      step += 1;
      if (step === 1) {
        return jsonResponse(302, 'go', { Location: 'https://cdn.example/b.mp4' });
      }
      if (step === 2) {
        return jsonResponse(302, 'go', { Location: 'https://cdn.example/c.mp4' });
      }
      return new fetch.Response(mp4Buffer(), { status: 200, headers: { 'Content-Type': 'video/mp4' } });
    };
    const saved = await downloadVideo('https://cdn.example/a.mp4', 'Maison sous la pluie');
    if (!saved.endsWith('.mp4')) throw new Error('extension');
    if (seenAuth.some((flag) => flag === 'yes')) throw new Error('authorization transmise');
    const info = await assertLocalVideo(saved);
    if (info.kind !== 'mp4') throw new Error('ftyp');
    step = 0;
    fetchImpl = async () => {
      step += 1;
      const next = 'https://cdn.example/hop-' + step + '.mp4';
      return jsonResponse(302, 'go', { Location: next });
    };
    let tooMany = false;
    try {
      await downloadVideo('https://cdn.example/start.mp4', 'Trop');
    } catch (err) {
      tooMany = err instanceof RejectedDownloadError;
    }
    if (!tooMany) throw new Error('redirections');
    fetchImpl = async () => new fetch.Response(webmBuffer(), { status: 200 });
    const webm = await downloadVideo('https://cdn.example/w.webm', 'Neige');
    if (!webm.endsWith('.webm')) throw new Error('webm');
    const target = path.join(tmp, 'secret.txt');
    await fs.promises.writeFile(target, 'SAFE');
    const link = path.join(outputDir, 'linked.mp4');
    await fs.promises.symlink(target, link);
    let symlinkRefused = false;
    try {
      await openExclusiveNewFile(link);
    } catch (err) {
      symlinkRefused = err instanceof RejectedDownloadError;
    }
    const after = await fs.promises.readFile(target, 'utf8');
    if (!symlinkRefused || after !== 'SAFE') throw new Error('symlink écrasé');
    const huge = path.join(outputDir, 'huge.mp4');
    const hugeFd = await fs.promises.open(huge, 'w');
    await hugeFd.truncate(MAX_VIDEO_BYTES);
    await hugeFd.close();
    let hugeRefused = false;
    try {
      await assertLocalVideo(huge);
    } catch (err) {
      hugeRefused = err instanceof RejectedDownloadError;
    }
    if (!hugeRefused) throw new Error('taille');
    await fs.promises.rm(tmp, { recursive: true, force: true });
  });

  await check('dépôt autorisé', async () => {
    if (assertAllowedRepoUrl(ALLOWED_REPO) !== ALLOWED_REPO) throw new Error('canonique');
    if (assertAllowedRepoUrl(ALLOWED_REPO + '.git') !== ALLOWED_REPO) throw new Error('suffixe git');
    const secretUrl = 'https://user:ghp_SUPERSECRET@github.com/urdirditfurd/videowan2g';
    let message = '';
    try {
      assertAllowedRepoUrl(secretUrl);
    } catch (err) {
      message = err.message;
    }
    if (message !== 'URL de dépôt refusée' || message.includes('SUPERSECRET') || message.includes('ghp_')) {
      throw new Error('fuite url');
    }
    let other = false;
    try {
      assertAllowedRepoUrl('https://github.com/urdirditfurd/other');
    } catch (err) {
      other = err.message === 'URL de dépôt refusée';
    }
    if (!other) throw new Error('autre dépôt');
    const scenes = loadScenes(path.join(REPO_ROOT, 'scenes.json'));
    if (scenes.length !== 5) throw new Error('scènes ' + scenes.length);
    if (scenes[0].description !== 'maison japonaise sous la pluie') throw new Error('première scène');
  });

  await check('askpass et git isolé', async () => {
    const canaryAgnes = 'CANARYAGNES8f3e1c0a9b74';
    const canaryGithub = 'CANARYGITHUB8f3e1c0a9b74';
    const previousAgnes = process.env.AGNES_API_KEY;
    const previousGithub = process.env.GITHUB_TOKEN;
    const previousRepo = process.env.GITHUB_REPO_URL;
    process.env.AGNES_API_KEY = canaryAgnes;
    process.env.GITHUB_TOKEN = canaryGithub;
    process.env.GITHUB_REPO_URL = ALLOWED_REPO;
    const tmp = await fs.promises.mkdtemp(path.join(os.tmpdir(), 'videowan2g-git-'));
    try {
      await withAskpass(canaryGithub, async (scriptPath) => {
        const user = spawnSync(process.execPath, [scriptPath, "Username for 'https://github.com'"], { encoding: 'utf8' });
        const pass = spawnSync(process.execPath, [scriptPath, "Password for 'https://github.com'"], { encoding: 'utf8' });
        if (String(user.stdout || '').trim() !== 'x-access-token') throw new Error('askpass utilisateur');
        if (String(pass.stdout || '').trim() !== canaryGithub) throw new Error('askpass mot de passe');
      });
      const dumpPath = path.join(tmp, 'env.jsonl');
      const argsPath = path.join(tmp, 'args.jsonl');
      const fakePath = path.join(tmp, 'fake-git.cjs');
      const fakeSource = [
        '#!/usr/bin/env node',
        '"use strict";',
        'const fs = require("fs");',
        'const env = process.env;',
        'fs.appendFileSync(' + JSON.stringify(dumpPath) + ', JSON.stringify({ keys: Object.keys(env), env: env }) + "\\n");',
        'const args = process.argv.slice(2);',
        'fs.appendFileSync(' + JSON.stringify(argsPath) + ', JSON.stringify(args) + "\\n");',
        'if (args[0] === "remote" && args[1] === "get-url") {',
        '  process.stdout.write(' + JSON.stringify(ALLOWED_REPO + '\n') + ');',
        '}',
        'process.exit(0);',
        '',
      ].join('\n');
      await fs.promises.writeFile(fakePath, fakeSource, { mode: 0o700 });
      const videos = path.join(tmp, 'generated_videos');
      await fs.promises.mkdir(videos);
      const mp4 = path.join(videos, 'pluie_1.mp4');
      await fs.promises.writeFile(mp4, mp4Buffer());
      await publishVideo(mp4, 'Pluie douce', {
        repoRoot: tmp,
        binary: [process.execPath, fakePath],
      });
      const dumps = fs.readFileSync(dumpPath, 'utf8').split('\n').filter(Boolean).map((line) => JSON.parse(line));
      if (dumps.length < 3) throw new Error('dumps ' + dumps.length);
      for (let i = 0; i < dumps.length; i += 1) {
        const blob = JSON.stringify(dumps[i]);
        if (dumps[i].keys.includes('AGNES_API_KEY') || dumps[i].keys.includes('GITHUB_TOKEN')) {
          throw new Error('nom de secret présent');
        }
        if (blob.includes(canaryAgnes) || blob.includes(canaryGithub)) throw new Error('valeur de secret présente');
      }
      const pushDump = dumps[dumps.length - 1];
      if (!pushDump.env.GIT_ASKPASS) throw new Error('askpass absent du push');
      if (fs.existsSync(pushDump.env.GIT_ASKPASS)) throw new Error('askpass non supprimé');
      if (pushDump.env.HOME !== '') throw new Error('HOME');
      if (pushDump.env.GIT_CONFIG_NOSYSTEM !== '1') throw new Error('nosystem');
      const argLines = fs.readFileSync(argsPath, 'utf8').split('\n').filter(Boolean).map((line) => JSON.parse(line));
      const flat = argLines.map((args) => args.join(' '));
      if (!flat.some((line) => line === 'add -f -- generated_videos/pluie_1.mp4')) throw new Error('add -f');
      if (!flat.some((line) => line.indexOf('commit.gpgsign=false') !== -1 && line.indexOf('-- generated_videos/pluie_1.mp4') !== -1)) {
        throw new Error('commit');
      }
      if (!flat.some((line) => line.indexOf('credential.helper=') !== -1 && line.indexOf('push origin HEAD') !== -1)) {
        throw new Error('push');
      }
      const bad = path.join(videos, 'notes.mp4');
      await fs.promises.writeFile(bad, 'pas une video!!');
      let rejected = false;
      try {
        await publishVideo(bad, 'Non', { repoRoot: tmp, binary: [process.execPath, fakePath] });
      } catch (err) {
        rejected = err instanceof RejectedDownloadError;
      }
      if (!rejected) throw new Error('mauvais en-tête publié');
    } finally {
      if (previousAgnes === undefined) delete process.env.AGNES_API_KEY;
      else process.env.AGNES_API_KEY = previousAgnes;
      if (previousGithub === undefined) delete process.env.GITHUB_TOKEN;
      else process.env.GITHUB_TOKEN = previousGithub;
      if (previousRepo === undefined) delete process.env.GITHUB_REPO_URL;
      else process.env.GITHUB_REPO_URL = previousRepo;
      await fs.promises.rm(tmp, { recursive: true, force: true });
    }
  });

  await check('git add -f réel', async () => {
    const tmp = await fs.promises.mkdtemp(path.join(os.tmpdir(), 'videowan2g-realgit-'));
    const env = {
      PATH: process.env.PATH || '/usr/bin:/bin',
      HOME: '',
      GIT_CONFIG_NOSYSTEM: '1',
      GIT_TERMINAL_PROMPT: '0',
    };
    const git = (args) => spawnSync('git', args, { cwd: tmp, env: env, encoding: 'utf8' });
    const init = git(['init', '-q']);
    if (init.status !== 0) throw new Error('init');
    await fs.promises.writeFile(path.join(tmp, '.gitignore'), 'generated_videos/*.mp4\n');
    await fs.promises.mkdir(path.join(tmp, 'generated_videos'));
    await fs.promises.writeFile(path.join(tmp, 'generated_videos', 'a.mp4'), mp4Buffer());
    const denied = git(['add', '--', 'generated_videos/a.mp4']);
    if (denied.status === 0) throw new Error('ignore ignoré');
    const forced = git(['add', '-f', '--', 'generated_videos/a.mp4']);
    if (forced.status !== 0) throw new Error('force refusé');
    await fs.promises.rm(tmp, { recursive: true, force: true });
  });

  await check('dotenv silencieux', async () => {
    const tmp = await fs.promises.mkdtemp(path.join(os.tmpdir(), 'videowan2g-env-'));
    const envFile = path.join(tmp, '.env');
    await fs.promises.writeFile(envFile, 'QUIET_TEST=1\n');
    const runner = [
      'const dotenv = require("dotenv");',
      'dotenv.config({ path: ' + JSON.stringify(envFile) + ', quiet: true });',
    ].join('\n');
    const result = spawnSync(process.execPath, ['-e', runner], {
      cwd: REPO_ROOT,
      encoding: 'utf8',
    });
    await fs.promises.rm(tmp, { recursive: true, force: true });
    if (result.status !== 0 || result.stdout) throw new Error('dotenv bavard');
  });

  await check('polling terminé', async () => {
    process.env.AGNES_API_KEY = process.env.AGNES_API_KEY || 'sk-self-test-placeholder';
    sleepImpl = async () => {};
    fetchImpl = async () => jsonResponse(200, JSON.stringify({
      status: 'completed',
      progress: 100,
      url: 'https://cdn.example/final.mp4',
    }));
    const url = await pollVideo('vid_123', () => {});
    if (url !== 'https://cdn.example/final.mp4') throw new Error('url');
    fetchImpl = async () => jsonResponse(200, JSON.stringify({ status: 'failed', progress: 0 }));
    let failed = false;
    try {
      await pollVideo('vid_123', () => {});
    } catch (err) {
      failed = /Échec définitif/.test(err.message);
    }
    if (!failed) throw new Error('échec non détecté');
  });

  if (failures.length > 0) {
    quietLogs = false;
    for (let i = 0; i < failures.length; i += 1) {
      console.error('SELF-TEST FAILED: ' + failures[i]);
    }
    process.exitCode = 1;
    return;
  }
  quietLogs = false;
  console.log('SELF-TEST PASSED (' + checks + ' checks)');
}

async function main() {
  if (process.argv.includes('--self-test')) {
    await runSelfTest();
    return;
  }
  dotenv.config({ path: path.join(REPO_ROOT, '.env'), quiet: true });
  await runBatch();
}

process.on('SIGINT', () => {
  state.stopRequested = true;
  log('warn', 'STOP', 'Arrêt demandé');
});

if (require.main === module) {
  main().catch((err) => {
    console.error('Erreur: ' + safeErrorMessage(err));
    process.exitCode = 1;
  });
}
