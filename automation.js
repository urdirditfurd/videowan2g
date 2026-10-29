'use strict';

// Installation et lancement :
//   npm install
//   cp .env.example .env
//   node automation.js --self-test
//   node automation.js

const fs = require('fs');
const os = require('os');
const path = require('path');
const fetch = require('node-fetch');
const { simpleGit } = require('simple-git');

require('dotenv').config({ quiet: true });

const API_BASE = 'https://apihub.agnes-ai.com/v1';
const POLL_BASE = 'https://apihub.agnes-ai.com/agnesapi';
const API_HOST = 'apihub.agnes-ai.com';
const MODEL_TEXT = 'agnes-2.5-flash';
const MODEL_IMAGE = 'agnes-image-2.5-flash';
const MODEL_VIDEO = 'agnes-video-2.5-flash';
const ALLOWED_REPO = 'https://github.com/urdirditfurd/videowan2g';
const ALLOWED_REPO_GIT = `${ALLOWED_REPO}.git`;

const IMAGE_INTERVAL_MIN = 3200;
const IMAGE_INTERVAL_MAX = 12000;
const VIDEO_INTERVAL_MIN = 62000;
const VIDEO_INTERVAL_MAX = 90000;
const VIDEO_INTERVAL_SAFE = 75000;
const VIDEO_FIRST_POLL_DELAY_SEC = 80;
const VIDEO_MAX_POLL_ATTEMPTS = 100;
const VIDEO_QUEUE_MAX_RETRIES = 12;
const QUEUE_BACKOFF_SECONDS = [60, 120, 300];
const QUEUE_WAIT_CAP_SECONDS = 360;
const MAX_VIDEO_BYTES = 100 * 1024 * 1024;
const MAX_REDIRECTS = 2;
const ALLOWED_DURATIONS = new Set([5, 6, 8, 10]);
const GIT_ENV_ALLOWLIST = [
  'GIT_AUTHOR_NAME',
  'GIT_AUTHOR_EMAIL',
  'GIT_COMMITTER_NAME',
  'GIT_COMMITTER_EMAIL',
  'GIT_CONFIG_NOSYSTEM',
  'GIT_TERMINAL_PROMPT',
  'GIT_ASKPASS',
];

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

const TIMES_OF_DAY = [
  'pale blue dawn',
  'misty morning',
  'soft late morning',
  'quiet overcast noon',
  'warm golden hour',
  'blue hour at dusk',
  'early night with fireflies',
  'quiet moonlit night',
];

const SEASONS = [
  'early spring blossoms',
  'full spring greenery',
  'early summer lushness',
  'late summer haze',
  'early autumn color',
  'deep autumn foliage',
  'first winter frost',
  'mild winter thaw',
];

const WEATHERS = [
  'fine rain',
  'drizzle on leaves',
  'air just after the rain',
  'low soft clouds',
  'sunbeams through mist',
  'light fog',
  'gentle breeze',
  'still clear air',
  'soft falling snow',
];

const SYSTEM_PROMPT = [
  'You are a Studio Ghibli-inspired art director specialized in lofi ambience videos.',
  'You always return strict JSON matching the requested schema exactly.',
  'Every scene includes a cute calm capybara.',
  'Every scene is peaceful, detailed, watercolor, and loopable with slow gentle motion.',
  'Each scene must be visually unique: honor the given time of day, season, and weather even when the base idea is similar to another scene.',
  'No text, no watermark. Return only JSON.',
].join(' ');

const OPEN_EXCLUSIVE = fs.constants.O_CREAT | fs.constants.O_EXCL | fs.constants.O_NOFOLLOW | fs.constants.O_WRONLY;
const OPEN_READ_NOFOLLOW = fs.constants.O_RDONLY | fs.constants.O_NOFOLLOW;

const runtime = {
  stopRequested: false,
  durationSeconds: 5,
  format: '16:9',
};

let createImageIntervalMs = 4000;
let createVideoIntervalMs = VIDEO_INTERVAL_SAFE;
let imageAttemptsSinceLastError = 0;
let videoAttemptsSinceLastError = 0;
let fetchImpl = fetch;
let sleepImpl = (ms) => new Promise((resolve) => setTimeout(resolve, ms));
let timingFile = null;
let sigintCount = 0;

function sleep(ms) {
  return sleepImpl(ms);
}

function redact(value) {
  return String(value == null ? '' : value)
    .replace(/\/\/[^/\s:@]+:[^/\s@]+@/g, '//***@')
    .replace(/\/\/[^/\s:@]+@/g, '//***@')
    .replace(/\bghp_[A-Za-z0-9_]+\b/g, 'ghp_***')
    .replace(/\bgithub_pat_[A-Za-z0-9_]+\b/g, 'github_pat_***')
    .replace(/\bsk-[A-Za-z0-9_-]+\b/g, 'sk-***');
}

function log(level, category, message) {
  const line = `${new Date().toISOString()} [${category}] ${redact(message)}`;
  if (level === 'error' || level === 'warn') console.error(line);
  else console.log(line);
}

function pick(list) {
  return list[Math.floor(Math.random() * list.length)];
}

function assert(condition, message) {
  if (!condition) throw new Error(message);
}

function sanitizeTitle(title) {
  const cleaned = String(title || '')
    .normalize('NFKD')
    .replace(/[\u0300-\u036f]/g, '')
    .replace(/[^A-Za-z0-9]+/g, '_')
    .replace(/^_+|_+$/g, '')
    .slice(0, 60);
  return cleaned || 'lofi_capybara';
}

function oneLine(title) {
  const cleaned = String(title || 'Ambiance Lofi').replace(/[\r\n\0]/g, ' ').trim();
  return cleaned.slice(0, 120) || 'Ambiance Lofi';
}

function shellSingleQuote(value) {
  return `'${String(value).replace(/'/g, `'\\''`)}'`;
}

function queueBackoffSeconds(index, random = Math.random) {
  const ladderIndex = Math.min(Math.max(0, index), QUEUE_BACKOFF_SECONDS.length - 1);
  const base = QUEUE_BACKOFF_SECONDS[ladderIndex];
  const factor = 0.85 + random() * 0.3;
  const jittered = Math.round(base * factor);
  return Math.min(QUEUE_WAIT_CAP_SECONDS, Math.max(1, jittered));
}

function createRateLimitedQueue(maxConcurrent, minIntervalMs) {
  let running = 0;
  let lastStart = 0;
  const queue = [];
  let resolveIdle = null;
  let timer = null;

  function intervalMs() {
    return typeof minIntervalMs === 'function' ? minIntervalMs() : minIntervalMs;
  }

  function schedule(wait) {
    if (timer !== null) return;
    timer = setTimeout(() => {
      timer = null;
      tryNext();
    }, wait);
  }

  function tryNext() {
    if (running >= maxConcurrent) return;
    if (queue.length === 0) {
      if (running === 0 && resolveIdle) {
        const resolve = resolveIdle;
        resolveIdle = null;
        resolve();
      }
      return;
    }
    const wait = Math.max(0, lastStart + intervalMs() - Date.now());
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
  } catch (e1) {
    // stratégie suivante
  }
  try {
    return JSON.parse(escapeControlCharsInStrings(raw));
  } catch (e2) {
    // stratégie suivante
  }
  try {
    const lastBrace = raw.lastIndexOf('}');
    if (lastBrace > 0) return JSON.parse(escapeControlCharsInStrings(raw.slice(0, lastBrace + 1)));
  } catch (e3) {
    // stratégie suivante
  }
  try {
    return JSON.parse(balanceJson(escapeControlCharsInStrings(raw)));
  } catch (e4) {
    // stratégie suivante
  }
  const salvaged = salvageExpansion(raw);
  if (salvaged && (salvaged.image_prompt || salvaged.video_motion_prompt)) return salvaged;
  return null;
}

function buildExpansionPrompt(userDescription, mood, variation) {
  const selected = MOODS[mood] || MOODS.rainy_forest;
  return [
    'You are an art director for Studio Ghibli-inspired lofi ambience videos.',
    '',
    'USER SCENE IDEA (French):',
    userDescription,
    '',
    'AMBIENCE (French mood chosen):',
    `${selected.label} — ${selected.desc}`,
    '',
    'UNIQUENESS — these three variables are mandatory and must be visible in the picture:',
    `Time of day (heure de la journée): ${variation.timeOfDay}`,
    `Season (saison): ${variation.season}`,
    `Weather (type de météo): ${variation.weather}`,
    'Even if the base idea resembles another scene, lighting, vegetation, sky, and air MUST change with these variables so the result is visually unique.',
    'The capybara must still be present, cute, and calm.',
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
    '  "image_prompt": "Ultra-detailed English visual description for an image model. Include: setting, lighting, colors, atmosphere, the capybara (with its location and pose), the unique time of day, season, and weather, Ghibli watercolor style, cinematic composition. 150-250 words.",',
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

function buildLofiImagePrompt(exp, variation) {
  return [
    'GHIBLI-STYLE LOFI AMBIENCE KEYFRAME.',
    exp.image_prompt,
    `UNIQUE ATMOSPHERE: ${variation.timeOfDay}, ${variation.season}, ${variation.weather}.`,
    buildGhibliStyleBlock(),
    'Ultra-detailed painterly image, calm and loopable composition.',
  ].join('\n');
}

function buildLofiVideoPrompt(exp, variation) {
  return [
    `LOFI AMBIENCE VIDEO — ${runtime.durationSeconds} seconds — seamless loop.`,
    'Studio Ghibli inspired watercolor scene with a cute calm capybara.',
    `BASE SCENE: ${exp.image_prompt}`,
    `GENTLE MOTION: ${exp.video_motion_prompt}`,
    `UNIQUE ATMOSPHERE: ${variation.timeOfDay}, ${variation.season}, ${variation.weather}.`,
    'ABSOLUTE MOTION RULE: SLOW, GENTLE, SUBTLE motion only. No fast movement, no camera shake, no action. The scene must feel peaceful and loopable.',
    'Perfect seamless loop, calm pacing.',
    'CAMERA: locked-off or extremely slow drift. No cuts.',
    'STYLE: hand-painted Ghibli watercolor, soft gradients, warm nostalgic light.',
    `PALETTE: ${GHIBLI_PALETTE}`,
    'SOUND: soft ambient nature sounds only — no music, no voice. Wind, birds, water, rustling leaves.',
    'NO readable text, NO watermark, NO logos.',
  ].join('\n');
}

function assertAgnesUrl(raw) {
  const url = new URL(raw);
  if (url.protocol !== 'https:' || url.hostname !== API_HOST || url.username || url.password) {
    throw Object.assign(new Error('Hôte API non autorisé'), { terminal: true });
  }
  return url.href;
}

async function apiFetch(url, options = {}, label = 'API', kind = 'default') {
  assertAgnesUrl(url);
  const maxAttempts = kind === 'video' ? 10 : kind === 'image' ? 10 : 8;
  let queueRetries = 0;

  for (let attempt = 0; attempt < maxAttempts; attempt += 1) {
    if (runtime.stopRequested) throw new Error('Arrêt demandé');
    const method = options.method || 'GET';
    log('api', `REQ ${label}`, `${method} ${url} (tentative ${attempt + 1}/${maxAttempts})`);

    let res;
    try {
      res = await fetchImpl(url, options);
    } catch (e) {
      if (e.message === 'Arrêt demandé' || e.terminal) throw e;
      const wait = Math.min(kind === 'video' ? 120 : 45, kind === 'video' ? 10 + attempt * 8 : 3 + attempt * 4);
      log('warn', 'NETWORK', `${e.message} — retry dans ${wait}s`);
      await sleep(wait * 1000);
      continue;
    }

    let retryAfter = 0;
    try {
      retryAfter = parseInt(res.headers.get('retry-after') || '0', 10) || 0;
    } catch (e) {
      retryAfter = 0;
    }

    let errorText = '';
    const readableStatus = [400, 401, 404, 429, 500, 502, 503, 504, 520].includes(res.status);
    if (!res.ok && (readableStatus || kind === 'video') && typeof res.clone === 'function') {
      try {
        errorText = (await res.clone().text()).slice(0, 1200);
      } catch (e) {
        errorText = '';
      }
    }
    if (!res.ok) log('response', `RÉP ${label}`, `HTTP ${res.status} ${errorText ? redact(errorText.slice(0, 300)) : ''}`.trim());

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
      if (queueRetries >= VIDEO_QUEUE_MAX_RETRIES) {
        throw new Error('File vidéo saturée — répétitions de file épuisées');
      }
      videoAttemptsSinceLastError = 0;
      const wait = queueBackoffSeconds(queueRetries);
      queueRetries += 1;
      attempt -= 1;
      log('warn', 'VIDEO_QUEUE', `File vidéo saturée — pause ${wait}s (${queueRetries}/${VIDEO_QUEUE_MAX_RETRIES})`);
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
      continue;
    }

    if ([500, 502, 503, 504, 520].includes(res.status)) {
      const wait = Math.min(kind === 'video' ? 180 : 60, kind === 'video' ? 20 + attempt * 10 : 5 + attempt * 5);
      log('warn', 'SERVICE', `HTTP ${res.status} — attente ${wait}s`);
      await sleep(wait * 1000);
      continue;
    }

    return res;
  }

  throw new Error(`Tentatives épuisées pour ${label} après ${maxAttempts} essais`);
}

async function retryTask(name, fn, maxAttempts = 3) {
  let lastError = null;
  for (let i = 1; i <= maxAttempts; i += 1) {
    if (runtime.stopRequested) throw new Error('Arrêt demandé');
    try {
      log('info', 'RETRY', `${name} — essai ${i}/${maxAttempts}`);
      return await fn();
    } catch (e) {
      lastError = e;
      if (e.message === 'Arrêt demandé') throw e;
      log('warn', 'RETRY', `${name} échoué : ${e.message}`);
      if (i < maxAttempts) {
        const wait = Math.min(90, 8 * i * i);
        log('warn', 'RETRY', `${name} — reprise dans ${wait}s`);
        await sleep(wait * 1000);
      }
    }
  }
  throw lastError || new Error(`${name} échoué`);
}

function authHeaders() {
  return {
    Authorization: `Bearer ${process.env.AGNES_API_KEY}`,
    'Content-Type': 'application/json',
  };
}

async function generateExpansion(userDescription, mood, variation) {
  const prompt = buildExpansionPrompt(userDescription, mood, variation);
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
    headers: authHeaders(),
    body: JSON.stringify(body),
  }, 'Expansion lofi', 'text');
  if (!res.ok) {
    const err = await res.text();
    throw new Error(`HTTP ${res.status} — ${err.slice(0, 300)}`);
  }
  const data = await res.json();
  const content = data.choices && data.choices[0] && data.choices[0].message && data.choices[0].message.content;
  const exp = extractJson(content);
  if (!exp || !exp.image_prompt) throw new Error('Expansion invalide');
  if (!exp.title) exp.title = 'Ambiance Lofi';
  if (!exp.video_motion_prompt) {
    exp.video_motion_prompt = 'Gentle slow motion: capybara breathing softly, leaves drifting down, faint mist, seamless loop.';
  }
  log('success', 'EXPANSION', `Titre : ${oneLine(exp.title)}`);
  return exp;
}

async function generateImage(prompt, format) {
  return imageQueue.add(() => retryTask('Image', async () => {
    const ratio = format === '9:16' ? '9:16' : '16:9';
    const body = { model: MODEL_IMAGE, prompt, size: '1K', ratio, n: 1 };
    const res = await apiFetch(`${API_BASE}/images/generations`, {
      method: 'POST',
      headers: authHeaders(),
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
      seconds: String(runtime.durationSeconds),
      mode: 'keyframe',
      size: '720P',
      aspect_ratio: aspect,
      first_frame: imageUrl,
      last_frame: imageUrl,
    };
    log('ai', 'PROMPT VIDÉO', `Mouvement : ${prompt.slice(0, 180)}`);
    const res = await apiFetch(`${API_BASE}/videos`, {
      method: 'POST',
      headers: authHeaders(),
      body: JSON.stringify(body),
    }, 'Vidéo', 'video');
    if (!res.ok) {
      const err = await res.text();
      throw new Error(`HTTP ${res.status} — ${err.slice(0, 500)}`);
    }
    const data = await res.json();
    const videoId = data.video_id || data.id || data.task_id;
    if (!videoId || !/^[A-Za-z0-9._:-]{1,200}$/.test(String(videoId))) throw new Error('Pas de video_id');
    return String(videoId);
  }, 3));
}

function timingCachePath() {
  return timingFile || path.join(process.cwd(), '.cache', 'lofi-timing.json');
}

function loadTimingCache() {
  try {
    return JSON.parse(fs.readFileSync(timingCachePath(), 'utf8'));
  } catch (e) {
    return {};
  }
}

function saveTimingCache(cache) {
  const file = timingCachePath();
  fs.mkdirSync(path.dirname(file), { recursive: true });
  fs.writeFileSync(file, JSON.stringify(cache));
}

function recordGenerationTime(key, ms) {
  const cache = loadTimingCache();
  if (!cache[key]) cache[key] = [];
  cache[key].push(ms);
  if (cache[key].length > 10) cache[key] = cache[key].slice(-10);
  saveTimingCache(cache);
}

function estimateGenerationTime(key, fallbackMs) {
  const samples = loadTimingCache()[key];
  if (!samples || samples.length === 0) return fallbackMs;
  const sorted = [...samples].sort((a, b) => a - b);
  const mid = Math.floor(sorted.length / 2);
  return sorted.length % 2 === 0 ? (sorted[mid - 1] + sorted[mid]) / 2 : sorted[mid];
}

async function pollVideo(videoId, onProgress) {
  if (!/^[A-Za-z0-9._:-]{1,200}$/.test(videoId)) throw new Error('video_id inattendu');
  const estimatedMs = estimateGenerationTime('video', 90000 + runtime.durationSeconds * 7000);
  const minWait = Math.max(25, Math.min(VIDEO_FIRST_POLL_DELAY_SEC, Math.floor((estimatedMs / 1000) * 0.35)));
  log('info', 'POLLING', `Attente initiale ${minWait}s pour ${videoId}`);
  for (let remaining = minWait; remaining > 0; remaining -= 1) {
    if (runtime.stopRequested) throw new Error('Arrêt demandé');
    onProgress(`File vidéo — ${remaining}s`);
    await sleep(1000);
  }

  let attempt = 0;
  while (attempt < VIDEO_MAX_POLL_ATTEMPTS) {
    if (runtime.stopRequested) throw new Error('Arrêt demandé');
    attempt += 1;
    const url = `${POLL_BASE}?video_id=${encodeURIComponent(videoId)}&model_name=${encodeURIComponent(MODEL_VIDEO)}`;
    let res;
    try {
      res = await apiFetch(url, { method: 'GET', headers: authHeaders() }, 'Polling', 'video');
    } catch (e) {
      if (e.message === 'Arrêt demandé') throw e;
      log('warn', 'POLLING', `Erreur transitoire : ${e.message}`);
      await sleep(Math.min(90000, 15000 + attempt * 1500));
      continue;
    }
    let data;
    try {
      data = await res.json();
    } catch (e) {
      log('warn', 'POLLING', 'Réponse non JSON, reprise');
      await sleep(Math.min(60000, 10000 + attempt * 1000));
      continue;
    }
    const status = String(data.status || 'unknown').toLowerCase();
    const progress = Number(data.progress || 0);
    onProgress(`Création — ${progress}%`);
    log('response', `POLL #${attempt}`, `Statut : ${status} (${progress}%)`);
    if (status === 'completed' || status === 'succeeded' || status === 'done') {
      const videoUrl = data.url || (data.metadata && data.metadata.url) || (data.output && data.output.url);
      if (!videoUrl) throw new Error("Terminé mais pas d'URL");
      return videoUrl;
    }
    if (status === 'failed' || status === 'error' || status === 'cancelled') {
      throw new Error(`Échec définitif (${status})`);
    }
    const wait = attempt < 5 ? 8 : Math.min(30, 10 + Math.floor(attempt / 5) * 3);
    onProgress(`File vidéo — nouvelle vérification dans ${wait}s`);
    await sleep(wait * 1000);
  }
  throw new Error('Délai maximal de polling dépassé');
}

function isIPv4(host) {
  return /^\d{1,3}(\.\d{1,3}){3}$/.test(host);
}

function isBlockedIPv4(ip) {
  const parts = ip.split('.').map((part) => Number(part));
  if (parts.length !== 4 || parts.some((part) => !Number.isInteger(part) || part < 0 || part > 255)) return true;
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

function ipv4FromMappedHex(bare) {
  const match = bare.match(/^::ffff:([0-9a-f]{1,4}):([0-9a-f]{1,4})$/i);
  if (!match) return null;
  const hi = parseInt(match[1], 16);
  const lo = parseInt(match[2], 16);
  return `${(hi >> 8) & 255}.${hi & 255}.${(lo >> 8) & 255}.${lo & 255}`;
}

function isBlockedIPv6(bare) {
  if (bare === '::1' || bare === '::' || bare === '0:0:0:0:0:0:0:1') return true;
  const dotted = bare.match(/^::ffff:(\d{1,3}(?:\.\d{1,3}){3})$/i);
  if (dotted) return isBlockedIPv4(dotted[1]);
  const mapped = ipv4FromMappedHex(bare);
  if (mapped) return isBlockedIPv4(mapped);
  if (/^fe[89ab]/i.test(bare)) return true;
  if (/^f[cd]/i.test(bare)) return true;
  if (/^ff/i.test(bare)) return true;
  if (bare.startsWith('2001:db8:')) return true;
  return false;
}

function isBlockedHostname(hostname) {
  const host = String(hostname || '').trim().toLowerCase();
  if (!host) return true;
  const bare = host.startsWith('[') && host.endsWith(']') ? host.slice(1, -1) : host;
  if (!bare || bare === 'localhost' || bare.endsWith('.localhost') || bare.endsWith('.local')) return true;
  if (bare === 'metadata.google.internal' || bare === 'metadata.google.com' || bare === 'metadata') return true;
  if (bare.includes(':')) return isBlockedIPv6(bare);
  if (isIPv4(bare)) return isBlockedIPv4(bare);
  if (/^\d+$/.test(bare)) return true;
  return false;
}

function assertPublicHttpsUrl(raw) {
  let url;
  try {
    url = new URL(raw);
  } catch (e) {
    throw new Error('URL vidéo invalide');
  }
  if (url.protocol !== 'https:' || url.username || url.password || isBlockedHostname(url.hostname)) {
    throw new Error('Hôte vidéo refusé');
  }
  return url.href;
}

function hasVideoMagic(buf) {
  if (!Buffer.isBuffer(buf) || buf.length < 4) return false;
  if (buf[0] === 0x1a && buf[1] === 0x45 && buf[2] === 0xdf && buf[3] === 0xa3) return true;
  const head = buf.subarray(0, Math.min(buf.length, 32)).toString('latin1');
  return head.includes('ftyp');
}

function fakeMp4(bytes = 64) {
  const buf = Buffer.alloc(bytes, 0);
  buf.writeUInt32BE(bytes, 0);
  buf.write('ftyp', 4, 'ascii');
  buf.write('isom', 8, 'ascii');
  return buf;
}

async function assertPathHasNoSymlink(root, target) {
  const resolvedRoot = path.resolve(root);
  const resolvedTarget = path.resolve(target);
  if (resolvedTarget !== resolvedRoot && !resolvedTarget.startsWith(`${resolvedRoot}${path.sep}`)) {
    throw new Error('Chemin hors racine');
  }
  let current = resolvedRoot;
  const rootStat = await fs.promises.lstat(current);
  if (rootStat.isSymbolicLink()) throw new Error('Racine est un lien symbolique');
  const parts = path.relative(resolvedRoot, resolvedTarget).split(path.sep).filter(Boolean);
  for (let i = 0; i < parts.length; i += 1) {
    current = path.join(current, parts[i]);
    let st;
    try {
      st = await fs.promises.lstat(current);
    } catch (e) {
      if (e.code === 'ENOENT') return;
      throw e;
    }
    if (st.isSymbolicLink()) throw new Error('Lien symbolique refusé');
  }
}

async function writeExclusiveFile(filePath, buffer) {
  await fs.promises.mkdir(path.dirname(filePath), { recursive: true });
  await assertPathHasNoSymlink(process.cwd(), filePath);
  try {
    const existing = await fs.promises.lstat(filePath);
    if (existing.isSymbolicLink()) throw new Error('Refus : la cible est un lien symbolique');
    throw new Error('Le fichier vidéo existe déjà');
  } catch (e) {
    if (e.code !== 'ENOENT') throw e;
  }
  const handle = await fs.promises.open(filePath, OPEN_EXCLUSIVE, 0o644);
  try {
    await handle.writeFile(buffer);
  } catch (e) {
    await handle.close();
    await fs.promises.unlink(filePath).catch(() => {});
    throw e;
  }
  await handle.close();
}

async function assertLocalVideo(filePath) {
  await assertPathHasNoSymlink(process.cwd(), filePath);
  const st = await fs.promises.lstat(filePath);
  if (st.isSymbolicLink()) throw new Error('Refus : lien symbolique');
  if (!st.isFile()) throw new Error('Refus : pas un fichier régulier');
  if (st.size <= 0 || st.size >= MAX_VIDEO_BYTES) throw new Error('Taille de vidéo refusée');
  const handle = await fs.promises.open(filePath, OPEN_READ_NOFOLLOW);
  try {
    const buf = Buffer.alloc(32);
    const { bytesRead } = await handle.read(buf, 0, 32, 0);
    if (!hasVideoMagic(buf.subarray(0, bytesRead))) throw new Error('En-tête vidéo invalide');
  } finally {
    await handle.close();
  }
}

function drain(res) {
  try {
    if (res.body && typeof res.body.resume === 'function') res.body.resume();
    else if (res.body && typeof res.body.cancel === 'function') res.body.cancel();
  } catch (e) {
    // corps déjà clos
  }
}

async function readBodyCapped(res, maxBytes) {
  if (!res.body) throw new Error('Réponse vide');
  const chunks = [];
  let total = 0;
  for await (const chunk of res.body) {
    const buf = Buffer.isBuffer(chunk) ? chunk : Buffer.from(chunk);
    total += buf.length;
    if (total > maxBytes) {
      if (typeof res.body.destroy === 'function') res.body.destroy();
      throw new Error('Vidéo trop volumineuse');
    }
    chunks.push(buf);
  }
  return Buffer.concat(chunks, total);
}

async function downloadVideo(startUrl, destPath) {
  let current = assertPublicHttpsUrl(startUrl);
  for (let hop = 0; hop <= MAX_REDIRECTS; hop += 1) {
    const res = await fetchImpl(current, {
      redirect: 'manual',
      headers: { Accept: 'video/mp4,video/webm,*/*' },
      size: MAX_VIDEO_BYTES,
    });
    if ([301, 302, 303, 307, 308].includes(res.status)) {
      const location = res.headers.get('location');
      drain(res);
      if (!location || hop === MAX_REDIRECTS) throw new Error('Trop de redirections');
      current = assertPublicHttpsUrl(new URL(location, current).href);
      continue;
    }
    if (!res.ok) {
      drain(res);
      throw new Error(`Téléchargement HTTP ${res.status}`);
    }
    const buf = await readBodyCapped(res, MAX_VIDEO_BYTES);
    if (!hasVideoMagic(buf)) throw new Error('En-tête vidéo invalide');
    await writeExclusiveFile(destPath, buf);
    await assertLocalVideo(destPath);
    return destPath;
  }
  throw new Error('Trop de redirections');
}

function safeGitIdentity(value, fallback) {
  const cleaned = String(value || '').replace(/[\r\n\0]/g, '').trim();
  return cleaned || fallback;
}

function buildGitEnv(extraEnv = {}) {
  const env = {
    PATH: process.env.PATH || '',
    HOME: '',
    GIT_CONFIG_NOSYSTEM: '1',
    GIT_TERMINAL_PROMPT: '0',
    GIT_AUTHOR_NAME: safeGitIdentity(process.env.GIT_AUTHOR_NAME, 'Lofi Capybara Bot'),
    GIT_AUTHOR_EMAIL: safeGitIdentity(process.env.GIT_AUTHOR_EMAIL, 'lofi-capybara-bot@users.noreply.github.com'),
    GIT_COMMITTER_NAME: safeGitIdentity(process.env.GIT_COMMITTER_NAME || process.env.GIT_AUTHOR_NAME, 'Lofi Capybara Bot'),
    GIT_COMMITTER_EMAIL: safeGitIdentity(process.env.GIT_COMMITTER_EMAIL || process.env.GIT_AUTHOR_EMAIL, 'lofi-capybara-bot@users.noreply.github.com'),
  };
  for (const [key, value] of Object.entries(extraEnv)) {
    if (key === 'AGNES_API_KEY' || key === 'GITHUB_TOKEN') {
      throw new Error('Secret interdit dans l’environnement git');
    }
    env[key] = value;
  }
  delete env.AGNES_API_KEY;
  delete env.GITHUB_TOKEN;
  return env;
}

function createGit(baseDir, extraEnv = {}, options = {}) {
  const gitOptions = {
    baseDir,
    unsafe: {
      allowUnsafeAskPass: true,
      allowUnsafeCredentialHelper: true,
    },
    allowEnvironment: GIT_ENV_ALLOWLIST,
  };
  if (options.binary) gitOptions.binary = options.binary;
  return simpleGit(gitOptions).env(buildGitEnv(extraEnv));
}

function assertToken(token) {
  if (!token || token.length > 300 || /[\r\n\0]/.test(token)) throw new Error('GITHUB_TOKEN invalide');
}

function assertApiKey(key) {
  if (!key || key.length > 400 || /[\r\n\0]/.test(key)) throw new Error('AGNES_API_KEY invalide');
}

async function withAskPass(baseDir, fn, options = {}) {
  assertToken(process.env.GITHUB_TOKEN);
  const dir = await fs.promises.mkdtemp(path.join(os.tmpdir(), 'lofi-askpass-'));
  const scriptPath = path.join(dir, 'askpass.cjs');
  const script = [
    `#!${process.execPath}`,
    "'use strict';",
    'const prompt = String(process.argv[2] || "");',
    "if (/username/i.test(prompt)) process.stdout.write('x-access-token\\n');",
    `else process.stdout.write(${JSON.stringify(process.env.GITHUB_TOKEN)} + '\\n');`,
    '',
  ].join('\n');
  await fs.promises.writeFile(scriptPath, script, { mode: 0o700 });
  try {
    const git = createGit(baseDir, { GIT_ASKPASS: scriptPath, ...(options.extraEnv || {}) }, options);
    return await fn(git);
  } finally {
    await fs.promises.rm(dir, { recursive: true, force: true });
  }
}

function normalizeRepo(raw) {
  try {
    const url = new URL(raw);
    const pathname = url.pathname.replace(/\.git$/i, '').replace(/\/$/, '');
    return `${url.protocol}//${url.hostname.toLowerCase()}${pathname}`;
  } catch (e) {
    return null;
  }
}

function assertAllowedRepoUrl(raw) {
  let url;
  try {
    url = new URL(String(raw || ''));
  } catch (e) {
    throw new Error('GITHUB_REPO_URL invalide');
  }
  if (url.protocol !== 'https:' || url.username || url.password) {
    throw new Error('GITHUB_REPO_URL refusé');
  }
  if (normalizeRepo(url.href) !== ALLOWED_REPO) throw new Error('Dépôt non autorisé');
  return ALLOWED_REPO_GIT;
}

async function ensureOrigin(git, repoUrl) {
  const canonical = assertAllowedRepoUrl(repoUrl);
  const remotes = await git.getRemotes(true);
  const origin = remotes.find((remote) => remote.name === 'origin');
  if (!origin) {
    await git.addRemote('origin', canonical);
    return;
  }
  const current = origin.refs.push || origin.refs.fetch || '';
  if (normalizeRepo(current) !== normalizeRepo(canonical)) {
    throw new Error('Le remote origin ne pointe pas vers le dépôt autorisé');
  }
  if (current !== canonical) await git.raw(['remote', 'set-url', 'origin', canonical]);
}

async function stageAndCommit(git, relPath, message) {
  await git.raw(['add', '--', relPath]);
  await git.raw(['-c', 'commit.gpgsign=false', 'commit', '-m', message, '--', relPath]);
}

async function pushHead(git) {
  const branch = String(await git.revparse(['--abbrev-ref', 'HEAD'])).trim();
  if (!branch || branch === 'HEAD' || !/^[A-Za-z0-9._/-]+$/.test(branch) || branch.includes('..')) {
    throw new Error('Branche inattendue');
  }
  await git.raw(['-c', 'credential.helper=', 'push', 'origin', `HEAD:${branch}`]);
}

async function publishVideo(filePath, title) {
  await assertLocalVideo(filePath);
  const repoUrl = assertAllowedRepoUrl(process.env.GITHUB_REPO_URL);
  const rel = path.relative(process.cwd(), path.resolve(filePath));
  if (!rel || rel.startsWith('..') || path.isAbsolute(rel)) throw new Error('Vidéo hors du dépôt');
  const message = `feat: add new lofi capybara video: ${oneLine(title)}`;
  await withAskPass(process.cwd(), async (git) => {
    await ensureOrigin(git, repoUrl);
    await stageAndCommit(git, rel, message);
    await pushHead(git);
  });
}

function resolveInsideCwd(raw) {
  if (!raw || path.isAbsolute(raw)) throw new Error('Chemin relatif requis');
  const root = path.resolve(process.cwd());
  const resolved = path.resolve(root, raw);
  if (resolved !== root && !resolved.startsWith(`${root}${path.sep}`)) throw new Error('Chemin hors du projet');
  return resolved;
}

function readDuration() {
  const value = Number(process.env.VIDEO_DURATION_SECONDS || '5');
  if (!ALLOWED_DURATIONS.has(value)) throw new Error('VIDEO_DURATION_SECONDS invalide');
  return value;
}

function readFormat() {
  const value = process.env.VIDEO_FORMAT || '16:9';
  if (value !== '16:9' && value !== '9:16') throw new Error('VIDEO_FORMAT invalide');
  return value;
}

function normalizeScene(entry, index) {
  if (typeof entry === 'string') {
    const description = entry.trim();
    if (description.length < 4) throw new Error(`Scène #${index + 1} trop courte`);
    return { description, mood: null, durationSeconds: null, format: null };
  }
  if (!entry || typeof entry !== 'object' || typeof entry.description !== 'string') {
    throw new Error(`Scène #${index + 1} invalide`);
  }
  const description = entry.description.trim();
  if (description.length < 4) throw new Error(`Scène #${index + 1} trop courte`);
  const mood = entry.mood || null;
  if (mood && !MOODS[mood]) throw new Error(`Ambiance inconnue pour la scène #${index + 1}`);
  let durationSeconds = null;
  if (entry.durationSeconds != null) {
    durationSeconds = Number(entry.durationSeconds);
    if (!ALLOWED_DURATIONS.has(durationSeconds)) throw new Error('Durée invalide');
  }
  let format = null;
  if (entry.format != null) {
    if (entry.format !== '16:9' && entry.format !== '9:16') throw new Error('Format invalide');
    format = entry.format;
  }
  return { description, mood, durationSeconds, format };
}

function loadScenes(filePath) {
  const data = JSON.parse(fs.readFileSync(filePath, 'utf8'));
  if (!Array.isArray(data) || data.length === 0) throw new Error('scenes.json doit être un tableau non vide');
  return data.map((entry, index) => normalizeScene(entry, index));
}

function buildVideoPath(outputDir, title) {
  const name = `${sanitizeTitle(title)}_${Date.now()}.mp4`;
  return path.join(outputDir, name);
}

async function runScene(scene, index, outputDir) {
  const variation = {
    timeOfDay: pick(TIMES_OF_DAY),
    season: pick(SEASONS),
    weather: pick(WEATHERS),
  };
  const mood = scene.mood && MOODS[scene.mood] ? scene.mood : 'rainy_forest';
  runtime.durationSeconds = scene.durationSeconds || readDuration();
  runtime.format = scene.format || readFormat();
  log('state', 'SCENE', `${index + 1} — ${scene.description}`);
  log('state', 'VAR', `${variation.timeOfDay} · ${variation.season} · ${variation.weather}`);
  const started = Date.now();
  const exp = await generateExpansion(scene.description, mood, variation);
  const imageUrl = await generateImage(buildLofiImagePrompt(exp, variation), runtime.format);
  log('success', 'IMAGE', 'Image Ghibli prête');
  createVideoIntervalMs = VIDEO_INTERVAL_SAFE;
  videoAttemptsSinceLastError = 0;
  const videoId = await generateVideo(imageUrl, buildLofiVideoPrompt(exp, variation), runtime.format);
  log('success', 'VIDÉO', `Tâche ${videoId}`);
  const videoUrl = await pollVideo(videoId, (message) => log('info', 'VIDÉO', message));
  const dest = buildVideoPath(outputDir, exp.title);
  await downloadVideo(videoUrl, dest);
  recordGenerationTime('video', Date.now() - started);
  log('success', 'FICHIER', path.relative(process.cwd(), dest));
  await publishVideo(dest, exp.title);
  log('success', 'GIT', `Publié : ${oneLine(exp.title)}`);
}

function singleUseResponse(status, body, headers = {}) {
  let used = false;
  const read = async () => {
    if (used) throw new Error('corps déjà consommé');
    used = true;
    return body;
  };
  return {
    ok: status >= 200 && status < 300,
    status,
    headers: {
      get(name) {
        return headers[String(name).toLowerCase()] || null;
      },
    },
    text: read,
    json: async () => JSON.parse(await read()),
    clone() {
      return singleUseResponse(status, body, headers);
    },
  };
}

function assertEnvClean(env, canaries) {
  const keys = Object.keys(env);
  if (keys.includes('AGNES_API_KEY') || keys.includes('GITHUB_TOKEN')) {
    throw new Error('Environnement git contient un nom de secret');
  }
  const blob = JSON.stringify(env);
  for (const canary of canaries) {
    if (blob.includes(canary)) throw new Error('Environnement git contient une valeur secrète');
  }
}

function writeFakeGit(dir) {
  const fakePath = path.join(dir, 'fake-git.cjs');
  const wrapperPath = path.join(dir, 'gitwrap.sh');
  const source = [
    "'use strict';",
    "const fs = require('fs');",
    'const logPath = process.env.LOFI_FAKE_GIT_LOG;',
    'if (logPath) {',
    '  const payload = { argv: process.argv.slice(2), env: process.env };',
    "  fs.appendFileSync(logPath, JSON.stringify(payload) + '\\n');",
    '}',
    'const args = process.argv.slice(2);',
    "if (args.includes('--version')) { process.stdout.write('git version 2.43.0\\n'); process.exit(0); }",
    "if (args.includes('rev-parse')) { process.stdout.write('main\\n'); process.exit(0); }",
    'process.exit(0);',
    '',
  ].join('\n');
  fs.writeFileSync(fakePath, source, { mode: 0o700 });
  const wrapper = `#!/bin/sh\nexec ${shellSingleQuote(process.execPath)} ${shellSingleQuote(fakePath)} "$@"\n`;
  fs.writeFileSync(wrapperPath, wrapper, { mode: 0o700 });
  return wrapperPath;
}

function readGitLog(logFile) {
  const raw = fs.readFileSync(logFile, 'utf8').trim();
  if (!raw) return [];
  return raw.split('\n').map((line) => JSON.parse(line));
}

async function runSelfTest() {
  const canaryKey = 'sk-canary-agnes-7f3a9c';
  const canaryToken = 'ghp_canarytokenvalue123456';
  process.env.AGNES_API_KEY = canaryKey;
  process.env.GITHUB_TOKEN = canaryToken;
  const canaries = [canaryKey, canaryToken];

  assert(queueBackoffSeconds(0, () => 0.5) === 60, 'backoff 60');
  assert(queueBackoffSeconds(1, () => 0.5) === 120, 'backoff 120');
  assert(queueBackoffSeconds(2, () => 0.5) === 300, 'backoff 300');
  assert(queueBackoffSeconds(9, () => 0.5) === 300, 'backoff plafond');
  assert(queueBackoffSeconds(2, () => 0) === Math.round(300 * 0.85), 'backoff plancher');
  for (let i = 0; i < 30; i += 1) {
    for (const index of [0, 1, 2, 8]) {
      const wait = queueBackoffSeconds(index);
      assert(wait >= 1 && wait <= QUEUE_WAIT_CAP_SECONDS, 'backoff hors plafond');
    }
  }

  const fenced = extractJson('```json\n{"title":"Brume","image_prompt":"a pond","video_motion_prompt":"slow mist"}\n```');
  assert(fenced && fenced.title === 'Brume' && fenced.image_prompt === 'a pond', 'JSON clôturé');
  const repaired = extractJson('note {"title":"Soir","image_prompt":"rain","video_motion_prompt":"mist"');
  assert(repaired && repaired.image_prompt === 'rain', 'JSON réparé');
  assert(extractJson('aucun objet') === null, 'JSON absent');
  assert(sanitizeTitle('Café sous la pluie !!!') === 'Cafe_sous_la_pluie', 'titre sanitisé');
  assert(sanitizeTitle('../etc/passwd') === 'etc_passwd', 'titre sans chemin');
  assert(sanitizeTitle('') === 'lofi_capybara', 'titre de repli');

  const variation = { timeOfDay: 'UNIQUE-TIME', season: 'UNIQUE-SEASON', weather: 'UNIQUE-WEATHER' };
  const prompt = buildExpansionPrompt('maison japonaise sous la pluie', 'cozy_room', variation);
  assert(prompt.includes('UNIQUE-TIME') && prompt.includes('UNIQUE-SEASON') && prompt.includes('UNIQUE-WEATHER'), 'variables uniques');
  assert(prompt.includes('capybara') && prompt.includes('Studio Ghibli'), 'contrainte capybara');
  assert(buildLofiImagePrompt({ image_prompt: 'pond' }, variation).includes('UNIQUE-WEATHER'), 'prompt image');

  const scenes = loadScenes(path.join(__dirname, 'scenes.json'));
  assert(scenes.length === 5 && scenes[0].description.includes('pluie'), 'scènes exemple');

  const blockedUrls = [
    'https://127.0.0.1/a.mp4',
    'https://2130706433/a.mp4',
    'https://0x7f000001/a.mp4',
    'https://0177.0.0.1/a.mp4',
    'https://[::ffff:127.0.0.1]/a.mp4',
    'https://[::ffff:7f00:1]/a.mp4',
    'https://[::1]/a.mp4',
    'https://169.254.169.254/latest/meta-data',
    'https://metadata.google.internal/a.mp4',
    'https://10.1.2.3/a.mp4',
    'https://192.168.0.5/a.mp4',
    'https://172.16.5.5/a.mp4',
    'http://cdn.example.com/a.mp4',
    'https://user:pass@cdn.example.com/a.mp4',
    'https://localhost/a.mp4',
  ];
  for (const blocked of blockedUrls) {
    let refused = false;
    try {
      assertPublicHttpsUrl(blocked);
    } catch (e) {
      refused = e.message === 'Hôte vidéo refusé';
    }
    assert(refused, 'URL privée acceptée');
  }
  assert(assertPublicHttpsUrl('https://cdn.example.com/video/a.mp4').includes('cdn.example.com'), 'CDN public');
  assert(normalizeRepo('https://[::ffff:7f00:1]/') !== null, 'hostname IPv6 parsé');

  let leaked = false;
  try {
    assertAllowedRepoUrl('https://x-access-token:SuperSecretValue@github.com/urdirditfurd/videowan2g.git');
  } catch (e) {
    leaked = e.message.includes('SuperSecretValue');
    assert(e.message === 'GITHUB_REPO_URL refusé', 'refus dépôt');
  }
  assert(!leaked, 'fuite dans le message');
  assert(assertAllowedRepoUrl(ALLOWED_REPO) === ALLOWED_REPO_GIT, 'dépôt canonique');
  assert(assertAllowedRepoUrl(ALLOWED_REPO_GIT) === ALLOWED_REPO_GIT, 'dépôt avec .git');
  for (const bad of ['http://github.com/urdirditfurd/videowan2g.git', 'https://github.com/other/videowan2g.git', 'git@github.com:urdirditfurd/videowan2g.git']) {
    let refused = false;
    try {
      assertAllowedRepoUrl(bad);
    } catch (e) {
      refused = true;
    }
    assert(refused, 'dépôt incorrect accepté');
  }

  const previousFetch = fetchImpl;
  const previousSleep = sleepImpl;
  try {
    const waits = [];
    sleepImpl = async (ms) => {
      waits.push(ms);
    };
    let calls = 0;
    fetchImpl = async () => {
      calls += 1;
      if (calls <= 11) return singleUseResponse(503, '{"error":"video_queue_full"}');
      return singleUseResponse(200, '{"ok":true}');
    };
    const accepted = await apiFetch('https://apihub.agnes-ai.com/v1/videos', { method: 'POST', body: '{}' }, 'Vidéo', 'video');
    assert(accepted.ok, 'file saturée non relancée');
    assert(calls === 12, 'budget de tentatives consommé par la file');
    assert(waits.length === 11, 'pauses de file');
    assert(waits[0] >= 51000 && waits[0] <= 69000, 'première pause');
    assert(waits[1] >= 102000 && waits[1] <= 138000, 'deuxième pause');
    assert(waits[2] >= 255000 && waits[2] <= 345000, 'troisième pause');
    assert(waits.every((wait) => wait <= QUEUE_WAIT_CAP_SECONDS * 1000), 'pause trop longue');

    calls = 0;
    waits.length = 0;
    fetchImpl = async () => {
      calls += 1;
      return singleUseResponse(429, '{"error":"video_queue_full"}');
    };
    let exhausted = false;
    try {
      await apiFetch('https://apihub.agnes-ai.com/v1/videos', {}, 'Vidéo', 'video');
    } catch (e) {
      exhausted = e.message.includes('répétitions de file');
    }
    assert(exhausted, 'plafond de file');
    assert(calls === VIDEO_QUEUE_MAX_RETRIES + 1, 'compteur de file');

    calls = 0;
    fetchImpl = async () => {
      calls += 1;
      return singleUseResponse(400, '{"error":"bad-request-marker"}');
    };
    let textError = '';
    try {
      await apiFetch('https://apihub.agnes-ai.com/v1/chat/completions', {}, 'Expansion', 'text');
    } catch (e) {
      textError = e.message;
    }
    assert(textError.includes('bad-request-marker') && calls === 1, 'erreur HTTP terminale');

    fetchImpl = async () => singleUseResponse(400, '{"error":"video-body-marker"}');
    const videoRes = await apiFetch('https://apihub.agnes-ai.com/v1/videos', {}, 'Vidéo', 'video');
    assert((await videoRes.text()).includes('video-body-marker'), 'corps vidéo conservé');

    let networkCalls = 0;
    fetchImpl = async () => {
      networkCalls += 1;
      if (networkCalls === 1) throw new Error('socket hang up');
      return singleUseResponse(200, '{"ok":true}');
    };
    const recovered = await apiFetch('https://apihub.agnes-ai.com/v1/chat/completions', {}, 'Expansion', 'text');
    assert(recovered.ok && networkCalls === 2, 'reprise réseau');

    let tries = 0;
    const retried = await retryTask('Image', async () => {
      tries += 1;
      if (tries < 3) throw new Error('boom');
      return 'ok';
    }, 4);
    assert(retried === 'ok' && waits.includes(8000) && waits.includes(32000), 'retryTask');

    const tempMedia = await fs.promises.mkdtemp(path.join(os.tmpdir(), 'lofi-media-'));
    const previousCwd = process.cwd();
    try {
      process.chdir(tempMedia);
      let downloadCalls = 0;
      fetchImpl = async () => {
        downloadCalls += 1;
        return {
          status: 302,
          ok: false,
          headers: { get: (name) => (String(name).toLowerCase() === 'location' ? 'https://2130706433/secret.mp4' : null) },
          body: null,
        };
      };
      let blockedRedirect = false;
      try {
        await downloadVideo('https://cdn.example.com/start.mp4', path.join(tempMedia, 'nope.mp4'));
      } catch (e) {
        blockedRedirect = e.message === 'Hôte vidéo refusé';
      }
      assert(blockedRedirect && downloadCalls === 1, 'redirection privée');

      const mp4 = fakeMp4();
      fetchImpl = async (url) => {
        if (String(url).endsWith('/final.mp4')) {
          return {
            status: 200,
            ok: true,
            headers: { get: () => null },
            body: (async function* () { yield mp4; })(),
          };
        }
        return {
          status: 302,
          ok: false,
          headers: { get: (name) => (String(name).toLowerCase() === 'location' ? '/final.mp4' : null) },
          body: null,
        };
      };
      const dest = path.join(tempMedia, 'generated_videos', 'ok.mp4');
      await downloadVideo('https://cdn.example.com/start.mp4', dest);
      await assertLocalVideo(dest);

      fetchImpl = async () => ({
        status: 200,
        ok: true,
        headers: { get: () => null },
        body: (async function* () { yield Buffer.from('this is not a media file!!!!'); })(),
      });
      const badDest = path.join(tempMedia, 'generated_videos', 'bad.mp4');
      let badHeader = false;
      try {
        await downloadVideo('https://cdn.example.com/bad.mp4', badDest);
      } catch (e) {
        badHeader = e.message === 'En-tête vidéo invalide';
      }
      assert(badHeader && !fs.existsSync(badDest), 'fichier sans en-tête');

      const target = path.join(tempMedia, 'target.mp4');
      const link = path.join(tempMedia, 'link.mp4');
      await fs.promises.writeFile(target, 'original-content');
      await fs.promises.symlink(target, link);
      let linkRefused = false;
      try {
        await writeExclusiveFile(link, mp4);
      } catch (e) {
        linkRefused = true;
      }
      assert(linkRefused, 'lien accepté');
      assert((await fs.promises.readFile(target, 'utf8')) === 'original-content', 'cible écrasée');
      let assertLink = false;
      try {
        await assertLocalVideo(link);
      } catch (e) {
        assertLink = true;
      }
      assert(assertLink, 'lecture de lien');

      const realDir = path.join(tempMedia, 'real-outside');
      await fs.promises.mkdir(realDir);
      const linkedDir = path.join(tempMedia, 'linked-dir');
      await fs.promises.symlink(realDir, linkedDir);
      let parentRefused = false;
      try {
        await writeExclusiveFile(path.join(linkedDir, 'escape.mp4'), mp4);
      } catch (e) {
        parentRefused = true;
      }
      assert(parentRefused && !fs.existsSync(path.join(realDir, 'escape.mp4')), 'dossier lien');
    } finally {
      process.chdir(previousCwd);
      await fs.promises.rm(tempMedia, { recursive: true, force: true });
    }
  } finally {
    fetchImpl = previousFetch;
    sleepImpl = previousSleep;
  }

  const starts = [];
  const paced = createRateLimitedQueue(1, 80);
  await Promise.all([
    paced.add(async () => {
      starts.push(Date.now());
      await new Promise((resolve) => setTimeout(resolve, 15));
    }),
    paced.add(async () => {
      starts.push(Date.now());
    }),
  ]);
  assert(starts.length === 2 && starts[1] - starts[0] >= 70, 'file cadencée');

  const gitDir = await fs.promises.mkdtemp(path.join(os.tmpdir(), 'lofi-git-'));
  try {
    const wrapper = writeFakeGit(gitDir);
    const logFile = path.join(gitDir, 'git-log.jsonl');
    await withAskPass(gitDir, async (git) => {
      await stageAndCommit(git, 'generated_videos/brume.mp4', 'feat: add new lofi capybara video: Brume');
      await pushHead(git);
    }, { binary: wrapper, extraEnv: { LOFI_FAKE_GIT_LOG: logFile } });
    const entries = readGitLog(logFile);
    assert(entries.length > 0, 'journal git vide');
    for (const entry of entries) assertEnvClean(entry.env, canaries);
    const commit = entries.find((entry) => entry.argv.includes('commit'));
    assert(commit && commit.argv.includes('--') && commit.argv.includes('commit.gpgsign=false'), 'commit pathspec');
    assert(commit.argv.includes('generated_videos/brume.mp4'), 'fichier commité');
    const pushed = entries.find((entry) => entry.argv.includes('push'));
    assert(pushed && pushed.argv.includes('credential.helper='), 'helper désactivé');
    const askpass = entries[0].env.GIT_ASKPASS;
    assert(askpass && !fs.existsSync(askpass), 'askpass supprimé');

    const repoDir = path.join(gitDir, 'repo');
    await fs.promises.mkdir(repoDir);
    const git = createGit(repoDir);
    await git.raw(['init', '-b', 'main']);
    await git.addRemote('origin', ALLOWED_REPO);
    await ensureOrigin(git, ALLOWED_REPO_GIT);
    const remotes = await git.getRemotes(true);
    const remoteUrl = (remotes[0] && (remotes[0].refs.push || remotes[0].refs.fetch)) || '';
    if (/x-access-token|ghp_|github_pat_|@/.test(remoteUrl)) throw new Error('Remote de test réécrit');
    if (normalizeRepo(remoteUrl) !== ALLOWED_REPO) throw new Error('Remote de test inattendu');

    await fs.promises.writeFile(path.join(repoDir, 'note.txt'), 'brouillon\n');
    await git.raw(['add', '--', 'note.txt']);
    const videoRel = path.join('generated_videos', 'scene_1.mp4');
    const videoAbs = path.join(repoDir, videoRel);
    await fs.promises.mkdir(path.dirname(videoAbs), { recursive: true });
    await fs.promises.writeFile(videoAbs, fakeMp4());
    const previousCwd = process.cwd();
    try {
      process.chdir(repoDir);
      await assertLocalVideo(videoAbs);
      await stageAndCommit(git, videoRel, 'feat: add new lofi capybara video: Test');
    } finally {
      process.chdir(previousCwd);
    }
    const shown = await git.raw(['show', '--name-only', '--pretty=format:', 'HEAD']);
    assert(shown.includes(videoRel) && !shown.includes('note.txt'), 'commit limité au mp4');
    const status = await git.status();
    assert(status.staged.includes('note.txt'), 'autre fichier laissé indexé');
  } finally {
    await fs.promises.rm(gitDir, { recursive: true, force: true });
  }

  log('success', 'TEST', 'Auto-test OK');
}

async function main() {
  if (process.argv.includes('--self-test')) {
    await runSelfTest();
    return;
  }
  const apiKey = String(process.env.AGNES_API_KEY || '').trim();
  const token = String(process.env.GITHUB_TOKEN || '').trim();
  assertApiKey(apiKey);
  assertToken(token);
  process.env.AGNES_API_KEY = apiKey;
  process.env.GITHUB_TOKEN = token;
  assertAllowedRepoUrl(process.env.GITHUB_REPO_URL);
  const outputDir = resolveInsideCwd(process.env.OUTPUT_DIR || 'generated_videos');
  const scenes = loadScenes(resolveInsideCwd(process.env.SCENES_FILE || 'scenes.json'));
  await fs.promises.mkdir(outputDir, { recursive: true });
  let failures = 0;
  for (let i = 0; i < scenes.length; i += 1) {
    if (runtime.stopRequested) break;
    try {
      await runScene(scenes[i], i, outputDir);
    } catch (e) {
      failures += 1;
      log('error', 'SCENE', e.message);
      if (e.message === 'Arrêt demandé') break;
    }
  }
  if (runtime.stopRequested) process.exitCode = 130;
  else if (failures > 0) process.exitCode = 1;
}

process.on('SIGINT', () => {
  sigintCount += 1;
  runtime.stopRequested = true;
  log('warn', 'STOP', 'Arrêt demandé');
  if (sigintCount >= 2) process.exit(130);
});

if (require.main === module) {
  main().catch((e) => {
    console.error(redact(e && e.stack ? e.stack : e));
    process.exitCode = 1;
  });
}
