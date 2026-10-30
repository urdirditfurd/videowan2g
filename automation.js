'use strict';

const dns = require('dns').promises;
const fs = require('fs');
const http = require('http');
const net = require('net');
const os = require('os');
const path = require('path');
const { Readable } = require('stream');
const { pipeline } = require('stream/promises');
const nodeFetch = require('node-fetch');
const { simpleGit } = require('simple-git');

require('dotenv').config({ quiet: true });

const API_BASE = 'https://apihub.agnes-ai.com/v1';
const POLL_BASE = 'https://apihub.agnes-ai.com/agnesapi';
const MODEL_TEXT = 'agnes-2.5-flash';
const MODEL_IMAGE = 'agnes-image-2.5-flash';
const MODEL_VIDEO = 'agnes-video-2.5-flash';
const ALLOWED_REPO = 'https://github.com/urdirditfurd/videowan2g';
const MAX_VIDEO_BYTES = 100 * 1024 * 1024;
const MAX_DOWNLOAD_REDIRECTS = 2;
const QUEUE_BACKOFF_SECONDS = [60, 120, 300];
const QUEUE_WAIT_CAP_SECONDS = 360;
const IMAGE_INTERVAL_MIN = 3200;
const IMAGE_INTERVAL_MAX = 12000;
const VIDEO_INTERVAL_MIN = 62000;
const VIDEO_INTERVAL_MAX = 90000;
const VIDEO_INTERVAL_SAFE = 75000;
const VIDEO_FIRST_POLL_DELAY_SEC = 80;
const VIDEO_MAX_POLL_ATTEMPTS = 100;
const OPEN_EXCL = fs.constants.O_CREAT | fs.constants.O_EXCL | fs.constants.O_NOFOLLOW | fs.constants.O_WRONLY;
const OPEN_READ = fs.constants.O_RDONLY | fs.constants.O_NOFOLLOW;
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
  'gentle noon light',
  'warm mid-afternoon',
  'golden hour',
  'dusk with paper lanterns',
  'quiet starry night',
];
const SEASONS = ['early spring', 'humid summer', 'late autumn', 'quiet winter'];
const WEATHERS = [
  'fine rain',
  'light drizzle',
  'clear soft sky',
  'layered gentle clouds',
  'low mist',
  'a few slow snowflakes',
  'the moment just after rain',
  'a light breeze',
];

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

const hooks = {
  sleep(ms) {
    return new Promise((resolve, reject) => {
      const step = 250;
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
  },
  random() {
    return Math.random();
  },
};

function sleep(ms) {
  return hooks.sleep(ms);
}

function redact(value) {
  return String(value == null ? '' : value)
    .replace(/sk-[A-Za-z0-9_-]{4,}/g, 'sk-***')
    .replace(/ghp_[A-Za-z0-9_]{4,}/g, 'ghp_***')
    .replace(/github_pat_[A-Za-z0-9_]{4,}/g, 'github_pat_***')
    .replace(/x-access-token:[^\s@]+/gi, 'x-access-token:***')
    .replace(/Bearer\s+[A-Za-z0-9._~+/-]{8,}/gi, 'Bearer ***');
}

function log(level, category, message) {
  const line = `${new Date().toISOString()} [${level}] [${category}] ${redact(message)}`;
  if (level === 'error' || level === 'warn') console.error(line);
  else console.log(line);
}

function getApiKey() {
  const key = String(process.env.AGNES_API_KEY || '').trim();
  if (!key) throw new Error('AGNES_API_KEY manquante');
  return key;
}

function queueMaxRetries() {
  const parsed = parseInt(process.env.VIDEO_QUEUE_MAX_RETRIES || '12', 10);
  if (!Number.isFinite(parsed) || parsed < 1) return 12;
  return Math.min(parsed, 100);
}

function computeQueueWaitSeconds(queueAttempt, retryAfterSeconds, randomValue) {
  const roll = Number.isFinite(randomValue) ? randomValue : 0;
  let seconds;
  if (retryAfterSeconds > 0) {
    seconds = retryAfterSeconds;
  } else {
    const index = Math.min(Math.max(0, queueAttempt), QUEUE_BACKOFF_SECONDS.length - 1);
    const jitter = Math.floor(roll * 31);
    seconds = QUEUE_BACKOFF_SECONDS[index] + jitter;
  }
  return Math.min(QUEUE_WAIT_CAP_SECONDS, Math.max(1, Math.round(seconds)));
}

function parseRetryAfter(res) {
  try {
    const raw = res.headers.get('retry-after');
    const seconds = parseInt(raw || '0', 10);
    return Number.isFinite(seconds) && seconds > 0 ? seconds : 0;
  } catch (err) {
    return 0;
  }
}

function authHeaders() {
  return {
    Authorization: `Bearer ${getApiKey()}`,
    'Content-Type': 'application/json',
  };
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

function noteRateLimit(kind) {
  if (kind === 'image') createImageIntervalMs = Math.min(IMAGE_INTERVAL_MAX, createImageIntervalMs + 1000);
  if (kind === 'video') createVideoIntervalMs = Math.min(VIDEO_INTERVAL_MAX, createVideoIntervalMs + 6000);
}

async function apiFetch(url, options = {}, label = 'API', kind = 'default') {
  const maxAttempts = kind === 'video' ? 10 : kind === 'image' ? 10 : 8;
  let attempt = 0;
  let queueRetries = 0;
  const queueLimit = queueMaxRetries();

  while (attempt < maxAttempts) {
    if (state.stopRequested) throw new Error('Arrêt demandé');
    const method = options.method || 'GET';
    log('api', `REQ ${label}`, `${method} ${url} (tentative ${attempt + 1}/${maxAttempts})`);

    let res;
    try {
      res = await nodeFetch(url, { timeout: 120000, ...options, redirect: 'error' });
    } catch (err) {
      if (err && err.message === 'Arrêt demandé') throw err;
      const wait = Math.min(kind === 'video' ? 120 : 45, kind === 'video' ? 10 + attempt * 8 : 3 + attempt * 4);
      log('warn', 'NETWORK', `${err.message} — reprise dans ${wait}s`);
      await sleep(wait * 1000);
      attempt += 1;
      continue;
    }

    const retryAfter = parseRetryAfter(res);
    let errorText = '';
    if (!res.ok && ([400, 401, 404, 429, 500, 502, 503, 504, 520].includes(res.status) || kind === 'video')) {
      try {
        errorText = (await res.clone().text()).slice(0, 1200);
      } catch (err) {
        errorText = '';
      }
    }
    log('response', `RÉP ${label}`, `HTTP ${res.status}`);

    if (res.ok) {
      noteSuccess(kind);
      return res;
    }

    if ([400, 401, 404].includes(res.status) && kind !== 'video') {
      throw new Error(redact(`HTTP ${res.status} — ${errorText.slice(0, 300)}`));
    }

    const queueFull = kind === 'video' && /video_queue_full|queue is full/i.test(errorText);
    if (queueFull) {
      if (queueRetries >= queueLimit) {
        throw new Error('File vidéo saturée — plafond VIDEO_QUEUE_MAX_RETRIES atteint');
      }
      const wait = computeQueueWaitSeconds(queueRetries, retryAfter, hooks.random());
      queueRetries += 1;
      videoAttemptsSinceLastError = 0;
      log('warn', 'VIDEO_QUEUE', `File saturée — pause ${wait}s (${queueRetries}/${queueLimit})`);
      await sleep(wait * 1000);
      continue;
    }

    if (res.status === 429) {
      const fallback = Math.min(kind === 'video' ? 180 : 120, kind === 'video' ? 60 + attempt * 8 : 10 + attempt * 12);
      const wait = Math.min(QUEUE_WAIT_CAP_SECONDS, retryAfter > 0 ? retryAfter : fallback);
      noteRateLimit(kind);
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
  return {
    title: titleMatch ? titleMatch[1].replace(/\\n/g, ' ').replace(/\\"/g, '"') : 'Ambiance Lofi',
    image_prompt: imgMatch ? imgMatch[1].replace(/\\n/g, ' ').replace(/\\"/g, '"') : '',
    video_motion_prompt: vidMatch ? vidMatch[1].replace(/\\n/g, ' ').replace(/\\"/g, '"') : '',
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
  try {
    return JSON.parse(source);
  } catch (err) {
    /* stratégie suivante */
  }
  try {
    return JSON.parse(escapeControlCharsInStrings(source));
  } catch (err) {
    /* stratégie suivante */
  }
  try {
    const last = source.lastIndexOf('}');
    if (last > 0) return JSON.parse(escapeControlCharsInStrings(source.slice(0, last + 1)));
  } catch (err) {
    /* stratégie suivante */
  }
  try {
    return JSON.parse(balanceJson(escapeControlCharsInStrings(source)));
  } catch (err) {
    /* stratégie suivante */
  }
  const salvaged = salvageExpansion(source);
  if (salvaged && (salvaged.image_prompt || salvaged.video_motion_prompt)) return salvaged;
  return null;
}

function rollVariation(randomFn = hooks.random) {
  const pick = (list) => {
    const index = Math.floor(randomFn() * list.length);
    return list[Math.min(list.length - 1, Math.max(0, index))];
  };
  return {
    timeOfDay: pick(TIMES_OF_DAY),
    season: pick(SEASONS),
    weather: pick(WEATHERS),
  };
}

function buildExpansionPrompt(userDescription, mood, variation) {
  const selected = MOODS[mood] || MOODS.rainy_forest;
  const unique = variation || rollVariation();
  return [
    'You are an art director for Studio Ghibli-inspired lofi ambience videos.',
    '',
    'USER SCENE IDEA (French):',
    userDescription,
    '',
    'AMBIENCE (French mood chosen):',
    `${selected.label} — ${selected.desc}`,
    '',
    'UNIQUENESS VARIABLES (mandatory — they must visibly change the picture, even when the user idea resembles another scene):',
    `Time of day: ${unique.timeOfDay}`,
    `Season: ${unique.season}`,
    `Weather: ${unique.weather}`,
    'Reflect all three in the title, the image_prompt and the video_motion_prompt. Produce a scene that cannot be confused with a generic version of the same idea.',
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

async function generateExpansion(userDescription, mood, variation) {
  const unique = variation || rollVariation();
  const prompt = buildExpansionPrompt(userDescription, mood, unique);
  const body = {
    model: MODEL_TEXT,
    messages: [
      {
        role: 'system',
        content: 'You are a Studio Ghibli-inspired art director specialized in lofi ambience videos. You always return strict JSON matching the requested schema exactly. Every scene includes a cute calm capybara. Every scene is peaceful, detailed, watercolor, and loopable with slow gentle motion. Honor the requested time of day, season and weather so each scene stays visually unique. No text, no watermark. Return only JSON.',
      },
      { role: 'user', content: prompt },
    ],
    temperature: 0.85,
    max_tokens: 2000,
    response_format: { type: 'json_object' },
  };
  const res = await apiFetch(API_BASE + '/chat/completions', {
    method: 'POST',
    headers: authHeaders(),
    body: JSON.stringify(body),
  }, 'Expansion lofi', 'text');
  if (!res.ok) {
    const err = await res.text();
    throw new Error(redact(`HTTP ${res.status} — ${err.slice(0, 300)}`));
  }
  const data = await res.json();
  const content = data.choices && data.choices[0] && data.choices[0].message && data.choices[0].message.content;
  const expansion = extractJson(content);
  if (!expansion || !expansion.image_prompt) throw new Error('Expansion invalide');
  if (!expansion.title) expansion.title = 'Ambiance Lofi';
  if (!expansion.video_motion_prompt) {
    expansion.video_motion_prompt = 'Gentle slow motion: capybara breathing softly, leaves drifting down, faint mist, seamless loop.';
  }
  expansion.variation = unique;
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
    } catch (err) {
      lastError = err;
      if (err.message === 'Arrêt demandé') throw err;
      log('warn', 'RETRY', `${name} échoué : ${err.message}`);
      if (i < maxAttempts) {
        const wait = Math.min(90, 8 * i * i);
        log('info', 'RETRY', `${name} — reprise automatique dans ${wait}s`);
        await sleep(wait * 1000);
      }
    }
  }
  throw lastError || new Error(`${name} échoué`);
}

function createRateLimitedQueue(maxConcurrent, minIntervalMs) {
  let running = 0;
  let lastStart = 0;
  const queue = [];
  let resolveIdle = null;

  function interval() {
    return typeof minIntervalMs === 'function' ? minIntervalMs() : minIntervalMs;
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
      setTimeout(tryNext, wait);
      return;
    }
    const task = queue.shift();
    running += 1;
    lastStart = Date.now();
    Promise.resolve()
      .then(task.fn)
      .then(task.resolve, task.reject)
      .finally(() => {
        running -= 1;
        setTimeout(tryNext, 0);
      });
  }

  return {
    add(fn) {
      return new Promise((resolve, reject) => {
        queue.push({ fn, resolve, reject });
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

const imageQueue = createRateLimitedQueue(1, () => createImageIntervalMs);
const videoQueue = createRateLimitedQueue(1, () => createVideoIntervalMs);

async function generateImage(prompt, format) {
  return retryTask('Image', async () => {
    const ratio = format === '9:16' ? '9:16' : '16:9';
    const res = await apiFetch(API_BASE + '/images/generations', {
      method: 'POST',
      headers: authHeaders(),
      body: JSON.stringify({ model: MODEL_IMAGE, prompt, size: '1K', ratio, n: 1 }),
    }, 'Image', 'image');
    if (!res.ok) {
      const err = await res.text();
      throw new Error(redact(`HTTP ${res.status} — ${err.slice(0, 500)}`));
    }
    const data = await res.json();
    const item = data.data && data.data[0];
    if (!item) throw new Error('Pas de donnée image');
    if (item.url) return item.url;
    if (item.b64_json) return `data:image/png;base64,${item.b64_json}`;
    throw new Error('Format image inconnu');
  }, 4);
}

function assertVideoId(videoId) {
  if (!/^[\w.-]{1,128}$/.test(String(videoId || ''))) throw new Error('video_id inattendu');
  return String(videoId);
}

async function generateVideo(imageUrl, prompt, format) {
  return retryTask('Vidéo', async () => {
    const aspect = format === '9:16' ? '9:16' : '16:9';
    const res = await apiFetch(API_BASE + '/videos', {
      method: 'POST',
      headers: authHeaders(),
      body: JSON.stringify({
        model: MODEL_VIDEO,
        prompt,
        seconds: String(state.durationSeconds),
        mode: 'keyframe',
        size: '720P',
        aspect_ratio: aspect,
        first_frame: imageUrl,
        last_frame: imageUrl,
      }),
    }, 'Vidéo', 'video');
    if (!res.ok) {
      const err = await res.text();
      throw new Error(redact(`HTTP ${res.status} — ${err.slice(0, 500)}`));
    }
    const data = await res.json();
    const videoId = data.video_id || data.id || data.task_id;
    if (!videoId) throw new Error('Pas de video_id');
    return assertVideoId(videoId);
  }, 3);
}

async function pollVideo(videoId, onProgress) {
  const safeId = assertVideoId(videoId);
  const estimatedMs = 90000 + state.durationSeconds * 7000;
  const minWait = Math.max(25, Math.min(VIDEO_FIRST_POLL_DELAY_SEC, Math.floor(estimatedMs / 1000 * 0.35)));
  let remaining = minWait;
  log('info', 'POLLING', `Attente initiale ${remaining}s pour ${safeId}`);
  while (remaining > 0) {
    if (state.stopRequested) throw new Error('Arrêt demandé');
    onProgress(`File vidéo — ${remaining}s`);
    await sleep(1000);
    remaining -= 1;
  }
  let attempt = 0;
  while (attempt < VIDEO_MAX_POLL_ATTEMPTS) {
    if (state.stopRequested) throw new Error('Arrêt demandé');
    attempt += 1;
    const url = `${POLL_BASE}?video_id=${encodeURIComponent(safeId)}&model_name=${encodeURIComponent(MODEL_VIDEO)}`;
    let res;
    try {
      res = await apiFetch(url, { method: 'GET', headers: { Authorization: `Bearer ${getApiKey()}` } }, 'Polling', 'video');
    } catch (err) {
      if (err.message === 'Arrêt demandé') throw err;
      log('warn', 'POLLING', `Erreur transitoire : ${err.message}`);
      await sleep(Math.min(90000, 15000 + attempt * 1500));
      continue;
    }
    let payload;
    try {
      payload = await res.json();
    } catch (err) {
      log('warn', 'POLLING', 'Réponse non JSON, reprise');
      await sleep(Math.min(60000, 10000 + attempt * 1000));
      continue;
    }
    const status = String(payload.status || 'unknown').toLowerCase();
    const progress = Number(payload.progress || 0);
    onProgress(`Création — ${progress}%`);
    log('response', `POLL #${attempt}`, `Statut : ${status} (${progress}%)`);
    if (status === 'completed' || status === 'succeeded' || status === 'done') {
      const videoUrl = payload.url || (payload.metadata && payload.metadata.url) || (payload.output && payload.output.url);
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

function stripHost(hostname) {
  return String(hostname || '').toLowerCase().replace(/^\[|\]$/g, '').replace(/\.$/, '');
}

function ipv4ToInt(host) {
  const parts = host.split('.');
  if (parts.length !== 4) return null;
  let value = 0;
  for (const part of parts) {
    if (!/^\d{1,3}$/.test(part)) return null;
    const octet = Number(part);
    if (octet > 255) return null;
    value = ((value << 8) + octet) >>> 0;
  }
  return value;
}

function inCidr(ip, base, bits) {
  const mask = bits === 0 ? 0 : (0xffffffff << (32 - bits)) >>> 0;
  return (ip & mask) === ((base >>> 0) & mask);
}

function isBlockedIpv4(host) {
  const ip = ipv4ToInt(host);
  if (ip == null) return true;
  const ranges = [
    [0x00000000, 8],
    [0x0a000000, 8],
    [0x64400000, 10],
    [0x7f000000, 8],
    [0xa9fe0000, 16],
    [0xac100000, 12],
    [0xc0a80000, 16],
    [0xc0000000, 24],
    [0xc0000200, 24],
    [0xc6120000, 15],
    [0xc6336400, 24],
    [0xcb007100, 24],
    [0xe0000000, 4],
    [0xf0000000, 4],
  ];
  return ranges.some(([base, bits]) => inCidr(ip, base, bits));
}

function isBlockedIpv6(host) {
  const value = host.toLowerCase();
  if (value === '::' || value === '::1') return true;
  if (value.startsWith('fc') || value.startsWith('fd')) return true;
  if (/^fe[89ab]/.test(value)) return true;
  if (value.startsWith('ff')) return true;
  if (value.startsWith('2001:db8')) return true;
  if (value.startsWith('::ffff:')) return true;
  return false;
}

function isBlockedAddress(hostname) {
  const host = stripHost(hostname);
  if (!host) return true;
  if (
    host === 'localhost'
    || host.endsWith('.localhost')
    || host.endsWith('.local')
    || host.endsWith('.internal')
    || host === 'metadata'
    || host === 'metadata.google.internal'
    || host === 'metadata.google.com'
  ) {
    return true;
  }
  const mappedDotted = host.match(/^::ffff:(\d{1,3}(?:\.\d{1,3}){3})$/);
  if (mappedDotted) return isBlockedAddress(mappedDotted[1]);
  const mappedHex = host.match(/^::ffff:([0-9a-f]{1,4}):([0-9a-f]{1,4})$/);
  if (mappedHex) {
    const hi = parseInt(mappedHex[1], 16);
    const lo = parseInt(mappedHex[2], 16);
    const ip = `${(hi >> 8) & 255}.${hi & 255}.${(lo >> 8) & 255}.${lo & 255}`;
    return isBlockedAddress(ip);
  }
  if (host.includes(':')) return isBlockedIpv6(host);
  if (/^\d{1,3}(\.\d{1,3}){3}$/.test(host)) return isBlockedIpv4(host);
  return false;
}

function assertPublicHttpUrl(raw) {
  let url;
  try {
    url = new URL(raw);
  } catch (err) {
    throw new Error('URL média invalide');
  }
  if (url.username || url.password) throw new Error('URL média avec identifiant refusée');
  if (isBlockedAddress(url.hostname)) throw new Error('Hôte média refusé');
  if (url.protocol !== 'https:') throw new Error('Seul le téléchargement HTTPS est autorisé');
  return url;
}

async function assertResolvedPublic(hostname) {
  const host = stripHost(hostname);
  if (net.isIP(host) || host.includes(':')) return;
  let records;
  try {
    records = await dns.lookup(host, { all: true, verbatim: true });
  } catch (err) {
    throw new Error('Résolution DNS impossible');
  }
  if (!records.length) throw new Error('Résolution DNS vide');
  for (const record of records) {
    if (isBlockedAddress(record.address)) throw new Error('Hôte résolu vers une adresse privée');
  }
}

function hasMediaHeader(buf) {
  if (buf.length >= 8 && buf.subarray(4, 8).toString('ascii') === 'ftyp') return true;
  if (buf.length >= 4 && buf[0] === 0x1a && buf[1] === 0x45 && buf[2] === 0xdf && buf[3] === 0xa3) return true;
  return false;
}

function safeUnlinkRegular(filePath) {
  let st;
  try {
    st = fs.lstatSync(filePath);
  } catch (err) {
    if (err.code === 'ENOENT') return;
    throw err;
  }
  if (st.isSymbolicLink() || !st.isFile()) {
    throw new Error('Refus de supprimer un chemin qui n’est pas un fichier régulier');
  }
  fs.unlinkSync(filePath);
}

async function writeExclusive(dest, bodyStream) {
  fs.mkdirSync(path.dirname(dest), { recursive: true });
  let existing = null;
  try {
    existing = fs.lstatSync(dest);
  } catch (err) {
    if (err.code !== 'ENOENT') throw err;
  }
  if (existing) {
    if (existing.isSymbolicLink()) throw new Error('Refus : la destination est un lien symbolique');
    throw new Error('Refus : le fichier vidéo existe déjà');
  }
  let fd;
  try {
    fd = fs.openSync(dest, OPEN_EXCL, 0o644);
  } catch (err) {
    if (err.code === 'ELOOP' || err.code === 'EEXIST') {
      throw new Error('Refus : destination existante ou lien symbolique');
    }
    throw err;
  }
  let handedOff = false;
  try {
    let written = 0;
    await pipeline(
      bodyStream,
      async function* limit(source) {
        for await (const chunk of source) {
          const buffer = Buffer.isBuffer(chunk) ? chunk : Buffer.from(chunk);
          written += buffer.length;
          if (written > MAX_VIDEO_BYTES) throw new Error('Vidéo trop volumineuse');
          yield buffer;
        }
      },
      fs.createWriteStream(dest, { fd, autoClose: true }),
    );
    handedOff = true;
  } catch (err) {
    if (!handedOff) {
      try { fs.closeSync(fd); } catch (closeErr) { /* déjà fermé */ }
    }
    safeUnlinkRegular(dest);
    throw err;
  }
}

function assertLocalVideo(filePath) {
  const st = fs.lstatSync(filePath);
  if (st.isSymbolicLink()) throw new Error('Lien symbolique refusé');
  if (!st.isFile()) throw new Error('La vidéo n’est pas un fichier régulier');
  if (st.size >= MAX_VIDEO_BYTES) throw new Error('Vidéo trop volumineuse (≥ 100 Mo), commit annulé');
  if (st.size < 16) throw new Error('Fichier vidéo trop petit');
  const fd = fs.openSync(filePath, OPEN_READ);
  try {
    const buf = Buffer.alloc(32);
    const read = fs.readSync(fd, buf, 0, 32, 0);
    if (!hasMediaHeader(buf.subarray(0, read))) throw new Error('En-tête média invalide');
  } finally {
    fs.closeSync(fd);
  }
}

function isRedirectStatus(status) {
  return status === 301 || status === 302 || status === 303 || status === 307 || status === 308;
}

async function downloadVideo(videoUrl, dest, deps = {}) {
  const doFetch = deps.fetchImpl || nodeFetch;
  let current = videoUrl;
  let redirects = 0;
  let response;
  while (true) {
    const url = assertPublicHttpUrl(current);
    await assertResolvedPublic(url.hostname);
    response = await doFetch(current, {
      redirect: 'manual',
      timeout: 180000,
      headers: { 'User-Agent': 'videowan2g-lofi/1.0' },
    });
    if (isRedirectStatus(response.status)) {
      redirects += 1;
      if (redirects > MAX_DOWNLOAD_REDIRECTS) throw new Error('Trop de redirections (max 2)');
      const location = response.headers.get('location');
      if (!location) throw new Error('Redirection sans en-tête Location');
      current = new URL(location, current).href;
      continue;
    }
    break;
  }
  if (!response.ok) throw new Error(`Téléchargement HTTP ${response.status}`);
  const advertised = Number(response.headers.get('content-length') || 0);
  if (Number.isFinite(advertised) && advertised >= MAX_VIDEO_BYTES) {
    throw new Error('Vidéo trop volumineuse (≥ 100 Mo), commit annulé');
  }
  await writeExclusive(dest, response.body);
  try {
    assertLocalVideo(dest);
  } catch (err) {
    safeUnlinkRegular(dest);
    throw err;
  }
  return dest;
}

function canonicalRepoUrl(raw) {
  let url;
  try {
    url = new URL(String(raw || '').trim());
  } catch (err) {
    throw new Error('URL de dépôt invalide');
  }
  if (url.username || url.password) throw new Error('URL de dépôt refusée : ne pas inclure d’identifiant');
  const repoPath = url.pathname.replace(/\.git$/i, '').replace(/\/+$/, '');
  if (url.protocol !== 'https:' || url.hostname.toLowerCase() !== 'github.com' || repoPath !== '/urdirditfurd/videowan2g') {
    throw new Error('URL de dépôt refusée : seul https://github.com/urdirditfurd/videowan2g est autorisé');
  }
  return ALLOWED_REPO;
}

function sameRepoIgnoringCredentials(raw) {
  try {
    const url = new URL(String(raw || '').trim());
    const repoPath = url.pathname.replace(/\.git$/i, '').replace(/\/+$/, '');
    return url.protocol === 'https:' && url.hostname.toLowerCase() === 'github.com' && repoPath === '/urdirditfurd/videowan2g';
  } catch (err) {
    return false;
  }
}

function classifyOrigin(raw) {
  if (!raw) return 'missing';
  if (!sameRepoIgnoringCredentials(raw)) return 'reject';
  try {
    canonicalRepoUrl(raw);
    return 'ok';
  } catch (err) {
    if (String(err.message).includes('identifiant')) return 'reset';
    return 'reject';
  }
}

function buildGitEnv(extra = {}) {
  const authorName = extra.GIT_AUTHOR_NAME || process.env.GIT_AUTHOR_NAME || 'Lofi Capybara Bot';
  const authorEmail = extra.GIT_AUTHOR_EMAIL || process.env.GIT_AUTHOR_EMAIL || 'lofi-capybara-bot@users.noreply.github.com';
  const env = {
    PATH: extra.PATH || process.env.PATH || '/usr/bin:/bin',
    HOME: '',
    GIT_CONFIG_NOSYSTEM: '1',
    GIT_TERMINAL_PROMPT: '0',
    GIT_AUTHOR_NAME: authorName,
    GIT_AUTHOR_EMAIL: authorEmail,
    GIT_COMMITTER_NAME: extra.GIT_COMMITTER_NAME || process.env.GIT_COMMITTER_NAME || authorName,
    GIT_COMMITTER_EMAIL: extra.GIT_COMMITTER_EMAIL || process.env.GIT_COMMITTER_EMAIL || authorEmail,
  };
  if (extra.GIT_ASKPASS) env.GIT_ASKPASS = extra.GIT_ASKPASS;
  return env;
}

function createGit(extra = {}) {
  const env = buildGitEnv(extra);
  const git = simpleGit({
    baseDir: extra.baseDir || process.cwd(),
    trimmed: true,
    unsafe: {
      allowUnsafeAskPass: true,
      allowUnsafeCredentialHelper: true,
    },
    allowEnvironment: GIT_ENV_ALLOW,
  });
  git.env(env);
  return git;
}

function assertGitHubToken(token) {
  if (!token || typeof token !== 'string') throw new Error('GITHUB_TOKEN manquant');
  if (token.length < 8 || token.length > 300 || /[\s\0]/.test(token)) throw new Error('GITHUB_TOKEN invalide');
  return token;
}

function createAskPass(token) {
  const secret = assertGitHubToken(token);
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'lofi-askpass-'));
  const tokenFile = path.join(dir, 'token');
  const script = path.join(dir, 'askpass.cjs');
  fs.writeFileSync(tokenFile, secret, { mode: 0o600 });
  const source = [
    '#!/usr/bin/env node',
    "'use strict';",
    "const fs = require('fs');",
    'const prompt = String(process.argv[2] || "");',
    `const token = fs.readFileSync(${JSON.stringify(tokenFile)}, 'utf8');`,
    "if (/password/i.test(prompt)) process.stdout.write(token + '\\n');",
    "else process.stdout.write('x-access-token\\n');",
    '',
  ].join('\n');
  fs.writeFileSync(script, source, { mode: 0o700 });
  return {
    script,
    cleanup() {
      fs.rmSync(dir, { recursive: true, force: true });
    },
  };
}

async function ensureOrigin(git) {
  let raw = '';
  try {
    raw = String(await git.raw(['remote', 'get-url', 'origin'])).trim();
  } catch (err) {
    raw = '';
  }
  const action = classifyOrigin(raw);
  if (action === 'missing') {
    await git.raw(['remote', 'add', 'origin', ALLOWED_REPO]);
    return;
  }
  if (action === 'ok') return;
  if (action === 'reset') {
    await git.raw(['remote', 'set-url', 'origin', ALLOWED_REPO]);
    return;
  }
  throw new Error('Remote origin refusé : seul https://github.com/urdirditfurd/videowan2g est autorisé');
}

function sanitizeTitle(title) {
  const base = String(title || 'ambiance-lofi')
    .normalize('NFKD')
    .replace(/[\u0300-\u036f]/g, '')
    .replace(/[^a-zA-Z0-9]+/g, '_')
    .replace(/^_+|_+$/g, '')
    .slice(0, 80);
  return base || 'ambiance-lofi';
}

function sanitizeCommitTitle(title) {
  const clean = String(title || 'ambiance lofi').replace(/[\r\n\0]/g, ' ').replace(/\s+/g, ' ').trim().slice(0, 120);
  return clean || 'ambiance lofi';
}

function videoOutputPath(title) {
  const dir = path.resolve(process.cwd(), 'generated_videos');
  fs.mkdirSync(dir, { recursive: true });
  const name = `${sanitizeTitle(title)}_${Date.now()}.mp4`;
  const full = path.resolve(dir, name);
  const relative = path.relative(dir, full);
  if (relative.startsWith('..') || path.isAbsolute(relative)) throw new Error('Nom de fichier refusé');
  return full;
}

async function publishVideo(filePath, title, options = {}) {
  const repoRoot = options.repoRoot || process.cwd();
  const relative = path.relative(repoRoot, filePath);
  if (!relative || relative.startsWith('..') || path.isAbsolute(relative)) {
    throw new Error('La vidéo doit rester dans le dépôt');
  }
  assertLocalVideo(filePath);
  canonicalRepoUrl(options.repoUrl || process.env.GITHUB_REPO_URL);
  const token = assertGitHubToken(options.token || process.env.GITHUB_TOKEN);
  const ask = createAskPass(token);
  try {
    const git = createGit({
      baseDir: repoRoot,
      GIT_ASKPASS: ask.script,
      PATH: options.path,
    });
    await ensureOrigin(git);
    await git.raw(['add', '-f', '--', relative]);
    const message = `feat: add new lofi capybara video: ${sanitizeCommitTitle(title)}`;
    await git.raw(['-c', 'commit.gpgsign=false', 'commit', '-m', message, '--', relative]);
    const push = ['-c', 'credential.helper=', 'push', 'origin', 'HEAD'];
    try {
      await git.raw(push);
    } catch (err) {
      log('warn', 'GIT', `Push interrompu, second essai : ${err.message}`);
      await sleep(5000);
      await git.raw(push);
    }
    log('success', 'GIT', `Publié : ${relative}`);
  } finally {
    ask.cleanup();
  }
}

function normalizeDuration(value) {
  const raw = value != null && value !== '' ? value : process.env.VIDEO_DURATION;
  const parsed = parseInt(raw == null || raw === '' ? '5' : String(raw), 10);
  if (!Number.isFinite(parsed) || parsed < 5 || parsed > 10) throw new Error('Durée invalide (5 à 10 secondes)');
  return parsed;
}

function normalizeFormat(value) {
  const format = value || process.env.VIDEO_FORMAT || '16:9';
  if (format === '16:9' || format === '9:16') return format;
  throw new Error('Format vidéo inconnu');
}

function normalizeMood(value) {
  const mood = value || process.env.VIDEO_MOOD || 'rainy_forest';
  if (!Object.prototype.hasOwnProperty.call(MOODS, mood)) throw new Error('Ambiance inconnue');
  return mood;
}

function loadScenes(file = path.resolve(process.cwd(), 'scenes.json')) {
  const raw = JSON.parse(fs.readFileSync(file, 'utf8'));
  if (!Array.isArray(raw) || raw.length === 0) throw new Error('scenes.json doit être un tableau non vide');
  return raw.map((item, index) => {
    if (typeof item === 'string') {
      const description = item.trim();
      if (description.length < 4) throw new Error(`Scène ${index + 1} trop courte`);
      return { description };
    }
    if (item && typeof item.description === 'string' && item.description.trim().length >= 4) {
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

function validateRuntimeConfig() {
  getApiKey();
  canonicalRepoUrl(process.env.GITHUB_REPO_URL);
  assertGitHubToken(process.env.GITHUB_TOKEN);
}

async function processScene(scene, index) {
  state.durationSeconds = normalizeDuration(scene.durationSeconds);
  state.format = normalizeFormat(scene.format);
  state.mood = normalizeMood(scene.mood);
  const variation = rollVariation();
  log('info', 'SCENE', `#${index + 1} ${scene.description} · ${state.mood} · ${variation.timeOfDay} / ${variation.season} / ${variation.weather}`);
  const expansion = await generateExpansion(scene.description, state.mood, variation);
  const imageUrl = await imageQueue.add(() => generateImage(buildLofiImagePrompt(expansion), state.format));
  log('success', 'IMAGE', 'Image keyframe prête');
  createVideoIntervalMs = VIDEO_INTERVAL_SAFE;
  videoAttemptsSinceLastError = 0;
  const videoId = await videoQueue.add(() => generateVideo(imageUrl, buildLofiVideoPrompt(expansion), state.format));
  log('success', 'VIDÉO', `Tâche ${videoId}`);
  const videoUrl = await pollVideo(videoId, (progress) => log('info', 'VIDÉO', progress));
  const dest = videoOutputPath(expansion.title);
  await downloadVideo(videoUrl, dest);
  log('success', 'FICHIER', dest);
  await publishVideo(dest, expansion.title);
  return dest;
}

function acquireLock() {
  const dir = path.resolve(process.cwd(), 'generated_videos');
  fs.mkdirSync(dir, { recursive: true });
  const lockPath = path.join(dir, '.generation.lock');
  try {
    const fd = fs.openSync(lockPath, fs.constants.O_CREAT | fs.constants.O_EXCL | fs.constants.O_WRONLY, 0o644);
    fs.writeSync(fd, String(process.pid));
    fs.closeSync(fd);
    return lockPath;
  } catch (err) {
    if (err.code !== 'EEXIST') throw err;
    let pid = 0;
    try {
      pid = parseInt(fs.readFileSync(lockPath, 'utf8'), 10);
    } catch (readErr) {
      pid = 0;
    }
    let alive = false;
    if (pid > 0) {
      try {
        process.kill(pid, 0);
        alive = true;
      } catch (killErr) {
        alive = killErr.code === 'EPERM';
      }
    }
    if (alive) throw new Error(`Une génération est déjà en cours (pid ${pid})`);
    fs.unlinkSync(lockPath);
    return acquireLock();
  }
}

function releaseLock(lockPath) {
  if (!lockPath) return;
  try {
    const current = fs.readFileSync(lockPath, 'utf8').trim();
    if (current === String(process.pid)) fs.unlinkSync(lockPath);
  } catch (err) {
    if (err.code !== 'ENOENT') log('warn', 'LOCK', err.message);
  }
}

async function main() {
  validateRuntimeConfig();
  const scenes = loadScenes();
  const lockPath = acquireLock();
  let failed = 0;
  try {
    for (let index = 0; index < scenes.length; index += 1) {
      if (state.stopRequested) break;
      try {
        await processScene(scenes[index], index);
      } catch (err) {
        failed += 1;
        log('error', 'SCENE', err.message || String(err));
        if (err.message === 'Arrêt demandé') break;
      }
    }
  } finally {
    releaseLock(lockPath);
  }
  log('info', 'FIN', `${scenes.length - failed} vidéo(s) publiée(s), ${failed} échec(s)`);
  if (failed > 0 || state.stopRequested) process.exitCode = 1;
}

function assert(condition, message) {
  if (!condition) throw new Error(message || 'assertion échouée');
}

async function expectThrow(fn, pattern) {
  let thrown = null;
  try {
    await fn();
  } catch (err) {
    thrown = err;
  }
  assert(thrown, 'une erreur était attendue');
  if (pattern && !pattern.test(thrown.message)) {
    throw new Error(`message inattendu : ${thrown.message}`);
  }
  return thrown;
}

function listen(handler) {
  const server = http.createServer(handler);
  return new Promise((resolve) => {
    server.listen(0, '127.0.0.1', () => {
      const { port } = server.address();
      resolve({
        port,
        close: () => new Promise((done) => server.close(done)),
      });
    });
  });
}

function mp4Buffer() {
  return Buffer.concat([Buffer.from([0x00, 0x00, 0x00, 0x18]), Buffer.from('ftypisom'), Buffer.alloc(32, 1)]);
}

async function withMockedSleep(fn) {
  const calls = [];
  const previousSleep = hooks.sleep;
  const previousRandom = hooks.random;
  hooks.sleep = async (ms) => {
    calls.push(ms);
  };
  hooks.random = () => 0;
  try {
    await fn(calls);
  } finally {
    hooks.sleep = previousSleep;
    hooks.random = previousRandom;
    state.stopRequested = false;
  }
}

async function runSelfTest() {
  const previous = {
    AGNES_API_KEY: process.env.AGNES_API_KEY,
    GITHUB_TOKEN: process.env.GITHUB_TOKEN,
    GITHUB_REPO_URL: process.env.GITHUB_REPO_URL,
    VIDEO_QUEUE_MAX_RETRIES: process.env.VIDEO_QUEUE_MAX_RETRIES,
  };
  const agnesSecret = `selftest-agnes-${process.pid}-do-not-leak`;
  const githubSecret = `selftest-github-${process.pid}-do-not-leak`;
  process.env.AGNES_API_KEY = agnesSecret;
  process.env.GITHUB_TOKEN = githubSecret;
  process.env.GITHUB_REPO_URL = ALLOWED_REPO;

  const tests = [
    ['package', async () => {
      const pkg = JSON.parse(fs.readFileSync(path.resolve(process.cwd(), 'package.json'), 'utf8'));
      assert(pkg.type !== 'module', 'le paquet doit rester en CommonJS');
      assert(pkg.dependencies['node-fetch'] === '2.7.0', 'node-fetch 2.7.0');
      assert(pkg.dependencies['simple-git'] === '4.0.2', 'simple-git 4.0.2');
      assert(pkg.dependencies.dotenv === '18.0.9', 'dotenv 18.0.9');
      const ignore = fs.readFileSync(path.resolve(process.cwd(), '.gitignore'), 'utf8');
      assert(ignore.includes('generated_videos/*.mp4'), 'les mp4 doivent être ignorés sans -f');
      assert(ignore.includes('.env'), '.env ignoré');
    }],
    ['scènes', async () => {
      const scenes = loadScenes();
      assert(scenes.length === 5, '5 scènes');
      assert(scenes[0].description === 'maison japonaise sous la pluie', 'première scène');
    }],
    ['json', async () => {
      const clean = extractJson('```json\n{"title":"Pluie","image_prompt":"capybara","video_motion_prompt":"mist"}\n```');
      assert(clean && clean.title === 'Pluie', 'stratégie 1');
      const newline = '{"title":"A","image_prompt":"line1\nline2","video_motion_prompt":"mist"}';
      const fixed = extractJson(newline);
      assert(fixed && fixed.image_prompt === 'line1\nline2', 'stratégie 2');
      const partial = '{"title":"A","image_prompt":"capybara under rain","video_motion_prompt":"slow mist"';
      const balanced = extractJson(partial);
      assert(balanced && balanced.image_prompt.includes('capybara'), 'stratégie 4');
      const garbage = '{note "image_prompt":"capybara on a porch" "video_motion_prompt":"breathing"';
      const salvaged = extractJson(garbage);
      assert(salvaged && salvaged.image_prompt.includes('porch'), 'stratégie 5');
    }],
    ['prompt unique', async () => {
      const variation = { timeOfDay: 'golden hour', season: 'late autumn', weather: 'fine rain' };
      const prompt = buildExpansionPrompt('maison japonaise sous la pluie', 'rainy_forest', variation);
      assert(prompt.includes('golden hour') && prompt.includes('late autumn') && prompt.includes('fine rain'), 'variables');
      assert(/capybara/i.test(prompt), 'capybara');
      assert(prompt.includes('strict JSON'), 'schéma');
    }],
    ['backoff file', async () => {
      assert(computeQueueWaitSeconds(0, 0, 0) === 60, '60s');
      assert(computeQueueWaitSeconds(1, 0, 0) === 120, '120s');
      assert(computeQueueWaitSeconds(2, 0, 0) === 300, '300s');
      assert(computeQueueWaitSeconds(8, 0, 0) === 300, 'plafond de palier');
      assert(computeQueueWaitSeconds(0, 4000, 0) === 360, 'plafond 360');
      assert(computeQueueWaitSeconds(0, 15, 0.9) === 15, 'Retry-After sans jitter');
      assert(computeQueueWaitSeconds(0, 0, 0.5) === 75, 'jitter');
    }],
    ['hôtes média', async () => {
      const blocked = [
        'http://2130706433/a.mp4',
        'https://2130706433/a.mp4',
        'https://0x7f000001/a.mp4',
        'https://0177.0.0.1/a.mp4',
        'https://127.1/a.mp4',
        'https://127.0.0.1/a.mp4',
        'https://[::ffff:127.0.0.1]/a.mp4',
        'https://[::ffff:7f00:1]/a.mp4',
        'https://[::1]/a.mp4',
        'https://169.254.169.254/latest/meta-data',
        'https://10.1.2.3/a.mp4',
        'https://192.168.1.8/a.mp4',
        'https://localhost/a.mp4',
        'https://metadata.google.internal/a.mp4',
        'https://user:secret@1.1.1.1/a.mp4',
      ];
      for (const sample of blocked) {
        await expectThrow(() => assertPublicHttpUrl(sample), /Hôte média refusé|identifiant|HTTPS/);
      }
      assert(assertPublicHttpUrl('https://1.1.1.1/video.mp4').hostname === '1.1.1.1', 'IP publique');
      await expectThrow(() => assertPublicHttpUrl('http://8.8.8.8/a.mp4'), /HTTPS/);
      assert(isBlockedAddress('::ffff:7f00:1'), 'IPv4 mappée loopback');
    }],
    ['api clone et file', async () => {
      await withMockedSleep(async (calls) => {
        let hits = 0;
        const server = await listen((req, res) => {
          hits += 1;
          if (hits <= 12) {
            res.statusCode = 503;
            res.end(JSON.stringify({ error: 'video_queue_full' }));
            return;
          }
          res.statusCode = 200;
          res.end(JSON.stringify({ ok: true }));
        });
        process.env.VIDEO_QUEUE_MAX_RETRIES = '20';
        try {
          const res = await apiFetch(`http://127.0.0.1:${server.port}/videos`, { method: 'POST' }, 'Vidéo test', 'video');
          const body = await res.json();
          assert(body.ok === true, 'succès après file pleine');
          assert(calls.length === 12, 'les pauses de file ne comptent pas comme des tentatives normales');
          assert(calls[0] === 60000 && calls[1] === 120000 && calls[2] === 300000, 'pauses 60/120/300');
        } finally {
          delete process.env.VIDEO_QUEUE_MAX_RETRIES;
          await server.close();
        }
      });
      await withMockedSleep(async () => {
        const server = await listen((req, res) => {
          res.statusCode = 400;
          res.end('detail-corps-video-intact');
        });
        try {
          const res = await apiFetch(`http://127.0.0.1:${server.port}/videos`, { method: 'POST' }, 'Vidéo 400', 'video');
          const text = await res.text();
          assert(text === 'detail-corps-video-intact', 'le corps vidéo reste lisible après clone');
        } finally {
          await server.close();
        }
      });
      await withMockedSleep(async () => {
        const server = await listen((req, res) => {
          res.statusCode = 400;
          res.end('detail-texte-terminal');
        });
        try {
          await expectThrow(
            () => apiFetch(`http://127.0.0.1:${server.port}/chat`, { method: 'POST' }, 'Texte 400', 'text'),
            /detail-texte-terminal/,
          );
        } finally {
          await server.close();
        }
      });
    }],
    ['téléchargement', async () => {
      const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'lofi-dl-'));
      const dest = path.join(dir, 'ok.mp4');
      const payload = mp4Buffer();
      let calls = 0;
      await downloadVideo('https://1.1.1.1/video.mp4', dest, {
        fetchImpl: async () => {
          calls += 1;
          return {
            ok: true,
            status: 200,
            headers: { get: (name) => (String(name).toLowerCase() === 'content-length' ? String(payload.length) : null) },
            body: Readable.from([payload]),
          };
        },
      });
      assert(calls === 1, 'un seul appel');
      assertLocalVideo(dest);

      const redirected = path.join(dir, 'redir.mp4');
      let hops = 0;
      await expectThrow(() => downloadVideo('https://1.1.1.1/start.mp4', redirected, {
        fetchImpl: async () => {
          hops += 1;
          return {
            ok: false,
            status: 302,
            headers: { get: () => 'https://127.0.0.1/secret.mp4' },
            body: Readable.from([]),
          };
        },
      }), /Hôte média refusé/);
      assert(hops === 1, 'pas de second saut vers un hôte privé');

      let chain = 0;
      await expectThrow(() => downloadVideo('https://1.1.1.1/chain.mp4', path.join(dir, 'chain.mp4'), {
        fetchImpl: async () => {
          chain += 1;
          return {
            ok: false,
            status: 302,
            headers: { get: () => `https://1.1.1.1/hop${chain}` },
            body: Readable.from([]),
          };
        },
      }), /Trop de redirections/);
      assert(chain === 3, 'deux redirections suivies, la troisième est refusée');

      const bad = path.join(dir, 'bad.mp4');
      await expectThrow(() => downloadVideo('https://1.1.1.1/bad.mp4', bad, {
        fetchImpl: async () => ({
          ok: true,
          status: 200,
          headers: { get: () => null },
          body: Readable.from([Buffer.from('this is not a media file!!')]),
        }),
      }), /En-tête média invalide/);
      assert(!fs.existsSync(bad), 'fichier invalide supprimé');

      const target = path.join(dir, 'target.mp4');
      fs.writeFileSync(target, 'secret-target');
      const link = path.join(dir, 'link.mp4');
      fs.symlinkSync(target, link);
      await expectThrow(() => downloadVideo('https://1.1.1.1/sym.mp4', link, {
        fetchImpl: async () => ({
          ok: true,
          status: 200,
          headers: { get: () => String(payload.length) },
          body: Readable.from([payload]),
        }),
      }), /lien symbolique/);
      assert(fs.readFileSync(target, 'utf8') === 'secret-target', 'la cible du lien est intacte');

      const big = path.join(dir, 'big.mp4');
      const bigFd = fs.openSync(big, 'w');
      fs.writeSync(bigFd, mp4Buffer());
      fs.ftruncateSync(bigFd, MAX_VIDEO_BYTES);
      fs.closeSync(bigFd);
      await expectThrow(() => assertLocalVideo(big), /100 Mo/);
      fs.rmSync(dir, { recursive: true, force: true });
    }],
    ['file d’attente', async () => {
      const starts = [];
      const queue = createRateLimitedQueue(1, 40);
      await Promise.all([0, 1, 2].map((item) => queue.add(async () => {
        starts.push(Date.now());
        await new Promise((resolve) => setTimeout(resolve, 5));
        return item;
      })));
      assert(starts.length === 3, 'trois tâches');
      assert(starts[1] - starts[0] >= 30 && starts[2] - starts[1] >= 30, 'intervalle respecté');
    }],
    ['dépôt autorisé', async () => {
      assert(canonicalRepoUrl(`${ALLOWED_REPO}.git`) === ALLOWED_REPO, 'suffixe .git');
      assert(classifyOrigin('https://user:SHOULDNOTAPPEAR@github.com/urdirditfurd/videowan2g') === 'reset', 'jeton dans l’URL');
      assert(classifyOrigin('https://github.com/other/videowan2g') === 'reject', 'autre dépôt');
      assert(classifyOrigin('http://github.com/urdirditfurd/videowan2g') === 'reject', 'http');
      assert(classifyOrigin('ssh://git@github.com/urdirditfurd/videowan2g.git') === 'reject', 'ssh');
      await expectThrow(() => canonicalRepoUrl('https://user:pass@github.com/urdirditfurd/videowan2g'), /identifiant/);
    }],
    ['git isolé', async () => {
      const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'lofi-git-'));
      const dumpPath = path.join(dir, 'dump.json');
      const fakeGit = path.join(dir, 'fake-git.cjs');
      const binDir = path.join(dir, 'bin');
      fs.mkdirSync(binDir);
      fs.writeFileSync(fakeGit, `'use strict';
const fs = require('fs');
const dumpPath = ${JSON.stringify(dumpPath)};
let prev = { invocations: [] };
try { prev = JSON.parse(fs.readFileSync(dumpPath, 'utf8')); } catch (err) { prev = { invocations: [] }; }
const askpass = process.env.GIT_ASKPASS && fs.existsSync(process.env.GIT_ASKPASS)
  ? fs.readFileSync(process.env.GIT_ASKPASS, 'utf8')
  : '';
prev.env = process.env;
prev.askpass = askpass;
prev.invocations.push(process.argv.slice(2));
fs.writeFileSync(dumpPath, JSON.stringify(prev));
const args = process.argv.slice(2);
if (args.includes('get-url')) {
  process.stdout.write('https://x-access-token:SHOULDNOTAPPEAR@github.com/urdirditfurd/videowan2g\\n');
} else {
  process.stdout.write('ok\\n');
}
`);
      const wrapper = path.join(binDir, 'git');
      fs.writeFileSync(wrapper, `#!/bin/sh\nexec ${JSON.stringify(process.execPath)} ${JSON.stringify(fakeGit)} "$@"\n`, { mode: 0o755 });
      const videoDir = path.join(dir, 'generated_videos');
      fs.mkdirSync(videoDir);
      const videoPath = path.join(videoDir, 'pluie_1.mp4');
      fs.writeFileSync(videoPath, mp4Buffer());
      const logs = [];
      const origLog = console.log;
      const origErr = console.error;
      console.log = (...args) => logs.push(args.map(String).join(' '));
      console.error = (...args) => logs.push(args.map(String).join(' '));
      try {
        await publishVideo(videoPath, 'Pluie douce', {
          repoRoot: dir,
          repoUrl: ALLOWED_REPO,
          token: githubSecret,
          path: `${binDir}:${process.env.PATH || ''}`,
        });
      } finally {
        console.log = origLog;
        console.error = origErr;
      }
      const dumped = JSON.parse(fs.readFileSync(dumpPath, 'utf8'));
      const blob = JSON.stringify(dumped);
      for (const banned of ['AGNES_API_KEY', 'GITHUB_TOKEN', 'GIT_CONFIG_COUNT', 'GIT_CONFIG_KEY_0', 'GIT_CONFIG_VALUE_0', 'GIT_CONFIG_PARAMETERS']) {
        assert(!Object.prototype.hasOwnProperty.call(dumped.env, banned), `fuite d'environnement : ${banned}`);
      }
      assert(!blob.includes(agnesSecret) && !blob.includes(githubSecret), 'valeur secrète visible par git');
      assert(!blob.includes('SHOULDNOTAPPEAR'), 'jeton d’URL recopié');
      assert(dumped.env.HOME === '', 'HOME vide');
      assert(dumped.env.GIT_CONFIG_NOSYSTEM === '1', 'config système ignorée');
      assert(dumped.env.GIT_ASKPASS && dumped.env.GIT_ASKPASS.endsWith('askpass.cjs'), 'askpass temporaire');
      assert(!String(dumped.askpass || '').includes(githubSecret), 'le script askpass ne contient pas le jeton');
      const flat = dumped.invocations.map((args) => args.join(' '));
      assert(flat.some((line) => line.includes('add') && line.includes('-f')), 'git add -f');
      assert(flat.some((line) => line.includes('commit.gpgsign=false') && line.includes('--')), 'commit ciblé');
      assert(flat.some((line) => line.includes('credential.helper=') && line.includes('push')), 'push sans helper');
      assert(flat.some((line) => line.includes('set-url') && line.includes(ALLOWED_REPO)), 'origin nettoyée');
      assert(!logs.join('\n').includes('SHOULDNOTAPPEAR'), 'URL avec jeton journalisée');
      assert(!logs.join('\n').includes(githubSecret) && !logs.join('\n').includes(agnesSecret), 'secret journalisé');
      assert(!fs.existsSync(path.dirname(dumped.env.GIT_ASKPASS)), 'askpass supprimé');
      const real = createGit({ baseDir: process.cwd() });
      const inside = String(await real.raw(['rev-parse', '--is-inside-work-tree'])).trim();
      assert(inside === 'true', 'git réel avec environnement filtré');
      fs.rmSync(dir, { recursive: true, force: true });
    }],
  ];

  try {
    for (const [name, fn] of tests) {
      try {
        await fn();
        console.log(`ok ${name}`);
      } catch (err) {
        throw new Error(`${name}: ${redact(err.message || String(err))}`);
      }
    }
    console.log('SELF-TEST PASSED');
  } finally {
    for (const [key, value] of Object.entries(previous)) {
      if (value == null) delete process.env[key];
      else process.env[key] = value;
    }
  }
}

process.once('SIGINT', () => {
  state.stopRequested = true;
  log('warn', 'STOP', 'Arrêt demandé');
});
process.once('SIGTERM', () => {
  state.stopRequested = true;
  log('warn', 'STOP', 'Arrêt demandé');
});

if (require.main === module) {
  const run = process.argv.includes('--self-test') ? runSelfTest : main;
  run().catch((err) => {
    console.error(redact(err && err.stack ? err.stack : err));
    process.exit(1);
  });
}
