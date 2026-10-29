'use strict';

const fs = require('fs');
const os = require('os');
const path = require('path');
const { spawnSync } = require('child_process');
const dotenv = require('dotenv');
const fetch = require('node-fetch');
const { simpleGit } = require('simple-git');

dotenv.config({ quiet: true });

const API_BASE = 'https://apihub.agnes-ai.com/v1';
const POLL_BASE = 'https://apihub.agnes-ai.com/agnesapi';
const MODEL_TEXT = 'agnes-2.5-flash';
const MODEL_IMAGE = 'agnes-image-2.5-flash';
const MODEL_VIDEO = 'agnes-video-2.5-flash';

const ROOT = __dirname;
const SCENES_PATH = path.join(ROOT, 'scenes.json');
const VIDEO_DIR = path.join(ROOT, 'generated_videos');
const TIMING_CACHE_PATH = path.join(VIDEO_DIR, '.timing-cache.json');

const ALLOWED_REPO = 'https://github.com/urdirditfurd/videowan2g';
const PUSH_URL = 'https://github.com/urdirditfurd/videowan2g.git';

const IMAGE_INTERVAL_MS = 3400;
const IMAGE_INTERVAL_MIN = 3200;
const IMAGE_INTERVAL_MAX = 12000;
const VIDEO_INTERVAL_MIN = 62000;
const VIDEO_INTERVAL_MAX = 90000;
const VIDEO_INTERVAL_SAFE = 75000;
const VIDEO_FIRST_POLL_DELAY_SEC = 80;
const VIDEO_MAX_POLL_ATTEMPTS = 100;
const VIDEO_QUEUE_BACKOFFS_SEC = [60, 120, 300];
const VIDEO_QUEUE_MAX_WAIT_SEC = 360;
const VIDEO_QUEUE_MAX_RETRIES = 12;
const MAX_VIDEO_BYTES = 100 * 1024 * 1024;
const MAX_REDIRECTS = 2;

const GHIBLI_PALETTE = [
  'soft pastel palette inspired by Studio Ghibli watercolor backgrounds:',
  'sage green #8fad88, warm cream #f2efe9, dusky blue #7a9eb8, soft gold #c4a882,',
  'gentle forest greens, muted earth tones, warm amber highlights.',
  'Never grey, never desaturated, never neon. Rich but calm.',
  'Hand-painted watercolor feel with soft gradients.',
].join(' ');

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
    label: 'Feuilles d\'automne',
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
  'aube pâle',
  'matin brumeux',
  'fin de matinée claire',
  'midi doux',
  'après-midi doré',
  'heure bleue',
  'crépuscule',
  'nuit claire',
];

const SEASONS = ['début de printemps', 'plein été', 'automne avancé', 'hiver calme'];

const WEATHERS = [
  'pluie fine',
  'bruine',
  'ciel dégagé',
  'nuages légers',
  'brume basse',
  'neige douce',
  'éclaircie après l\'orage',
  'vent léger dans les feuilles',
];

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

const state = {
  stopRequested: false,
  durationSeconds: 5,
  format: '16:9',
};

let createImageIntervalMs = 4000;
let createVideoIntervalMs = VIDEO_INTERVAL_SAFE;
let imageAttemptsSinceLastError = 0;
let videoAttemptsSinceLastError = 0;
let loggingEnabled = true;
let sleepFn = (ms) => new Promise((resolve) => setTimeout(resolve, ms));
let randomFn = Math.random;

function sleep(ms) {
  return sleepFn(ms);
}

function redact(value) {
  return String(value == null ? '' : value)
    .replace(/sk-[A-Za-z0-9_-]{4,}/g, 'sk-[REDACTED]')
    .replace(/ghp_[A-Za-z0-9]{4,}/g, 'ghp_[REDACTED]')
    .replace(/github_pat_[A-Za-z0-9_]{4,}/g, 'github_pat_[REDACTED]')
    .replace(/x-access-token:[^@\s]+/gi, 'x-access-token:[REDACTED]');
}

function log(level, category, message) {
  if (!loggingEnabled) return;
  const line = `${new Date().toISOString()} [${level}] [${category}] ${redact(message)}`;
  if (level === 'error') console.error(line);
  else console.log(line);
}

function pick(list) {
  const index = Math.floor(randomFn() * list.length);
  return list[Math.max(0, Math.min(list.length - 1, index))];
}

function loadTimingCache() {
  try {
    return JSON.parse(fs.readFileSync(TIMING_CACHE_PATH, 'utf8'));
  } catch (e) {
    return {};
  }
}

function saveTimingCache(cache) {
  fs.mkdirSync(VIDEO_DIR, { recursive: true });
  fs.writeFileSync(TIMING_CACHE_PATH, JSON.stringify(cache));
}

function recordGenerationTime(key, ms) {
  const cache = loadTimingCache();
  if (!Array.isArray(cache[key])) cache[key] = [];
  cache[key].push(ms);
  if (cache[key].length > 10) cache[key] = cache[key].slice(-10);
  saveTimingCache(cache);
}

function estimateGenerationTime(key, fallbackMs) {
  const samples = loadTimingCache()[key];
  if (!samples || samples.length === 0) return fallbackMs;
  const sorted = [...samples].sort((a, b) => a - b);
  const mid = Math.floor(sorted.length / 2);
  if (sorted.length % 2 === 0) return (sorted[mid - 1] + sorted[mid]) / 2;
  return sorted[mid];
}

function videoQueueDelaySeconds(retryIndex, retryAfterSeconds) {
  const index = Math.min(Math.max(retryIndex, 0), VIDEO_QUEUE_BACKOFFS_SEC.length - 1);
  const base = VIDEO_QUEUE_BACKOFFS_SEC[index];
  const jitter = Math.floor(randomFn() * 15);
  const chosen = retryAfterSeconds > 0 ? retryAfterSeconds : base + jitter;
  return Math.min(VIDEO_QUEUE_MAX_WAIT_SEC, Math.max(1, Math.round(chosen)));
}

function retryAfterSeconds(res) {
  try {
    const raw = res.headers && res.headers.get ? res.headers.get('retry-after') : null;
    const parsed = parseInt(raw || '0', 10);
    return Number.isFinite(parsed) && parsed > 0 ? parsed : 0;
  } catch (e) {
    return 0;
  }
}

function createRateLimitedQueue(maxConcurrent, minIntervalMs) {
  let running = 0;
  let lastStart = 0;
  const queue = [];
  let resolveIdle = null;

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
    const wait = Math.max(0, lastStart + minIntervalMs - Date.now());
    if (wait > 0) {
      setTimeout(tryNext, wait);
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
        queue.push(() => Promise.resolve().then(fn).then(resolve, reject));
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

async function apiFetch(url, options = {}, label = 'API', kind = 'default', fetchImpl = fetch) {
  const maxAttempts = kind === 'video' ? 10 : kind === 'image' ? 10 : 8;
  let attempt = 0;
  let queueRetries = 0;

  while (attempt < maxAttempts) {
    if (state.stopRequested) throw new Error('Arrêt demandé');
    const method = options.method || 'GET';
    log('api', `REQ ${label}`, `${method} ${url} (tentative ${attempt + 1}/${maxAttempts})`);

    let response;
    try {
      response = await fetchImpl(url, options);
    } catch (e) {
      if (e && e.message === 'Arrêt demandé') throw e;
      const wait = Math.min(kind === 'video' ? 120 : 45, kind === 'video' ? 10 + attempt * 8 : 3 + attempt * 4);
      log('warn', 'NETWORK', `${e && e.message ? e.message : e} — retry dans ${wait}s`);
      await sleep(wait * 1000);
      attempt += 1;
      continue;
    }

    let errorText = '';
    const shouldReadError = !response.ok && (
      [400, 401, 404, 429, 500, 502, 503, 504, 520].includes(response.status) || kind === 'video'
    );
    if (shouldReadError) {
      try {
        errorText = (await response.clone().text()).slice(0, 1200);
      } catch (e) {
        errorText = '';
      }
    }
    log('response', `RÉP ${label}`, `HTTP ${response.status}${errorText ? '' : ''}`);

    if (response.ok) {
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
      return response;
    }

    if ([400, 401, 404].includes(response.status) && kind !== 'video') {
      throw new Error(`HTTP ${response.status} — ${errorText.slice(0, 300)}`);
    }

    const queueFull = /video_queue_full|queue is full/i.test(errorText);
    if (kind === 'video' && queueFull) {
      if (queueRetries >= VIDEO_QUEUE_MAX_RETRIES) {
        throw new Error('File vidéo saturée — nombre maximal d\'attentes atteint');
      }
      const wait = videoQueueDelaySeconds(queueRetries, retryAfterSeconds(response));
      queueRetries += 1;
      videoAttemptsSinceLastError = 0;
      log('warn', 'VIDEO_QUEUE', `Queue vidéo saturée — nouvelle tentative dans ${wait}s (${queueRetries}/${VIDEO_QUEUE_MAX_RETRIES})`);
      await sleep(wait * 1000);
      continue;
    }

    if (response.status === 429) {
      const headerWait = retryAfterSeconds(response);
      const wait = headerWait > 0
        ? headerWait
        : Math.min(kind === 'video' ? 180 : 120, kind === 'video' ? 60 + attempt * 8 : 10 + attempt * 12);
      if (kind === 'image') createImageIntervalMs = Math.min(IMAGE_INTERVAL_MAX, createImageIntervalMs + 1000);
      if (kind === 'video') createVideoIntervalMs = Math.min(VIDEO_INTERVAL_MAX, createVideoIntervalMs + 6000);
      log('warn', 'RATELIMIT', `429 — attente ${wait}s`);
      await sleep(wait * 1000);
      attempt += 1;
      continue;
    }

    if ([500, 502, 503, 504, 520].includes(response.status)) {
      const wait = Math.min(kind === 'video' ? 180 : 60, kind === 'video' ? 20 + attempt * 10 : 5 + attempt * 5);
      log('warn', 'SERVICE', `HTTP ${response.status} — attente ${wait}s`);
      await sleep(wait * 1000);
      attempt += 1;
      continue;
    }

    return response;
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
    } catch (e) {
      lastError = e;
      if (e && e.message === 'Arrêt demandé') throw e;
      log('warn', 'RETRY', `${name} échoué : ${e && e.message ? e.message : e}`);
      if (i < maxAttempts) {
        const wait = Math.min(90, 8 * i * i);
        log('info', 'RETRY', `${name} — reprise automatique dans ${wait}s`);
        await sleep(wait * 1000);
      }
    }
  }
  throw lastError || new Error(`${name} échoué`);
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
  let escape = false;
  for (let i = 0; i < text.length; i += 1) {
    const ch = text[i];
    if (escape) { escape = false; continue; }
    if (ch === '\\') { escape = true; continue; }
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
  try { return JSON.parse(raw); } catch (e) { /* stratégie suivante */ }
  try { return JSON.parse(escapeControlCharsInStrings(raw)); } catch (e) { /* stratégie suivante */ }
  try {
    const end = raw.lastIndexOf('}');
    if (end > 0) return JSON.parse(escapeControlCharsInStrings(raw.slice(0, end + 1)));
  } catch (e) { /* stratégie suivante */ }
  try { return JSON.parse(balanceJson(escapeControlCharsInStrings(raw))); } catch (e) { /* stratégie suivante */ }
  const salvaged = salvageExpansion(raw);
  if (salvaged && (salvaged.image_prompt || salvaged.video_motion_prompt)) return salvaged;
  return null;
}

function rollVariation(existing) {
  if (existing) return existing;
  return {
    timeOfDay: pick(TIMES_OF_DAY),
    season: pick(SEASONS),
    weather: pick(WEATHERS),
  };
}

function buildExpansionPrompt(userDescription, mood, variation) {
  const selected = MOODS[mood] || MOODS.rainy_forest;
  const unique = rollVariation(variation);
  return [
    'You are an art director for Studio Ghibli-inspired lofi ambience videos.',
    'Every scene must be visually unique, even if the user idea resembles a previous one.',
    '',
    'USER SCENE IDEA (French):',
    userDescription,
    '',
    'AMBIENCE (French mood chosen):',
    `${selected.label} — ${selected.desc}`,
    '',
    'UNIQUENESS VARIATION — this exact combination must be visibly obvious in the image:',
    `- Time of day: ${unique.timeOfDay}`,
    `- Season: ${unique.season}`,
    `- Weather: ${unique.weather}`,
    'Do not reuse a generic version of the idea. Change the light, the vegetation, the sky and the small props so this frame cannot be confused with another.',
    '',
    'YOUR TASK:',
    'Expand the user idea into a rich, poetic, ultra-detailed visual scene for a single lofi ambience video.',
    'ALWAYS include a cute, calm, peaceful capybara (capybara mascot) somewhere visible in the scene, integrated naturally into the environment.',
    'The capybara is mandatory and must be clearly readable: round body, gentle eyes, resting or soaking, never hidden, never cropped out.',
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
    '  "image_prompt": "Ultra-detailed English visual description for an image model. Include: setting, the uniqueness variation (time of day, season, weather), lighting, colors, atmosphere, the capybara (with its location and pose), Ghibli watercolor style, cinematic composition. 150-250 words.",',
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

function buildLofiImagePrompt(exp) {
  return [
    'GHIBLI-STYLE LOFI AMBIENCE KEYFRAME.',
    exp.image_prompt,
    buildGhibliStyleBlock(),
    'Ultra-detailed painterly image, calm and loopable composition.',
  ].join('\n');
}

function buildLofiVideoPrompt(exp) {
  return [
    `LOFI AMBIENCE VIDEO — ${state.durationSeconds} seconds — seamless loop.`,
    'Studio Ghibli inspired watercolor scene with a cute calm capybara.',
    `BASE SCENE: ${exp.image_prompt}`,
    `GENTLE MOTION: ${exp.video_motion_prompt}`,
    'ABSOLUTE MOTION RULE: SLOW, GENTLE, SUBTLE motion only. No fast movement, no camera shake, no action. The scene must feel peaceful and loopable.',
    'Perfect seamless loop, calm pacing.',
    'CAMERA: locked-off or extremely slow drift. No cuts.',
    'STYLE: hand-painted Ghibli watercolor, soft gradients, warm nostalgic light.',
    `PALETTE: ${GHIBLI_PALETTE}`,
    'SOUND: soft ambient nature sounds only — no music, no voice. Wind, birds, water, rustling leaves.',
    'NO readable text, NO watermark, NO logos.',
  ].join('\n');
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
      {
        role: 'system',
        content: 'You are a Studio Ghibli-inspired art director specialized in lofi ambience videos. You always return strict JSON matching the requested schema exactly. Every scene includes a cute calm capybara. Every scene is peaceful, detailed, watercolor, visually unique, and loopable with slow gentle motion. No text, no watermark. Return only JSON.',
      },
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
  const content = data && data.choices && data.choices[0] && data.choices[0].message
    ? data.choices[0].message.content
    : '';
  const exp = extractJson(content);
  if (!exp || !exp.image_prompt) throw new Error('Expansion invalide');
  if (!exp.title) exp.title = 'Ambiance Lofi';
  if (!exp.video_motion_prompt) {
    exp.video_motion_prompt = 'Gentle slow motion: capybara breathing softly, leaves drifting down, faint mist, seamless loop.';
  }
  log('success', 'EXPANSION', `Titre : ${exp.title}`);
  return exp;
}

async function generateImage(prompt, format) {
  return retryTask('Image', async () => {
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
    const item = data && data.data && data.data[0];
    if (!item) throw new Error('Pas de donnée image');
    if (item.url) return item.url;
    if (item.b64_json) return `data:image/png;base64,${item.b64_json}`;
    throw new Error('Format image inconnu');
  }, 4);
}

async function generateVideo(imageUrl, prompt, format) {
  return retryTask('Vidéo', async () => {
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
      headers: authHeaders(),
      body: JSON.stringify(body),
    }, 'Vidéo', 'video');
    if (!res.ok) {
      const err = await res.text();
      throw new Error(`HTTP ${res.status} — ${err.slice(0, 500)}`);
    }
    const data = await res.json();
    const videoId = data.video_id || data.id || data.task_id;
    if (!videoId) throw new Error('Pas de video_id');
    return videoId;
  }, 3);
}

async function pollVideo(videoId, onProgress) {
  const estimatedMs = estimateGenerationTime('video', 90000 + state.durationSeconds * 7000);
  const minWait = Math.max(25, Math.min(VIDEO_FIRST_POLL_DELAY_SEC, Math.floor(estimatedMs / 1000 * 0.35)));
  let remaining = minWait;
  log('info', 'POLLING', `Attente initiale ${remaining}s pour ${videoId}`);
  while (remaining > 0) {
    if (state.stopRequested) throw new Error('Arrêt demandé');
    if (onProgress) onProgress(`File vidéo — ${remaining}s`);
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
      res = await apiFetch(url, { method: 'GET', headers: authHeaders() }, 'Polling', 'video');
    } catch (e) {
      if (e && e.message === 'Arrêt demandé') throw e;
      log('warn', 'POLLING', `Erreur transitoire : ${e && e.message ? e.message : e}`);
      await sleep(Math.min(90000, 15000 + attempt * 1500));
      continue;
    }
    let payload;
    try {
      payload = await res.json();
    } catch (e) {
      log('warn', 'POLLING', 'Réponse non JSON, reprise');
      await sleep(Math.min(60000, 10000 + attempt * 1000));
      continue;
    }
    const status = String((payload && payload.status) || 'unknown').toLowerCase();
    const progress = Number((payload && payload.progress) || 0);
    if (onProgress) onProgress(`Création — ${progress}%`);
    log('response', `POLL #${attempt}`, `Statut : ${status} (${progress}%)`);
    if (status === 'completed' || status === 'succeeded' || status === 'done') {
      const videoUrl = payload.url || (payload.metadata && payload.metadata.url) || (payload.output && payload.output.url);
      if (!videoUrl) throw new Error('Terminé mais pas d\'URL');
      return videoUrl;
    }
    if (status === 'failed' || status === 'error' || status === 'cancelled') {
      throw new Error(`Échec définitif (${status})`);
    }
    const wait = attempt < 5 ? 8 : Math.min(30, 10 + Math.floor(attempt / 5) * 3);
    if (onProgress) onProgress(`File vidéo — nouvelle vérification dans ${wait}s`);
    await sleep(wait * 1000);
  }
  throw new Error('Délai maximal de polling dépassé');
}

function bareHostname(hostname) {
  if (hostname.startsWith('[') && hostname.endsWith(']')) return hostname.slice(1, -1);
  return hostname;
}

function isPrivateIPv4(ip) {
  const parts = String(ip).split('.');
  if (parts.length !== 4) return false;
  const nums = parts.map((part) => (/^\d+$/.test(part) ? Number(part) : NaN));
  if (nums.some((n) => !Number.isInteger(n) || n < 0 || n > 255)) return false;
  const a = nums[0];
  const b = nums[1];
  if (a === 0 || a === 10 || a === 127) return true;
  if (a === 169 && b === 254) return true;
  if (a === 172 && b >= 16 && b <= 31) return true;
  if (a === 192 && b === 168) return true;
  if (a === 100 && b >= 64 && b <= 127) return true;
  if (a >= 224) return true;
  return false;
}

function mappedIPv4(host) {
  const lower = host.toLowerCase();
  const hex = lower.match(/^::ffff:([0-9a-f]{1,4}):([0-9a-f]{1,4})$/);
  if (hex) {
    const hi = parseInt(hex[1], 16);
    const lo = parseInt(hex[2], 16);
    return [
      (hi >> 8) & 255,
      hi & 255,
      (lo >> 8) & 255,
      lo & 255,
    ].join('.');
  }
  const dotted = lower.match(/^::ffff:(\d{1,3}(?:\.\d{1,3}){3})$/);
  if (dotted) return dotted[1];
  return null;
}

function isBlockedHostname(hostname) {
  const host = bareHostname(hostname).toLowerCase().replace(/\.$/, '');
  if (!host) return true;
  if (host === 'localhost' || host.endsWith('.localhost')) return true;
  if (host === 'metadata.google.internal' || host.endsWith('.metadata.google.internal')) return true;
  if (host === 'metadata') return true;
  if (host === '::1' || host === '::') return true;
  if (host.startsWith('::ffff:')) {
    const mapped = mappedIPv4(host);
    if (!mapped) return true;
    return isPrivateIPv4(mapped);
  }
  if (isPrivateIPv4(host)) return true;
  if (/^[0-9a-f:]+$/i.test(host)) {
    if (host.startsWith('fe80:') || host.startsWith('fc') || host.startsWith('fd') || host.startsWith('ff')) return true;
  }
  return false;
}

function assertPublicHttpsUrl(raw) {
  let url;
  try {
    url = new URL(raw);
  } catch (e) {
    throw new Error('URL de vidéo invalide');
  }
  if (url.protocol !== 'https:') throw new Error('URL de vidéo refusée');
  if (url.username || url.password) throw new Error('URL de vidéo avec identifiant refusée');
  if (isBlockedHostname(url.hostname)) throw new Error('Hôte de vidéo refusé');
  return url;
}

function hasVideoMagic(buf) {
  if (buf.length >= 8 && buf.slice(4, 8).toString('ascii') === 'ftyp') return true;
  if (buf.length >= 4 && buf[0] === 0x1a && buf[1] === 0x45 && buf[2] === 0xdf && buf[3] === 0xa3) return true;
  return false;
}

function openExclusiveNoFollow(filePath) {
  fs.mkdirSync(path.dirname(filePath), { recursive: true });
  let existing = null;
  try {
    existing = fs.lstatSync(filePath);
  } catch (e) {
    if (e.code !== 'ENOENT') throw e;
  }
  if (existing) {
    if (existing.isSymbolicLink()) throw new Error('Lien symbolique refusé');
    throw new Error('Le fichier vidéo existe déjà');
  }
  const flags = fs.constants.O_CREAT | fs.constants.O_EXCL | fs.constants.O_NOFOLLOW | fs.constants.O_WRONLY;
  try {
    return fs.openSync(filePath, flags, 0o644);
  } catch (e) {
    if (e.code === 'ELOOP' || e.code === 'EEXIST') throw new Error('Lien symbolique refusé');
    throw e;
  }
}

function safeUnlinkRegular(filePath) {
  let st;
  try {
    st = fs.lstatSync(filePath);
  } catch (e) {
    if (e.code === 'ENOENT') return;
    throw e;
  }
  if (st.isSymbolicLink() || !st.isFile()) return;
  fs.unlinkSync(filePath);
}

function assertLocalVideo(filePath) {
  const st = fs.lstatSync(filePath);
  if (st.isSymbolicLink()) throw new Error('Lien symbolique refusé');
  if (!st.isFile()) throw new Error('La vidéo n\'est pas un fichier régulier');
  if (st.size <= 0 || st.size >= MAX_VIDEO_BYTES) throw new Error('Taille de vidéo refusée');
  const fd = fs.openSync(filePath, fs.constants.O_RDONLY | fs.constants.O_NOFOLLOW);
  try {
    const buf = Buffer.alloc(16);
    const read = fs.readSync(fd, buf, 0, 16, 0);
    if (!hasVideoMagic(buf.subarray(0, read))) throw new Error('Signature vidéo invalide');
  } finally {
    fs.closeSync(fd);
  }
}

async function writeExclusiveVideo(filePath, byteSource) {
  const fd = openExclusiveNoFollow(filePath);
  try {
    let received = 0;
    for await (const chunk of byteSource) {
      const buf = Buffer.isBuffer(chunk) ? chunk : Buffer.from(chunk);
      received += buf.length;
      if (received >= MAX_VIDEO_BYTES) throw new Error('Vidéo trop volumineuse');
      fs.writeSync(fd, buf);
    }
    if (received < 12) throw new Error('Vidéo trop petite');
  } catch (e) {
    fs.closeSync(fd);
    safeUnlinkRegular(filePath);
    throw e;
  }
  fs.closeSync(fd);
  try {
    assertLocalVideo(filePath);
  } catch (e) {
    safeUnlinkRegular(filePath);
    throw e;
  }
}

async function downloadVideo(rawUrl, filePath, fetchImpl = fetch) {
  assertPublicHttpsUrl(rawUrl);
  let current = rawUrl;
  let redirects = 0;
  while (true) {
    if (state.stopRequested) throw new Error('Arrêt demandé');
    const res = await fetchImpl(current, {
      method: 'GET',
      redirect: 'manual',
      headers: { 'User-Agent': 'videowan2g-lofi/1.0' },
    });
    if ([301, 302, 303, 307, 308].includes(res.status)) {
      if (res.body && typeof res.body.resume === 'function') res.body.resume();
      redirects += 1;
      if (redirects > MAX_REDIRECTS) throw new Error('Trop de redirections');
      const location = res.headers.get('location');
      if (!location) throw new Error('Redirection sans cible');
      const next = new URL(location, current);
      assertPublicHttpsUrl(next.href);
      current = next.href;
      continue;
    }
    if (!res.ok) throw new Error(`Téléchargement HTTP ${res.status}`);
    if (!res.body) throw new Error('Réponse vidéo vide');
    await writeExclusiveVideo(filePath, res.body);
    return filePath;
  }
}

function sanitizeTitle(title) {
  const cleaned = String(title || 'ambiance-lofi')
    .normalize('NFKD')
    .replace(/[\u0300-\u036f]/g, '')
    .replace(/[^A-Za-z0-9]+/g, '-')
    .replace(/-+/g, '-')
    .replace(/^-|-$/g, '')
    .slice(0, 60);
  return cleaned || 'ambiance-lofi';
}

function timestampForFile(date = new Date()) {
  const pad = (n, size = 2) => String(n).padStart(size, '0');
  return [
    date.getUTCFullYear(),
    pad(date.getUTCMonth() + 1),
    pad(date.getUTCDate()),
    'T',
    pad(date.getUTCHours()),
    pad(date.getUTCMinutes()),
    pad(date.getUTCSeconds()),
    pad(date.getUTCMilliseconds(), 3),
    'Z',
  ].join('');
}

function videoFilePath(title, date = new Date()) {
  const name = `${sanitizeTitle(title)}_${timestampForFile(date)}.mp4`;
  if (!/^[A-Za-z0-9_-]+\.mp4$/.test(name)) throw new Error('Nom de fichier invalide');
  const full = path.resolve(VIDEO_DIR, name);
  const root = path.resolve(VIDEO_DIR);
  if (path.dirname(full) !== root) throw new Error('Chemin vidéo invalide');
  return full;
}

function commitMessage(title) {
  const clean = String(title || 'ambiance lofi').replace(/[\r\n\0]+/g, ' ').trim().slice(0, 120);
  return `feat: add new lofi capybara video: ${clean}`;
}

function shellSingleQuote(value) {
  return `'${String(value).replace(/'/g, `'\\''`)}'`;
}

function assertAllowedRepo(input) {
  let url;
  try {
    url = new URL(String(input || '').trim());
  } catch (e) {
    throw new Error('URL de dépôt invalide');
  }
  if (url.protocol !== 'https:') throw new Error('Le dépôt doit utiliser https');
  if (url.username || url.password) throw new Error('URL de dépôt avec identifiant refusée');
  if (url.port) throw new Error('URL de dépôt invalide');
  if (url.search || url.hash) throw new Error('URL de dépôt invalide');
  const host = url.hostname.toLowerCase().replace(/\.$/, '');
  const pathname = url.pathname.replace(/\/+$/, '').replace(/\.git$/, '');
  if (host !== 'github.com' || pathname !== '/urdirditfurd/videowan2g') {
    throw new Error('Dépôt non autorisé');
  }
  return PUSH_URL;
}

function baseGitEnv() {
  const name = process.env.GIT_AUTHOR_NAME || 'Lofi Capybara Bot';
  const email = process.env.GIT_AUTHOR_EMAIL || 'lofi-bot@users.noreply.github.com';
  return {
    PATH: process.env.PATH || '',
    HOME: '',
    GIT_CONFIG_NOSYSTEM: '1',
    GIT_TERMINAL_PROMPT: '0',
    GIT_AUTHOR_NAME: name,
    GIT_AUTHOR_EMAIL: email,
    GIT_COMMITTER_NAME: process.env.GIT_COMMITTER_NAME || name,
    GIT_COMMITTER_EMAIL: process.env.GIT_COMMITTER_EMAIL || email,
  };
}

function createGit(cwd = ROOT, binary) {
  const options = {
    baseDir: cwd,
    unsafe: {
      allowUnsafeAskPass: true,
      allowUnsafeCredentialHelper: true,
    },
    allowEnvironment: GIT_ENV_ALLOW,
  };
  if (binary) options.binary = binary;
  return simpleGit(options).env(baseGitEnv());
}

function createAskpass(token) {
  if (!token || /[\r\n\0]/.test(token)) throw new Error('Jeton GitHub invalide');
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'lofi-askpass-'));
  const script = path.join(dir, 'askpass.sh');
  const contents = `#!/bin/sh
case "$1" in
  *[Uu]ser*) printf '%s\\n' 'x-access-token' ;;
  *) printf '%s\\n' ${shellSingleQuote(token)} ;;
esac
`;
  fs.writeFileSync(script, contents, { mode: 0o700 });
  return {
    path: script,
    cleanup() {
      fs.rmSync(dir, { recursive: true, force: true });
    },
  };
}

async function stageAndCommit(git, cwd, filePath, message) {
  const rel = path.relative(cwd, path.resolve(filePath));
  if (!rel || rel.startsWith('..') || path.isAbsolute(rel)) throw new Error('Chemin vidéo hors dépôt');
  const posixRel = rel.split(path.sep).join('/');
  await git.env(baseGitEnv()).raw(['add', '--', posixRel]);
  await git.env(baseGitEnv()).raw([
    '-c', 'commit.gpgsign=false',
    'commit',
    '-m', message,
    '--',
    posixRel,
  ]);
}

async function pushRepo(git, repoUrl) {
  const token = process.env.GITHUB_TOKEN;
  if (!token) throw new Error('GITHUB_TOKEN manquant');
  assertAllowedRepo(repoUrl);
  const ask = createAskpass(token);
  try {
    await git.env({ ...baseGitEnv(), GIT_ASKPASS: ask.path }).raw([
      '-c', 'credential.helper=',
      'push',
      PUSH_URL,
      'HEAD',
    ]);
  } finally {
    ask.cleanup();
    git.env(baseGitEnv());
  }
}

function loadScenes(filePath) {
  const raw = JSON.parse(fs.readFileSync(filePath, 'utf8'));
  if (!Array.isArray(raw) || raw.length === 0) throw new Error('scenes.json doit être un tableau non vide');
  return raw.map((item, index) => {
    if (typeof item === 'string') return { description: item.trim(), mood: null };
    if (item && typeof item.description === 'string') {
      return { description: item.description.trim(), mood: item.mood || null };
    }
    throw new Error(`Scène ${index + 1} invalide`);
  }).filter((scene) => scene.description.length >= 4);
}

function resolveMood(mood) {
  if (!mood) return 'rainy_forest';
  if (Object.prototype.hasOwnProperty.call(MOODS, mood)) return mood;
  throw new Error(`Ambiance inconnue : ${mood}`);
}

function readRuntimeOptions() {
  const duration = parseInt(process.env.VIDEO_DURATION_SECONDS || '5', 10);
  if (![5, 6, 8, 10].includes(duration)) throw new Error('VIDEO_DURATION_SECONDS doit valoir 5, 6, 8 ou 10');
  const format = process.env.VIDEO_ASPECT_RATIO || '16:9';
  if (format !== '16:9' && format !== '9:16') throw new Error('VIDEO_ASPECT_RATIO doit valoir 16:9 ou 9:16');
  state.durationSeconds = duration;
  state.format = format;
}

async function publishVideo(filePath, title) {
  assertLocalVideo(filePath);
  const repoUrl = process.env.GIT_REPO_URL;
  if (!repoUrl) throw new Error('GIT_REPO_URL manquant');
  assertAllowedRepo(repoUrl);
  const git = createGit(ROOT);
  await stageAndCommit(git, ROOT, filePath, commitMessage(title));
  await pushRepo(git, repoUrl);
}

async function processScene(scene, imageQueue, videoQueue) {
  const mood = resolveMood(scene.mood);
  const variation = rollVariation(null);
  log('state', 'SCENE', `${scene.description} · ${MOODS[mood].label} · ${variation.timeOfDay}, ${variation.season}, ${variation.weather}`);
  const exp = await generateExpansion(scene.description, mood, variation);
  const imageUrl = await imageQueue.add(() => generateImage(buildLofiImagePrompt(exp), state.format));
  log('success', 'IMAGE', 'Image générée');
  createVideoIntervalMs = VIDEO_INTERVAL_SAFE;
  videoAttemptsSinceLastError = 0;
  const started = Date.now();
  const videoId = await videoQueue.add(() => generateVideo(imageUrl, buildLofiVideoPrompt(exp), state.format));
  log('success', 'VIDÉO', `Tâche : ${videoId}`);
  const videoUrl = await pollVideo(videoId, (progress) => log('info', 'VIDÉO', progress));
  recordGenerationTime('video', Date.now() - started);
  const filePath = videoFilePath(exp.title);
  await downloadVideo(videoUrl, filePath);
  log('success', 'FICHIER', path.basename(filePath));
  await publishVideo(filePath, exp.title);
  log('success', 'GIT', `Publié : ${exp.title}`);
  return filePath;
}

async function runPipeline() {
  if (!process.env.AGNES_API_KEY) throw new Error('AGNES_API_KEY manquant');
  if (!process.env.GITHUB_TOKEN) throw new Error('GITHUB_TOKEN manquant');
  if (!process.env.GIT_REPO_URL) throw new Error('GIT_REPO_URL manquant');
  readRuntimeOptions();
  assertAllowedRepo(process.env.GIT_REPO_URL);
  const scenes = loadScenes(SCENES_PATH);
  const imageQueue = createRateLimitedQueue(1, IMAGE_INTERVAL_MS);
  const videoQueue = createRateLimitedQueue(1, VIDEO_INTERVAL_SAFE);
  let failed = 0;
  for (let i = 0; i < scenes.length; i += 1) {
    if (state.stopRequested) throw new Error('Arrêt demandé');
    const scene = scenes[i];
    log('state', 'START', `Scène ${i + 1}/${scenes.length}`);
    try {
      await processScene(scene, imageQueue, videoQueue);
    } catch (e) {
      if (e && e.message === 'Arrêt demandé') throw e;
      failed += 1;
      log('error', 'SCENE', e && e.message ? e.message : String(e));
    }
  }
  if (failed > 0) throw new Error(`${failed} scène(s) en échec`);
}

function mockResponse({ status, body, headers = {} }) {
  let consumed = false;
  const normalized = {};
  Object.keys(headers).forEach((key) => {
    normalized[key.toLowerCase()] = headers[key];
  });
  return {
    status,
    ok: status >= 200 && status < 300,
    headers: {
      get(name) {
        return normalized[String(name).toLowerCase()] || null;
      },
    },
    body: {
      resume() {},
      async *[Symbol.asyncIterator]() {
        yield Buffer.from(body);
      },
    },
    clone() {
      return mockResponse({ status, body, headers });
    },
    async text() {
      if (consumed) throw new Error('body used');
      consumed = true;
      return body;
    },
    async json() {
      return JSON.parse(await this.text());
    },
  };
}

function assert(condition, message) {
  if (!condition) throw new Error(message);
}

async function assertRejects(fn, pattern) {
  try {
    await fn();
  } catch (e) {
    if (pattern && !pattern.test(e.message)) {
      throw new Error(`Message inattendu : ${e.message}`);
    }
    return e;
  }
  throw new Error('Une erreur était attendue');
}

async function withInstantSleep(fn) {
  const previousSleep = sleepFn;
  const previousRandom = randomFn;
  const waits = [];
  sleepFn = async (ms) => { waits.push(ms); };
  randomFn = () => 0;
  try {
    await fn(waits);
  } finally {
    sleepFn = previousSleep;
    randomFn = previousRandom;
  }
}

function sampleMp4() {
  const buf = Buffer.alloc(32);
  buf.write('ftyp', 4);
  buf.write('isom', 8);
  return buf;
}

function sampleWebm() {
  const buf = Buffer.alloc(16);
  buf[0] = 0x1a;
  buf[1] = 0x45;
  buf[2] = 0xdf;
  buf[3] = 0xa3;
  return buf;
}

async function* once(buf) {
  yield buf;
}

async function runSelfTest() {
  const previousLogging = loggingEnabled;
  loggingEnabled = false;
  state.stopRequested = false;
  const failures = [];
  async function check(name, fn) {
    try {
      await fn();
      console.log(`ok  ${name}`);
    } catch (e) {
      failures.push(`${name} — ${e.message}`);
      console.error(`échec  ${name} — ${e.message}`);
    }
  }

  await check('variation dans le prompt', async () => {
    const prompt = buildExpansionPrompt('maison japonaise sous la pluie', 'rainy_forest', {
      timeOfDay: 'heure bleue',
      season: 'automne avancé',
      weather: 'brume basse',
    });
    assert(prompt.includes('heure bleue'), 'heure absente');
    assert(prompt.includes('automne avancé'), 'saison absente');
    assert(prompt.includes('brume basse'), 'météo absente');
    assert(/capybara/i.test(prompt), 'capybara absente');
    assert(/unique/i.test(prompt), 'unicité absente');
  });

  await check('extraction JSON', async () => {
    const parsed = extractJson('```json\n{"title":"Lac","image_prompt":"lake","video_motion_prompt":"ripples"}\n```');
    assert(parsed.title === 'Lac' && parsed.image_prompt === 'lake', 'fence');
    const broken = extractJson('{"title":"Pluie","image_prompt":"wet leaves","video_motion_prompt":"slow"} trailing');
    assert(broken && broken.image_prompt === 'wet leaves', 'salvage');
    const multiline = extractJson('{"title":"A","image_prompt":"ligne\ndeux","video_motion_prompt":"calm"}');
    assert(multiline && multiline.image_prompt.includes('ligne'), 'contrôle');
  });

  await check('backoff de file saturée', async () => {
    const previous = randomFn;
    randomFn = () => 0;
    try {
      assert(videoQueueDelaySeconds(0, 0) === 60, '60');
      assert(videoQueueDelaySeconds(1, 0) === 120, '120');
      assert(videoQueueDelaySeconds(2, 0) === 300, '300');
      assert(videoQueueDelaySeconds(8, 0) === 300, 'plafond de palier');
      assert(videoQueueDelaySeconds(0, 500) === 360, 'plafond 360');
      assert(videoQueueDelaySeconds(1, 30) === 30, 'retry-after');
    } finally {
      randomFn = previous;
    }
  });

  await check('file saturée hors budget de tentatives', async () => {
    await withInstantSleep(async (waits) => {
      let calls = 0;
      const fetchImpl = async () => {
        calls += 1;
        return mockResponse({ status: 503, body: '{"error":"video_queue_full"}', headers: {} });
      };
      await assertRejects(
        () => apiFetch('https://apihub.agnes-ai.com/v1/videos', { method: 'POST' }, 'Vidéo', 'video', fetchImpl),
        /saturée/,
      );
      assert(calls === VIDEO_QUEUE_MAX_RETRIES + 1, `appels ${calls}`);
      assert(waits.length === VIDEO_QUEUE_MAX_RETRIES, `attentes ${waits.length}`);
      assert(waits[0] === 60000 && waits[1] === 120000 && waits[2] === 300000, 'paliers');
    });
  });

  await check('erreur HTTP terminale non rejouée', async () => {
    await withInstantSleep(async () => {
      let calls = 0;
      const fetchImpl = async () => {
        calls += 1;
        return mockResponse({ status: 401, body: 'invalid api key' });
      };
      const error = await assertRejects(
        () => apiFetch('https://apihub.agnes-ai.com/v1/chat/completions', { method: 'POST' }, 'Expansion', 'text', fetchImpl),
        /401/,
      );
      assert(error.message.includes('invalid api key'), 'corps perdu');
      assert(calls === 1, 'rejouée');
    });
  });

  await check('clone du corps vidéo', async () => {
    const fetchImpl = async () => mockResponse({ status: 400, body: 'bad request details' });
    const res = await apiFetch('https://apihub.agnes-ai.com/v1/videos', { method: 'POST' }, 'Vidéo', 'video', fetchImpl);
    assert(res.status === 400, 'statut');
    assert(await res.text() === 'bad request details', 'corps déjà lu');
  });

  await check('hôtes de vidéo', async () => {
    const blocked = [
      'https://2130706433/a.mp4',
      'https://0x7f000001/a.mp4',
      'https://0177.0.0.1/a.mp4',
      'https://127.0.0.1/a.mp4',
      'https://[::ffff:127.0.0.1]/a.mp4',
      'https://[::ffff:7f00:1]/a.mp4',
      'https://[::1]/a.mp4',
      'https://169.254.169.254/latest/meta-data/',
      'https://[::ffff:a9fe:a9fe]/a.mp4',
      'https://10.1.2.3/a.mp4',
      'https://192.168.1.8/a.mp4',
      'https://172.16.0.4/a.mp4',
      'https://metadata.google.internal/a.mp4',
      'https://localhost/a.mp4',
      'http://cdn.example/a.mp4',
      'https://user:secret@cdn.example/a.mp4',
      'https://[fd00::1]/a.mp4',
    ];
    for (const url of blocked) {
      await assertRejects(() => Promise.resolve().then(() => assertPublicHttpsUrl(url)), /refus/i);
    }
    assert(assertPublicHttpsUrl('https://cdn.example/a.mp4').hostname === 'cdn.example', 'public');
    assert(assertPublicHttpsUrl('https://[::ffff:0808:0808]/a.mp4'), 'ipv4 publique mappée');
  });

  await check('redirections et écriture', async () => {
    const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'lofi-dl-'));
    const dest = path.join(dir, 'clip.mp4');
    let step = 0;
    const fetchImpl = async (url) => {
      step += 1;
      if (step === 1) return mockResponse({ status: 302, body: '', headers: { location: 'https://cdn.example/hop' } });
      if (step === 2) return mockResponse({ status: 302, body: '', headers: { location: 'https://cdn.example/final.mp4' } });
      assert(url === 'https://cdn.example/final.mp4', 'cible');
      return mockResponse({ status: 200, body: sampleMp4() });
    };
    await downloadVideo('https://cdn.example/start.mp4', dest, fetchImpl);
    assertLocalVideo(dest);

    const tooMany = path.join(dir, 'too.mp4');
    let hops = 0;
    await assertRejects(() => downloadVideo('https://cdn.example/start.mp4', tooMany, async () => {
      hops += 1;
      return mockResponse({ status: 302, body: '', headers: { location: `https://cdn.example/h${hops}` } });
    }), /redirections/);

    await assertRejects(() => downloadVideo('https://cdn.example/start.mp4', path.join(dir, 'priv.mp4'), async () => {
      return mockResponse({ status: 302, body: '', headers: { location: 'https://127.0.0.1/secret.mp4' } });
    }), /refus/);
  });

  await check('lien symbolique et signature', async () => {
    const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'lofi-link-'));
    const target = path.join(dir, 'secret.txt');
    fs.writeFileSync(target, 'SECRET_CONTENT');
    const link = path.join(dir, 'video.mp4');
    fs.symlinkSync(target, link);
    await assertRejects(() => writeExclusiveVideo(link, once(sampleMp4())), /symbolique/i);
    assert(fs.readFileSync(target, 'utf8') === 'SECRET_CONTENT', 'cible écrasée');
    assert(fs.lstatSync(link).isSymbolicLink(), 'lien remplacé');
    await assertRejects(() => Promise.resolve().then(() => assertLocalVideo(link)), /symbolique/i);

    const bad = path.join(dir, 'bad.mp4');
    await assertRejects(() => writeExclusiveVideo(bad, once(Buffer.from('pas une video!!'))), /Signature|petite/);
    assert(!fs.existsSync(bad), 'fichier invalide conservé');

    const webm = path.join(dir, 'ok.webm');
    await writeExclusiveVideo(webm, once(sampleWebm()));
    assertLocalVideo(webm);
  });

  await check('nom de fichier', async () => {
    const file = videoFilePath('Maison japonaise / pluie', new Date(Date.UTC(2026, 9, 29, 4, 46, 1, 7)));
    assert(path.dirname(file) === path.resolve(VIDEO_DIR), 'dossier');
    assert(path.basename(file) === 'Maison-japonaise-pluie_20261029T044601007Z.mp4', path.basename(file));
    assert(!sanitizeTitle('../../etc/passwd').includes('/'), 'traversal');
    assert(commitMessage('Brume\nsecrète').startsWith('feat: add new lofi capybara video: '), 'message');
    assert(!commitMessage('A\nB').includes('\n'), 'newline');
  });

  await check('dépôt autorisé', async () => {
    assert(assertAllowedRepo(ALLOWED_REPO) === PUSH_URL, 'canonique');
    assert(assertAllowedRepo(`${ALLOWED_REPO}.git`) === PUSH_URL, 'suffixe git');
    const rejected = [
      'http://github.com/urdirditfurd/videowan2g',
      'https://user:token@github.com/urdirditfurd/videowan2g',
      'https://github.com/other/videowan2g',
      'https://github.com/urdirditfurd/videowan2g/extra',
      'https://evil.example/urdirditfurd/videowan2g',
    ];
    for (const url of rejected) {
      await assertRejects(() => Promise.resolve().then(() => assertAllowedRepo(url)));
    }
  });

  await check('file à intervalle', async () => {
    const starts = [];
    const queue = createRateLimitedQueue(1, 40);
    await Promise.all([
      queue.add(async () => { starts.push(Date.now()); }),
      queue.add(async () => { starts.push(Date.now()); }),
    ]);
    assert(starts.length === 2, 'deux tâches');
    assert(starts[1] - starts[0] >= 30, 'intervalle');
  });

  await check('git sans secrets', async () => {
    const sentinelKey = 'sk-SELFTESTAGNESVALUE';
    const sentinelToken = 'ghp_SELFTESTGITHUBVALUE';
    const previousKey = process.env.AGNES_API_KEY;
    const previousToken = process.env.GITHUB_TOKEN;
    process.env.AGNES_API_KEY = sentinelKey;
    process.env.GITHUB_TOKEN = sentinelToken;
    const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'lofi-git-'));
    const dump = path.join(dir, 'dump.json');
    const fake = path.join(dir, 'fake-git.cjs');
    fs.writeFileSync(fake, `#!/usr/bin/env node
const fs = require('fs');
const prev = fs.existsSync(${JSON.stringify(dump)}) ? JSON.parse(fs.readFileSync(${JSON.stringify(dump)}, 'utf8')) : [];
prev.push({ argv: process.argv.slice(2), env: process.env });
fs.writeFileSync(${JSON.stringify(dump)}, JSON.stringify(prev));
process.exit(0);
`);
    fs.chmodSync(fake, 0o755);
    const clip = path.join(dir, 'clip.mp4');
    fs.writeFileSync(clip, sampleMp4());
    try {
      const git = createGit(dir, fake);
      await stageAndCommit(git, dir, clip, commitMessage('Brume du lac'));
      await pushRepo(git, ALLOWED_REPO);
      const calls = JSON.parse(fs.readFileSync(dump, 'utf8'));
      assert(calls.length === 3, `commandes ${calls.length}`);
      const commit = calls[1];
      assert(commit.argv.includes('commit.gpgsign=false'), 'gpgsign');
      assert(commit.argv.includes('--'), 'pathspec');
      assert(commit.argv.includes('clip.mp4'), 'fichier');
      const push = calls[2];
      assert(push.argv.includes('credential.helper='), 'helper');
      assert(push.argv.includes(PUSH_URL), 'url');
      const allowed = new Set([
        'PATH', 'HOME',
        'GIT_CONFIG_NOSYSTEM', 'GIT_TERMINAL_PROMPT',
        'GIT_AUTHOR_NAME', 'GIT_AUTHOR_EMAIL',
        'GIT_COMMITTER_NAME', 'GIT_COMMITTER_EMAIL',
        'GIT_TEST_DISALLOW_ABBREVIATED_OPTIONS',
      ]);
      calls.forEach((call, index) => {
        const blob = JSON.stringify(call.env);
        assert(!Object.prototype.hasOwnProperty.call(call.env, 'AGNES_API_KEY'), `nom agnes #${index}`);
        assert(!Object.prototype.hasOwnProperty.call(call.env, 'GITHUB_TOKEN'), `nom github #${index}`);
        assert(!blob.includes(sentinelKey) && !blob.includes(sentinelToken), `valeur #${index}`);
        Object.keys(call.env).forEach((key) => {
          if (index === calls.length - 1 && key === 'GIT_ASKPASS') return;
          assert(allowed.has(key), `variable inattendue ${key}`);
        });
      });
      assert(calls[2].env.GIT_ASKPASS, 'askpass absent');
      assert(!fs.existsSync(calls[2].env.GIT_ASKPASS), 'askpass non supprimé');
    } finally {
      if (previousKey === undefined) delete process.env.AGNES_API_KEY;
      else process.env.AGNES_API_KEY = previousKey;
      if (previousToken === undefined) delete process.env.GITHUB_TOKEN;
      else process.env.GITHUB_TOKEN = previousToken;
    }
  });

  await check('askpass temporaire', async () => {
    const token = 'ghp_ASKPASSVALUE';
    const ask = createAskpass(token);
    const user = spawnSync(ask.path, ['Username for https://github.com'], { encoding: 'utf8' });
    const pass = spawnSync(ask.path, ['Password for https://github.com'], { encoding: 'utf8' });
    const userOk = user.stdout.trim() === 'x-access-token';
    const passOk = pass.stdout.trim() === token;
    ask.cleanup();
    assert(userOk, 'username');
    assert(passOk, 'password');
    assert(!fs.existsSync(ask.path), 'fichier restant');
  });

  await check('scenes.json', async () => {
    const scenes = loadScenes(SCENES_PATH);
    assert(scenes.length === 5, `compte ${scenes.length}`);
    assert(scenes[0].description === 'maison japonaise sous la pluie', 'première scène');
  });

  loggingEnabled = previousLogging;
  if (failures.length > 0) {
    console.error(`auto-test échoué : ${failures.length}`);
    return false;
  }
  console.log('auto-test réussi');
  return true;
}

async function main() {
  if (process.argv.includes('--self-test')) {
    const ok = await runSelfTest();
    process.exit(ok ? 0 : 1);
  }
  await runPipeline();
}

process.on('SIGINT', () => {
  state.stopRequested = true;
  log('warn', 'STOP', 'Arrêt demandé');
});

if (require.main === module) {
  main().catch((e) => {
    console.error(redact(e && e.message ? e.message : e));
    process.exit(1);
  });
}
