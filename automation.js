#!/usr/bin/env node
'use strict';

const dns = require('dns');
const fs = require('fs');
const os = require('os');
const path = require('path');
const { Readable } = require('stream');
const dotenv = require('dotenv');
const fetch = require('node-fetch');
const { simpleGit } = require('simple-git');

dotenv.config({ quiet: true, path: path.join(__dirname, '.env') });

const API_BASE = 'https://apihub.agnes-ai.com/v1';
const POLL_BASE = 'https://apihub.agnes-ai.com/agnesapi';
const MODEL_TEXT = 'agnes-2.5-flash';
const MODEL_IMAGE = 'agnes-image-2.5-flash';
const MODEL_VIDEO = 'agnes-video-2.5-flash';

const ALLOWED_REPO = 'https://github.com/urdirditfurd/videowan2g.git';
const VIDEO_QUEUE_BACKOFFS_SEC = [60, 120, 300];
const VIDEO_QUEUE_MAX_WAIT_SEC = 360;
const VIDEO_QUEUE_MAX_RETRIES = 12;
const MAX_VIDEO_BYTES = 100 * 1024 * 1024;
const MAX_REDIRECTS = 2;
const VIDEO_FIRST_POLL_DELAY_SEC = 80;
const VIDEO_MAX_POLL_ATTEMPTS = 100;
const IMAGE_INTERVAL_MIN = 3200;
const IMAGE_INTERVAL_MAX = 12000;
const VIDEO_INTERVAL_MIN = 62000;
const VIDEO_INTERVAL_MAX = 90000;
const VIDEO_INTERVAL_SAFE = 75000;

const OPEN_NEW = fs.constants.O_CREAT | fs.constants.O_EXCL | fs.constants.O_NOFOLLOW | fs.constants.O_WRONLY;
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

const GHIBLI_PALETTE = 'soft pastel palette inspired by Studio Ghibli watercolor backgrounds: sage green #8fad88, warm cream #f2efe9, dusky blue #7a9eb8, soft gold #c4a882, gentle forest greens, muted earth tones, warm amber highlights. Never grey, never desaturated, never neon. Rich but calm. Hand-painted watercolor feel with soft gradients.';

const TIMES_OF_DAY = [
  'pale blue dawn',
  'misty morning',
  'soft late morning',
  'gentle noon light',
  'golden late afternoon',
  'blue hour dusk',
  'quiet moonlit night',
];
const SEASONS = [
  'early spring with young leaves',
  'humid midsummer',
  'late summer cicada haze',
  'deep autumn',
  'quiet early winter',
];
const WEATHERS = [
  'fine warm rain',
  'after a rain shower',
  'low drifting mist',
  'clear soft sky',
  'light breeze',
  'gentle snowfall',
  'humid air after thunder',
];

const MOOD_KEYS = Object.keys(MOODS);

const state = {
  stopRequested: false,
  durationSeconds: 5,
  format: '16:9',
  apiKey: '',
  silent: false,
};

let createImageIntervalMs = 4000;
let createVideoIntervalMs = VIDEO_INTERVAL_SAFE;
let imageAttemptsSinceLastError = 0;
let videoAttemptsSinceLastError = 0;
let activeAskpass = null;

const deps = {
  fetch: (url, options) => fetch(url, options),
  sleep: (ms) => sleepInterruptible(ms),
  random: () => Math.random(),
  lookup: (hostname) => dns.promises.lookup(hostname, { all: true, verbatim: true }),
};

function redact(value) {
  let text = String(value == null ? '' : value);
  const secrets = [process.env.AGNES_API_KEY, process.env.GITHUB_TOKEN, state.apiKey];
  for (const secret of secrets) {
    if (secret && secret.length > 6) text = text.split(secret).join('[redacted]');
  }
  return text;
}

function log(level, category, message, data) {
  if (state.silent && level !== 'error') return;
  const line = `${new Date().toISOString()} [${level}] [${category}] ${redact(message)}`;
  if (level === 'error') console.error(line);
  else console.log(line);
  if (data === undefined) return;
  let extra;
  try {
    extra = typeof data === 'string' ? data : JSON.stringify(data);
  } catch (err) {
    extra = String(data);
  }
  if (extra.length > 2000) extra = `${extra.slice(0, 2000)}…`;
  console.log(redact(extra));
}

function sleepInterruptible(ms) {
  const step = 1000;
  let left = ms;
  return (async () => {
    while (left > 0) {
      if (state.stopRequested) throw new Error('Arrêt demandé');
      const chunk = Math.min(step, left);
      await new Promise((resolve) => setTimeout(resolve, chunk));
      left -= chunk;
    }
  })();
}

function resetRuntime() {
  state.stopRequested = false;
  state.durationSeconds = 5;
  state.format = '16:9';
  createImageIntervalMs = 4000;
  createVideoIntervalMs = VIDEO_INTERVAL_SAFE;
  imageAttemptsSinceLastError = 0;
  videoAttemptsSinceLastError = 0;
}

function attemptsForKind(kind) {
  switch (kind) {
    case 'video':
      return 10;
    case 'image':
      return 10;
    case 'text':
      return 8;
    case 'default':
      return 8;
    default: {
      const unexpected = kind;
      throw new Error(`type de requête inconnu: ${unexpected}`);
    }
  }
}

function computeQueueWaitSeconds(queueRetryIndex, rng) {
  const index = Math.min(Math.max(queueRetryIndex, 0), VIDEO_QUEUE_BACKOFFS_SEC.length - 1);
  const base = VIDEO_QUEUE_BACKOFFS_SEC[index];
  const span = Math.max(1, Math.round(base * 0.1));
  const jitter = Math.floor(rng() * span);
  return Math.min(VIDEO_QUEUE_MAX_WAIT_SEC, base + jitter);
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

async function readErrorText(res) {
  try {
    const source = typeof res.clone === 'function' ? res.clone() : res;
    return String(await source.text()).slice(0, 1200);
  } catch (err) {
    return '';
  }
}

async function apiFetch(url, options = {}, label = 'API', kind = 'default') {
  const maxAttempts = attemptsForKind(kind);
  let attempt = 0;
  let queueRetries = 0;

  while (attempt < maxAttempts) {
    if (state.stopRequested) throw new Error('Arrêt demandé');
    const method = options.method || 'GET';
    log('api', `REQ ${label}`, `${method} ${safeUrlForLog(url)} (tentative ${attempt + 1}/${maxAttempts})`);

    let res;
    try {
      res = await deps.fetch(url, options);
    } catch (err) {
      if (err && err.message === 'Arrêt demandé') throw err;
      const wait = Math.min(kind === 'video' ? 120 : 45, kind === 'video' ? 10 + attempt * 8 : 3 + attempt * 4);
      log('warn', 'NETWORK', `${err.message} — retry dans ${wait}s`);
      await deps.sleep(wait * 1000);
      attempt += 1;
      continue;
    }

    let retryAfter = 0;
    try {
      retryAfter = parseInt(res.headers.get('Retry-After') || '0', 10) || 0;
    } catch (err) {
      retryAfter = 0;
    }

    if (res.ok) {
      noteSuccess(kind);
      return res;
    }

    const errorText = await readErrorText(res);
    log('response', `RÉP ${label}`, `HTTP ${res.status}`, errorText || undefined);

    if ([400, 401, 404].includes(res.status) && kind !== 'video') {
      throw new Error(`HTTP ${res.status} — ${errorText.slice(0, 300)}`);
    }

    const queueFull = /video_queue_full|queue is full/i.test(errorText);
    if (kind === 'video' && queueFull) {
      if (queueRetries >= VIDEO_QUEUE_MAX_RETRIES) {
        throw new Error('File vidéo saturée — tentatives de file épuisées');
      }
      videoAttemptsSinceLastError = 0;
      const wait = computeQueueWaitSeconds(queueRetries, deps.random);
      queueRetries += 1;
      log('warn', 'VIDEO_QUEUE', `Queue vidéo saturée — nouvelle tentative dans ${wait}s`);
      await deps.sleep(wait * 1000);
      continue;
    }

    if (res.status === 429) {
      const wait = retryAfter > 0
        ? retryAfter
        : Math.min(kind === 'video' ? 180 : 120, kind === 'video' ? 60 + attempt * 8 : 10 + attempt * 12);
      if (kind === 'image') createImageIntervalMs = Math.min(IMAGE_INTERVAL_MAX, createImageIntervalMs + 1000);
      if (kind === 'video') createVideoIntervalMs = Math.min(VIDEO_INTERVAL_MAX, createVideoIntervalMs + 6000);
      log('warn', 'RATELIMIT', `429 — attente ${wait}s`);
      await deps.sleep(wait * 1000);
      attempt += 1;
      continue;
    }

    if ([500, 502, 503, 504, 520].includes(res.status)) {
      const wait = Math.min(kind === 'video' ? 180 : 60, kind === 'video' ? 20 + attempt * 10 : 5 + attempt * 5);
      log('warn', 'SERVICE', `HTTP ${res.status} — attente ${wait}s`);
      await deps.sleep(wait * 1000);
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
  } catch (err) {
    /* stratégie suivante */
  }
  try {
    return JSON.parse(escapeControlCharsInStrings(raw));
  } catch (err) {
    /* stratégie suivante */
  }
  try {
    const end = raw.lastIndexOf('}');
    if (end > 0) return JSON.parse(escapeControlCharsInStrings(raw.slice(0, end + 1)));
  } catch (err) {
    /* stratégie suivante */
  }
  try {
    return JSON.parse(balanceJson(escapeControlCharsInStrings(raw)));
  } catch (err) {
    /* stratégie suivante */
  }
  const salvaged = salvageExpansion(raw);
  if (salvaged && (salvaged.image_prompt || salvaged.video_motion_prompt)) return salvaged;
  return null;
}

function pickFrom(list, rng) {
  const index = Math.floor(rng() * list.length);
  return list[Math.min(list.length - 1, Math.max(0, index))];
}

function pickVariation(rng = deps.random) {
  return {
    timeOfDay: pickFrom(TIMES_OF_DAY, rng),
    season: pickFrom(SEASONS, rng),
    weather: pickFrom(WEATHERS, rng),
    seed: Math.floor(rng() * 1e9),
  };
}

function buildExpansionPrompt(userDescription, mood, variation) {
  const chosen = variation || pickVariation();
  const moodInfo = MOODS[mood] || MOODS.rainy_forest;
  return [
    'You are an art director for Studio Ghibli-inspired lofi ambience videos.',
    '',
    'USER SCENE IDEA (French):',
    userDescription,
    '',
    'AMBIENCE (French mood chosen):',
    moodInfo.label,
    '—',
    moodInfo.desc,
    '',
    'UNIQUE SCENE VARIATION — every generation must look different, even if the user idea is similar.',
    `Time of day: ${chosen.timeOfDay}.`,
    `Season: ${chosen.season}.`,
    `Weather: ${chosen.weather}.`,
    `Variation seed (do not render this number, do not write it anywhere in the image): ${chosen.seed}.`,
    'Let the seed change small concrete details: cloud shapes, flower species, wood grain, cup, blanket pattern, lantern placement.',
    'The capybara stays present, cute, and calm.',
    '',
    'YOUR TASK:',
    'Expand the user idea into a rich, poetic, ultra-detailed visual scene for a single lofi ambience video.',
    'ALWAYS include a cute, calm, peaceful capybara (capybara mascot) somewhere visible in the scene, integrated naturally into the environment.',
    'The mood must be lofi: slow, nostalgic, peaceful, detailed, soothing, hand-painted watercolor, Studio Ghibli / Hayao Miyazaki inspired.',
    'The result must be visually unique. Do not reuse a generic template.',
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
    '  "image_prompt": "Ultra-detailed English visual description for an image model. Include: setting, lighting, colors, atmosphere, the capybara (with its location and pose), the unique time/season/weather, Ghibli watercolor style, cinematic composition. 150-250 words.",',
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
    Authorization: `Bearer ${state.apiKey}`,
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
  log('success', 'EXPANSION', `Titre : ${exp.title}`);
  return exp;
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
        await deps.sleep(wait * 1000);
      }
    }
  }
  throw lastError || new Error(`${name} échoué`);
}

function assertImageReference(imageUrl) {
  const value = String(imageUrl || '');
  if (value.startsWith('data:image/')) return value;
  let url;
  try {
    url = new URL(value);
  } catch (err) {
    throw new Error('URL image invalide');
  }
  if (url.protocol !== 'https:') throw new Error('image https requise');
  if (url.username || url.password) throw new Error('image URL refusée');
  if (isBlockedHostname(url.hostname)) throw new Error('hôte image refusé');
  return url.toString();
}

async function generateImage(prompt, format) {
  return retryTask('Image', async () => {
    const ratio = format === '9:16' ? '9:16' : '16:9';
    const body = {
      model: MODEL_IMAGE,
      prompt,
      size: '1K',
      ratio,
      n: 1,
    };
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
    if (item.url) return assertImageReference(item.url);
    if (item.b64_json) return `data:image/png;base64,${item.b64_json}`;
    throw new Error('Format image inconnu');
  }, 4);
}

async function generateVideo(imageUrl, prompt, format) {
  return retryTask('Vidéo', async () => {
    const aspect = format === '9:16' ? '9:16' : '16:9';
    const safeImage = assertImageReference(imageUrl);
    const body = {
      model: MODEL_VIDEO,
      prompt,
      seconds: String(state.durationSeconds),
      mode: 'keyframe',
      size: '720P',
      aspect_ratio: aspect,
      first_frame: safeImage,
      last_frame: safeImage,
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
    return String(videoId);
  }, 3);
}

function timingPath(cwd) {
  return path.join(cwd, 'generated_videos', '.timing.json');
}

function loadTiming(cwd) {
  try {
    return JSON.parse(fs.readFileSync(timingPath(cwd), 'utf8'));
  } catch (err) {
    return {};
  }
}

function estimateGenerationTime(cwd, key, fallbackMs) {
  const cache = loadTiming(cwd);
  const samples = cache[key];
  if (!samples || samples.length === 0) return fallbackMs;
  const sorted = [...samples].sort((a, b) => a - b);
  const mid = Math.floor(sorted.length / 2);
  return sorted.length % 2 === 0 ? (sorted[mid - 1] + sorted[mid]) / 2 : sorted[mid];
}

function recordGenerationTime(cwd, key, ms) {
  const cache = loadTiming(cwd);
  if (!cache[key]) cache[key] = [];
  cache[key].push(ms);
  if (cache[key].length > 10) cache[key] = cache[key].slice(-10);
  try {
    fs.writeFileSync(timingPath(cwd), JSON.stringify(cache));
  } catch (err) {
    log('warn', 'TIMING', err.message);
  }
}

async function pollVideo(videoId, onProgress, cwd) {
  const estimatedMs = estimateGenerationTime(cwd, 'video', 90000 + state.durationSeconds * 7000);
  const minWait = Math.max(25, Math.min(VIDEO_FIRST_POLL_DELAY_SEC, Math.floor(estimatedMs / 1000 * 0.35)));
  let remaining = minWait;
  log('info', 'POLLING', `Attente initiale ${remaining}s pour ${videoId}`);
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
      res = await apiFetch(url, { method: 'GET', headers: { Authorization: `Bearer ${state.apiKey}` } }, 'Polling', 'video');
    } catch (err) {
      log('warn', 'POLLING', `Erreur transitoire : ${err.message}`);
      await deps.sleep(Math.min(90000, 15000 + attempt * 1500));
      continue;
    }
    let data;
    try {
      data = await res.json();
    } catch (err) {
      log('warn', 'POLLING', 'Réponse non JSON, reprise');
      await deps.sleep(Math.min(60000, 10000 + attempt * 1000));
      continue;
    }
    const status = String(data.status || 'unknown').toLowerCase();
    const progress = Number(data.progress || 0);
    onProgress(`Création — ${progress}%`);
    log('response', `POLL #${attempt}`, `Statut : ${status} (${progress}%)`);
    if (status === 'completed' || status === 'succeeded' || status === 'done') {
      const videoUrl = data.url || (data.metadata && data.metadata.url) || (data.output && data.output.url);
      if (!videoUrl) throw new Error('Terminé mais pas d\'URL');
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

function createRateLimitedQueue(maxConcurrent, minIntervalMs) {
  let running = 0;
  let lastStart = 0;
  const queue = [];
  let resolveIdle = null;
  const intervalOf = () => (typeof minIntervalMs === 'function' ? minIntervalMs() : minIntervalMs);

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
    const wait = Math.max(0, lastStart + intervalOf() - Date.now());
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

function stripHost(hostname) {
  return String(hostname || '').replace(/^\[|\]$/g, '').split('%')[0].toLowerCase();
}

function isIPv4Syntax(ip) {
  const parts = String(ip).split('.');
  if (parts.length !== 4) return false;
  return parts.every((part) => /^\d{1,3}$/.test(part) && Number(part) >= 0 && Number(part) <= 255);
}

function isBlockedIPv4(ip) {
  if (!isIPv4Syntax(ip)) return true;
  const parts = ip.split('.').map((part) => Number(part));
  const a = parts[0];
  const b = parts[1];
  const c = parts[2];
  if (a === 0 || a === 10 || a === 127) return true;
  if (a === 169 && b === 254) return true;
  if (a === 172 && b >= 16 && b <= 31) return true;
  if (a === 192 && b === 168) return true;
  if (a === 100 && b >= 64 && b <= 127) return true;
  if (a === 192 && b === 0 && (c === 0 || c === 2)) return true;
  if (a === 198 && b === 51 && c === 100) return true;
  if (a === 203 && b === 0 && c === 113) return true;
  if (a >= 224) return true;
  return false;
}

function expandIPv6(host) {
  let input = stripHost(host);
  let v4tail = null;
  const v4 = input.match(/:(\d{1,3}(?:\.\d{1,3}){3})$/);
  if (v4) {
    v4tail = v4[1];
    input = `${input.slice(0, v4.index)}:0:0`;
  }
  const halves = input.split('::');
  if (halves.length > 2) return null;
  const head = halves[0] ? halves[0].split(':') : [];
  const tail = halves.length === 2 ? (halves[1] ? halves[1].split(':') : []) : [];
  if (halves.length === 1 && head.length !== 8) return null;
  const missing = 8 - (head.length + tail.length);
  if (missing < 0) return null;
  const parts = halves.length === 1 ? head : [...head, ...Array(missing).fill('0'), ...tail];
  if (parts.length !== 8) return null;
  const hextets = parts.map((part) => {
    if (!/^[0-9a-f]{1,4}$/.test(part)) return Number.NaN;
    return parseInt(part, 16);
  });
  if (hextets.some((part) => Number.isNaN(part))) return null;
  if (v4tail) {
    if (!isIPv4Syntax(v4tail)) return null;
    const nums = v4tail.split('.').map((part) => Number(part));
    hextets[6] = (nums[0] << 8) + nums[1];
    hextets[7] = (nums[2] << 8) + nums[3];
  }
  return hextets;
}

function isBlockedIPv6(host) {
  const hextets = expandIPv6(host);
  if (!hextets) return true;
  const mapped = hextets[0] === 0 && hextets[1] === 0 && hextets[2] === 0
    && hextets[3] === 0 && hextets[4] === 0 && hextets[5] === 0xffff;
  if (mapped) {
    const v4 = `${(hextets[6] >> 8) & 255}.${hextets[6] & 255}.${(hextets[7] >> 8) & 255}.${hextets[7] & 255}`;
    return isBlockedIPv4(v4);
  }
  if (hextets.every((part, index) => (index === 7 ? part === 1 : part === 0))) return true;
  if (hextets.every((part) => part === 0)) return true;
  if ((hextets[0] & 0xfe00) === 0xfc00) return true;
  if ((hextets[0] & 0xffc0) === 0xfe80) return true;
  if ((hextets[0] & 0xff00) === 0xff00) return true;
  return false;
}

function isBlockedHostname(hostname) {
  const host = stripHost(hostname);
  if (!host) return true;
  if (host === 'localhost' || host.endsWith('.localhost') || host.endsWith('.local')) return true;
  if (host === 'metadata.google.internal' || host === 'metadata' || host === 'metadata.internal') return true;
  if (isIPv4Syntax(host)) return isBlockedIPv4(host);
  if (host.includes(':')) return isBlockedIPv6(host);
  return false;
}

function assertSafeDownloadUrl(raw) {
  let url;
  try {
    url = new URL(raw);
  } catch (err) {
    throw new Error('URL vidéo invalide');
  }
  if (url.protocol !== 'https:') throw new Error('HTTPS requis pour le téléchargement');
  if (url.username || url.password) throw new Error('identifiants interdits dans l\'URL vidéo');
  if (isBlockedHostname(url.hostname)) throw new Error('hôte refusé');
  return url;
}

function isAgnesHost(hostname) {
  const host = stripHost(hostname);
  return host === 'agnes-ai.com' || host.endsWith('.agnes-ai.com');
}

async function assertResolvedPublic(url) {
  const host = stripHost(url.hostname);
  if (isIPv4Syntax(host) || host.includes(':')) return;
  const records = await deps.lookup(host);
  const list = Array.isArray(records) ? records : [records];
  if (!list.length) throw new Error('résolution DNS vide');
  for (const record of list) {
    const address = typeof record === 'string' ? record : record.address;
    if (isBlockedHostname(address)) throw new Error('résolution vers adresse privée refusée');
  }
}

function discardBody(res) {
  if (res && res.body && typeof res.body.destroy === 'function') res.body.destroy();
}

function safeUrlForLog(raw) {
  try {
    const url = new URL(raw);
    return `${url.origin}${url.pathname}`;
  } catch (err) {
    return '[url]';
  }
}

async function fetchFollowing(rawUrl, redirects) {
  if (redirects > MAX_REDIRECTS) throw new Error('trop de redirections');
  const url = assertSafeDownloadUrl(rawUrl);
  await assertResolvedPublic(url);
  const headers = {};
  if (isAgnesHost(url.hostname) && state.apiKey) headers.Authorization = `Bearer ${state.apiKey}`;
  const res = await deps.fetch(url.toString(), { method: 'GET', redirect: 'manual', headers });
  if ([301, 302, 303, 307, 308].includes(res.status)) {
    if (redirects >= MAX_REDIRECTS) {
      discardBody(res);
      throw new Error('trop de redirections');
    }
    const location = res.headers.get('location');
    discardBody(res);
    if (!location) throw new Error('redirection sans location');
    const next = new URL(location, url).toString();
    return fetchFollowing(next, redirects + 1);
  }
  return res;
}

function isVideoHeader(buf) {
  if (!buf || buf.length < 12) return false;
  if (buf[0] === 0x1a && buf[1] === 0x45 && buf[2] === 0xdf && buf[3] === 0xa3) return true;
  return buf.slice(4, 8).toString('ascii') === 'ftyp';
}

function assertRealDirectory(dir) {
  const st = fs.lstatSync(dir);
  if (st.isSymbolicLink() || !st.isDirectory()) throw new Error('dossier invalide');
}

function ensureOutputDir(dir) {
  try {
    assertRealDirectory(dir);
  } catch (err) {
    if (err.code !== 'ENOENT') throw err;
    fs.mkdirSync(dir, { recursive: true });
    assertRealDirectory(dir);
  }
}

function prepareDestination(destPath) {
  assertRealDirectory(path.dirname(destPath));
  try {
    const st = fs.lstatSync(destPath);
    if (st.isSymbolicLink()) throw new Error('lien symbolique refusé');
    throw new Error('fichier déjà présent');
  } catch (err) {
    if (err.code === 'ENOENT') return;
    throw err;
  }
}

function safeUnlink(file) {
  try {
    const st = fs.lstatSync(file);
    if (st.isSymbolicLink() || !st.isFile()) return;
    fs.unlinkSync(file);
  } catch (err) {
    if (err.code !== 'ENOENT') throw err;
  }
}

function assertLocalVideo(filePath) {
  const st = fs.lstatSync(filePath);
  if (st.isSymbolicLink()) throw new Error('lien symbolique refusé');
  if (!st.isFile()) throw new Error('pas un fichier régulier');
  if (st.size <= 0 || st.size >= MAX_VIDEO_BYTES) throw new Error('taille refusée');
  const fd = fs.openSync(filePath, OPEN_READ);
  try {
    const buf = Buffer.alloc(32);
    const read = fs.readSync(fd, buf, 0, 32, 0);
    if (!isVideoHeader(buf.subarray(0, read))) throw new Error('en-tête vidéo invalide');
  } finally {
    fs.closeSync(fd);
  }
}

async function writeVideoStream(res, destPath) {
  const declared = Number(res.headers.get('content-length'));
  if (Number.isFinite(declared) && declared >= MAX_VIDEO_BYTES) {
    discardBody(res);
    throw new Error('vidéo trop volumineuse');
  }
  prepareDestination(destPath);
  if (!fs.constants.O_NOFOLLOW) throw new Error('O_NOFOLLOW indisponible');
  const fd = fs.openSync(destPath, OPEN_NEW, 0o644);
  let written = 0;
  let header = Buffer.alloc(0);
  let headerChecked = false;
  let settled = false;
  try {
    await new Promise((resolve, reject) => {
      const fail = (err) => {
        if (settled) return;
        settled = true;
        discardBody(res);
        reject(err);
      };
      const succeed = () => {
        if (settled) return;
        settled = true;
        resolve();
      };
      if (!res.body || typeof res.body.on !== 'function') {
        fail(new Error('flux vidéo absent'));
        return;
      }
      res.body.on('data', (chunk) => {
        if (settled) return;
        try {
          const buf = Buffer.isBuffer(chunk) ? chunk : Buffer.from(chunk);
          written += buf.length;
          if (written > MAX_VIDEO_BYTES) {
            fail(new Error('vidéo trop volumineuse'));
            return;
          }
          if (header.length < 32) header = Buffer.concat([header, buf]).subarray(0, 32);
          if (!headerChecked && header.length >= 12) {
            headerChecked = true;
            if (!isVideoHeader(header)) {
              fail(new Error('contenu non vidéo'));
              return;
            }
          }
          fs.writeSync(fd, buf);
        } catch (err) {
          fail(err);
        }
      });
      res.body.on('end', () => {
        if (!headerChecked || !isVideoHeader(header) || written <= 0) {
          fail(new Error('contenu non vidéo'));
          return;
        }
        succeed();
      });
      res.body.on('error', (err) => fail(err));
    });
  } catch (err) {
    try { fs.closeSync(fd); } catch (closeErr) { /* déjà fermé */ }
    safeUnlink(destPath);
    throw err;
  }
  fs.closeSync(fd);
}

async function downloadVideo(videoUrl, destPath) {
  const res = await fetchFollowing(videoUrl, 0);
  if (!res.ok) {
    const errText = await readErrorText(res);
    discardBody(res);
    throw new Error(`Téléchargement HTTP ${res.status} — ${errText.slice(0, 200)}`);
  }
  await writeVideoStream(res, destPath);
  assertLocalVideo(destPath);
  return destPath;
}

function sanitizeTitle(title) {
  const cleaned = String(title || '')
    .normalize('NFKD')
    .replace(/[\u0300-\u036f]/g, '')
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, '_')
    .replace(/^_+|_+$/g, '')
    .slice(0, 60);
  if (!cleaned || cleaned === '.' || cleaned === '..') return 'ambiance_lofi';
  return cleaned;
}

function sanitizeCommitTitle(title) {
  return String(title || 'Ambiance Lofi').replace(/[\r\n]+/g, ' ').slice(0, 120);
}

function videoFileName(title, now = Date.now()) {
  return `${sanitizeTitle(title)}_${now}.mp4`;
}

function assertAllowedRepoUrl(raw) {
  const trimmed = String(raw || '').trim();
  let url;
  try {
    url = new URL(trimmed);
  } catch (err) {
    throw new Error('URL dépôt invalide');
  }
  if (url.protocol !== 'https:') throw new Error('HTTPS requis pour le dépôt');
  if (url.username || url.password) throw new Error('identifiants interdits dans l\'URL du dépôt');
  if (url.hostname !== 'github.com') throw new Error('dépôt non autorisé');
  const parts = url.pathname.replace(/\.git$/, '').replace(/\/$/, '').split('/').filter(Boolean);
  if (parts.length !== 2 || parts[0] !== 'urdirditfurd' || parts[1] !== 'videowan2g') {
    throw new Error('dépôt non autorisé');
  }
  return ALLOWED_REPO;
}

function assertGithubToken(token) {
  const value = String(token || '');
  if (value.length < 8) throw new Error('GITHUB_TOKEN manquant');
  if (/[\r\n\0]/.test(value)) throw new Error('GITHUB_TOKEN invalide');
  return value;
}

function assertApiKey(key) {
  const value = String(key || '').trim();
  if (value.length < 8) throw new Error('AGNES_API_KEY manquante');
  if (/[\r\n\0\s]/.test(value)) throw new Error('AGNES_API_KEY invalide');
  return value;
}

function buildGitEnv(extra) {
  const env = {
    PATH: process.env.PATH || '',
    HOME: '',
    GIT_CONFIG_NOSYSTEM: '1',
    GIT_TERMINAL_PROMPT: '0',
    GIT_AUTHOR_NAME: 'Lofi Capybara Studio',
    GIT_AUTHOR_EMAIL: 'lofi-capybara@users.noreply.github.com',
    GIT_COMMITTER_NAME: 'Lofi Capybara Studio',
    GIT_COMMITTER_EMAIL: 'lofi-capybara@users.noreply.github.com',
  };
  if (extra && extra.GIT_ASKPASS) env.GIT_ASKPASS = extra.GIT_ASKPASS;
  return env;
}

function createGit(cwd, env, binary) {
  const options = {
    baseDir: cwd,
    unsafe: {
      allowUnsafeAskPass: true,
      allowUnsafeCredentialHelper: true,
    },
    allowEnvironment: GIT_ENV_ALLOW,
  };
  if (binary) {
    options.binary = binary;
    options.unsafe.allowUnsafeCustomBinary = true;
  }
  const git = simpleGit(options);
  git.env(env);
  return git;
}

function writeAskpass(token) {
  const file = path.join(os.tmpdir(), `lofi-askpass-${process.pid}-${Date.now()}.cjs`);
  const source = [
    "'use strict';",
    `const token = ${JSON.stringify(token)};`,
    'const prompt = String(process.argv[2] || \'\');',
    'if (/username/i.test(prompt)) process.stdout.write(\'x-access-token\\n\');',
    'else process.stdout.write(token + \'\\n\');',
    '',
  ].join('\n');
  fs.writeFileSync(file, source, { mode: 0o700, flag: 'wx' });
  return file;
}

function cleanupAskpass() {
  if (!activeAskpass) return;
  const file = activeAskpass;
  activeAskpass = null;
  try {
    fs.unlinkSync(file);
  } catch (err) {
    if (err.code !== 'ENOENT') log('warn', 'ASKPASS', err.message);
  }
}

process.on('exit', cleanupAskpass);

function repoRelativeVideo(cwd, videoPath) {
  const root = path.resolve(cwd, 'generated_videos');
  assertRealDirectory(root);
  const abs = path.resolve(videoPath);
  const relToRoot = path.relative(root, abs);
  if (!relToRoot || relToRoot.startsWith('..') || path.isAbsolute(relToRoot)) {
    throw new Error('vidéo hors de generated_videos');
  }
  if (path.basename(abs) !== relToRoot) throw new Error('sous-dossier vidéo refusé');
  return path.posix.join('generated_videos', path.basename(abs));
}

async function ensureOrigin(git, repoUrl) {
  const clean = assertAllowedRepoUrl(repoUrl);
  let remotes = '';
  try {
    remotes = await git.raw(['remote']);
  } catch (err) {
    remotes = '';
  }
  const names = String(remotes || '').split(/\s+/).filter(Boolean);
  if (names.includes('origin')) await git.raw(['remote', 'set-url', 'origin', clean]);
  else await git.raw(['remote', 'add', 'origin', clean]);
}

async function publishVideo(options) {
  const cwd = options.cwd || process.cwd();
  const repoUrl = assertAllowedRepoUrl(options.repoUrl || process.env.GITHUB_REPO_URL);
  const token = assertGithubToken(options.token || process.env.GITHUB_TOKEN);
  assertLocalVideo(options.videoPath);
  const rel = repoRelativeVideo(cwd, options.videoPath);
  const message = `feat: add new lofi capybara video: ${sanitizeCommitTitle(options.title)}`;
  const git = createGit(cwd, buildGitEnv(), options.binary);
  await ensureOrigin(git, repoUrl);
  await git.raw(['add', '--', rel]);
  await git.raw(['-c', 'commit.gpgsign=false', 'commit', '-m', message, '--', rel]);
  const askpass = writeAskpass(token);
  activeAskpass = askpass;
  try {
    git.env(buildGitEnv({ GIT_ASKPASS: askpass }));
    await git.raw(['-c', 'credential.helper=', 'push', 'origin', 'HEAD']);
  } finally {
    cleanupAskpass();
  }
}

function loadScenes(file) {
  const raw = JSON.parse(fs.readFileSync(file, 'utf8'));
  if (!Array.isArray(raw) || raw.length === 0) throw new Error('scenes.json doit être un tableau non vide');
  return raw.map((item, index) => {
    const description = typeof item === 'string' ? item : item && item.description;
    if (typeof description !== 'string' || description.trim().length < 4) {
      throw new Error(`scène ${index + 1} invalide`);
    }
    const mood = item && typeof item === 'object' ? item.mood : null;
    return {
      description: description.trim(),
      mood: mood && MOODS[mood] ? mood : null,
    };
  });
}

function readSettings() {
  const duration = parseInt(process.env.VIDEO_DURATION_SECONDS || '5', 10);
  if (![5, 6, 8, 10].includes(duration)) throw new Error('VIDEO_DURATION_SECONDS invalide');
  const format = process.env.VIDEO_FORMAT || '16:9';
  if (format !== '16:9' && format !== '9:16') throw new Error('VIDEO_FORMAT invalide');
  state.durationSeconds = duration;
  state.format = format;
}

async function generateScene(scene, queues, cwd) {
  const variation = pickVariation();
  const mood = scene.mood || pickFrom(MOOD_KEYS, deps.random);
  log('info', 'SCENE', `${scene.description} · ${MOODS[mood].label} · ${variation.timeOfDay} / ${variation.season} / ${variation.weather}`);
  const exp = await generateExpansion(scene.description, mood, variation);
  const imagePrompt = buildLofiImagePrompt(exp);
  const imageUrl = await queues.image.add(() => generateImage(imagePrompt, state.format));
  log('success', 'IMAGE', safeUrlForLog(imageUrl.startsWith('data:') ? 'https://local.invalid/inline' : imageUrl));
  const videoPrompt = buildLofiVideoPrompt(exp);
  createVideoIntervalMs = VIDEO_INTERVAL_SAFE;
  videoAttemptsSinceLastError = 0;
  const videoId = await queues.video.add(() => generateVideo(imageUrl, videoPrompt, state.format));
  log('success', 'VIDÉO', `Tâche ${videoId}`);
  const started = Date.now();
  const videoUrl = await pollVideo(videoId, (progress) => log('info', 'POLL', progress), cwd);
  recordGenerationTime(cwd, 'video', Date.now() - started);
  const dir = path.join(cwd, 'generated_videos');
  ensureOutputDir(dir);
  const dest = path.join(dir, videoFileName(exp.title));
  await downloadVideo(videoUrl, dest);
  return { title: exp.title, videoPath: dest, videoUrl };
}

async function runPipeline(scenesFile) {
  state.apiKey = assertApiKey(process.env.AGNES_API_KEY);
  assertGithubToken(process.env.GITHUB_TOKEN);
  assertAllowedRepoUrl(process.env.GITHUB_REPO_URL);
  readSettings();
  const cwd = process.cwd();
  const scenes = loadScenes(scenesFile || path.join(cwd, 'scenes.json'));
  ensureOutputDir(path.join(cwd, 'generated_videos'));
  const queues = {
    image: createRateLimitedQueue(1, () => createImageIntervalMs),
    video: createRateLimitedQueue(1, () => createVideoIntervalMs),
  };
  let failed = 0;
  for (const scene of scenes) {
    if (state.stopRequested) break;
    try {
      const result = await generateScene(scene, queues, cwd);
      await publishVideo({
        cwd,
        videoPath: result.videoPath,
        title: result.title,
        repoUrl: process.env.GITHUB_REPO_URL,
        token: process.env.GITHUB_TOKEN,
      });
      log('success', 'GIT', `publié : ${path.basename(result.videoPath)}`);
    } catch (err) {
      failed += 1;
      log('error', 'SCENE', err.message);
    }
  }
  if (failed > 0) process.exitCode = 1;
  log('info', 'FIN', `${scenes.length - failed} vidéo(s) publiée(s), ${failed} échec(s)`);
}

function parseArgs(argv) {
  const args = { selfTest: false, scenes: null };
  for (let i = 2; i < argv.length; i += 1) {
    const token = argv[i];
    if (token === '--self-test') args.selfTest = true;
    else if (token === '--scenes') {
      i += 1;
      args.scenes = argv[i];
      if (!args.scenes) throw new Error('--scenes attend un fichier');
    } else {
      throw new Error(`argument inconnu: ${token}`);
    }
  }
  return args;
}

function onStop() {
  state.stopRequested = true;
  log('warn', 'STOP', 'Arrêt demandé');
}

function ftypBuffer() {
  const buf = Buffer.alloc(32);
  buf.writeUInt32BE(24, 0);
  buf.write('ftypisom', 4);
  return buf;
}

function webmBuffer() {
  const buf = Buffer.alloc(32);
  buf[0] = 0x1a;
  buf[1] = 0x45;
  buf[2] = 0xdf;
  buf[3] = 0xa3;
  return buf;
}

function mockRes(status, body, headers = {}) {
  function create() {
    let consumed = false;
    return {
      status,
      ok: status >= 200 && status < 300,
      headers: {
        get(name) {
          const key = String(name || '').toLowerCase();
          for (const [header, value] of Object.entries(headers)) {
            if (header.toLowerCase() === key) return value;
          }
          return null;
        },
      },
      async text() {
        if (consumed) throw new Error('corps déjà lu');
        consumed = true;
        return body;
      },
      async json() {
        return JSON.parse(await this.text());
      },
      clone() {
        return create();
      },
      body: null,
    };
  }
  return create();
}

function streamRes(status, buf, headers = {}) {
  const res = mockRes(status, '', headers);
  res.ok = status >= 200 && status < 300;
  res.status = status;
  res.body = Readable.from([buf]);
  return res;
}

async function expectThrow(fn, pattern) {
  let thrown = null;
  try {
    await fn();
  } catch (err) {
    thrown = err;
  }
  if (!thrown) throw new Error('exception attendue');
  if (pattern && !pattern.test(thrown.message)) throw new Error(`message inattendu: ${thrown.message}`);
  return thrown;
}

async function runSelfTest() {
  state.silent = true;
  const failures = [];
  async function check(name, fn) {
    try {
      await fn();
      console.log(`OK ${name}`);
    } catch (err) {
      const message = redact(err && err.stack ? err.stack : err);
      failures.push(name);
      console.error(`FAIL ${name}\n${message}`);
    }
  }

  await check('backoff file 60/120/300', async () => {
    if (computeQueueWaitSeconds(0, () => 0) !== 60) throw new Error('60');
    if (computeQueueWaitSeconds(1, () => 0) !== 120) throw new Error('120');
    if (computeQueueWaitSeconds(2, () => 0) !== 300) throw new Error('300');
    if (computeQueueWaitSeconds(8, () => 0) !== 300) throw new Error('plafond palier');
    const jittered = computeQueueWaitSeconds(2, () => 0.5);
    if (jittered < 300 || jittered > VIDEO_QUEUE_MAX_WAIT_SEC) throw new Error(`jitter ${jittered}`);
  });

  await check('hôtes privés et IP obfusquées', async () => {
    const blocked = [
      'https://2130706433/a.mp4',
      'https://0x7f000001/a.mp4',
      'https://0177.0.0.1/a.mp4',
      'https://127.0.0.1/a.mp4',
      'https://127.1/a.mp4',
      'http://0/',
      'https://[::ffff:127.0.0.1]/a.mp4',
      'https://[::ffff:7f00:1]/a.mp4',
      'https://[::1]/a.mp4',
      'https://169.254.169.254/a.mp4',
      'https://[::ffff:169.254.169.254]/latest',
      'https://10.1.2.3/a.mp4',
      'https://192.168.0.2/a.mp4',
      'https://localhost/a.mp4',
      'https://metadata.google.internal/a',
    ];
    for (const raw of blocked) {
      let url;
      try {
        url = new URL(raw);
      } catch (err) {
        throw new Error(`parse ${raw}`);
      }
      if (url.protocol === 'https:' && !isBlockedHostname(url.hostname)) {
        throw new Error(`hôte accepté à tort: ${raw} -> ${url.hostname}`);
      }
      if (url.protocol === 'https:') {
        await expectThrow(() => Promise.resolve().then(() => assertSafeDownloadUrl(raw)), /hôte refusé/);
      }
    }
    if (isBlockedHostname('cdn.example.com')) throw new Error('domaine public bloqué');
    if (isBlockedHostname('8.8.8.8')) throw new Error('IPv4 publique bloquée');
    if (isBlockedHostname('2001:4860:4860::8888')) throw new Error('IPv6 publique bloquée');
    await expectThrow(() => Promise.resolve().then(() => assertSafeDownloadUrl('http://cdn.example.com/a.mp4')), /HTTPS/);
    await expectThrow(() => Promise.resolve().then(() => assertSafeDownloadUrl('https://user:secret@cdn.example.com/a.mp4')), /identifiants/);
  });

  await check('dépôt autorisé', async () => {
    if (assertAllowedRepoUrl('https://github.com/urdirditfurd/videowan2g') !== ALLOWED_REPO) throw new Error('canonique');
    if (assertAllowedRepoUrl('https://github.com/urdirditfurd/videowan2g.git') !== ALLOWED_REPO) throw new Error('suffixe git');
    const rejected = [
      'http://github.com/urdirditfurd/videowan2g',
      'https://user:token@github.com/urdirditfurd/videowan2g',
      'https://github.com/other/videowan2g',
      'https://github.com/urdirditfurd/videowan2g/extra',
      'git@github.com:urdirditfurd/videowan2g.git',
    ];
    for (const raw of rejected) {
      await expectThrow(() => Promise.resolve().then(() => assertAllowedRepoUrl(raw)));
    }
  });

  await check('nom de fichier', async () => {
    const name = videoFileName('../../etc/passwd\npluie', 1700000000000);
    if (name !== 'etc_passwd_pluie_1700000000000.mp4' && name !== 'ambiance_lofi_1700000000000.mp4') {
      if (name.includes('/') || name.includes('..')) throw new Error(name);
    }
    if (videoFileName('Maison japonaise', 5) !== 'maison_japonaise_5.mp4') {
      throw new Error(videoFileName('Maison japonaise', 5));
    }
  });

  await check('variation de prompt', async () => {
    const morning = { timeOfDay: 'pale blue dawn', season: 'quiet early winter', weather: 'gentle snowfall', seed: 11 };
    const noon = { timeOfDay: 'gentle noon light', season: 'humid midsummer', weather: 'clear soft sky', seed: 22 };
    const a = buildExpansionPrompt('maison japonaise sous la pluie', 'rainy_forest', morning);
    const b = buildExpansionPrompt('maison japonaise sous la pluie', 'rainy_forest', noon);
    if (a === b) throw new Error('prompts identiques');
    if (!a.includes('pale blue dawn') || !a.includes('capybara') || !a.includes('11')) throw new Error('variation absente');
    if (!b.includes('humid midsummer')) throw new Error('saison absente');
  });

  await check('extractJson', async () => {
    const fenced = extractJson('```json\n{"title":"Brume","image_prompt":"a capybara","video_motion_prompt":"slow"}\n```');
    if (!fenced || fenced.title !== 'Brume') throw new Error('fence');
    const broken = extractJson('{"title":"A","image_prompt":"line\nbreak","video_motion_prompt":"soft"');
    if (!broken || !broken.image_prompt) throw new Error('réparation');
  });

  await check('apiFetch clone le corps', async () => {
    resetRuntime();
    deps.fetch = async () => mockRes(400, 'détail-image-refusée');
    const res = await apiFetch('https://apihub.agnes-ai.com/v1/videos', { method: 'POST' }, 'Vidéo', 'video');
    if (await res.text() !== 'détail-image-refusée') throw new Error('corps consommé');
  });

  await check('erreur HTTP terminale hors retry', async () => {
    resetRuntime();
    let calls = 0;
    const sleeps = [];
    deps.fetch = async () => {
      calls += 1;
      return mockRes(401, 'clé refusée');
    };
    deps.sleep = async (ms) => { sleeps.push(ms); };
    await expectThrow(() => apiFetch('https://apihub.agnes-ai.com/v1/chat/completions', {}, 'Expansion', 'text'), /clé refusée/);
    if (calls !== 1) throw new Error(`appels ${calls}`);
    if (sleeps.length !== 0) throw new Error('retry inattendu');
  });

  await check('video_queue_full ne consomme pas le budget', async () => {
    resetRuntime();
    let calls = 0;
    const sleeps = [];
    deps.random = () => 0;
    deps.sleep = async (ms) => { sleeps.push(ms); };
    deps.fetch = async () => {
      calls += 1;
      if (calls <= 2) return mockRes(503, '{"error":"video_queue_full"}');
      return mockRes(200, '{"ok":true}');
    };
    const res = await apiFetch('https://apihub.agnes-ai.com/v1/videos', { method: 'POST' }, 'Vidéo', 'video');
    if (calls !== 3) throw new Error(`appels ${calls}`);
    if (sleeps[0] !== 60000 || sleeps[1] !== 120000) throw new Error(`waits ${sleeps.join(',')}`);
    if (await res.text() !== '{"ok":true}') throw new Error('corps succès');
  });

  await check('plafond de file', async () => {
    resetRuntime();
    let calls = 0;
    const sleeps = [];
    deps.random = () => 0;
    deps.sleep = async (ms) => { sleeps.push(ms); };
    deps.fetch = async () => {
      calls += 1;
      return mockRes(429, 'queue is full');
    };
    await expectThrow(
      () => apiFetch('https://apihub.agnes-ai.com/v1/videos', {}, 'Vidéo', 'video'),
      /tentatives de file épuisées/,
    );
    if (sleeps.length !== VIDEO_QUEUE_MAX_RETRIES) throw new Error(`sleeps ${sleeps.length}`);
    if (calls !== VIDEO_QUEUE_MAX_RETRIES + 1) throw new Error(`appels ${calls}`);
    if (sleeps[2] !== 300000) throw new Error('troisième palier');
  });

  await check('retryTask', async () => {
    resetRuntime();
    const sleeps = [];
    deps.sleep = async (ms) => { sleeps.push(ms); };
    let n = 0;
    const value = await retryTask('démo', async () => {
      n += 1;
      if (n === 1) throw new Error('temporaire');
      return 'ok';
    }, 3);
    if (value !== 'ok' || sleeps.length !== 1) throw new Error('reprise');
  });

  await check('file à intervalle', async () => {
    const starts = [];
    const queue = createRateLimitedQueue(1, 80);
    await Promise.all([
      queue.add(async () => { starts.push(Date.now()); }),
      queue.add(async () => { starts.push(Date.now()); }),
    ]);
    if (starts.length !== 2 || starts[1] - starts[0] < 50) throw new Error(`écart ${starts[1] - starts[0]}`);
  });

  await check('expansion et polling simulés', async () => {
    resetRuntime();
    state.apiKey = 'sk-test-key-123456';
    deps.sleep = async () => {};
    deps.random = () => 0;
    let step = 0;
    deps.fetch = async (url) => {
      step += 1;
      if (String(url).includes('/chat/completions')) {
        return mockRes(200, JSON.stringify({
          choices: [{ message: { content: '{"title":"Pluie douce","image_prompt":"a calm capybara","video_motion_prompt":"slow mist"}' } }],
        }));
      }
      if (String(url).includes('video_id=')) {
        if (step < 4) return mockRes(200, JSON.stringify({ status: 'processing', progress: 20 }));
        return mockRes(200, JSON.stringify({ status: 'completed', progress: 100, url: 'https://cdn.example.com/out.mp4' }));
      }
      throw new Error(`url inattendue ${url}`);
    };
    const exp = await generateExpansion('maison sous la pluie', 'rainy_forest', pickVariation(() => 0));
    if (exp.title !== 'Pluie douce' || !exp.image_prompt.includes('capybara')) throw new Error('expansion');
    const url = await pollVideo('vid123', () => {}, process.cwd());
    if (url !== 'https://cdn.example.com/out.mp4') throw new Error(url);
  });

  await check('téléchargement, redirections et lien symbolique', async () => {
    resetRuntime();
    state.apiKey = 'sk-test-key-123456';
    const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'lofi-dl-'));
    const seen = [];
    deps.lookup = async () => [{ address: '8.8.8.8', family: 4 }];
    deps.fetch = async (url, options) => {
      seen.push({ url, auth: options.headers && options.headers.Authorization ? 'yes' : 'no' });
      if (seen.length === 1) return mockRes(302, '', { location: 'https://cdn.example.com/a.mp4' });
      return streamRes(200, ftypBuffer(), { 'content-length': '32' });
    };
    const dest = path.join(dir, 'ok.mp4');
    await downloadVideo('https://cdn.agnes-ai.com/source.mp4', dest);
    assertLocalVideo(dest);
    if (seen[0].auth !== 'yes' || seen[1].auth !== 'no') throw new Error(`auth ${seen.map((item) => item.auth).join(',')}`);
    if (seen.length !== 2) throw new Error('redirections');

    deps.fetch = async () => mockRes(302, '', { location: 'https://127.0.0.1/secret.mp4' });
    await expectThrow(() => downloadVideo('https://cdn.example.com/start.mp4', path.join(dir, 'nope.mp4')), /hôte refusé/);

    deps.fetch = async () => mockRes(302, '', { location: 'https://2130706433/secret.mp4' });
    await expectThrow(() => downloadVideo('https://cdn.example.com/start.mp4', path.join(dir, 'nope2.mp4')), /hôte refusé/);

    let hops = 0;
    deps.fetch = async () => {
      hops += 1;
      return mockRes(302, '', { location: `https://cdn.example.com/hop${hops}` });
    };
    await expectThrow(() => downloadVideo('https://cdn.example.com/start.mp4', path.join(dir, 'hops.mp4')), /trop de redirections/);
    if (hops !== 3) throw new Error(`hops ${hops}`);

    deps.lookup = async () => [{ address: '169.254.169.254', family: 4 }];
    deps.fetch = async () => streamRes(200, ftypBuffer());
    await expectThrow(() => downloadVideo('https://evil.example/a.mp4', path.join(dir, 'dns.mp4')), /privée/);

    const secret = path.join(dir, 'secret.txt');
    fs.writeFileSync(secret, 'SECRET');
    const link = path.join(dir, 'linked.mp4');
    fs.symlinkSync(secret, link);
    await expectThrow(() => writeVideoStream(streamRes(200, ftypBuffer()), link), /lien symbolique/);
    if (fs.readFileSync(secret, 'utf8') !== 'SECRET') throw new Error('cible écrasée');

    const webm = path.join(dir, 'clip.webm');
    await writeVideoStream(streamRes(200, webmBuffer()), webm);
    assertLocalVideo(webm);

    const textFile = path.join(dir, 'notes.mp4');
    fs.writeFileSync(textFile, 'pas une video !!!!!');
    await expectThrow(() => Promise.resolve().then(() => assertLocalVideo(textFile)), /en-tête/);

    const big = path.join(dir, 'big.mp4');
    const fd = fs.openSync(big, 'w');
    fs.writeSync(fd, ftypBuffer());
    fs.ftruncateSync(fd, MAX_VIDEO_BYTES);
    fs.closeSync(fd);
    await expectThrow(() => Promise.resolve().then(() => assertLocalVideo(big)), /taille/);
    safeUnlink(big);
    fs.rmSync(dir, { recursive: true, force: true });
  });

  await check('git sans secrets', async () => {
    const sentinelAgnes = 'sentinel-agnes-7f3c9a';
    const sentinelGithub = 'sentinel-github-b81e04';
    const previousAgnes = process.env.AGNES_API_KEY;
    const previousGithub = process.env.GITHUB_TOKEN;
    process.env.AGNES_API_KEY = sentinelAgnes;
    process.env.GITHUB_TOKEN = sentinelGithub;
    const root = fs.mkdtempSync(path.join(os.tmpdir(), 'lofi-git-'));
    const outDir = path.join(root, 'generated_videos');
    fs.mkdirSync(outDir);
    const videoPath = path.join(outDir, 'pluie_1.mp4');
    fs.writeFileSync(videoPath, ftypBuffer());
    const logPath = path.join(root, 'fake-git-log.jsonl');
    const fakePath = path.join(root, 'fake-git.cjs');
    fs.writeFileSync(fakePath, `#!/usr/bin/env node
'use strict';
const fs = require('fs');
const env = {};
for (const [key, value] of Object.entries(process.env)) env[key] = value;
const rec = { argv: process.argv.slice(2), env, askpassExists: false };
if (process.env.GIT_ASKPASS) {
  try { rec.askpassExists = fs.existsSync(process.env.GIT_ASKPASS); } catch (err) { rec.askpassExists = false; }
}
fs.appendFileSync(${JSON.stringify(logPath)}, JSON.stringify(rec) + '\\n');
if (process.argv.includes('--version')) process.stdout.write('git version 2.43.0\\n');
process.exit(0);
`);
    fs.chmodSync(fakePath, 0o755);
    let askpassDuring = null;
    try {
      await publishVideo({
        cwd: root,
        videoPath,
        title: 'Pluie douce',
        repoUrl: 'https://github.com/urdirditfurd/videowan2g',
        token: sentinelGithub,
        binary: fakePath,
      });
      const lines = fs.readFileSync(logPath, 'utf8').trim().split('\n').map((line) => JSON.parse(line));
      if (lines.length < 4) throw new Error(`commandes ${lines.length}`);
      const joined = JSON.stringify(lines);
      if (joined.includes(sentinelAgnes) || joined.includes(sentinelGithub)) throw new Error('secret dans le processus git');
      if (joined.includes('AGNES_API_KEY') || joined.includes('GITHUB_TOKEN')) throw new Error('nom de secret dans le processus git');
      for (const line of lines) {
        if (line.env.HOME !== '') throw new Error('HOME non vide');
        if (line.env.GIT_CONFIG_NOSYSTEM !== '1') throw new Error('GIT_CONFIG_NOSYSTEM');
        if (line.env.GIT_TERMINAL_PROMPT !== '0') throw new Error('GIT_TERMINAL_PROMPT');
        if (!line.env.GIT_AUTHOR_NAME) throw new Error('auteur absent');
      }
      const commit = lines.find((line) => line.argv.includes('commit'));
      const push = lines.find((line) => line.argv.includes('push'));
      if (!commit || !commit.argv.includes('--') || !commit.argv.some((arg) => arg.includes('pluie_1.mp4'))) {
        throw new Error(`commit ${commit && commit.argv.join(' ')}`);
      }
      if (!commit.argv.some((arg) => String(arg).includes('commit.gpgsign=false'))) throw new Error('gpgsign');
      if (commit.env.GIT_ASKPASS) throw new Error('askpass pendant le commit');
      if (!push || !push.argv.some((arg) => String(arg).includes('credential.helper='))) throw new Error('credential.helper');
      if (!push.env.GIT_ASKPASS || !push.askpassExists) throw new Error('askpass absent au push');
      askpassDuring = push.env.GIT_ASKPASS;
      if (fs.existsSync(askpassDuring)) throw new Error('askpass non supprimé');
      const remoteArgs = lines.map((line) => line.argv.join(' ')).join('\n');
      if (!remoteArgs.includes('https://github.com/urdirditfurd/videowan2g.git')) throw new Error('remote');
      if (remoteArgs.includes('x-access-token')) throw new Error('jeton dans l\'URL');
    } finally {
      if (previousAgnes === undefined) delete process.env.AGNES_API_KEY;
      else process.env.AGNES_API_KEY = previousAgnes;
      if (previousGithub === undefined) delete process.env.GITHUB_TOKEN;
      else process.env.GITHUB_TOKEN = previousGithub;
      fs.rmSync(root, { recursive: true, force: true });
    }
  });

  await check('scenes.json', async () => {
    const scenes = loadScenes(path.join(__dirname, 'scenes.json'));
    if (scenes.length !== 5) throw new Error(`compte ${scenes.length}`);
    if (scenes.some((scene) => scene.description.trim().length < 4)) throw new Error('description');
  });

  deps.fetch = (url, options) => fetch(url, options);
  deps.sleep = (ms) => sleepInterruptible(ms);
  deps.random = () => Math.random();
  deps.lookup = (hostname) => dns.promises.lookup(hostname, { all: true, verbatim: true });
  state.apiKey = '';
  state.silent = false;
  resetRuntime();

  if (failures.length) {
    console.error(`SELF-TEST FAILED (${failures.length})`);
    process.exitCode = 1;
    return;
  }
  console.log('SELF-TEST PASSED');
}

async function main() {
  const args = parseArgs(process.argv);
  if (args.selfTest) {
    await runSelfTest();
    return;
  }
  await runPipeline(args.scenes);
}

if (require.main === module) {
  process.on('SIGINT', onStop);
  process.on('SIGTERM', onStop);
  main().catch((err) => {
    log('error', 'FATAL', err && err.message ? err.message : String(err));
    process.exitCode = 1;
  });
}

module.exports = {
  apiFetch,
  assertAllowedRepoUrl,
  assertLocalVideo,
  buildExpansionPrompt,
  computeQueueWaitSeconds,
  createRateLimitedQueue,
  extractJson,
  isBlockedHostname,
  loadScenes,
  publishVideo,
  sanitizeTitle,
};
