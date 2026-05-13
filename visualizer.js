import * as THREE from 'https://cdn.jsdelivr.net/npm/three@0.160.0/build/three.module.js';
import Meyda from 'https://cdn.jsdelivr.net/npm/meyda@5.6.3/+esm';
import { presets, DEFAULT_PRESET } from './presets.js';

const preset = presets[DEFAULT_PRESET];

// ---------- Three.js scene ----------
const canvas = document.getElementById('bg');
const renderer = new THREE.WebGLRenderer({ canvas, antialias: false });
renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
renderer.setSize(window.innerWidth, window.innerHeight);

const scene = new THREE.Scene();
const camera = new THREE.OrthographicCamera(-1, 1, 1, -1, 0, 1);

const PATH_LEN = 16;
const PATH_MIN_PX = 7;      // smaller = denser polyline, smoother joints
const PATH_TTL = 1.6;       // seconds before a sample fully fades

const uniforms = {
  iTime:       { value: 0 },
  iResolution:  { value: new THREE.Vector2(window.innerWidth, window.innerHeight) },
  iPixelRatio:  { value: Math.min(window.devicePixelRatio, 2) },
  iMouse:       { value: new THREE.Vector2(window.innerWidth / 2, window.innerHeight / 2) },
  iMousePath:    { value: Array.from({ length: PATH_LEN }, () => new THREE.Vector2(0, 0)) },
  iMousePathAge: { value: new Array(PATH_LEN).fill(PATH_TTL + 1) },
  iPathTtl:      { value: PATH_TTL },
  iSubFloor:    { value: preset.iSubFloor    },
  iSubStrength: { value: preset.iSubStrength },
  iSubExp:      { value: preset.iSubExp      },
  iSubBass:    { value: 0 }, // 0-130Hz, heavily smoothed — drives slow large-scale motion
  iBass:       { value: 0 }, // 130-345Hz — drives subtle pulse
  iMid:        { value: 0 },
  iTreble:     { value: 0 },
  iHarmonic:    { value: new THREE.Vector2(0, 0) },
  iTransient:   { value: 0 },
  iHarmRotation:      { value: preset.iHarmRotation      },
  iHarmSensLow:       { value: preset.iHarmSensLow       },
  iHarmSensHigh:      { value: preset.iHarmSensHigh      },
  iHarmSat:           { value: preset.iHarmSat           },
  iTransientStrength: { value: preset.iTransientStrength },
};

const vertexShader = `
  varying vec2 vUv;
  void main() {
    vUv = uv;
    gl_Position = vec4(position, 1.0);
  }
`;

const fragmentShader = `
  precision highp float;
  #define PATH_LEN 16
  varying vec2 vUv;
  uniform float iTime;
  uniform vec2 iResolution;
  uniform float iPixelRatio;
  uniform vec2 iMouse;
  uniform vec2 iMousePath[PATH_LEN];
  uniform float iMousePathAge[PATH_LEN];
  uniform float iPathTtl;
  uniform float iSubFloor;
  uniform float iSubStrength;
  uniform float iSubExp;
  uniform float iSubBass;
  uniform float iBass;
  uniform float iMid;
  uniform float iTreble;
  uniform vec2 iHarmonic;
  uniform float iTransient;
  uniform float iTransientStrength;
  uniform float iHarmRotation;
  uniform float iHarmSensLow;
  uniform float iHarmSensHigh;
  uniform float iHarmSat;

  float hash(vec2 p) {
    return fract(sin(dot(p, vec2(127.1, 311.7))) * 43758.5453123);
  }

  float noise(vec2 p) {
    vec2 i = floor(p);
    vec2 f = fract(p);
    vec2 u = f * f * (3.0 - 2.0 * f);
    float a = hash(i);
    float b = hash(i + vec2(1.0, 0.0));
    float c = hash(i + vec2(0.0, 1.0));
    float d = hash(i + vec2(1.0, 1.0));
    return mix(mix(a, b, u.x), mix(c, d, u.x), u.y);
  }

  float fbm(vec2 p) {
    float v = 0.0;
    float a = 0.5;
    for (int i = 0; i < 5; i++) {
      v += a * noise(p);
      p *= 2.0;
      a *= 0.5;
    }
    return v;
  }

  // Rotate hue around the luminance axis (1,1,1) by 'angle' radians, preserving
  // perceived brightness. Standard Rodrigues rotation of RGB around the grey axis.
  vec3 hueShift(vec3 col, float angle) {
    const vec3 k = vec3(0.57735); // 1/sqrt(3)
    float c = cos(angle);
    float s = sin(angle);
    return col * c + cross(k, col) * s + k * dot(k, col) * (1.0 - c);
  }

  // Convert CSS-pixel coord to physical-normalized centered space.
  // gl_FragCoord is physical pixels; iResolution is CSS pixels; iPixelRatio bridges them.
  vec2 pxToP(vec2 pxCSS) {
    vec2 r = pxCSS / iResolution - 0.5;
    r.y = -r.y;
    r.x *= iResolution.x / iResolution.y;
    return r;
  }

  float sdSegment(vec2 p, vec2 a, vec2 b) {
    vec2 pa = p - a;
    vec2 ba = b - a;
    float h = clamp(dot(pa, ba) / max(dot(ba, ba), 1e-6), 0.0, 1.0);
    return length(pa - ba * h);
  }

  // Returns (intensity, dirX, dirY).
  //   intensity = max-pooled exp falloff to nearest segment (the visible trail)
  //   dir       = smoothly-blended unit-ish vector pointing away from the path,
  //               averaged across all segments by an exp weight. This avoids the
  //               direction "snapping" at segment vertices that produced visible
  //               kinks in the carved channel.
  vec3 trailField(vec2 p) {
    float intensity = 0.0;
    vec2 wDir = vec2(0.0);
    float wSum = 0.0;
    for (int i = 0; i < PATH_LEN - 1; i++) {
      vec2 a = pxToP(iMousePath[i]);
      vec2 b = pxToP(iMousePath[i + 1]);
      float age = max(iMousePathAge[i], iMousePathAge[i + 1]);
      // Smoothstep fade for a gentler tail
      float fade = 1.0 - smoothstep(0.0, iPathTtl, age);

      vec2 pa = p - a;
      vec2 ba = b - a;
      float h = clamp(dot(pa, ba) / max(dot(ba, ba), 1e-6), 0.0, 1.0);
      vec2 cp = a + ba * h;
      vec2 toP = p - cp;
      float d = length(toP);

      // Visible trail — wide, soft falloff for a "smoke parting" feel
      float k = mix(17.0, 11.0, fade);
      intensity = max(intensity, exp(-d * k) * fade);

      // Direction blend uses an even wider kernel — keeps the displacement
      // direction smooth and gradual across the whole influence area.
      float dirW = exp(-d * 5.0) * fade;
      vec2 unit = toP / max(d, 1e-4);
      wDir += unit * dirW;
      wSum += dirW;
    }
    vec2 dir = wSum > 0.0 ? wDir / wSum : vec2(0.0);
    return vec3(intensity, dir);
  }

  void main() {
    // p: physical-normalized [0..1] UV, correct for cursor/trail matching.
    // pScene: scaled by pixelRatio to preserve noise visual density regardless of DPI.
    vec2 uv = gl_FragCoord.xy / (iResolution.xy * iPixelRatio);
    vec2 p = uv - 0.5;
    p.x *= iResolution.x / iResolution.y;
    vec2 pScene = p * iPixelRatio;
    vec2 m = pxToP(iMouse);

    // --- Trail field, computed early so we can deform the noise itself ---
    vec2 wob = vec2(
      fbm(pScene * 3.5 + iTime * 0.25),
      fbm(pScene * 3.5 + vec2(7.3, 2.9) + iTime * 0.25)
    ) - 0.5;
    vec2 pTrail = p + wob * 0.030;

    vec3 tf = trailField(pTrail);
    float trail = tf.x;
    vec2 awayDir = tf.yz; // already smoothly averaged across segments

    // Wider but gentle push — takes advantage of the broader influence area
    float pushAmt = smoothstep(0.02, 0.55, trail) * 0.20;
    // Sub-bass adds a radial "breath" to the noise field. Curve aligned to
    // musical structure: ambient intros (pad reverb tails in bins 0-2) read
    // ~0.10-0.22 and produce ZERO effect. The bassline/kick proper sits
    // 0.35+; a steep exponent compresses near-floor noise and keeps growing
    // through the loudest drops with no upper cap.
    float bassDrive = pow(max(0.0, iSubBass - iSubFloor), iSubExp);
    float bassBreath = 1.0 + bassDrive * iSubStrength;
    // pNoise: cursor displacement applied in cursor-space (p), then scaled to scene-space
    vec2 pNoise = (pScene + awayDir * pushAmt * iPixelRatio) * bassBreath;

    // --- Ambient flow, sampled at the displaced coord ---
    float t = iTime * 0.05;

    vec2 q = vec2(
      fbm(pNoise * 1.4 + t * 0.3),
      fbm(pNoise * 1.4 + vec2(5.2, 1.3) + t * 0.25)
    );
    vec2 warp = 2.5 * q;
    vec2 r = vec2(
      fbm(pNoise + warp + vec2(1.7, 9.2) + t * 0.4),
      fbm(pNoise + warp + vec2(8.3, 2.8) + t * 0.35)
    );
    float n = fbm(pNoise + 3.0 * r);

    // --- Harmonic-driven hue rotation around the blue/green baseline ---
    // Angle of the chroma circular mean → significant hue rotation across the
    // full color wheel. Magnitude controls how strongly we deviate from baseline.
    float harmAngle = atan(iHarmonic.y, iHarmonic.x);
    float harmStrength = smoothstep(iHarmSensLow, iHarmSensHigh, length(iHarmonic));
    float hueOffset = harmAngle * iHarmRotation * harmStrength;

    // Palette: near-black → deep blue → teal → soft green, hue-shifted by harmonics
    vec3 cBlack = vec3(0.0, 0.015, 0.03);
    vec3 cBlue  = hueShift(vec3(0.02, 0.12, 0.28), hueOffset);
    vec3 cTeal  = hueShift(vec3(0.05, 0.40, 0.50), hueOffset);
    vec3 cGreen = hueShift(vec3(0.18, 0.65, 0.45), hueOffset);

    vec3 col = mix(cBlack, cBlue,  smoothstep(0.25, 0.55, n));
    col      = mix(col,    cTeal,  smoothstep(0.55, 0.78, n) * (0.45 + iMid * 0.60));
    col      = mix(col,    cGreen, smoothstep(0.78, 0.95, n) * (0.25 + iTreble * 0.85));

    float harmLum = dot(col, vec3(0.299, 0.587, 0.114));
    col = mix(vec3(harmLum), col, 1.0 + harmStrength * iHarmSat);

    // --- The "cut" — wide soft darkening, never fully obscures the smoke ---
    float cutCenter = smoothstep(0.02, 0.85, trail);
    col = mix(col, cBlack, cutCenter * 0.28);

    // Broader, softer rim highlight — gentle gradient rather than a defined line.
    float edge = smoothstep(0.15, 0.65, trail) - smoothstep(0.65, 0.95, trail);
    col += vec3(0.15, 0.48, 0.42) * edge * 0.22;

    // Faint pinpoint at the live cursor so a stationary pointer remains visible.
    float dToMouse = length(p - m);
    col += vec3(0.35, 0.65, 0.58) * exp(-dToMouse * 95.0) * 0.04;

    // Base brightness + treble shimmer (always on)
    col *= 0.86 + iTreble * 0.06;
    float shimmer = (hash(gl_FragCoord.xy + iTime * 60.0) - 0.5) * iTreble * 0.06;
    col += shimmer;

    // Hi-hat / transient flash — gated by iTransientStrength (0 = off)
    col += vec3(0.20, 0.55, 0.50) * iTransient * iTransientStrength * 0.10;
    float sparkle = (hash(gl_FragCoord.xy + iTime * 120.0) - 0.5) * iTransient * iTransientStrength * 0.18;
    col += sparkle;
    col *= 1.0 + iTransient * iTransientStrength * 0.12;

    // Vignette
    col *= 1.0 - 0.35 * dot(uv - 0.5, uv - 0.5);

    gl_FragColor = vec4(col, 1.0);
  }
`;

const material = new THREE.ShaderMaterial({ uniforms, vertexShader, fragmentShader });
const quad = new THREE.Mesh(new THREE.PlaneGeometry(2, 2), material);
scene.add(quad);

window.addEventListener('resize', () => {
  renderer.setSize(window.innerWidth, window.innerHeight);
  uniforms.iResolution.value.set(window.innerWidth, window.innerHeight);
  uniforms.iPixelRatio.value = renderer.getPixelRatio();
});

// ---------- Mouse tracking + path history ----------
const mouse = { x: window.innerWidth / 2, y: window.innerHeight / 2 };
const smoothMouse = { x: mouse.x, y: mouse.y };

const path = []; // [{ x, y, time }]  newest at end
let lastPathSample = { x: -1e9, y: -1e9 };

function maybeRecordPath(x, y) {
  const dx = x - lastPathSample.x;
  const dy = y - lastPathSample.y;
  if (dx * dx + dy * dy < PATH_MIN_PX * PATH_MIN_PX) return;
  lastPathSample = { x, y };
  path.push({ x, y, time: performance.now() / 1000 });
  if (path.length > PATH_LEN) path.shift();
}

function setPointer(x, y) {
  mouse.x = x;
  mouse.y = y;
  maybeRecordPath(x, y);
}

window.addEventListener('mousemove', e => setPointer(e.clientX, e.clientY));
window.addEventListener('touchmove', e => {
  const t = e.touches[0];
  setPointer(t.clientX, t.clientY);
}, { passive: true });

function updatePathUniforms() {
  const now = performance.now() / 1000;
  // Drop entries past their TTL
  while (path.length > 0 && now - path[0].time > PATH_TTL) path.shift();

  const pr = renderer.getPixelRatio();
  for (let i = 0; i < PATH_LEN; i++) {
    if (i < path.length) {
      uniforms.iMousePath.value[i].set(path[i].x, path[i].y);
      uniforms.iMousePathAge.value[i] = now - path[i].time;
    } else {
      // Stale slot — position doesn't matter since age >= TTL kills the contribution
      uniforms.iMousePathAge.value[i] = PATH_TTL + 1;
    }
  }
}

// ---------- Audio + FFT analysis ----------
const audio = document.getElementById('audio');
const npBtn = document.getElementById('npBtn');
const npIcon = document.getElementById('npIcon');
const nowplaying = document.getElementById('nowplaying');
const startOverlay = document.getElementById('startOverlay');

let audioCtx, analyser, dataArray, prevDataArray, meydaAnalyzer;
let latestFeatures = null;
let subBassSmoothed = 0, bassSmoothed = 0, midSmoothed = 0, trebleSmoothed = 0;
let harmonicX = 0, harmonicY = 0;
// Smoothing setting 0..1 where 1 = maximum lag. Exponential map gives a
// much wider range at the slow end: 0→0.5 (fast), 0.5→0.022, 1.0→0.001 (~10s TC).
let harmSmoothingSetting = preset.harmSmoothingSetting;
const harmAlpha = () => 0.5 * Math.pow(0.002, harmSmoothingSetting);
let transientLevel = 0;
let fluxBaseline = 0; // adaptive baseline of spectral flux

// FFT bin layout (44.1kHz sample rate, fftSize 1024 → ~43Hz per bin)
const SUB_BASS_START = 0,  SUB_BASS_END = 3;    // ~0-130Hz    (deep sub + kick)
const BASS_START     = 3,  BASS_END     = 8;    // ~130-345Hz  (body)
const MID_START      = 8,  MID_END      = 80;   // ~345-3.4kHz
const TREB_START     = 80, TREB_END     = 250;  // ~3.4-11kHz
const HI_FLUX_START  = 100, HI_FLUX_END = 220;  // ~4.3-9.5kHz (hats/perc)

function initAudio() {
  if (audioCtx) return;
  audioCtx = new (window.AudioContext || window.webkitAudioContext)();
  const source = audioCtx.createMediaElementSource(audio);
  analyser = audioCtx.createAnalyser();
  analyser.fftSize = 1024;
  analyser.smoothingTimeConstant = 0.85;
  source.connect(analyser);
  analyser.connect(audioCtx.destination);
  dataArray = new Uint8Array(analyser.frequencyBinCount);
  prevDataArray = new Uint8Array(analyser.frequencyBinCount);

  // Meyda: start AFTER context is running, not before.
  // We defer to the next microtask so resume() has already been called.
  meydaAnalyzer = Meyda.createMeydaAnalyzer({
    audioContext: audioCtx,
    source,
    bufferSize: 1024,
    featureExtractors: ['chroma'],
    callback: features => {
      if (!latestFeatures && DEV) console.log('[Meyda] first callback, chroma:', features?.chroma);
      latestFeatures = features;
    },
  });
}

function avgBins(start, end) {
  let sum = 0;
  for (let i = start; i < end; i++) sum += dataArray[i];
  return sum / ((end - start) * 255);
}

// Twelve unit vectors on the pitch-class circle, precomputed
const PC_COS = new Float32Array(12);
const PC_SIN = new Float32Array(12);
for (let i = 0; i < 12; i++) {
  const a = (i * Math.PI * 2) / 12;
  PC_COS[i] = Math.cos(a);
  PC_SIN[i] = Math.sin(a);
}

function readBands() {
  if (!analyser) return;
  analyser.getByteFrequencyData(dataArray);

  const subBass = avgBins(SUB_BASS_START, SUB_BASS_END);
  const bass    = avgBins(BASS_START,     BASS_END);
  const mid     = avgBins(MID_START,      MID_END);
  const treble  = avgBins(TREB_START,     TREB_END);

  const easeAsym = (cur, target, atk, rel) =>
    target > cur ? cur + (target - cur) * atk : cur + (target - cur) * rel;

  // Sub-bass envelope: slightly faster attack so sustained sub-bass climbs
  // promptly, slow release so it decays as a swell instead of cutting off.
  subBassSmoothed = easeAsym(subBassSmoothed, subBass, 0.09, 0.03);
  bassSmoothed    = easeAsym(bassSmoothed,    bass,    0.45, 0.10);
  midSmoothed     = easeAsym(midSmoothed,     mid,     0.50, 0.10);
  trebleSmoothed  = easeAsym(trebleSmoothed,  treble,  0.55, 0.10);

  // --- Meyda features → harmonic + transient signals ---
  if (latestFeatures) {
    // Chroma → circular mean = tonal direction in the pitch-class circle.
    const chroma = latestFeatures.chroma;
    if (chroma && chroma.length === 12) {
      let cx = 0, cy = 0, total = 0;
      for (let i = 0; i < 12; i++) {
        const e = chroma[i];
        cx += e * PC_COS[i];
        cy += e * PC_SIN[i];
        total += e;
      }
      if (total > 0) { cx /= total; cy /= total; }
      const a = harmAlpha();
      harmonicX += (cx - harmonicX) * a;
      harmonicY += (cy - harmonicY) * a;
    }

  }

  // Spectral flux from our own analyser — high band only (hats/perc)
  let flux = 0;
  for (let i = HI_FLUX_START; i < HI_FLUX_END; i++) {
    const diff = dataArray[i] - prevDataArray[i];
    if (diff > 0) flux += diff;
  }
  const fluxNorm = flux / ((HI_FLUX_END - HI_FLUX_START) * 255);
  fluxBaseline += (fluxNorm - fluxBaseline) * 0.04;
  const onset = Math.max(0, fluxNorm - fluxBaseline * 1.6);
  transientLevel = Math.max(transientLevel * 0.78, Math.min(1.0, onset * 3.0));
  prevDataArray.set(dataArray);
}

// ---------- Player controls ----------
const ICON_PLAY  = '<path d="M8 5v14l11-7z"/>';
const ICON_PAUSE = '<path d="M6 5h4v14H6zm8 0h4v14h-4z"/>';

function updatePlayState() {
  const playing = !audio.paused;
  nowplaying.classList.toggle('playing', playing);
  npIcon.innerHTML = playing ? ICON_PAUSE : ICON_PLAY;
}

async function startPlayback() {
  initAudio();
  try {
    if (audioCtx.state === 'suspended') await audioCtx.resume();
    await audio.play();
    // Start Meyda only after context is definitely running
    meydaAnalyzer.start();
    if (DEV) console.log('[Meyda] started, context state:', audioCtx.state);
  } catch (err) {
    console.error('Playback failed:', err);
  }
}

audio.addEventListener('play', updatePlayState);
audio.addEventListener('pause', updatePlayState);

npBtn.addEventListener('click', async () => {
  initAudio();
  if (audioCtx.state === 'suspended') {
    await audioCtx.resume();
    meydaAnalyzer.start();
  }
  if (audio.paused) await audio.play(); else audio.pause();
});

startOverlay.addEventListener('click', async () => {
  await startPlayback();
  startOverlay.classList.add('hidden');
});

// ---------- Debug panel (localhost only) ----------
const DEV = location.hostname === 'localhost' || location.hostname === '127.0.0.1';
const harmPanel = document.getElementById('debug-harm');
if (!DEV) harmPanel.style.display = 'none';

if (DEV) {
  const LS_KEY = 'ww:debug:harm';

  function loadDebugSettings() {
    try { return JSON.parse(localStorage.getItem(LS_KEY)) || {}; } catch { return {}; }
  }
  function saveDebugSettings(patch) {
    try {
      localStorage.setItem(LS_KEY, JSON.stringify({ ...loadDebugSettings(), ...patch }));
    } catch { /* quota or private browsing */ }
  }

  const saved = loadDebugSettings();

  const harmSliders = [
    ['s-hrot', 'v-hrot', 'iHarmRotation',      v => v.toFixed(2)],
    ['s-hslo', 'v-hslo', 'iHarmSensLow',        v => v.toFixed(3)],
    ['s-hshi', 'v-hshi', 'iHarmSensHigh',       v => v.toFixed(3)],
    ['s-hsat', 'v-hsat', 'iHarmSat',            v => v.toFixed(2)],
    ['s-tstr', 'v-tstr', 'iTransientStrength',  v => v.toFixed(2)],
  ];

  // Lag slider refs needed by reset button
  const smoEl  = document.getElementById('s-hsmo');
  const smoLbl = document.getElementById('v-hsmo');

  function applyPreset() {
    for (const [sliderId, labelId, uniform, fmt] of harmSliders) {
      const el  = document.getElementById(sliderId);
      const lbl = document.getElementById(labelId);
      const v   = preset[uniform] ?? 0;
      el.value  = v;
      uniforms[uniform].value = v;
      lbl.textContent = fmt(v);
    }
    harmSmoothingSetting = preset.harmSmoothingSetting;
    smoEl.value  = harmSmoothingSetting;
    smoLbl.textContent = harmSmoothingSetting.toFixed(2);
  }

  function applySliderValue(sliderId, labelId, uniform, fmt, v) {
    const el  = document.getElementById(sliderId);
    const lbl = document.getElementById(labelId);
    el.value  = v;
    uniforms[uniform].value = v;
    lbl.textContent = fmt(v);
  }

  // Init: preset as base, localStorage overrides on top
  applyPreset();
  for (const [sliderId, labelId, uniform, fmt] of harmSliders) {
    if (sliderId in saved) applySliderValue(sliderId, labelId, uniform, fmt, saved[sliderId]);
  }
  if ('s-hsmo' in saved) {
    harmSmoothingSetting = saved['s-hsmo'];
    smoEl.value  = harmSmoothingSetting;
    smoLbl.textContent = harmSmoothingSetting.toFixed(2);
  }

  // Live changes
  for (const [sliderId, labelId, uniform, fmt] of harmSliders) {
    document.getElementById(sliderId).addEventListener('input', () => {
      const v = parseFloat(document.getElementById(sliderId).value);
      uniforms[uniform].value = v;
      document.getElementById(labelId).textContent = fmt(v);
      saveDebugSettings({ [sliderId]: v });
    });
  }
  smoEl.addEventListener('input', () => {
    harmSmoothingSetting = parseFloat(smoEl.value);
    smoLbl.textContent   = harmSmoothingSetting.toFixed(2);
    saveDebugSettings({ 's-hsmo': harmSmoothingSetting });
  });

  // Reset: wipe cache and re-apply preset immediately, no reload needed
  document.getElementById('btn-preset').addEventListener('click', () => {
    localStorage.removeItem(LS_KEY);
    applyPreset();
  });
}

// ---------- Harmonic display (throttled) ----------
let lastHarmDisplay = 0;
function updateHarmDisplay() {
  const now = performance.now();
  if (now - lastHarmDisplay < 250) return;
  lastHarmDisplay = now;
  const mag = Math.hypot(harmonicX, harmonicY);
  document.getElementById('v-hmag').textContent = mag.toFixed(3);
  document.getElementById('v-hang').textContent = Math.atan2(harmonicY, harmonicX).toFixed(2);
  document.getElementById('v-tlvl').textContent = transientLevel.toFixed(3);
}

// ---------- Render loop ----------
const clock = new THREE.Clock();
function tick() {
  uniforms.iTime.value += clock.getDelta();

  // Smooth cursor — pass CSS coords; shader handles pixel ratio internally
  smoothMouse.x += (mouse.x - smoothMouse.x) * 0.08;
  smoothMouse.y += (mouse.y - smoothMouse.y) * 0.08;
  uniforms.iMouse.value.set(smoothMouse.x, smoothMouse.y);

  updatePathUniforms();

  readBands();
  uniforms.iSubBass.value   = subBassSmoothed;
  uniforms.iBass.value      = bassSmoothed;
  uniforms.iMid.value       = midSmoothed;
  uniforms.iTreble.value    = trebleSmoothed;
  uniforms.iHarmonic.value.set(harmonicX, harmonicY);
  if (DEV) updateHarmDisplay();
  uniforms.iTransient.value = transientLevel;

  renderer.render(scene, camera);
  requestAnimationFrame(tick);
}
tick();
