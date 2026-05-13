import * as THREE from 'https://cdn.jsdelivr.net/npm/three@0.160.0/build/three.module.js';
import Meyda from 'https://cdn.jsdelivr.net/npm/meyda@5.6.3/+esm';

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
  iResolution: { value: new THREE.Vector2(window.innerWidth, window.innerHeight) },
  iMouse:      { value: new THREE.Vector2(window.innerWidth / 2, window.innerHeight / 2) },
  iMousePath:    { value: Array.from({ length: PATH_LEN }, () => new THREE.Vector2(0, 0)) },
  iMousePathAge: { value: new Array(PATH_LEN).fill(PATH_TTL + 1) },
  iPathTtl:      { value: PATH_TTL },
  iSubFloor:    { value: 0.60 },
  iSubStrength: { value: 0.15 },
  iSubExp:      { value: 2.5  },
  iSubBass:    { value: 0 }, // 0-130Hz, heavily smoothed — drives slow large-scale motion
  iBass:       { value: 0 }, // 130-345Hz — drives subtle pulse
  iMid:        { value: 0 },
  iTreble:     { value: 0 },
  iHarmonic:   { value: new THREE.Vector2(0, 0) }, // tonal center (smoothed circular mean of chroma)
  iTransient:  { value: 0 },                       // 0..1 impulse on hi-band onsets
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

  // Convert pixel-space coord to the same aspect-corrected, centered space
  // we use for the noise field (so the trail lines up visually).
  vec2 pxToP(vec2 px) {
    vec2 r = px / iResolution - 0.5;
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
    vec2 uv = gl_FragCoord.xy / iResolution.xy;
    vec2 p = uv - 0.5;
    p.x *= iResolution.x / iResolution.y;
    vec2 m = pxToP(iMouse);

    // --- Trail field, computed early so we can deform the noise itself ---
    // Very slow, large-scale fractal wobble — adds an organic edge without
    // introducing high-frequency jitter.
    vec2 wob = vec2(
      fbm(p * 3.5 + iTime * 0.25),
      fbm(p * 3.5 + vec2(7.3, 2.9) + iTime * 0.25)
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
    vec2 pNoise = (p + awayDir * pushAmt) * bassBreath;

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
    // Aggressive curve: typical dub-techno chroma magnitudes are ~0.04-0.15,
    // so we want to saturate the response within that range.
    float harmStrength = smoothstep(0.005, 0.10, length(iHarmonic));
    // Up to ~140° rotation — full palette excursion through purple/amber/teal
    float hueOffset = harmAngle * 2.4 * harmStrength;

    // Palette: near-black → deep blue → teal → soft green, hue-shifted by harmonics
    vec3 cBlack = vec3(0.0, 0.015, 0.03);
    vec3 cBlue  = hueShift(vec3(0.02, 0.12, 0.28), hueOffset);
    vec3 cTeal  = hueShift(vec3(0.05, 0.40, 0.50), hueOffset);
    vec3 cGreen = hueShift(vec3(0.18, 0.65, 0.45), hueOffset);

    vec3 col = mix(cBlack, cBlue,  smoothstep(0.25, 0.55, n));
    col      = mix(col,    cTeal,  smoothstep(0.55, 0.78, n) * (0.45 + iMid * 0.60));
    col      = mix(col,    cGreen, smoothstep(0.78, 0.95, n) * (0.25 + iTreble * 0.85));

    // Saturation lift on strongly tonal content — pads feel more vivid, washes stay muted
    float harmLum = dot(col, vec3(0.299, 0.587, 0.114));
    col = mix(vec3(harmLum), col, 1.0 + harmStrength * 0.45);

    // --- The "cut" — wide soft darkening, never fully obscures the smoke ---
    float cutCenter = smoothstep(0.02, 0.85, trail);
    col = mix(col, cBlack, cutCenter * 0.28);

    // Broader, softer rim highlight — gentle gradient rather than a defined line.
    float edge = smoothstep(0.15, 0.65, trail) - smoothstep(0.65, 0.95, trail);
    col += vec3(0.15, 0.48, 0.42) * edge * 0.22;

    // Faint pinpoint at the live cursor so a stationary pointer remains visible.
    float dToMouse = length(p - m);
    col += vec3(0.35, 0.65, 0.58) * exp(-dToMouse * 95.0) * 0.04;

    // Percussion-driven brightness — subtle bloom on hat/transient hits
    col *= 0.86 + iTransient * 0.12 + iTreble * 0.06;

    // Treble shimmer
    float shimmer = (hash(gl_FragCoord.xy + iTime * 60.0) - 0.5) * iTreble * 0.06;
    col += shimmer;

    // Transient pop — sharp brief flash + sparkle on hat/perc onsets
    col += vec3(0.20, 0.55, 0.50) * iTransient * 0.10;
    float sparkle = (hash(gl_FragCoord.xy + iTime * 120.0) - 0.5) * iTransient * 0.18;
    col += sparkle;

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
      uniforms.iMousePath.value[i].set(path[i].x * pr, path[i].y * pr);
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

let audioCtx, analyser, dataArray, meydaAnalyzer;
let latestFeatures = null;
let subBassSmoothed = 0, bassSmoothed = 0, midSmoothed = 0, trebleSmoothed = 0;
let harmonicX = 0, harmonicY = 0;
let transientLevel = 0;
let fluxBaseline = 0; // adaptive baseline of spectral flux

// FFT bin layout (44.1kHz sample rate, fftSize 1024 → ~43Hz per bin)
const SUB_BASS_START = 0,  SUB_BASS_END = 3;    // ~0-130Hz    (deep sub + kick)
const BASS_START     = 3,  BASS_END     = 8;    // ~130-345Hz  (body)
const MID_START      = 8,  MID_END      = 80;   // ~345-3.4kHz
const TREB_START     = 80, TREB_END     = 250;  // ~3.4-11kHz

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

  // Meyda: chroma (pitch class energies) + spectral flux (transients).
  // Callback fires every bufferSize samples (~43Hz at 44.1k/1024).
  meydaAnalyzer = Meyda.createMeydaAnalyzer({
    audioContext: audioCtx,
    source,
    bufferSize: 1024,
    featureExtractors: ['chroma', 'spectralFlux'],
    callback: features => { latestFeatures = features; },
  });
  meydaAnalyzer.start();
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
      // Moderate smoothing — slow enough to read as chord-paced motion,
      // fast enough that you can see it shift within a phrase.
      harmonicX += (cx - harmonicX) * 0.10;
      harmonicY += (cy - harmonicY) * 0.10;
    }

    // Spectral flux → onset detection via adaptive baseline
    const flux = latestFeatures.spectralFlux;
    if (typeof flux === 'number' && isFinite(flux)) {
      fluxBaseline += (flux - fluxBaseline) * 0.04;
      const onset = Math.max(0, flux - fluxBaseline * 1.6);
      // Fast attack, fast decay — transient pop, not a swell
      transientLevel = Math.max(transientLevel * 0.78, Math.min(1.0, onset * 3.0));
    } else {
      transientLevel *= 0.78;
    }
  }
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
  } catch (err) {
    console.error('Playback failed:', err);
  }
}

audio.addEventListener('play', updatePlayState);
audio.addEventListener('pause', updatePlayState);

npBtn.addEventListener('click', async () => {
  initAudio();
  if (audioCtx.state === 'suspended') await audioCtx.resume();
  if (audio.paused) await audio.play(); else audio.pause();
});

startOverlay.addEventListener('click', async () => {
  await startPlayback();
  startOverlay.classList.add('hidden');
});

// ---------- Render loop ----------
const clock = new THREE.Clock();
function tick() {
  uniforms.iTime.value += clock.getDelta();

  // Smooth cursor
  smoothMouse.x += (mouse.x - smoothMouse.x) * 0.08;
  smoothMouse.y += (mouse.y - smoothMouse.y) * 0.08;
  const pr = renderer.getPixelRatio();
  uniforms.iMouse.value.set(smoothMouse.x * pr, smoothMouse.y * pr);

  updatePathUniforms();

  readBands();
  uniforms.iSubBass.value   = subBassSmoothed;
  uniforms.iBass.value      = bassSmoothed;
  uniforms.iMid.value       = midSmoothed;
  uniforms.iTreble.value    = trebleSmoothed;
  uniforms.iHarmonic.value.set(harmonicX, harmonicY);
  uniforms.iTransient.value = transientLevel;

  renderer.render(scene, camera);
  requestAnimationFrame(tick);
}
tick();
