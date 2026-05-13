import * as THREE from 'three';

// ---------- Three.js scene ----------
const canvas = document.getElementById('bg');
const renderer = new THREE.WebGLRenderer({ canvas, antialias: false });
renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
renderer.setSize(window.innerWidth, window.innerHeight);

const scene = new THREE.Scene();
const camera = new THREE.OrthographicCamera(-1, 1, 1, -1, 0, 1);

const uniforms = {
  iTime:       { value: 0 },
  iResolution: { value: new THREE.Vector2(window.innerWidth, window.innerHeight) },
  iMouse:      { value: new THREE.Vector2(window.innerWidth / 2, window.innerHeight / 2) },
  iSubBass:    { value: 0 }, // 0-130Hz, heavily smoothed — drives slow large-scale motion
  iBass:       { value: 0 }, // 130-345Hz — drives subtle pulse
  iMid:        { value: 0 },
  iTreble:     { value: 0 },
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
  varying vec2 vUv;
  uniform float iTime;
  uniform vec2 iResolution;
  uniform vec2 iMouse;
  uniform float iSubBass;
  uniform float iBass;
  uniform float iMid;
  uniform float iTreble;

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

  void main() {
    vec2 uv = gl_FragCoord.xy / iResolution.xy;
    vec2 p = uv - 0.5;
    p.x *= iResolution.x / iResolution.y;

    // Mouse position in same aspect-corrected space (y flipped: canvas vs gl)
    vec2 m = iMouse / iResolution - 0.5;
    m.y = -m.y;
    m.x *= iResolution.x / iResolution.y;

    // Pure ambient flow — geometry never reacts to audio, so it can't jerk.
    // Audio only touches color/brightness below.
    float t = iTime * 0.05;

    vec2 q = vec2(
      fbm(p * 1.4 + t * 0.3),
      fbm(p * 1.4 + vec2(5.2, 1.3) + t * 0.25)
    );

    vec2 warp = 2.5 * q;

    vec2 r = vec2(
      fbm(p + warp + vec2(1.7, 9.2) + t * 0.4),
      fbm(p + warp + vec2(8.3, 2.8) + t * 0.35)
    );

    float n = fbm(p + 3.0 * r);

    // Palette: near-black → deep blue → teal → soft green (relaxed)
    vec3 cBlack = vec3(0.0, 0.015, 0.03);
    vec3 cBlue  = vec3(0.02, 0.12, 0.28);
    vec3 cTeal  = vec3(0.05, 0.40, 0.50);
    vec3 cGreen = vec3(0.18, 0.65, 0.45);

    vec3 col = mix(cBlack, cBlue,  smoothstep(0.25, 0.55, n));
    col      = mix(col,    cTeal,  smoothstep(0.55, 0.78, n) * (0.65 + iMid * 0.12));
    col      = mix(col,    cGreen, smoothstep(0.78, 0.95, n) * (0.5 + iTreble * 0.18));

    // --- Localized cursor effect ---
    float dToMouse = length(p - m);
    float cursorCore = exp(-dToMouse * 18.0);
    float cursorHalo = exp(-dToMouse * 6.0);
    col += vec3(0.55, 1.0, 0.85) * cursorCore * 0.55;
    col += vec3(0.10, 0.55, 0.65) * cursorHalo * 0.20;

    // Gentle brightness breathing with sub-bass
    col *= 0.85 + iSubBass * 0.12;

    // Trace shimmer
    float shimmer = (hash(gl_FragCoord.xy + iTime * 60.0) - 0.5) * iTreble * 0.025;
    col += shimmer;

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

// ---------- Mouse tracking ----------
const mouse = { x: window.innerWidth / 2, y: window.innerHeight / 2 };
const smoothMouse = { x: mouse.x, y: mouse.y };

window.addEventListener('mousemove', e => {
  mouse.x = e.clientX;
  mouse.y = e.clientY;
});
window.addEventListener('touchmove', e => {
  const t = e.touches[0];
  mouse.x = t.clientX;
  mouse.y = t.clientY;
}, { passive: true });

// ---------- Audio + FFT analysis ----------
const audio = document.getElementById('audio');
const npBtn = document.getElementById('npBtn');
const npIcon = document.getElementById('npIcon');
const nowplaying = document.getElementById('nowplaying');
const startOverlay = document.getElementById('startOverlay');

let audioCtx, analyser, dataArray;
let subBassSmoothed = 0, bassSmoothed = 0, midSmoothed = 0, trebleSmoothed = 0;

// FFT bin layout (44.1kHz sample rate, fftSize 1024 → ~43Hz per bin)
const SUB_BASS_START = 1,  SUB_BASS_END = 3;    // ~43-130Hz   (kick / sub)
const BASS_START     = 3,  BASS_END     = 8;    // ~130-345Hz  (body)
const MID_START      = 8,  MID_END      = 80;   // ~345-3.4kHz
const TREB_START     = 80, TREB_END     = 250;  // ~3.4-11kHz

function initAudio() {
  if (audioCtx) return;
  audioCtx = new (window.AudioContext || window.webkitAudioContext)();
  const source = audioCtx.createMediaElementSource(audio);
  analyser = audioCtx.createAnalyser();
  analyser.fftSize = 1024;
  analyser.smoothingTimeConstant = 0.85; // heavier built-in smoothing
  source.connect(analyser);
  analyser.connect(audioCtx.destination);
  dataArray = new Uint8Array(analyser.frequencyBinCount);
}

function avgBins(start, end) {
  let sum = 0;
  for (let i = start; i < end; i++) sum += dataArray[i];
  return sum / ((end - start) * 255);
}

function readBands() {
  if (!analyser) return;
  analyser.getByteFrequencyData(dataArray);

  const subBass = avgBins(SUB_BASS_START, SUB_BASS_END);
  const bass    = avgBins(BASS_START,     BASS_END);
  const mid     = avgBins(MID_START,      MID_END);
  const treble  = avgBins(TREB_START,     TREB_END);

  // Sub-bass: very slow envelope — these drive the big movements, so we want
  // gentle swells not flickers. Other bands: snappier for color/grain reactivity.
  const easeAsym = (cur, target, atk, rel) =>
    target > cur ? cur + (target - cur) * atk : cur + (target - cur) * rel;

  subBassSmoothed = easeAsym(subBassSmoothed, subBass, 0.08, 0.025);
  bassSmoothed    = easeAsym(bassSmoothed,    bass,    0.35, 0.08);
  midSmoothed     = easeAsym(midSmoothed,     mid,     0.30, 0.08);
  trebleSmoothed  = easeAsym(trebleSmoothed,  treble,  0.30, 0.08);
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

  readBands();
  uniforms.iSubBass.value = subBassSmoothed;
  uniforms.iBass.value    = bassSmoothed;
  uniforms.iMid.value     = midSmoothed;
  uniforms.iTreble.value  = trebleSmoothed;

  renderer.render(scene, camera);
  requestAnimationFrame(tick);
}
tick();
