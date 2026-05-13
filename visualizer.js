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
  iBass:       { value: 0 },
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

    float t = iTime * 0.18 + iBass * 1.5;

    // Domain-warped FBM with mouse attraction
    vec2 q = vec2(
      fbm(p * 1.4 + t * 0.3),
      fbm(p * 1.4 + vec2(5.2, 1.3) + t * 0.25)
    );

    vec2 warp = 3.0 * q + (m - p) * (0.6 + iBass * 0.8);

    vec2 r = vec2(
      fbm(p + warp + vec2(1.7, 9.2) + t * 0.4 + iMid * 0.5),
      fbm(p + warp + vec2(8.3, 2.8) + t * 0.35 + iTreble * 0.5)
    );

    float n = fbm(p + 3.5 * r);

    // Palette: deep purple → magenta → cyan → gold, modulated by audio bands
    vec3 cDeep    = vec3(0.05, 0.02, 0.18);
    vec3 cMagenta = vec3(0.9, 0.25, 0.7);
    vec3 cCyan    = vec3(0.15, 0.7, 0.95);
    vec3 cGold    = vec3(1.0, 0.85, 0.4);

    vec3 col = mix(cDeep, cMagenta, smoothstep(0.25, 0.55, n));
    col = mix(col, cCyan, smoothstep(0.55, 0.78, n) * (0.5 + iMid));
    col = mix(col, cGold, smoothstep(0.78, 0.95, n) * (0.3 + iTreble * 1.2));

    // Mouse glow
    float dToMouse = length(p - m);
    col += vec3(1.0, 0.6, 0.95) * smoothstep(0.55, 0.0, dToMouse) * (0.15 + iBass * 0.3);

    // Bass pulse (overall brightness)
    col *= 0.55 + iBass * 0.55;

    // Treble shimmer (high-freq grain)
    float shimmer = (hash(gl_FragCoord.xy + iTime * 60.0) - 0.5) * iTreble * 0.12;
    col += shimmer;

    // Subtle vignette
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
let bassSmoothed = 0, midSmoothed = 0, trebleSmoothed = 0;

function initAudio() {
  if (audioCtx) return;
  audioCtx = new (window.AudioContext || window.webkitAudioContext)();
  const source = audioCtx.createMediaElementSource(audio);
  analyser = audioCtx.createAnalyser();
  analyser.fftSize = 1024;
  analyser.smoothingTimeConstant = 0.7;
  source.connect(analyser);
  analyser.connect(audioCtx.destination);
  dataArray = new Uint8Array(analyser.frequencyBinCount);
}

function readBands() {
  if (!analyser) return;
  analyser.getByteFrequencyData(dataArray);

  const n = dataArray.length;
  const bassEnd = Math.floor(n * 0.06);
  const midEnd  = Math.floor(n * 0.25);
  const trebEnd = Math.floor(n * 0.6);

  let bass = 0, mid = 0, treble = 0;
  for (let i = 0; i < bassEnd; i++) bass += dataArray[i];
  for (let i = bassEnd; i < midEnd; i++) mid += dataArray[i];
  for (let i = midEnd; i < trebEnd; i++) treble += dataArray[i];

  bass   = bass   / (bassEnd * 255);
  mid    = mid    / ((midEnd - bassEnd) * 255);
  treble = treble / ((trebEnd - midEnd) * 255);

  // Fast attack, slow release — feels musical
  const attack = 0.55, release = 0.12;
  const ease = (cur, target) => target > cur ? cur + (target - cur) * attack : cur + (target - cur) * release;
  bassSmoothed   = ease(bassSmoothed, bass);
  midSmoothed    = ease(midSmoothed, mid);
  trebleSmoothed = ease(trebleSmoothed, treble);
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
  uniforms.iBass.value   = bassSmoothed;
  uniforms.iMid.value    = midSmoothed;
  uniforms.iTreble.value = trebleSmoothed;

  renderer.render(scene, camera);
  requestAnimationFrame(tick);
}
tick();
