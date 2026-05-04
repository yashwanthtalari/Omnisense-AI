/**
 * OmniSense AI v4
 * - Ref-based position engine (60fps gesture reactivity)
 * - Two-hand tracking with independent grab/move
 * - Groq LLaMA-3 intelligence + Llama Vision for environment scanning
 * - Room Space / Infinite Space modes
 * - Live voice-to-input with Web Speech API
 */

import React, { useRef, useEffect, useState, useCallback } from 'react';
import { Canvas, useFrame } from '@react-three/fiber';
import { Line, Grid, GizmoHelper, GizmoViewport, OrbitControls, Text, Float, TransformControls } from '@react-three/drei';
import { motion, AnimatePresence } from 'framer-motion';
import * as THREE from 'three';
import {
  Mic, MicOff, Video, VideoOff, Send, Plus, Trash2,
  Layers, Terminal, Globe, Scan, Box, Cpu, Loader, Play
} from 'lucide-react';
import { FilesetResolver, HandLandmarker } from '@mediapipe/tasks-vision';
import './index.css';

// ─────────────────────────────────────────────────────────────────────────────
// CONSTANTS
// ─────────────────────────────────────────────────────────────────────────────

const OLLAMA_URL   = import.meta.env.VITE_OLLAMA_URL || 'http://localhost:11434/v1/chat/completions';
const TEXT_MODEL   = import.meta.env.VITE_TEXT_MODEL || 'llama3.1';
const SCOUT_MODEL  = import.meta.env.VITE_TEXT_MODEL || 'llama3.1';
const VISION_MODEL = import.meta.env.VITE_VISION_MODEL || 'llama3.2-vision';

const WEBCAM_FOV      = 60;
const SCALE_X         = 9;
const SCALE_Y         = 7;
const SCALE_Z         = 4;
const PINCH_DIST      = 0.07;  // normalized MediaPipe units
const GRAB_RADIUS     = 1.6;   // Three.js units

// ─────────────────────────────────────────────────────────────────────────────
// HELPERS
// ─────────────────────────────────────────────────────────────────────────────

function lmToVec3(lm, mirror = true) {
  return new THREE.Vector3(
    (mirror ? -1 : 1) * ((lm.x - 0.5) * SCALE_X),
    -((lm.y - 0.5) * SCALE_Y),
     1 - lm.z * SCALE_Z
  );
}

function landmarksToWorld(rawLms) {
  return rawLms.map(lm => lmToVec3(lm));
}

function dist2D(a, b) {
  return Math.sqrt((a.x - b.x) ** 2 + (a.y - b.y) ** 2);
}

function classifyGesture(raw) {
  if (!raw?.length) return 'none';

  // Pinch: thumb tip (4) close to index tip (8)
  const pinchDist = dist2D(raw[4], raw[8]);
  if (pinchDist < 0.06) return 'pinch';

  // Finger extended: tip Y clearly ABOVE its pip (lower Y number in screen coords)
  const iExt = raw[8].y  < raw[6].y  - 0.03; // index
  const mExt = raw[12].y < raw[10].y - 0.03; // middle
  const rExt = raw[16].y < raw[14].y - 0.03; // ring
  const pExt = raw[20].y < raw[18].y - 0.03; // pinky
  // Thumb: compare x to avoid confusion
  const tExt = Math.abs(raw[4].x - raw[2].x) > 0.04;

  // Fist: all four fingers clearly curled
  if (!iExt && !mExt && !rExt && !pExt) return 'fist';

  // Open palm: all four fingers AND thumb extended
  if (iExt && mExt && rExt && pExt) return 'open';

  // Peace: index + middle extended, ring + pinky curled (strict)
  if (iExt && mExt && !rExt && !pExt) return 'peace';

  // Point: ONLY index extended, middle+ring+pinky all curled
  if (iExt && !mExt && !rExt && !pExt) return 'point';

  return 'none';
}

const HAND_CONNECTIONS = [
  [0,1],[1,2],[2,3],[3,4],
  [0,5],[5,6],[6,7],[7,8],
  [5,9],[9,10],[10,11],[11,12],
  [9,13],[13,14],[14,15],[15,16],
  [13,17],[0,17],[17,18],[18,19],[19,20]
];

const GESTURE_COLOR = { pinch: '#ec4899', open: '#10b981', point: '#38bdf8', fist: '#f59e0b', peace: '#a78bfa', none: '#6366f1' };

// ─────────────────────────────────────────────────────────────────────────────
// ── Speech Synthesis (Funny Male Personality) ──────────────────────────────
let maleVoice = null;
function getMaleVoice() {
  if (maleVoice) return maleVoice;
  const voices = window.speechSynthesis.getVoices();
  maleVoice = voices.find(v => 
    v.name.toLowerCase().includes('male') || 
    v.name.toLowerCase().includes('david') || 
    v.name.toLowerCase().includes('alex') ||
    v.name.toLowerCase().includes('daniel') ||
    v.name.toLowerCase().includes('guy') ||
    v.name.toLowerCase().includes('andrew')
  ) || voices.find(v => v.lang.startsWith('en')) || voices[0];
  return maleVoice;
}
if (window.speechSynthesis) {
  window.speechSynthesis.onvoiceschanged = () => { maleVoice = null; getMaleVoice(); };
}

function speak(text) {
  if (!window.speechSynthesis) return;
  console.log("AI SPEAKING:", text);
  window.speechSynthesis.cancel();
  const utterance = new SpeechSynthesisUtterance(text);
  utterance.voice = getMaleVoice();
  utterance.rate  = 1.1;
  utterance.pitch = 0.85;
  window.speechSynthesis.speak(utterance);
}

// ─────────────────────────────────────────────────────────────────────────────
// OLLAMA API
// ─────────────────────────────────────────────────────────────────────────────

async function ollamaText(messages, model = TEXT_MODEL) {
  const res = await fetch(OLLAMA_URL, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ model, messages, temperature: 0.4, max_tokens: 768 })
  });
  if (!res.ok) {
    const errText = await res.text();
    // If primary model fails, fall back to fast model
    if (model !== TEXT_MODEL) throw new Error(`Ollama ${res.status}: ${errText}`);
    console.warn('Primary model failed, falling back to llama3.1');
    return ollamaText(messages, 'llama3.1');
  }
  const d = await res.json();
  return d.choices[0].message.content;
}

async function ollamaVision(base64Image, prompt) {
  const res = await fetch(OLLAMA_URL, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      model: VISION_MODEL,
      messages: [{
        role: 'user',
        content: [
          { type: 'text', text: prompt },
          { type: 'image_url', image_url: { url: `data:image/jpeg;base64,${base64Image}` } }
        ]
      }],
      max_tokens: 512
    })
  });
  if (!res.ok) throw new Error(`Ollama vision ${res.status}: ${await res.text()}`);
  const d = await res.json();
  return d.choices[0].message.content;
}

function captureVideoFrame(videoEl, quality = 0.5) {
  const canvas = document.createElement('canvas');
  canvas.width  = videoEl.videoWidth  || 640;
  canvas.height = videoEl.videoHeight || 480;
  canvas.getContext('2d').drawImage(videoEl, 0, 0);
  return canvas.toDataURL('image/jpeg', quality).split(',')[1];
}

// ─────────────────────────────────────────────────────────────────────────────
// THREE.JS – HAND SKELETON (reads from ref, updates in useFrame)
// ─────────────────────────────────────────────────────────────────────────────

function HandSkeleton({ handsRef, handIdx }) {
  const groupRef = useRef();
  const jointRefs = useRef([]);
  const lineRefs  = useRef([]);

  useFrame(() => {
    const h = handsRef.current[handIdx];
    if (!groupRef.current) return;
    if (!h || !h.landmarks3D) {
      groupRef.current.visible = false;
      return;
    }
    groupRef.current.visible = true;
    const col = GESTURE_COLOR[h.gesture] || '#6366f1';
    h.landmarks3D.forEach((v, i) => {
      if (jointRefs.current[i]) {
        jointRefs.current[i].position.copy(v);
        jointRefs.current[i].material.color.set(col);
      }
    });
  });

  // Build static geometry, positions updated in useFrame
  return (
    <group ref={groupRef} visible={false}>
      {Array.from({ length: 21 }, (_, i) => (
        <mesh key={i} ref={el => jointRefs.current[i] = el}>
          <sphereGeometry args={[(i === 8 || i === 4) ? 0.14 : 0.07, 10, 10]} />
          <meshBasicMaterial color="#6366f1" />
        </mesh>
      ))}
      {HAND_CONNECTIONS.map(([a, b], i) => (
        <DynamicLine key={i} handsRef={handsRef} handIdx={handIdx} a={a} b={b} />
      ))}
    </group>
  );
}

function DynamicLine({ handsRef, handIdx, a, b }) {
  const lineRef = useRef();
  const posA = useRef(new THREE.Vector3());
  const posB = useRef(new THREE.Vector3());

  useFrame(() => {
    const h = handsRef.current[handIdx];
    if (!h?.landmarks3D || !lineRef.current) return;
    posA.current.copy(h.landmarks3D[a]);
    posB.current.copy(h.landmarks3D[b]);
    const pts = lineRef.current.geometry.attributes.position;
    pts.setXYZ(0, posA.current.x, posA.current.y, posA.current.z);
    pts.setXYZ(1, posB.current.x, posB.current.y, posB.current.z);
    pts.needsUpdate = true;
  });

  return (
    <line ref={lineRef}>
      <bufferGeometry>
        <bufferAttribute attach="attributes-position" count={2} array={new Float32Array(6)} itemSize={3} />
      </bufferGeometry>
      <lineBasicMaterial color="#10b981" linewidth={2} />
    </line>
  );
}

// ── Interaction Mic (Point-to-Toggle) ───────────────────────────────────
function InteractionMic({ isMicOn, onToggle, handsRef }) {
  const meshRef = useRef();
  const [hovered, setHovered] = useState(false);
  const hoverTime = useRef(0);

  useFrame((state, delta) => {
    if (meshRef.current) {
      meshRef.current.position.y = 2.5 + Math.sin(state.clock.elapsedTime * 2) * 0.1;
      meshRef.current.rotation.y += delta * 0.5;
    }

    let isPointingAt = false;
    for (const h of handsRef.current) {
      if (h.gesture === 'point' && h.landmarks3D) {
        const tip = h.landmarks3D[8];
        const dist = new THREE.Vector3(tip.x, tip.y, tip.z).distanceTo(new THREE.Vector3(-3.5, 2.5, 0));
        if (dist < 0.7) isPointingAt = true;
      }
    }

    if (isPointingAt) {
      hoverTime.current += delta;
      setHovered(true);
      if (hoverTime.current > 1.2) {
        onToggle();
        hoverTime.current = -3; // Cooldown
        addLog('system', 'Mic toggled via 3D point gesture.');
      }
    } else {
      hoverTime.current = Math.max(0, hoverTime.current - delta * 2);
      if (hoverTime.current === 0) setHovered(false);
    }
  });

  return (
    <group position={[-3.5, 2.5, 0]}>
      <Float speed={3} rotationIntensity={0.5} floatIntensity={0.5}>
        <mesh ref={meshRef}>
          <sphereGeometry args={[0.3, 32, 32]} />
          <meshStandardMaterial 
            color={isMicOn ? '#10b981' : '#3b82f6'} 
            emissive={isMicOn ? '#10b981' : '#3b82f6'} 
            emissiveIntensity={hovered ? 2 : 0.4} 
            transparent opacity={0.6}
          />
        </mesh>
      </Float>
      <Text position={[0, -0.6, 0]} fontSize={0.15} color="white" anchorX="center">
        {isMicOn ? 'MIC ON' : 'POINT TO MIC'}
      </Text>
    </group>
  );
}

// ─────────────────────────────────────────────────────────────────────────────
// THREE.JS – SHAPE MESH (reads position from posMap ref every frame)
// ─────────────────────────────────────────────────────────────────────────────

function ShapeMesh({ shapeId, type, color, posMap, selectedIdRef, grabbedMap, onClick }) {
  const meshRef = useRef();
  const ringRef = useRef();
  const rotVel  = useRef({ x: (Math.random() - 0.5) * 0.8, y: (Math.random() - 0.5) * 1.2 });

  useFrame((state, delta) => {
    if (!meshRef.current || !posMap.current[shapeId]) return;
    const { target, current } = posMap.current[shapeId];
    const isGrabbed = !!grabbedMap.current[shapeId];

    current.lerp(target, isGrabbed ? 0.3 : 0.08);
    meshRef.current.position.copy(current);

    const isSelected = selectedIdRef.current === shapeId;
    
    // Selection Ring
    if (ringRef.current) {
      ringRef.current.visible = isSelected;
      ringRef.current.position.copy(current);
      ringRef.current.rotation.z += delta * 1.2;
    }

    if (!isGrabbed) {
      meshRef.current.rotation.y += delta * rotVel.current.y * 0.5;
    }

    const targetScale = isGrabbed ? 1.2 : isSelected ? 1.08 : 1.0;
    meshRef.current.scale.lerp(new THREE.Vector3(targetScale, targetScale, targetScale), 0.15);

    if (meshRef.current.material) {
      meshRef.current.material.emissiveIntensity = THREE.MathUtils.lerp(
        meshRef.current.material.emissiveIntensity, isGrabbed ? 0.6 : isSelected ? 0.2 : 0, 0.15
      );
    }
  });

  return (
    <group>
      <mesh
        ref={meshRef}
        onClick={(e) => { e.stopPropagation(); onClick(shapeId); }}
      >
        {type === 'box'    && <boxGeometry args={[1,1,1]} />}
        {type === 'sphere' && <sphereGeometry args={[0.65, 32, 32]} />}
        {type === 'cone'   && <coneGeometry args={[0.6, 1.2, 32]} />}
        {type === 'torus'  && <torusGeometry args={[0.55, 0.2, 16, 40]} />}
        {type === 'cylinder' && <cylinderGeometry args={[0.5, 0.5, 1.2, 32]} />}
        <meshStandardMaterial color={color} metalness={0.5} roughness={0.3} emissive={color} emissiveIntensity={0} />
      </mesh>
      <mesh ref={ringRef}>
        <torusGeometry args={[0.9, 0.02, 16, 64]} />
        <meshBasicMaterial color="#f59e0b" transparent opacity={0.6} />
      </mesh>
      {selectedIdRef.current === shapeId && meshRef.current && (
        <TransformControls object={meshRef.current} mode="translate" size={0.6} />
      )}
    </group>
  );
}

// Grab cursor ring at index finger
function GrabCursor({ handsRef }) {
  const refs = [useRef(), useRef()];

  useFrame(() => {
    [0, 1].forEach(idx => {
      const mesh = refs[idx].current;
      const h    = handsRef.current[idx];
      if (!mesh) return;
      if (!h?.landmarks3D) { mesh.visible = false; return; }
      mesh.visible = true;
      mesh.position.copy(h.landmarks3D[8]);
      const isPinch = h.gesture === 'pinch';
      mesh.scale.setScalar(isPinch ? 0.8 : 1.0 + Math.sin(Date.now() * 0.005) * 0.1);
      mesh.material.color.set(isPinch ? '#ec4899' : '#38bdf8');
    });
  });

  return (
    <>
      {[0, 1].map(idx => (
        <mesh key={idx} ref={refs[idx]} visible={false}>
          <ringGeometry args={[0.15, 0.22, 32]} />
          <meshBasicMaterial color="#38bdf8" transparent opacity={0.9} side={THREE.DoubleSide} />
        </mesh>
      ))}
    </>
  );
}

// Scene root
function Scene({ shapes, posMap, grabbedMap, selectedIdRef, handsRef, spaceMode, onSelect }) {
  return (
    <>
      <ambientLight intensity={0.5} />
      <directionalLight position={[8, 12, 5]} intensity={1.3} />
      <pointLight position={[-6, 6, -6]} intensity={0.8} color="#6366f1" />
      <pointLight position={[6, -3,  6]} intensity={0.4} color="#ec4899" />

      {spaceMode === 'room' ? (
        <Grid position={[0, -3.5, 0]} args={[40, 40]} cellSize={0.5} sectionSize={2} cellColor="#1a2035" sectionColor="#2d3a55" infiniteGrid fadeDistance={20} />
      ) : (
        <Grid position={[0, -3, 0]} infiniteGrid fadeDistance={40} cellSize={0.5} sectionSize={3} cellColor="#1a2035" sectionColor="#243050" />
      )}

      <GizmoHelper alignment="bottom-right" margin={[60, 60]}>
        <GizmoViewport axisColors={['#ef4444', '#10b981', '#38bdf8']} labelColor="white" />
      </GizmoHelper>

      {spaceMode === 'infinite' && <OrbitControls makeDefault />}

      <InteractionMic isMicOn={document.getElementById('mic-btn')?.classList.contains('mic-on')} onToggle={() => document.getElementById('mic-btn')?.click()} handsRef={handsRef} />

      {shapes.map(s => (
        <ShapeMesh
          key={s.id}
          shapeId={s.id}
          type={s.type}
          color={s.color}
          posMap={posMap}
          selectedIdRef={selectedIdRef}
          grabbedMap={grabbedMap}
          onClick={onSelect}
        />
      ))}

      <HandSkeleton handsRef={handsRef} handIdx={0} />
      <HandSkeleton handsRef={handsRef} handIdx={1} />
      <GrabCursor handsRef={handsRef} />
    </>
  );
}

// ─────────────────────────────────────────────────────────────────────────────
// INITIAL STATE
// ─────────────────────────────────────────────────────────────────────────────

const INITIAL_SHAPES = [
  { id: 1, type: 'sphere',   color: '#6366f1', name: 'Sphere.001', position: [-2.5,  0.5,  0] },
  { id: 2, type: 'box',      color: '#38bdf8', name: 'Cube.001',   position: [ 2.5,  0.5, -1] },
  { id: 3, type: 'torus',    color: '#ec4899', name: 'Torus.001',  position: [ 0,    1.5, -2] },
];

// ─────────────────────────────────────────────────────────────────────────────
// MAIN APP
// ─────────────────────────────────────────────────────────────────────────────

export default function App() {
  // ── Refs (hot-path data, never trigger re-renders) ────────────────────────
  const videoRef          = useRef(null);
  const handLandmarkerRef = useRef(null);
  const reqFrameRef       = useRef(null);
  const recognitionRef    = useRef(null);
  const micOnRef          = useRef(false);
  const logScrollRef      = useRef(null);

  // posMap: { [id]: { target: Vector3, current: Vector3 } }  — moved by gestures
  const posMap     = useRef({});
  // grabbedMap: { [id]: handIdx }  — which hand holds which shape
  const grabbedMap = useRef({});
  // per-hand gesture state
  const handsRef   = useRef([
    { gesture: 'none', landmarks3D: null, rawLms: null, grabbedId: null, prevGesture: 'none' },
    { gesture: 'none', landmarks3D: null, rawLms: null, grabbedId: null, prevGesture: 'none' },
  ]);
  const selectedIdRef  = useRef(null);
  const shapesRef      = useRef(INITIAL_SHAPES);

  // ── React State (UI only) ─────────────────────────────────────────────────
  const [shapes,       setShapes]       = useState(INITIAL_SHAPES);
  const [selectedId,   setSelectedId]   = useState(null);
  const [isCamOn,      setIsCamOn]      = useState(false);
  const [isMicOn,      setIsMicOn]      = useState(false);
  const [stream,       setStream]       = useState(null);
  const [inputText,    setInputText]    = useState('');
  const [spaceMode,    setSpaceMode]    = useState('room');      // 'room' | 'infinite'
  const [fps,          setFps]          = useState(0);
  const [handStatus,   setHandStatus]   = useState(['NO HAND', 'NO HAND']);
  const [gestures,     setGestures]     = useState(['none', 'none']);
  const [grabbedNames, setGrabbedNames] = useState([null, null]);
  const [loading,      setLoading]      = useState(false);
  const [envInfo,      setEnvInfo]      = useState('');          // Groq vision analysis
  const [logs, setLogs] = useState([
    { sender: 'system', text: 'OmniSense v4 — Gesture Engine + Ollama Intelligence Online', time: new Date() },
    { sender: 'ai',     text: '📡 Gesture guide: ✋ OPEN PALM (Superman pose) → hover near shape to GRAB & MOVE it. Make a ✊ FIST to DROP. 🤏 PINCH to SELECT nearest. ✌️ PEACE to SPAWN shape at hand. 👊 FIST = delete selected.', time: new Date() },
  ]);

  // ── Keep refs in sync with state ──────────────────────────────────────────
  useEffect(() => { shapesRef.current = shapes; }, [shapes]);
  useEffect(() => { selectedIdRef.current = selectedId; }, [selectedId]);
  useEffect(() => { micOnRef.current = isMicOn; }, [isMicOn]);

  // ── Initialize posMap for each shape ─────────────────────────────────────
  useEffect(() => {
    INITIAL_SHAPES.forEach(s => {
      posMap.current[s.id] = {
        target:  new THREE.Vector3(...s.position),
        current: new THREE.Vector3(...s.position),
      };
    });
  }, []);

  // ── Logging ───────────────────────────────────────────────────────────────
  const addLog = useCallback((sender, text) => {
    setLogs(prev => [...prev, { sender, text, time: new Date() }]);
  }, []);

  useEffect(() => {
    if (logScrollRef.current) logScrollRef.current.scrollTop = logScrollRef.current.scrollHeight;
  }, [logs]);

  const startTour = useCallback(() => {
    addLog('system', 'Starting Training Walkthrough...');
    const steps = [
      "Sir, to get started, open your arm out like Superman to grab and move the shape.",
      "To release or drop the object, close your fingers into a fist.",
      "See the glowing blue sphere in your 3D space? Point your index finger at it for one second to toggle the microphone.",
      "When the mic is active, speak your commands. Say 'spawn a torus', or 'move the box to the left'.",
      "Notice the Blender-style gizmos on selected shapes? You can now edit shapes perfectly in 3D."
    ];
    let i = 0;
    const nextStep = () => {
      if (i >= steps.length) return;
      speak(steps[i]);
      addLog('ai', steps[i]);
      i++;
      setTimeout(nextStep, 6500); // Wait long enough for narration
    };
    nextStep();
  }, [addLog]);

  // ── Shape Management ──────────────────────────────────────────────────────
  const spawnShape = useCallback((type = 'box', posArr) => {
    const x = posArr?.[0] ?? (Math.random() - 0.5) * 5;
    const y = posArr?.[1] ?? Math.random() * 2;
    const z = posArr?.[2] ?? -1 + (Math.random() - 0.5) * 2;
    const colorMap = { box:'#ec4899', sphere:'#10b981', cone:'#f59e0b', torus:'#6366f1', cylinder:'#38bdf8' };
    const nameMap  = { box:'Cube', sphere:'Sphere', cone:'Cone', torus:'Torus', cylinder:'Cylinder' };
    const id = Date.now();
    const shape = { id, type, color: colorMap[type] || '#ffffff', name: `${nameMap[type]||'Obj'}.${String(id).slice(-3)}`, position: [x,y,z] };
    posMap.current[id] = { target: new THREE.Vector3(x,y,z), current: new THREE.Vector3(x,y,z) };
    setShapes(prev => [...prev, shape]);
    setSelectedId(id);
    addLog('system', `Spawned ${type} at (${x.toFixed(1)}, ${y.toFixed(1)}, ${z.toFixed(1)})`);
    return id;
  }, [addLog]);

  const deleteShape = useCallback((id) => {
    delete posMap.current[id];
    delete grabbedMap.current[id];
    handsRef.current.forEach(h => { if (h.grabbedId === id) { h.grabbedId = null; } });
    setShapes(prev => prev.filter(s => s.id !== id));
    if (selectedIdRef.current === id) { setSelectedId(null); }
    addLog('system', `Deleted object #${id}`);
  }, [addLog]);

  const clearScene = useCallback(() => {
    setShapes([]);
    posMap.current = {};
    grabbedMap.current = {};
    handsRef.current.forEach(h => { h.grabbedId = null; });
    setSelectedId(null);
    addLog('system', 'Scene cleared.');
  }, [addLog]);

  // ── Camera ────────────────────────────────────────────────────────────────
  useEffect(() => {
    if (isCamOn) {
      navigator.mediaDevices.getUserMedia({ video: { facingMode: 'user', width: { ideal: 1280 } } })
        .then(s => {
          setStream(s);
          if (videoRef.current) { videoRef.current.srcObject = s; videoRef.current.play(); }
          addLog('system', 'Camera stream online.');
        })
        .catch(e => { addLog('system', `Camera error: ${e.message}`); setIsCamOn(false); });
    } else {
      stream?.getTracks().forEach(t => t.stop());
      setStream(null);
      handsRef.current.forEach((h, i) => {
        h.gesture = 'none'; h.landmarks3D = null; h.rawLms = null; h.grabbedId = null;
      });
      setHandStatus(['NO HAND', 'NO HAND']);
      setGestures(['none', 'none']);
    }
  }, [isCamOn]);

  // ── MediaPipe Init ────────────────────────────────────────────────────────
  useEffect(() => {
    (async () => {
      try {
        const vision = await FilesetResolver.forVisionTasks(
          'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@latest/wasm'
        );
        handLandmarkerRef.current = await HandLandmarker.createFromOptions(vision, {
          baseOptions: {
            modelAssetPath: 'https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task',
            delegate: 'GPU'
          },
          runningMode: 'VIDEO',
          numHands: 2   // ← Two hands
        });
        addLog('system', 'MediaPipe HandLandmarker ready (2 hands, GPU).');
      } catch (e) {
        addLog('system', `MediaPipe error: ${e.message}`);
      }
    })();
  }, []);

  // ── Gesture + Hand Tracking rAF Loop ─────────────────────────────────────
  useEffect(() => {
    if (!isCamOn) { cancelAnimationFrame(reqFrameRef.current); return; }

    let lastTime  = -1;
    let frameCnt  = 0;
    let lastFpsTm = performance.now();

    const loop = () => {
      const vid = videoRef.current;
      reqFrameRef.current = requestAnimationFrame(loop);

      if (!vid || vid.readyState < 2 || !handLandmarkerRef.current) return;

      const now = performance.now();
      frameCnt++;
      if (now - lastFpsTm >= 1000) { setFps(frameCnt); frameCnt = 0; lastFpsTm = now; }
      if (lastTime === vid.currentTime) return;
      lastTime = vid.currentTime;

      let results;
      try { results = handLandmarkerRef.current.detectForVideo(vid, now); }
      catch (_) { return; }

      const numDetected = results.landmarks?.length || 0;
      const newStatus   = ['NO HAND', 'NO HAND'];
      const newGestures = ['none', 'none'];

      // Reset hands not detected this frame
      handsRef.current.forEach((h, idx) => {
        if (idx >= numDetected) {
          // Drop any grabbed object for this hand
          if (h.grabbedId !== null) {
            delete grabbedMap.current[h.grabbedId];
            h.grabbedId = null;
          }
          h.gesture = 'none';
          h.landmarks3D = null;
          h.rawLms = null;
        }
      });

      // Process each detected hand
      for (let idx = 0; idx < Math.min(numDetected, 2); idx++) {
        const rawLms = results.landmarks[idx];
        const lms3D  = landmarksToWorld(rawLms);
        const g      = classifyGesture(rawLms);
        const h      = handsRef.current[idx];
        const prev   = h.prevGesture;

        h.rawLms      = rawLms;
        h.landmarks3D = lms3D;
        h.gesture     = g;
        h.prevGesture = g;

        newStatus[idx]   = 'DETECTED';
        newGestures[idx] = g;

        const indexTip = lms3D[8]; // index finger tip in world space
        // Palm center = average of wrist + 4 knuckle bases (more stable anchor for open-hand grab)
        const palmCenter = new THREE.Vector3(
          (lms3D[0].x + lms3D[5].x + lms3D[9].x + lms3D[13].x + lms3D[17].x) / 5,
          (lms3D[0].y + lms3D[5].y + lms3D[9].y + lms3D[13].y + lms3D[17].y) / 5,
          (lms3D[0].z + lms3D[5].z + lms3D[9].z + lms3D[13].z + lms3D[17].z) / 5,
        );

        // ── OPEN PALM (Superman Pose) = GRAB & MOVE ────────────────────────
        if (g === 'open') {
          if (prev !== 'open') {
            // Entered open-palm — find nearest shape within grab radius
            let nearestId = null, minD = GRAB_RADIUS;
            for (const s of shapesRef.current) {
              const pm = posMap.current[s.id];
              if (!pm) continue;
              const d = palmCenter.distanceTo(pm.current);
              if (d < minD) { minD = d; nearestId = s.id; }
            }
            if (nearestId !== null && !grabbedMap.current[nearestId]) {
              h.grabbedId = nearestId;
              grabbedMap.current[nearestId] = idx;
              setSelectedId(nearestId);
              const name = shapesRef.current.find(s => s.id === nearestId)?.name || nearestId;
              addLog('system', `Hand ${idx}: ✋ GRAB — ${name}`);
              speak(`Grabbed ${name}, Sir.`);
            }
          }
          // Every open-palm frame — drag grabbed object to palm center
          if (h.grabbedId !== null && posMap.current[h.grabbedId]) {
            posMap.current[h.grabbedId].target.copy(palmCenter);
          }
        }

        // ── DROP: leaving open-palm releases held object (NO delete) ─────
        if (g !== 'open' && prev === 'open') {
          if (h.grabbedId !== null) {
            const finalPos = posMap.current[h.grabbedId]?.current;
            if (finalPos) {
              const pos = [finalPos.x, finalPos.y, finalPos.z];
              setShapes(prev => prev.map(s => s.id === h.grabbedId ? { ...s, position: pos } : s));
              addLog('system', `Hand ${idx}: ✊ DROP at (${finalPos.x.toFixed(1)}, ${finalPos.y.toFixed(1)}, ${finalPos.z.toFixed(1)})`);
            }
            delete grabbedMap.current[h.grabbedId];
            h.grabbedId = null;
          }
        }

        // ── PINCH: select nearest object ─────────────────────────────────
        if (g === 'pinch' && prev !== 'pinch') {
          let nearestId = null, minD = GRAB_RADIUS * 2;
          for (const s of shapesRef.current) {
            const pm = posMap.current[s.id];
            if (!pm) continue;
            const d = indexTip.distanceTo(pm.current);
            if (d < minD) { minD = d; nearestId = s.id; }
          }
          if (nearestId !== null) {
            setSelectedId(nearestId);
            addLog('system', `Hand ${idx}: 🤏 PINCH — selected ${shapesRef.current.find(s=>s.id===nearestId)?.name || nearestId}`);
          }
        }

        // ── POINT: select nearest ─────────────────────────────────────────
        if (g === 'point' && prev !== 'point') {
          let nearestId = null, minD = GRAB_RADIUS * 1.8;
          for (const s of shapesRef.current) {
            const pm = posMap.current[s.id];
            if (!pm) continue;
            const d = indexTip.distanceTo(pm.current);
            if (d < minD) { minD = d; nearestId = s.id; }
          }
          if (nearestId !== null) setSelectedId(nearestId);
        }

        // ── FIST: delete selected ONLY if not currently holding anything ──
        if (g === 'fist' && prev !== 'fist' && h.grabbedId === null) {
          const sel = selectedIdRef.current;
          if (sel) {
            deleteShape(sel);
            addLog('ai', '✊ Fist — object deleted, Sir.');
            speak('Deleted, Sir.');
          }
        }

        // ── PEACE ✌️: spawn shape at palm position ────────────────────────
        if (g === 'peace' && prev !== 'peace') {
          const types = ['sphere', 'box', 'cone', 'torus', 'cylinder'];
          const picked = types[Math.floor(Math.random() * types.length)];
          spawnShape(picked, [palmCenter.x, palmCenter.y, palmCenter.z]);
          addLog('ai', `✌️ Peace — spawned ${picked} at hand!`);
          speak(`Spawning a ${picked} right at your hand, Sir.`);
        }
      }

      // Update UI state (React — throttled by rAF so ~60fps max)
      setHandStatus([...newStatus]);
      setGestures([...newGestures]);

      // Update grabbed names for UI
      const names = [null, null];
      handsRef.current.forEach((h, i) => {
        if (h.grabbedId) {
          const s = shapesRef.current.find(s => s.id === h.grabbedId);
          names[i] = s?.name || `#${h.grabbedId}`;
        }
      });
      setGrabbedNames([...names]);
    };

    reqFrameRef.current = requestAnimationFrame(loop);
    return () => cancelAnimationFrame(reqFrameRef.current);
  }, [isCamOn, stream, deleteShape, spawnShape, addLog]);

  // ── Voice Recognition — initialize ONCE ──────────────────────────────────
  useEffect(() => {
    const SR = window.SpeechRecognition || window.webkitSpeechRecognition;
    if (!SR) { addLog('system', 'SpeechRecognition not supported in this browser.'); return; }
    const r = new SR();
    r.continuous      = true;
    r.interimResults  = true;
    r.lang            = 'en-US';
    r.onresult = (ev) => {
      let interim = '', final = '';
      for (let i = ev.resultIndex; i < ev.results.length; i++) {
        const t = ev.results[i][0].transcript;
        ev.results[i].isFinal ? (final += t) : (interim += t);
      }
      setInputText(final || interim);
    };
    r.onerror = (e) => {
      if (e.error === 'not-allowed') { addLog('system', 'Microphone permission denied.'); setIsMicOn(false); }
    };
    r.onend = () => { if (micOnRef.current) { try { r.start(); } catch (_) {} } };
    recognitionRef.current = r;
    addLog('system', 'Speech recognition engine ready.');
  }, []); // ← Init once only

  useEffect(() => {
    if (!recognitionRef.current) return;
    if (isMicOn) {
      try { recognitionRef.current.start(); addLog('system', 'Mic ON → speak, transcript appears in input.'); }
      catch (_) {}
    } else {
      try { recognitionRef.current.stop(); addLog('system', 'Mic OFF.'); }
      catch (_) {}
    }
  }, [isMicOn, addLog]);

  // ── Ollama Intelligence ───────────────────────────────────────────────────
  const handleSubmit = useCallback(async (e) => {
    e?.preventDefault();
    if (!inputText.trim() || loading) return;
    const userMsg = inputText.trim();
    addLog('user', userMsg);
    setInputText('');
    setLoading(true);

    const sceneContext = shapesRef.current.map(s => {
      const pm = posMap.current[s.id];
      const pos = pm ? [pm.current.x.toFixed(2), pm.current.y.toFixed(2), pm.current.z.toFixed(2)] : s.position;
      return `${s.name} (${s.type}) at [${pos}]`;
    }).join(', ');

    const gestureContext = handsRef.current
      .map((h, i) => h.landmarks3D ? `Hand${i}: ${h.gesture}${h.grabbedId ? ` holding #${h.grabbedId}` : ''}` : '')
      .filter(Boolean).join('; ') || 'No hands visible';

    const systemPrompt = `You are OmniSense AI — a highly capable, slightly witty, yet deeply respectful male spatial-intelligence core.

PERSONALITY & RULES:
- Address the user as "Sir" (e.g., "As you wish, Sir.", "The sphere is in position, Sir.").
- Be conversational and "bro-ish" but with a professional 'Jarvis' or 'engineer-buddy' edge.
- Use humor and confidence: "Boom! Done.", "Another masterpiece for the collection, Sir.", "Spatial systems are green."
- You "talk back" using a male voice (lower pitch). 
- Keep 'reply' concise (1-2 sentences).

CURRENT SCENE (live positions):
${sceneContext || '(empty scene)'}

HAND STATE: ${gestureContext}
SPACE MODE: ${spaceMode}
${envInfo ? `ENVIRONMENT: ${envInfo}` : ''}

You MUST respond with ONLY a raw JSON object. No markdown, no code fences, no prose outside JSON.
Schema:
{
  "reply": "<witty, conversational response to the user>",
  "actions": [
    { "type": "spawn",   "shape": "box|sphere|cone|torus|cylinder", "position": [x, y, z] },
    { "type": "move",    "id": <object_id_number>, "position": [x, y, z] },
    { "type": "delete",  "id": <object_id_number> },
    { "type": "clear" },
    { "type": "setMode", "mode": "room|infinite" },
    { "type": "select",  "id": <object_id_number> }
  ]
}
Rules:
- Use exact numeric id values from the scene list above for delete/move/select
- For spatial words like "left", "right", "above", "between" — compute actual [x,y,z] coordinates
- If no 3D action needed, set actions: []
- NEVER output anything outside the JSON object. Keep the 'reply' characterful.`;

    try {
      // Use Scout for spatial reasoning (it understands 3D positions natively)
      const raw = await ollamaText([
        { role: 'system',    content: systemPrompt },
        { role: 'user',      content: userMsg }
      ], SCOUT_MODEL);

      let parsed;
      try {
        // Strip markdown code fences if the model wraps response in ```json ... ```
        const clean = raw.replace(/```json|```/gi, '').trim();
        const jsonMatch = clean.match(/\{[\s\S]*\}/);
        parsed = JSON.parse(jsonMatch ? jsonMatch[0] : clean);
      } catch {
        parsed = { reply: raw, actions: [] };
      }

      addLog('ai', parsed.reply || 'Done.');
      if (parsed.reply) speak(parsed.reply);

      for (const action of (parsed.actions || [])) {
        // Normalise: models sometimes return 'object' or 'shape' interchangeably
        const shapeType = action.shape || action.object || action.geometry || 'box';
        if (action.type === 'spawn')   spawnShape(shapeType, action.position);
        if (action.type === 'move' && action.id && posMap.current[action.id]) {
          posMap.current[action.id].target.set(...(action.position || [0,0,0]));
          // sync React state after short delay
          setTimeout(() => {
            const p = posMap.current[action.id]?.current;
            if (p) setShapes(prev => prev.map(s => s.id === action.id ? { ...s, position: [p.x, p.y, p.z] } : s));
          }, 600);
        }
        if (action.type === 'delete')  deleteShape(action.id);
        if (action.type === 'clear')   clearScene();
        if (action.type === 'setMode') setSpaceMode(action.mode);
        if (action.type === 'select')  setSelectedId(action.id);
      }
    } catch (e) {
      addLog('system', `Ollama error: ${e.message}`);
      // Local fallback
      const t = userMsg.toLowerCase();
      if      (t.includes('sphere') || t.includes('ball'))    spawnShape('sphere');
      else if (t.includes('cube')   || t.includes('box'))     spawnShape('box');
      else if (t.includes('torus')  || t.includes('ring'))    spawnShape('torus');
      else if (t.includes('cone'))                            spawnShape('cone');
      else if (t.includes('cylinder'))                        spawnShape('cylinder');
      else if (t.includes('clear')  || t.includes('reset'))  clearScene();
      else addLog('system', 'Ollama unavailable, local parser used.');
    } finally {
      setLoading(false);
    }
  }, [inputText, loading, spaceMode, envInfo, addLog, spawnShape, deleteShape, clearScene]);

  // ── Environment Scan (Ollama Vision) ────────────────────────────────────────
  const scanEnvironment = useCallback(async () => {
    if (!isCamOn || !videoRef.current) { addLog('system', 'Enable camera first.'); return; }
    setLoading(true);
    addLog('system', 'Scanning environment with Ollama Vision…');
    try {
      const frame = captureVideoFrame(videoRef.current, 0.6);
      const result = await ollamaVision(frame,
        `Analyze this room/environment for an AR 3D workspace. Identify:
1. Floor position/surface (estimate how far down)
2. Major objects visible (desk, chair, walls, etc.)
3. Available empty space for placing 3D objects
4. Estimated room depth/size
Keep response to 3-4 sentences, focus on spatial layout.`
      );
      setEnvInfo(result);
      addLog('ai', `[Vision] ${result}`);
    } catch (e) {
      addLog('system', `Vision scan failed: ${e.message}`);
    } finally {
      setLoading(false);
    }
  }, [isCamOn, addLog]);



  // ── Derived for UI ────────────────────────────────────────────────────────
  const selectedShape = shapes.find(s => s.id === selectedId);
  // Updated to match new gesture mapping (open = grab, fist = drop/delete)
  const GESTURE_LABEL = {
    open:  '✋ GRAB & MOVE',
    pinch: '🤏 SELECT',
    point: '👆 SELECT',
    fist:  '✊ DROP / DELETE',
    peace: '✌️ SPAWN',
    none:  '— IDLE'
  };
  const GESTURE_COLOR_UI = {
    open:  'var(--accent-green)',
    pinch: 'var(--primary)',
    point: 'var(--accent-blue)',
    fist:  'var(--accent-red)',
    peace: '#a78bfa',
    none:  'var(--text-muted)'
  };

  // ─────────────────────────────────────────────────────────────────────────
  return (
    <div className="app-shell">

      {/* ══ TOP BAR ═══════════════════════════════════════════════════════════ */}
      <header className="topbar">
        <div className="topbar-logo">OMNI<span>SENSE</span> <span style={{ fontSize: '9px', color: 'var(--accent-green)', letterSpacing: 2 }}>v4</span></div>
        <div className="topbar-divider" />
        {/* Space mode toggle */}
        <div style={{ display: 'flex', gap: 4 }}>
          {['room', 'infinite'].map(m => (
            <button
              key={m}
              className={`topbar-tab${spaceMode === m ? ' active' : ''}`}
              onClick={() => { setSpaceMode(m); addLog('system', `Space mode → ${m}`); }}
            >
              {m === 'room' ? '🏠 Room Space' : '🌌 Infinite Space'}
            </button>
          ))}
        </div>
        <div className="topbar-divider" />
        <div className="topbar-tabs">
          {['Spatial', 'Analysis', 'Settings'].map(t => (
            <button key={t} className={`topbar-tab${t === 'Spatial' ? ' active' : ''}`}>{t}</button>
          ))}
        </div>
        <div className="topbar-status-pill">
          <div className={`status-dot ${isMicOn ? 'listening' : isCamOn ? 'online' : ''}`} />
          {isMicOn ? 'LISTENING' : isCamOn ? 'CAM ACTIVE' : 'STANDBY'}
          {loading && <><div className="topbar-divider" /><Loader size={10} style={{ animation: 'spin 1s linear infinite' }} /> OLLAMA THINKING</>}
          <div className="topbar-divider" />
          <Cpu size={10} /> OLLAMA · MEDIAPIPE
        </div>
      </header>

      {/* ══ LEFT PANEL ════════════════════════════════════════════════════════ */}
      <aside className="panel-left">
        <div className="panel-header"><Layers size={12} className="panel-header-icon" /> Scene</div>

        {/* Hand status per hand */}
        <div className="panel-section">
          <div className="panel-section-title"><span>Hand Tracking</span></div>
          {[0, 1].map(idx => (
            <div key={idx} style={{ padding: '6px 12px', borderTop: '1px solid var(--border)' }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 2 }}>
                <span className="prop-label">Hand {idx + 1}</span>
                <span className="prop-value" style={{ color: handStatus[idx] === 'DETECTED' ? 'var(--accent-green)' : 'var(--text-dim)' }}>
                  {handStatus[idx]}
                </span>
              </div>
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                <span className="prop-label">Gesture</span>
                <span style={{ fontFamily: 'var(--mono)', fontSize: '10px', color: GESTURE_COLOR_UI[gestures[idx]] }}>
                  {GESTURE_LABEL[gestures[idx]] || '—'}
                </span>
              </div>
              {grabbedNames[idx] && (
                <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                  <span className="prop-label">Holding</span>
                  <span className="prop-value pink">{grabbedNames[idx]}</span>
                </div>
              )}
            </div>
          ))}
        </div>

        {/* Gesture guide */}
        <div className="panel-section">
          <div className="panel-section-title"><span>Gesture Guide</span></div>
          {[
            ['✋', 'OPEN PALM', 'Superman pose → Grab & Move'],
            ['✊', 'FIST',     'Drop held / Delete selected'],
            ['🤏', 'PINCH',   'Select nearest object'],
            ['👆', 'POINT',   'Select nearest object'],
            ['✌️', 'PEACE',  'Spawn shape at palm'],
          ].map(([icon, name, desc]) => (
            <div key={name} className="prop-row" style={{ alignItems: 'flex-start', gap: 8 }}>
              <span style={{ fontSize: 14, flexShrink: 0 }}>{icon}</span>
              <div>
                <div style={{ fontFamily: 'var(--mono)', fontSize: 9, color: 'var(--text-main)' }}>{name}</div>
                <div className="prop-label" style={{ fontSize: 9 }}>{desc}</div>
              </div>
            </div>
          ))}
        </div>

        {/* Object outliner */}
        <div className="panel-section" style={{ display:'flex', flexDirection:'column', minHeight:0, flexShrink: 0 }}>
          <div className="panel-section-title">
            <span>Objects ({shapes.length})</span>
            <div style={{ display: 'flex', gap: 6 }}>
              <Plus size={11} style={{ cursor: 'pointer' }} onClick={() => spawnShape('box')} title="Spawn box" />
              <Trash2 size={11} style={{ cursor: 'pointer', color: 'var(--accent-red)' }} onClick={clearScene} title="Clear scene" />
            </div>
          </div>
          <div className="scene-list" style={{ maxHeight: '220px' }}>
            {shapes.map(s => {
              const isGrabbed = handsRef.current.some(h => h.grabbedId === s.id);
              return (
                <div
                  key={s.id}
                  className="scene-item"
                  style={{
                    background:  selectedId === s.id ? 'var(--bg-hover)' : '',
                    color:       selectedId === s.id ? 'var(--text-main)' : '',
                    borderLeft:  isGrabbed ? '2px solid var(--secondary)' : '2px solid transparent',
                  }}
                  onClick={() => setSelectedId(prev => prev === s.id ? null : s.id)}
                >
                  <div className="scene-item-dot" style={{ background: s.color }} />
                  <span style={{ flex: 1 }}>{s.name}</span>
                  {isGrabbed && <span style={{ fontSize: 8, color: 'var(--secondary)' }}>HELD</span>}
                  <Trash2 size={10} style={{ opacity: .4, cursor: 'pointer' }}
                    onClick={e => { e.stopPropagation(); deleteShape(s.id); }} />
                </div>
              );
            })}
          </div>
        </div>

        {/* Properties */}
        <div className="panel-section">
          <div className="panel-section-title"><span>Properties</span></div>
          {selectedShape ? (
            <>
              <div className="prop-row"><span className="prop-label">Name</span><span className="prop-value">{selectedShape.name}</span></div>
              <div className="prop-row"><span className="prop-label">Type</span><span className="prop-value orange">{selectedShape.type.toUpperCase()}</span></div>
              {['x','y','z'].map((ax, i) => {
                const pm = posMap.current[selectedShape.id];
                const v  = pm ? [pm.current.x, pm.current.y, pm.current.z][i] : selectedShape.position[i];
                return <div key={ax} className="prop-row"><span className="prop-label">{ax.toUpperCase()}</span><span className="prop-value green">{v.toFixed(3)}</span></div>;
              })}
            </>
          ) : (
            <div className="prop-row"><span className="prop-label" style={{ color: 'var(--text-dim)' }}>None selected</span></div>
          )}
        </div>

        {/* Telemetry */}
        <div className="panel-section" style={{ borderBottom: 'none', borderTop: '1px solid var(--border)', marginTop: 'auto' }}>
          <div className="panel-section-title"><span>Telemetry</span></div>
          <div className="prop-row"><span className="prop-label">FPS</span><span className={`prop-value ${fps > 24 ? 'green' : fps > 12 ? 'orange' : 'red'}`}>{fps}</span></div>
          <div className="prop-row"><span className="prop-label">Mode</span><span className="prop-value blue">{spaceMode.toUpperCase()}</span></div>
          <div className="prop-row"><span className="prop-label">AI</span><span className="prop-value green">OLLAMA</span></div>
        </div>
      </aside>

      {/* ══ VIEWPORT ══════════════════════════════════════════════════════════ */}
      <main className="viewport">
        {/* Mirrored webcam feed */}
        {isCamOn && (
          <video
            ref={videoRef}
            autoPlay playsInline muted
            style={{
              position: 'absolute', inset: 0, width: '100%', height: '100%',
              objectFit: 'cover', opacity: 0.85,
              transform: 'scaleX(-1)', zIndex: 1
            }}
          />
        )}

        {/* 3D AR Canvas */}
        <div style={{ position: 'absolute', inset: 0, zIndex: 2 }}>
          <Canvas camera={{ position: [0, 1, 8], fov: WEBCAM_FOV }}>
            <Scene
              shapes={shapes}
              posMap={posMap}
              grabbedMap={grabbedMap}
              selectedIdRef={selectedIdRef}
              handsRef={handsRef}
              spaceMode={spaceMode}
              onSelect={id => setSelectedId(prev => prev === id ? null : id)}
            />
          </Canvas>
        </div>

        {/* HUD */}
        <div className="viewport-label" style={{ zIndex: 10 }}>
          {isCamOn && <span style={{ color:'var(--accent-green)', fontFamily:'var(--mono)', fontSize:10 }}>● LIVE</span>}
          {(handStatus[0]==='DETECTED'||handStatus[1]==='DETECTED') && <span style={{ color:'var(--secondary)', fontFamily:'var(--mono)', fontSize:10 }}>✋ HANDS TRACKED</span>}
          {(grabbedNames[0]||grabbedNames[1]) && <span style={{ color:'#f59e0b', fontFamily:'var(--mono)', fontSize:10 }}>🤏 GRABBING</span>}
          {spaceMode === 'room' && <span style={{ color:'#a78bfa', fontFamily:'var(--mono)', fontSize:10 }}>🏠 ROOM SPACE</span>}
        </div>

        {/* Env info overlay */}
        {envInfo && (
          <div style={{
            position: 'absolute', bottom: 10, left: 10, zIndex: 10,
            background: 'rgba(10,12,16,0.9)', border: '1px solid rgba(99,102,241,0.3)',
            borderRadius: 8, padding: '8px 12px', maxWidth: 300,
            fontFamily: 'var(--mono)', fontSize: 10, color: 'var(--text-muted)', lineHeight: 1.6,
          }}>
            <div style={{ color: 'var(--primary)', marginBottom: 4 }}>🌍 ENVIRONMENT</div>
            {envInfo.slice(0, 180)}{envInfo.length > 180 ? '…' : ''}
          </div>
        )}

        {/* Quick spawn buttons */}
        <div className="viewport-controls" style={{ zIndex: 10 }}>
          <button className={`vp-btn${isCamOn ? ' active' : ''}`} onClick={() => setIsCamOn(v => !v)}>
            {isCamOn ? <Video size={10} /> : <VideoOff size={10} />} {isCamOn ? 'CAM ON' : 'CAM OFF'}
          </button>
          <button className="vp-btn" onClick={scanEnvironment} title="Scan room with AI vision">
            <Scan size={10} /> SCAN ENV
          </button>
          {['box','sphere','cone','torus','cylinder'].map(t => (
            <button key={t} className="vp-btn" onClick={() => spawnShape(t)}>
              <Plus size={10} /> {t.slice(0,3).toUpperCase()}
            </button>
          ))}
        </div>
      </main>

      {/* ══ RIGHT PANEL ═══════════════════════════════════════════════════════ */}
      <aside className="panel-right">
        <div className="panel-header"><Terminal size={12} className="panel-header-icon" /> AI Console · Ollama Local</div>
        <div className="log-scroll" ref={logScrollRef}>
          <AnimatePresence initial={false}>
            {logs.map((log, i) => (
              <motion.div
                key={i}
                initial={{ opacity: 0, y: 6 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.1 }}
                className={`log-entry ${log.sender}`}
              >
                <div className="log-sender">
                  {log.sender.toUpperCase()} · {log.time.toLocaleTimeString([], { hour:'2-digit', minute:'2-digit', second:'2-digit' })}
                </div>
                {log.text}
              </motion.div>
            ))}
          </AnimatePresence>
          {loading && (
            <div className="log-entry system" style={{ color: 'var(--primary)' }}>
              <div className="log-sender">OLLAMA</div>
              Thinking…
            </div>
          )}
        </div>
      </aside>

      {/* ══ BOTTOM BAR ════════════════════════════════════════════════════════ */}
      <div className="bottom-bar">
        <div className="cam-controls">
          <button className="vp-btn" onClick={startTour} style={{ background: 'var(--primary)', color: 'white' }}>
            <Play size={10} /> START TRAINING
          </button>
          <span style={{ fontFamily:'var(--mono)', fontSize:10, color:'var(--text-dim)', marginLeft: 12 }}>CAM</span>
          <button className={`icon-btn${isCamOn ? ' cam-on' : ''}`} onClick={() => setIsCamOn(v => !v)} title="Toggle Camera">
            {isCamOn ? <Video size={14}/> : <VideoOff size={14}/>}
          </button>
          <button id="mic-btn" className={`icon-btn${isMicOn ? ' mic-on' : ''}`} onClick={() => setIsMicOn(v => !v)} title="Toggle Microphone" style={isCamOn ? {} : { opacity: 0.4 }}>
            {isMicOn ? <Mic size={14}/> : <MicOff size={14}/>}
          </button>
          <button className="icon-btn" onClick={scanEnvironment} title="Scan Environment" style={isCamOn ? {} : { opacity: 0.4 }}>
            <Scan size={14}/>
          </button>
        </div>

        <div className="input-section">
          <form onSubmit={handleSubmit} style={{ display:'flex', flex:1, gap:8, alignItems:'center' }}>
            <span style={{ fontFamily:'var(--mono)', fontSize:10, color: isMicOn ? 'var(--secondary)' : 'var(--text-muted)', whiteSpace:'nowrap' }}>
              {isMicOn ? '🔴 VOICE →' : 'OLLAMA ›'}
            </span>
            <input
              className={`main-input${isMicOn ? ' mic-active' : ''}`}
              type="text"
              value={inputText}
              onChange={e => setInputText(e.target.value)}
              onKeyDown={e => e.key === 'Enter' && handleSubmit(e)}
              placeholder={isMicOn ? 'Listening… speak your command' : 'Ask Ollama AI or describe what to build…'}
              disabled={loading}
            />
            <button type="submit" className="send-btn" disabled={loading}>
              {loading ? <Loader size={12} style={{ animation:'spin 1s linear infinite' }}/> : <Send size={12}/>}
              {loading ? 'THINKING' : 'SEND'}
            </button>
          </form>
        </div>

        <div className="mic-controls">
          <span style={{ fontFamily:'var(--mono)', fontSize:10, color:'var(--text-muted)' }}>MIC</span>
          <button className={`icon-btn${isMicOn ? ' mic-on' : ''}`} onClick={() => setIsMicOn(v => !v)} title="Toggle Mic">
            {isMicOn ? <Mic size={14}/> : <MicOff size={14}/>}
            {isMicOn && <div className="mic-pulse"/>}
          </button>
          <button className="icon-btn spawn" onClick={() => spawnShape('box')} title="Spawn box">
            <Plus size={14}/>
          </button>
          <button
            className="icon-btn"
            style={{ borderColor:'var(--accent-red)', color:'var(--accent-red)' }}
            onClick={() => selectedId ? deleteShape(selectedId) : addLog('system','Select an object first.')}
            title="Delete selected"
          >
            <Trash2 size={14}/>
          </button>
        </div>
      </div>

      <style>{`
        @keyframes spin { to { transform: rotate(360deg); } }
        .blue { color: var(--accent-blue) !important; }
      `}</style>
    </div>
  );
}
