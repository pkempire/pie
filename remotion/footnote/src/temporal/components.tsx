import React from 'react';
import {interpolate, spring, useCurrentFrame, useVideoConfig} from 'remotion';
import {COLORS, FONT, MONO, smooth, prog, fadeWindow} from './theme';

/* ------------------------------------------------------------------ */
/* Chat bubble — a message that fades/rises in.                        */
/* ------------------------------------------------------------------ */
export const ChatBubble: React.FC<{
  text: string;
  accent: string;
  x: number;
  y: number;
  startFrame: number;
  width?: number;
}> = ({text, accent, x, y, startFrame, width = 460}) => {
  const frame = useCurrentFrame();
  const {fps} = useVideoConfig();
  const s = spring({frame: frame - startFrame, fps, config: {damping: 16}});
  const op = fadeWindow(frame, startFrame, startFrame + 8);
  const rise = interpolate(s, [0, 1], [28, 0]);
  return (
    <div
      style={{
        position: 'absolute',
        left: x,
        top: y,
        width,
        opacity: op,
        transform: `translateY(${rise}px)`,
        background: COLORS.panel,
        border: `2px solid ${accent}`,
        borderRadius: 18,
        padding: '20px 26px',
        color: COLORS.ink,
        fontFamily: FONT,
        fontSize: 30,
        lineHeight: 1.35,
        boxShadow: `0 0 40px ${accent}22`,
      }}
    >
      <span style={{color: accent, fontFamily: MONO, fontSize: 18}}>user</span>
      <div style={{marginTop: 6}}>{text}</div>
    </div>
  );
};

/* ------------------------------------------------------------------ */
/* Embedding plane — grid + points that spring in and can decay.       */
/* ------------------------------------------------------------------ */
export type EmbPoint = {
  id: string;
  cx: number; // 0..1 within plane
  cy: number; // 0..1
  color: string;
  label: string;
  landFrame: number;
  /** 0..1 brightness multiplier (decay handled by caller). */
  intensity?: number;
};

export const EmbeddingPlane: React.FC<{
  x: number;
  y: number;
  w: number;
  h: number;
  points: EmbPoint[];
  appearFrame: number;
  title?: string;
}> = ({x, y, w, h, points, appearFrame, title = 'embedding space'}) => {
  const frame = useCurrentFrame();
  const {fps} = useVideoConfig();
  const planeOp = fadeWindow(frame, appearFrame, appearFrame + 12);
  const lines = 8;
  return (
    <div style={{position: 'absolute', left: x, top: y, width: w, height: h, opacity: planeOp}}>
      <svg width={w} height={h} style={{position: 'absolute', inset: 0}}>
        <rect x={0} y={0} width={w} height={h} rx={14} fill={COLORS.panel} stroke={COLORS.panelStroke} strokeWidth={2} />
        {Array.from({length: lines + 1}).map((_, i) => (
          <line key={`v${i}`} x1={(i / lines) * w} y1={0} x2={(i / lines) * w} y2={h} stroke={COLORS.bgGrid} strokeWidth={1} />
        ))}
        {Array.from({length: lines + 1}).map((_, i) => (
          <line key={`h${i}`} x1={0} y1={(i / lines) * h} x2={w} y2={(i / lines) * h} stroke={COLORS.bgGrid} strokeWidth={1} />
        ))}
        {points.map((p) => {
          const s = spring({frame: frame - p.landFrame, fps, config: {damping: 14, stiffness: 120}});
          const r = interpolate(s, [0, 1], [0, 13]);
          const intensity = p.intensity ?? 1;
          const px = p.cx * w;
          const py = p.cy * h;
          return (
            <g key={p.id} opacity={Math.max(0.12, intensity)}>
              <circle cx={px} cy={py} r={r + 10} fill={p.color} opacity={0.18 * intensity} />
              <circle cx={px} cy={py} r={r} fill={p.color} />
              <text x={px + 18} y={py + 6} fill={COLORS.ink} fontFamily={MONO} fontSize={20} opacity={intensity}>
                {p.label}
              </text>
            </g>
          );
        })}
      </svg>
      <div style={{position: 'absolute', top: -34, left: 4, color: COLORS.inkDim, fontFamily: MONO, fontSize: 18}}>
        {title}
      </div>
    </div>
  );
};

/* ------------------------------------------------------------------ */
/* Decay curve — exponential half-life, animated draw + moving marker. */
/* ------------------------------------------------------------------ */
export const DecayCurve: React.FC<{
  x: number;
  y: number;
  w: number;
  h: number;
  color: string;
  halfLifeFrac: number; // half-life as fraction of x-axis (small = fast decay)
  label: string;
  startFrame: number;
  flat?: boolean; // draw a flat line at 1.0 instead (the DB's behavior)
}> = ({x, y, w, h, color, halfLifeFrac, label, startFrame, flat = false}) => {
  const frame = useCurrentFrame();
  const draw = prog(frame, startFrame, startFrame + 40);
  const N = 80;
  const pts: string[] = [];
  const lastIdx = Math.floor(draw * N);
  for (let i = 0; i <= lastIdx; i++) {
    const t = i / N;
    const val = flat ? 1 : Math.pow(2, -t / halfLifeFrac);
    pts.push(`${x + t * w},${y + (1 - val) * h}`);
  }
  const op = fadeWindow(frame, startFrame, startFrame + 10);
  return (
    <g opacity={op}>
      <polyline points={pts.join(' ')} fill="none" stroke={color} strokeWidth={4} strokeLinecap="round" />
      {!flat && lastIdx > 2 && (
        <circle
          cx={x + (lastIdx / N) * w}
          cy={y + (1 - Math.pow(2, -(lastIdx / N) / halfLifeFrac)) * h}
          r={6}
          fill={color}
        />
      )}
      <text x={x + w + 14} y={y + (flat ? 6 : (1 - 0.5) * h)} fill={color} fontFamily={MONO} fontSize={20}>
        {label}
      </text>
    </g>
  );
};

/* ------------------------------------------------------------------ */
/* Caption — lower-third line that fades in/out.                       */
/* ------------------------------------------------------------------ */
export const Caption: React.FC<{
  text: string;
  startFrame: number;
  endFrame?: number;
  y?: number;
  size?: number;
  color?: string;
  bold?: boolean;
}> = ({text, startFrame, endFrame, y = 880, size = 40, color = COLORS.ink, bold = false}) => {
  const frame = useCurrentFrame();
  const op = fadeWindow(frame, startFrame, startFrame + 12, endFrame ? endFrame - 12 : Infinity, endFrame ?? Infinity);
  return (
    <div
      style={{
        position: 'absolute',
        left: 0,
        right: 0,
        top: y,
        textAlign: 'center',
        opacity: op,
        color,
        fontFamily: FONT,
        fontSize: size,
        fontWeight: bold ? 700 : 400,
        letterSpacing: 0.2,
        padding: '0 160px',
      }}
    >
      {text}
    </div>
  );
};

/* ------------------------------------------------------------------ */
/* Time slider — a track + handle + advancing label.                   */
/* ------------------------------------------------------------------ */
export const TimeSlider: React.FC<{
  x: number;
  y: number;
  w: number;
  startFrame: number;
  endFrame: number;
  labels: string[]; // shown across the sweep
}> = ({x, y, w, startFrame, endFrame, labels}) => {
  const frame = useCurrentFrame();
  const p = interpolate(frame, [startFrame, endFrame], [0, 1], {
    extrapolateLeft: 'clamp',
    extrapolateRight: 'clamp',
    easing: smooth,
  });
  const op = fadeWindow(frame, startFrame, startFrame + 10);
  const idx = Math.min(labels.length - 1, Math.floor(p * labels.length));
  return (
    <div style={{position: 'absolute', left: x, top: y, width: w, opacity: op}}>
      <div style={{height: 6, background: COLORS.panelStroke, borderRadius: 3, position: 'relative'}}>
        <div style={{height: 6, width: `${p * 100}%`, background: COLORS.yellow, borderRadius: 3}} />
        <div
          style={{
            position: 'absolute',
            left: `${p * 100}%`,
            top: -9,
            width: 22,
            height: 22,
            borderRadius: 11,
            background: COLORS.yellow,
            transform: 'translateX(-50%)',
            boxShadow: `0 0 20px ${COLORS.yellow}88`,
          }}
        />
      </div>
      <div style={{marginTop: 16, color: COLORS.yellow, fontFamily: MONO, fontSize: 24, textAlign: 'center'}}>
        {labels[idx]}
      </div>
    </div>
  );
};

/* ------------------------------------------------------------------ */
/* Write stamp — "WRITE ✓ t=0" badge that pops onto a point.           */
/* ------------------------------------------------------------------ */
export const WriteStamp: React.FC<{
  x: number;
  y: number;
  startFrame: number;
  text?: string;
}> = ({x, y, startFrame, text = 'WRITE ✓  t=0'}) => {
  const frame = useCurrentFrame();
  const {fps} = useVideoConfig();
  const s = spring({frame: frame - startFrame, fps, config: {damping: 10, stiffness: 200}});
  const scale = interpolate(s, [0, 1], [1.6, 1]);
  const op = fadeWindow(frame, startFrame, startFrame + 6);
  return (
    <div
      style={{
        position: 'absolute',
        left: x,
        top: y,
        opacity: op,
        transform: `scale(${scale})`,
        transformOrigin: 'left center',
        color: COLORS.green,
        border: `2px solid ${COLORS.green}`,
        borderRadius: 8,
        padding: '4px 12px',
        fontFamily: MONO,
        fontSize: 20,
        background: COLORS.panel,
      }}
    >
      {text}
    </div>
  );
};
