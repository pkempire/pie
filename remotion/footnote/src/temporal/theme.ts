/**
 * 3Blue1Brown-style visual language for the "Working Memory" explainers.
 * Deep navy ground, restrained accent palette, generous whitespace, eased motion.
 */
import {interpolate, Easing} from 'remotion';

export const COLORS = {
  bg: '#0b1021',          // 3b1b deep navy
  bgGrid: '#1b2340',      // faint grid lines
  ink: '#e8ecf6',         // near-white text
  inkDim: '#8b93ad',      // captions / secondary
  blue: '#5b9bff',        // 3b1b signature blue
  yellow: '#ffd166',      // highlight / "the point"
  green: '#4ade80',       // durable / vegetarian
  red: '#ff6b6b',         // volatile / angry
  teal: '#2dd4bf',
  purple: '#a78bfa',
  panel: '#121a33',
  panelStroke: '#26304f',
};

export const FONT =
  '"Inter", "SF Pro Display", -apple-system, "Segoe UI", system-ui, sans-serif';
export const MONO =
  '"Berkeley Mono", "JetBrains Mono", "SF Mono", ui-monospace, monospace';

/** Smooth in/out used everywhere — matches Manim's default smooth(). */
export const smooth = Easing.bezier(0.45, 0, 0.55, 1);
export const easeOut = Easing.out(Easing.cubic);

/** Fade a value in over [a,b] and (optionally) out over [c,d] in frames. */
export function fadeWindow(
  frame: number,
  a: number,
  b: number,
  c = Infinity,
  d = Infinity,
): number {
  const inP = interpolate(frame, [a, b], [0, 1], {
    extrapolateLeft: 'clamp',
    extrapolateRight: 'clamp',
    easing: smooth,
  });
  const outP =
    c === Infinity
      ? 1
      : interpolate(frame, [c, d], [1, 0], {
          extrapolateLeft: 'clamp',
          extrapolateRight: 'clamp',
          easing: smooth,
        });
  return Math.min(inP, outP);
}

/** Eased 0->1 progress over [a,b]. */
export function prog(frame: number, a: number, b: number): number {
  return interpolate(frame, [a, b], [0, 1], {
    extrapolateLeft: 'clamp',
    extrapolateRight: 'clamp',
    easing: smooth,
  });
}

export const SECONDS = (s: number, fps: number) => Math.round(s * fps);
