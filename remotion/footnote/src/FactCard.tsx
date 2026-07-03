import React from 'react';
import {AbsoluteFill, interpolate, spring, useCurrentFrame, useVideoConfig, Img} from 'remotion';

export interface OverlayPayload {
  id: string;
  timestamp_sec: number;
  duration_sec: number;
  type: 'fact-card' | 'person-card' | 'image' | 'chart';
  text: string;
  citation_url: string;
  citation_label: string;
  image_url?: string | null;
  bbox: [number, number, number, number];   // x, y, w, h in 0-1 normalized
  entry_animation: 'fade' | 'slide-up' | 'slide-down' | 'slide-left' | 'slide-right';
  exit_animation: 'fade' | 'slide-up' | 'slide-down' | 'slide-right';
}

interface Props {
  overlay: OverlayPayload;
}

export const FactCard: React.FC<Props> = ({overlay}) => {
  const frame = useCurrentFrame();
  const {fps, width: vidW, height: vidH, durationInFrames} = useVideoConfig();
  const totalFrames = Math.round(overlay.duration_sec * fps);

  // Spring-based entry over first 0.4s, exit over last 0.4s
  const entryFrames = Math.round(0.4 * fps);
  const exitFrames = Math.round(0.4 * fps);
  const localFrame = frame;

  const entryProgress = spring({frame: localFrame, fps, config: {damping: 18}});
  const inExit = localFrame > totalFrames - exitFrames;
  const exitProgress = inExit
    ? interpolate(localFrame, [totalFrames - exitFrames, totalFrames], [1, 0], {
        extrapolateLeft: 'clamp',
        extrapolateRight: 'clamp',
      })
    : 1;
  const visibility = Math.min(entryProgress, exitProgress);

  const [x, y, w, h] = overlay.bbox;
  const px = x * vidW;
  const py = y * vidH;
  const pw = w * vidW;
  const ph = h * vidH;

  // Animation transforms based on entry type
  let entryTransform = 'none';
  if (overlay.entry_animation === 'slide-up') {
    const offset = interpolate(entryProgress, [0, 1], [40, 0]);
    entryTransform = `translateY(${offset}px)`;
  } else if (overlay.entry_animation === 'slide-down') {
    const offset = interpolate(entryProgress, [0, 1], [-40, 0]);
    entryTransform = `translateY(${offset}px)`;
  } else if (overlay.entry_animation === 'slide-left') {
    const offset = interpolate(entryProgress, [0, 1], [80, 0]);
    entryTransform = `translateX(${offset}px)`;
  } else if (overlay.entry_animation === 'slide-right') {
    const offset = interpolate(entryProgress, [0, 1], [-80, 0]);
    entryTransform = `translateX(${offset}px)`;
  }

  const hasImage = Boolean(overlay.image_url);

  return (
    <AbsoluteFill style={{pointerEvents: 'none'}}>
      <div
        style={{
          position: 'absolute',
          left: px,
          top: py,
          width: pw,
          height: ph,
          opacity: visibility,
          transform: entryTransform,
          display: 'flex',
          flexDirection: 'row',
          background: 'rgba(255, 255, 255, 0.97)',
          borderRadius: 16,
          boxShadow: '0 12px 36px rgba(0, 0, 0, 0.25)',
          fontFamily: 'system-ui, -apple-system, "Segoe UI", sans-serif',
          overflow: 'hidden',
        }}
      >
        {/* Body text */}
        <div
          style={{
            flex: 1,
            padding: '20px 24px',
            display: 'flex',
            flexDirection: 'column',
            justifyContent: 'center',
          }}
        >
          <div
            style={{
              fontSize: 28,
              lineHeight: 1.35,
              color: '#1a1a1a',
              fontWeight: 500,
            }}
          >
            {overlay.text}
          </div>
          <div
            style={{
              marginTop: 12,
              fontSize: 18,
              color: '#888',
              display: 'flex',
              alignItems: 'center',
              gap: 6,
            }}
          >
            <span
              style={{
                padding: '2px 10px',
                background: '#f0f0f0',
                borderRadius: 6,
                color: '#444',
              }}
            >
              {overlay.citation_label}
            </span>
          </div>
        </div>

        {/* Optional image — sized to overlay height, square aspect */}
        {hasImage && (
          <div
            style={{
              width: ph,
              height: ph,
              flexShrink: 0,
              background: '#eee',
              overflow: 'hidden',
            }}
          >
            <Img
              src={overlay.image_url as string}
              style={{width: '100%', height: '100%', objectFit: 'cover'}}
            />
          </div>
        )}
      </div>
    </AbsoluteFill>
  );
};
