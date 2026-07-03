import React from 'react';
import {AbsoluteFill, useCurrentFrame} from 'remotion';
import {COLORS, FONT, MONO, prog} from '../theme';
import {
  ChatBubble,
  EmbeddingPlane,
  EmbPoint,
  DecayCurve,
  Caption,
  TimeSlider,
  WriteStamp,
} from '../components';

/**
 * SCENE 1 — "The Flat Database" (cold open, ~30s @ 30fps = 900 frames)
 *
 * Beat 1  (0.0s)  Two messages arrive: angry (volatile) + vegetarian (durable).
 * Beat 2  (4.0s)  Both embed into the same plane as identical points; WRITE t=0.
 * Beat 3  (9.0s)  Time slider sweeps forward — the world changes, the points don't.
 * Beat 4  (15.0s) Half-life curves: angry decays in minutes, vegetarian in years;
 *                 the DB stores both at a flat 1.0 forever.
 * Beat 5  (23.0s) Punch: "The store is flat. It has no time axis."
 */
export const IntroFlatDatabase: React.FC = () => {
  const frame = useCurrentFrame();

  const points: EmbPoint[] = [
    {id: 'angry', cx: 0.32, cy: 0.4, color: COLORS.red, label: '"angry"', landFrame: 150},
    {id: 'veg', cx: 0.66, cy: 0.62, color: COLORS.green, label: '"vegetarian"', landFrame: 165},
  ];

  // Curve panel geometry
  const cx = 250;
  const cy = 470;
  const cw = 620;
  const ch = 230;

  return (
    <AbsoluteFill style={{background: COLORS.bg}}>
      {/* Beat 1 — the two messages */}
      <ChatBubble
        text="ugh, I am SO angry right now"
        accent={COLORS.red}
        x={170}
        y={150}
        startFrame={10}
      />
      <ChatBubble
        text="btw I'm vegetarian"
        accent={COLORS.green}
        x={170}
        y={300}
        startFrame={55}
      />

      {/* Beat 2 — embedding plane (right side) + write stamps */}
      <EmbeddingPlane x={1010} y={210} w={720} h={520} points={points} appearFrame={120} />
      {frame >= 190 && <WriteStamp x={1010} y={150} startFrame={190} text='WRITE  "angry"   @ t=0' />}
      {frame >= 205 && <WriteStamp x={1390} y={150} startFrame={205} text='WRITE  "vegetarian"  @ t=0' />}
      <Caption
        text="Same operation. Same timestamp. The store treats them identically."
        startFrame={230}
        endFrame={290}
      />

      {/* Beat 3 — time advances; the world changes, the points don't */}
      {frame >= 270 && (
        <TimeSlider
          x={1010}
          y={770}
          w={720}
          startFrame={285}
          endFrame={430}
          labels={['now', '5 min later', '1 hour later', '6 months later']}
        />
      )}
      <Caption
        text="Five minutes later you've calmed down. You're still vegetarian."
        startFrame={320}
        endFrame={420}
      />

      {/* Beat 4 — half-life curves */}
      {frame >= 430 && (
        <svg width={1920} height={1080} style={{position: 'absolute', inset: 0}}>
          {/* axes */}
          <line x1={cx} y1={cy} x2={cx + cw} y2={cy} stroke={COLORS.panelStroke} strokeWidth={2} />
          <line x1={cx} y1={cy} x2={cx} y2={cy - ch} stroke={COLORS.panelStroke} strokeWidth={2} />
          <text x={cx - 10} y={cy - ch - 12} fill={COLORS.inkDim} fontFamily={MONO} fontSize={18}>
            how true is this fact?
          </text>
          <text x={cx + cw - 60} y={cy + 30} fill={COLORS.inkDim} fontFamily={MONO} fontSize={18}>
            time →
          </text>
          {/* reality: two very different half-lives */}
          <DecayCurve x={cx} y={cy - ch} w={cw} h={ch} color={COLORS.red} halfLifeFrac={0.06} label="anger · half-life: minutes" startFrame={450} />
          <DecayCurve x={cx} y={cy - ch} w={cw} h={ch} color={COLORS.green} halfLifeFrac={1.4} label="diet · half-life: years" startFrame={520} />
          {/* the database's actual behavior: flat at 1.0 forever */}
          <DecayCurve x={cx} y={cy - ch} w={cw} h={ch} color={COLORS.yellow} halfLifeFrac={1} label="what the vector DB stores" startFrame={610} flat />
        </svg>
      )}
      <Caption
        text="Every fact, stored at full confidence, forever."
        startFrame={640}
        endFrame={720}
        color={COLORS.yellow}
        bold
      />

      {/* Beat 5 — punch line */}
      {frame >= 730 && (
        <AbsoluteFill style={{justifyContent: 'center', alignItems: 'center'}}>
          <div
            style={{
              opacity: prog(frame, 740, 770),
              color: COLORS.ink,
              fontFamily: FONT,
              fontSize: 64,
              fontWeight: 800,
              textAlign: 'center',
              lineHeight: 1.2,
              transform: `translateY(${(1 - prog(frame, 740, 775)) * 24}px)`,
            }}
          >
            The store is <span style={{color: COLORS.yellow}}>flat</span>.<br />
            It has no time axis.
          </div>
        </AbsoluteFill>
      )}
    </AbsoluteFill>
  );
};
