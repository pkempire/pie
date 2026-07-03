import React from 'react';
import {AbsoluteFill, Sequence} from 'remotion';
import {COLORS} from './theme';
import {IntroFlatDatabase} from './scenes/IntroFlatDatabase';

/**
 * "The Clock You Cannot Feel" — Working Memory explainer.
 * 30 fps, 1920x1080. Scenes are stitched as Sequences so each owns its own
 * local frame clock (a scene's animations start at frame 0 inside it).
 *
 * Scene durations (frames @30fps):
 *   S1 Intro / flat database ...... 0    -> 900   (built)
 *   S2 Reasoning vs awareness ..... 900  -> 1620  (storyboarded)
 *   S3 Structural blindness ....... 1620 -> 2310  (storyboarded)
 *   S4 The receipts (65/4/11) ..... 2310 -> 3000  (storyboarded)
 *   S5 Urgency beats countdown .... 3000 -> 3660  (storyboarded)
 *   S6 Three layers of the fix .... 3660 -> 4560  (storyboarded)
 *   S7 Thesis / close ............. 4560 -> 5100  (storyboarded)
 */
export const TOTAL_FRAMES = 900; // grows as scenes are added

export const TemporalExplainer: React.FC = () => {
  return (
    <AbsoluteFill style={{background: COLORS.bg}}>
      <Sequence from={0} durationInFrames={900}>
        <IntroFlatDatabase />
      </Sequence>
      {/* Add S2..S7 here as they're built:
          <Sequence from={900} durationInFrames={720}><ReasoningVsAwareness /></Sequence>
          ... */}
    </AbsoluteFill>
  );
};
