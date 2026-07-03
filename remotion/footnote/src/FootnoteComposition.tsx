import React from 'react';
import {AbsoluteFill, OffthreadVideo, Sequence, useVideoConfig} from 'remotion';
import {FactCard, OverlayPayload} from './FactCard';

export interface FootnoteProps {
  videoSrc: string;
  overlays: OverlayPayload[];
}

export const FootnoteComposition: React.FC<FootnoteProps> = ({videoSrc, overlays}) => {
  const {fps} = useVideoConfig();

  return (
    <AbsoluteFill style={{backgroundColor: 'black'}}>
      {/* Underlying video */}
      {videoSrc && (
        <OffthreadVideo src={videoSrc} />
      )}

      {/* Overlays — each in its own Sequence so it appears at the right time */}
      {overlays.map((overlay) => {
        const startFrame = Math.round(overlay.timestamp_sec * fps);
        const durationFrames = Math.round(overlay.duration_sec * fps);
        return (
          <Sequence
            key={overlay.id}
            from={startFrame}
            durationInFrames={durationFrames}
          >
            <FactCard overlay={overlay} />
          </Sequence>
        );
      })}
    </AbsoluteFill>
  );
};
