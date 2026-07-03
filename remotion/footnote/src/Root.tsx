import React from 'react';
import {Composition, getInputProps} from 'remotion';
import {FootnoteComposition, FootnoteProps} from './FootnoteComposition';
import {TemporalExplainer, TOTAL_FRAMES} from './temporal/TemporalExplainer';
import {getVideoMetadata} from '@remotion/media-utils';

// Default props for studio preview. Real renders pass --props=footnote_props.json
const DEFAULT_PROPS: FootnoteProps = {
  videoSrc: '',
  overlays: [],
};

// We don't know the video duration until render time. Default to 10 min @ 30fps.
const DEFAULT_FPS = 30;
const DEFAULT_DURATION_FRAMES = 10 * 60 * DEFAULT_FPS;

export const RemotionRoot: React.FC = () => {
  const inputProps = getInputProps() as Partial<FootnoteProps>;
  const props: FootnoteProps = {
    videoSrc: inputProps.videoSrc || DEFAULT_PROPS.videoSrc,
    overlays: inputProps.overlays || DEFAULT_PROPS.overlays,
  };

  // Compute composition length: latest overlay end + tail OR video duration.
  const latestOverlayEnd = props.overlays.reduce((acc, o) => {
    return Math.max(acc, o.timestamp_sec + o.duration_sec);
  }, 0);
  const durationFrames = Math.max(
    DEFAULT_DURATION_FRAMES,
    Math.ceil((latestOverlayEnd + 10) * DEFAULT_FPS),
  );

  return (
    <>
      <Composition
        id="FootnoteComposition"
        component={FootnoteComposition}
        durationInFrames={durationFrames}
        fps={DEFAULT_FPS}
        width={1920}
        height={1080}
        defaultProps={props}
      />
      <Composition
        id="TemporalExplainer"
        component={TemporalExplainer}
        durationInFrames={TOTAL_FRAMES}
        fps={30}
        width={1920}
        height={1080}
      />
    </>
  );
};
