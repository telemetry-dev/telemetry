import { setTimeout as sleep } from "node:timers/promises";

import { HIDE_CURSOR, motionEnabled, SHOW_CURSOR, style } from "./terminal.ts";

// The telemetry.dev mark is a sphere cut into horizontal bands, so each band is one
// terminal row drawn as a three-quarter block from the bottom of its cells; the empty
// top quarters are the gaps, close to the logo's band-to-gap ratio. Shorter blocks
// thin the domed cap, the dip in the second band, the pill at the bottom, and the
// rounded band ends.
const MARK = [
  "   ▃▄▅▆▆▆▆▅▄▃",
  " ▅▆▆▆▅▃▃▃▃▅▆▆▆▅",
  "▅▆▆▆▆      ▆▆▆▆▅",
  "▆▆▆▆▆▆▆▆▆▆▆▆▆▆▆▆",
  "▅▆▆▆▆▆▆▆▆▆▆▆▆▆▆▅",
  " ▅▆▆▆▆▆▆▆▆▆▆▆▆▅",
  "   ▅▆▆▆▆▆▆▆▆▅",
  "      ▃▄▄▃",
] as const;

const MARK_WIDTH = 16;
const CENTER = MARK_WIDTH / 2;
const WORDMARK = "telemetry.dev";
const WORDMARK_ROW = 3;
const GUTTER = "    ";

const BAND_STAGGER_MS = 45;
const BAND_GROW_MS = 260;
const WORDMARK_START_MS = 300;
const LETTER_MS = 22;
const TAGLINE_START_MS = WORDMARK_START_MS + WORDMARK.length * LETTER_MS + 40;
const DURATION_MS = Math.max((MARK.length - 1) * BAND_STAGGER_MS + BAND_GROW_MS, TAGLINE_START_MS);
const FRAME_MS = 16;

const easeOut = (progress: number) => 1 - (1 - progress) ** 3;

const clamp = (value: number) => Math.min(1, Math.max(0, value));

/**
 * One frame of the intro at `elapsed` ms: each band grows outward from the center,
 * top to bottom, then the wordmark types in beside it. At DURATION_MS and after, this
 * is the finished lockup.
 */
const frame = (elapsed: number, tagline: string, stream: NodeJS.WriteStream) =>
  MARK.map((band, row) => {
    const progress = easeOut(clamp((elapsed - row * BAND_STAGGER_MS) / BAND_GROW_MS));
    const reach = progress * CENTER;

    let drawn = "";

    // Every glyph in the mark is one UTF-16 unit and one column wide.
    for (let column = 0; column < MARK_WIDTH; column += 1) {
      const glyph = band.charAt(column) || " ";
      const distance = Math.abs(column + 0.5 - CENTER);

      if (glyph === " " || distance > reach + 0.01) {
        drawn += " ";
      } else {
        // The leading edge of a band that is still growing draws dimmer.
        drawn += progress < 1 && reach - distance < 1.5 ? style.dim(glyph, stream) : glyph;
      }
    }

    if (row === WORDMARK_ROW) {
      const letters = Math.floor(
        clamp((elapsed - WORDMARK_START_MS) / (WORDMARK.length * LETTER_MS)) * WORDMARK.length,
      );

      return `${drawn}${GUTTER}${style.bold(WORDMARK.slice(0, letters), stream)}`;
    }

    if (row === WORDMARK_ROW + 1 && elapsed >= TAGLINE_START_MS) {
      return `${drawn}${GUTTER}${style.dim(tagline, stream)}`;
    }

    return drawn;
  }).map((line) => line.trimEnd());

/**
 * Prints the mark and wordmark. On a terminal it animates once, in about two thirds
 * of a second; elsewhere, or with motion off, it prints the finished lockup.
 */
export const printMark = async (
  tagline: string,
  { animate, stream = process.stdout }: { animate: boolean; stream?: NodeJS.WriteStream },
) => {
  if (!animate || !motionEnabled(stream)) {
    stream.write(`${frame(DURATION_MS, tagline, stream).join("\n")}\n`);

    return;
  }

  // A Ctrl-C mid-animation must not leave the cursor hidden.
  const restore = () => stream.write(SHOW_CURSOR);
  process.once("exit", restore);
  stream.write(HIDE_CURSOR);

  const start = performance.now();
  let first = true;

  for (;;) {
    const elapsed = performance.now() - start;
    const lines = frame(elapsed, tagline, stream);

    if (!first) stream.write(`\u001b[${MARK.length}A`);
    first = false;
    stream.write(lines.map((line) => `\r\u001b[2K${line}`).join("\n") + "\n");

    if (elapsed >= DURATION_MS) break;
    await sleep(FRAME_MS);
  }

  stream.write(SHOW_CURSOR);
  process.removeListener("exit", restore);
};
