import { stripVTControlCharacters } from "node:util";

type Stream = NodeJS.WriteStream;

let colorDisabled = false;

export const disableColor = () => {
  colorDisabled = true;
};

/** Color follows NO_COLOR, FORCE_COLOR, TERM=dumb, --no-color, and whether the stream is a terminal. */
export const colorEnabled = (stream: Stream) => {
  if (colorDisabled) return false;

  const force = process.env.FORCE_COLOR;

  if (force !== undefined && force !== "") return force !== "0" && force !== "false";

  if (process.env.NO_COLOR) return false;

  if (process.env.TERM === "dumb") return false;

  return stream.isTTY === true;
};

/** Motion only plays for a person: a terminal, outside CI, with color on. */
export const motionEnabled = (stream: Stream) =>
  stream.isTTY === true && !process.env.CI && colorEnabled(stream);

const sgr =
  (open: number, close: number) =>
  (text: string, stream: Stream = process.stdout) =>
    colorEnabled(stream) && text !== "" ? `\u001b[${open}m${text}\u001b[${close}m` : text;

export const style = {
  bold: sgr(1, 22),
  dim: sgr(2, 22),
  red: sgr(31, 39),
  green: sgr(32, 39),
  yellow: sgr(33, 39),
  blue: sgr(34, 39),
  magenta: sgr(35, 39),
  cyan: sgr(36, 39),
};

export type Paint = (text: string, stream?: Stream) => string;

/** Columns available on a terminal; unlimited when the output goes to a file or pipe. */
export const outputWidth = (stream: Stream = process.stdout) =>
  stream.isTTY ? Math.max(40, stream.columns || 80) : Number.POSITIVE_INFINITY;

// Captured text comes from the user's applications, models, and tools. Terminal escape
// sequences in it could rewrite the screen, set the clipboard (OSC 52), or forge links,
// and bidirectional controls can hide text from the reader, so none reach the terminal.
const CONTROL = new RegExp(
  String.raw`[\u0000-\u0008\u000b-\u001f\u007f-\u009f\u061c\u200e\u200f\u202a-\u202e\u2066-\u2069]`,
  "g",
);

/** Captured text, safe to print: escape sequences and control characters removed. */
export const clean = (text: string) => stripVTControlCharacters(text).replace(CONTROL, "");

/** Captured text on one line, for a table cell. */
export const oneLine = (text: string) => clean(text.replace(/\s+/g, " ")).trim();

const segmenter = new Intl.Segmenter();

// East Asian wide and fullwidth ranges and emoji take two columns in most terminals.
// Terminals disagree on some emoji sequences, so this is a close estimate, which is
// all column alignment needs.
const WIDE =
  /^(?:\p{Extended_Pictographic}|[\u1100-\u115f\u2e80-\u303e\u3041-\u33ff\u3400-\u4dbf\u4e00-\u9fff\ua000-\ua4cf\uac00-\ud7a3\uf900-\ufaff\ufe30-\ufe4f\uff00-\uff60\uffe0-\uffe6]|[\u{20000}-\u{3fffd}])/u;

const graphemeWidth = (grapheme: string) => (WIDE.test(grapheme) ? 2 : 1);

/** Display width in terminal columns of text without escape sequences. */
export const displayWidth = (text: string) => {
  let width = 0;

  for (const { segment } of segmenter.segment(text)) width += graphemeWidth(segment);

  return width;
};

/** Cuts plain text to at most `width` columns, ending with an ellipsis when cut. */
export const fit = (text: string, width: number) => {
  if (displayWidth(text) <= width) return text;

  if (width <= 0) return "";

  let used = 0;
  let result = "";

  for (const { segment } of segmenter.segment(text)) {
    const next = graphemeWidth(segment);

    if (used + next > width - 1) break;
    used += next;
    result += segment;
  }

  return `${result}…`;
};

const pad = (text: string, width: number, align: "left" | "right") => {
  const space = " ".repeat(Math.max(0, width - displayWidth(text)));

  return align === "right" ? space + text : text + space;
};

export type Cell = { readonly text: string; readonly paint?: Paint | undefined };

export type Column = {
  readonly header: string;
  readonly align?: "left" | "right";
  /** Takes the room left over and is cut to fit the terminal. One column per table. */
  readonly flex?: boolean;
};

const GAP = 2;

/**
 * Renders rows as a borderless table, one record per line, sized to the terminal.
 * Cells are plain text and painted after sizing, so widths never count escape codes.
 */
export const table = (
  columns: readonly Column[],
  rows: readonly (readonly (Cell | string)[])[],
  stream: Stream = process.stdout,
) => {
  const cells = rows.map((row) =>
    row.map((cell) => (typeof cell === "string" ? { text: cell } : cell)),
  );

  const widths = columns.map((column, index) =>
    Math.max(displayWidth(column.header), ...cells.map((row) => displayWidth(row[index]!.text))),
  );

  const flex = columns.findIndex((column) => column.flex);
  const available = outputWidth(stream);
  const total = widths.reduce((sum, width) => sum + width, 0) + GAP * (columns.length - 1);

  if (flex !== -1 && total > available) {
    widths[flex] = Math.max(
      displayWidth(columns[flex]!.header),
      widths[flex]! - (total - available),
    );
  }

  const line = (row: readonly Cell[], header: boolean) =>
    row
      .map((cell, index) => {
        const column = columns[index]!;
        const text = pad(fit(cell.text, widths[index]!), widths[index]!, column.align ?? "left");
        const last = index === row.length - 1;
        const shown = last && column.align !== "right" ? text.trimEnd() : text;

        if (header) return style.dim(shown, stream);

        return cell.paint ? cell.paint(shown, stream) : shown;
      })
      .join(" ".repeat(GAP))
      .trimEnd();

  return [
    line(
      columns.map((column) => ({ text: column.header })),
      true,
    ),
    ...cells.map((row) => line(row, false)),
  ].join("\n");
};

const SECOND = 1000;
const MINUTE = 60 * SECOND;
const HOUR = 60 * MINUTE;
const DAY = 24 * HOUR;

export const DASH = "–";

export const formatDuration = (ms: number | null | undefined) => {
  if (ms === null || ms === undefined) return DASH;

  if (ms < 1) return "<1ms";

  if (ms < SECOND) return `${Math.round(ms)}ms`;

  if (ms < 10 * SECOND) return `${(ms / SECOND).toFixed(2)}s`;

  if (ms < MINUTE) return `${(ms / SECOND).toFixed(1)}s`;

  const minutes = Math.floor(ms / MINUTE);
  const seconds = Math.round((ms % MINUTE) / SECOND);

  return `${minutes}m ${String(seconds).padStart(2, "0")}s`;
};

const compact = new Intl.NumberFormat("en-US", {
  notation: "compact",
  maximumFractionDigits: 1,
});

const grouped = new Intl.NumberFormat("en-US");

export const formatCount = (value: number | null | undefined) =>
  value === null || value === undefined
    ? DASH
    : value < 10_000
      ? grouped.format(value)
      : compact.format(value);

export const formatInteger = (value: number) => grouped.format(value);

export const formatCost = (usd: number | null | undefined) => {
  if (usd === null || usd === undefined) return DASH;

  if (usd === 0) return "$0";

  // Single calls cost fractions of a cent, so small amounts keep two significant digits.
  if (usd < 0.1) return `$${usd.toLocaleString("en-US", { maximumSignificantDigits: 2 })}`;

  return `$${usd.toLocaleString("en-US", { minimumFractionDigits: 2, maximumFractionDigits: 2 })}`;
};

export const formatPercent = (value: number) => {
  // A real but tiny rate must not round to a zero that reads as none.
  if (value > 0 && value < 0.1) return "<0.1%";

  return `${value < 10 && value !== 0 ? value.toFixed(1) : Math.round(value)}%`;
};

/** How long ago an ISO timestamp was, such as "3m ago", or the date after a week. */
export const formatAge = (iso: string, now = Date.now()) => {
  const time = Date.parse(iso);

  if (Number.isNaN(time)) return oneLine(iso);

  const elapsed = Math.max(0, now - time);

  if (elapsed < MINUTE) return "just now";

  if (elapsed < HOUR) return `${Math.floor(elapsed / MINUTE)}m ago`;

  if (elapsed < DAY) return `${Math.floor(elapsed / HOUR)}h ago`;

  if (elapsed < 7 * DAY) return `${Math.floor(elapsed / DAY)}d ago`;

  return new Date(time).toLocaleDateString("en-US", { month: "short", day: "numeric" });
};

/** A local wall-clock time for log lines, such as "Oct 29 14:03:07". */
export const formatTimestamp = (iso: string) => {
  const date = new Date(iso);

  if (Number.isNaN(date.getTime())) return oneLine(iso);

  const day = date.toLocaleDateString("en-US", { month: "short", day: "numeric" });
  const time = date.toLocaleTimeString("en-GB", { hour12: false });

  return `${day} ${time}`;
};

const SPINNER = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"];
const SPINNER_DELAY_MS = 120;
const SPINNER_FRAME_MS = 80;

export const HIDE_CURSOR = "\u001b[?25l";
export const SHOW_CURSOR = "\u001b[?25h";
export const CLEAR_LINE = "\r\u001b[2K";

/**
 * Shows a spinner on stderr while `work` runs, once it has taken long enough to notice.
 * Prints nothing when stderr is not a terminal.
 */
export const withSpinner = async <A>(label: string, work: Promise<A>): Promise<A> => {
  const stream = process.stderr;

  if (!motionEnabled(stream)) return work;

  let frame = 0;
  let interval: NodeJS.Timeout | undefined;

  const draw = () => {
    stream.write(
      `${CLEAR_LINE}${style.cyan(SPINNER[frame % SPINNER.length]!, stream)} ${style.dim(label, stream)}`,
    );
    frame += 1;
  };

  const delay = setTimeout(() => {
    stream.write(HIDE_CURSOR);
    draw();
    interval = setInterval(draw, SPINNER_FRAME_MS);
  }, SPINNER_DELAY_MS);

  try {
    return await work;
  } finally {
    clearTimeout(delay);

    if (interval) {
      clearInterval(interval);
      stream.write(CLEAR_LINE + SHOW_CURSOR);
    }
  }
};

/**
 * Reports what the program is doing to terminals that show program status (OSC 7501),
 * such as "needs you" while a sign-in waits. Other terminals ignore the sequence.
 */
export const reportStatus = (state: "working" | "blocked" | "done" | "error", kind?: "auth") => {
  if (!process.stderr.isTTY) return;

  const fields = [`state=${state}`, ...(kind ? [`kind=${kind}`] : []), "app=telemetry"];

  process.stderr.write(`\u001b]7501;${fields.join(":")}\u001b\\`);
};
