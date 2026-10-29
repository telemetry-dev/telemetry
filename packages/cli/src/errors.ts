/** Exit codes, documented in the README. */
export const EXIT = {
  ok: 0,
  failure: 1,
  usage: 2,
  auth: 4,
  interrupted: 130,
} as const;

/** A failure with a message written for the person running the command. */
export class CliError extends Error {
  readonly exitCode: number;

  constructor(message: string, exitCode: number = EXIT.failure) {
    super(message);
    this.exitCode = exitCode;
  }
}

export const usageError = (message: string) => new CliError(message, EXIT.usage);

export const notSignedIn = () =>
  new CliError("You're not signed in. Run `telemetry login` first.", EXIT.auth);
