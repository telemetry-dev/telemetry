import { telemetryDevExtension } from "./extension.ts";

/**
 * Ready-to-load pi extension entry configured from `TELEMETRY_DEV_*`
 * environment variables. Referenced by this package's `pi.extensions`
 * manifest so `@telemetry-dev/pi` works as an installed pi package.
 */
export default telemetryDevExtension();
