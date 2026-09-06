import { telemetryDevExtension } from "./extension.ts";

/**
 * Ready-to-load omp extension entry configured from `TELEMETRY_DEV_*`
 * environment variables. Referenced by this package's `omp.extensions`
 * manifest so `@telemetry-dev/omp` works as an installed omp package.
 */
export default telemetryDevExtension();
