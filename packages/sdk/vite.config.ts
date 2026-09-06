import { fileURLToPath } from "node:url";
import { defineConfig } from "vite-plus";
import { customExports } from "../../scripts/package-exports.mjs";

export default defineConfig(({ mode }) => ({
  resolve:
    mode === "test"
      ? {
          alias: {
            "@telemetry-dev/otel": fileURLToPath(new URL("../otel/src/index.ts", import.meta.url)),
          },
        }
      : undefined,
  pack: {
    entry: ["src/index.ts"],
    dts: true,
    exports: {
      devExports: "bun",
      customExports,
    },
  },
  lint: { options: { typeAware: true, typeCheck: true } },
  fmt: {},
}));
