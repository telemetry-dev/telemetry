import { fileURLToPath } from "node:url";

import { defineConfig } from "vite-plus";
import { customExports } from "../../scripts/package-exports.mjs";

export default defineConfig(({ mode }) => ({
  resolve:
    mode === "test"
      ? {
          alias: {
            "@telemetry-dev/sdk": fileURLToPath(new URL("../sdk/src/index.ts", import.meta.url)),
          },
        }
      : undefined,
  pack: {
    entry: ["src/index.ts", "src/register.ts"],
    dts: true,
    exports: {
      devExports: "bun",
      customExports,
    },
  },
  lint: {
    options: {
      typeAware: true,
      typeCheck: true,
    },
  },
  fmt: {},
}));
