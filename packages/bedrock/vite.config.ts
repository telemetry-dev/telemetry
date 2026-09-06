import { defineConfig } from "vite-plus";
import { customExports } from "../../scripts/package-exports.mjs";

export default defineConfig({
  resolve: {
    alias: {
      "@telemetry-dev/sdk": new URL("../sdk/src/index.ts", import.meta.url).pathname,
    },
  },
  pack: {
    entry: ["src/index.ts", "src/agents.ts"],
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
});
