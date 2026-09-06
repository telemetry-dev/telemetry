import { defineConfig } from "vite-plus";
import { customExports } from "../../scripts/package-exports.mjs";

export default defineConfig({
  pack: {
    entry: ["src/index.ts", "src/v6.ts"],
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
