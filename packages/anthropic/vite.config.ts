import { defineConfig } from "vite-plus";
import { customExports } from "../../scripts/package-exports.mjs";

const alias = [
  {
    find: "@telemetry-dev/sdk",
    replacement: new URL("../sdk/src/index.ts", import.meta.url).pathname,
  },
];

// The tests run against the catalog's 1.x client and a 0.x client, the two majors of the peer range.
const anthropicSdkVersion = process.env.ANTHROPIC_SDK_VERSION ?? "1";

if (anthropicSdkVersion !== "0" && anthropicSdkVersion !== "1") {
  throw new Error(`Unsupported ANTHROPIC_SDK_VERSION: ${JSON.stringify(anthropicSdkVersion)}`);
}

if (anthropicSdkVersion === "0") {
  alias.push({ find: "@anthropic-ai/sdk", replacement: "anthropic-sdk-v0" });
}

export default defineConfig({
  resolve: {
    alias,
  },
  pack: {
    entry: ["src/index.ts"],
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
