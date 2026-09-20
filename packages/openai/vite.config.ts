import { defineConfig } from "vite-plus";
import { customExports } from "../../scripts/package-exports.mjs";

const alias = [
  {
    find: "@telemetry-dev/sdk",
    replacement: new URL("../sdk/src/index.ts", import.meta.url).pathname,
  },
];

const openAiSdkVersion = process.env.OPENAI_SDK_VERSION ?? "7";

if (openAiSdkVersion !== "6" && openAiSdkVersion !== "7") {
  throw new Error(`Unsupported OPENAI_SDK_VERSION: ${JSON.stringify(openAiSdkVersion)}`);
}

if (openAiSdkVersion === "6") {
  alias.push({ find: "openai", replacement: "openai-v6" });
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
