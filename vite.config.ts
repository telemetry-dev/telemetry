import { defineConfig } from "vite-plus";

export default defineConfig({
  staged: {
    "*": "vp check --fix --no-error-on-unmatched-pattern",
  },
  fmt: { ignorePatterns: ["sdks/**", "packages/*/CHANGELOG.md"] },
  lint: {
    ignorePatterns: ["sdks/**"],
    options: { typeAware: true, typeCheck: true },
  },
  run: {
    cache: true,
  },
});
