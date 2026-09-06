import { defineConfig } from "vite-plus";

export default defineConfig({
  fmt: { ignorePatterns: ["sdks/**"] },
  lint: {
    ignorePatterns: ["sdks/**"],
    options: { typeAware: true, typeCheck: true },
  },
  run: {
    cache: true,
  },
});
