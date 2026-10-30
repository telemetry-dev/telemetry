import { defineConfig } from "vite-plus";

export default defineConfig({
  pack: {
    entry: ["src/cli.ts"],
    platform: "node",
    deps: {
      // The MCP client is a dev dependency so the published package installs nothing
      // else: npx and global installs start without resolving a dependency tree.
      alwaysBundle: [/^@modelcontextprotocol\//],
      // Every package in the bundle has its license in THIRD_PARTY_LICENSES.md; the build
      // fails when a new one appears until it is added there and here.
      onlyBundle: [
        "@modelcontextprotocol/client",
        "@modelcontextprotocol/core",
        "eventsource-parser",
        "pkce-challenge",
        "zod",
      ],
    },
  },
  lint: {
    options: {
      typeAware: true,
      typeCheck: true,
    },
  },
  fmt: {},
  test: {
    // Each test starts the CLI as a process, several times; one start takes up to 700ms
    // on a CI runner, which pushes the longer tests past the 5s default.
    testTimeout: 30_000,
  },
});
