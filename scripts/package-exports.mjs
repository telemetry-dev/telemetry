/** Bun and Vite use source files. Node and published packages use dist files. */
export const customExports = (exportsMap, { isPublish }) => {
  for (const [key, value] of Object.entries(exportsMap)) {
    if (key === "./package.json") continue;

    if (typeof value === "string") {
      exportsMap[key] = {
        types: value.replace(/\.mjs$/, ".d.mts"),
        import: value,
        default: value,
      };
      continue;
    }

    if (
      value &&
      typeof value === "object" &&
      "default" in value &&
      typeof value.default === "string"
    ) {
      const source = "bun" in value && typeof value.bun === "string" ? value.bun : value.default;
      const types = isPublish ? value.default.replace(/\.mjs$/, ".d.mts") : source;
      exportsMap[key] =
        "bun" in value
          ? { bun: value.bun, types, development: source, default: value.default }
          : { types, import: source, default: value.default };
    }
  }

  return exportsMap;
};
