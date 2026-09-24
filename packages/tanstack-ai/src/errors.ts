export function errorDetails(error: unknown, fallbackType = "Error") {
  const details = { type: fallbackType, message: "Unknown error" };

  if (typeof error === "string") return { ...details, message: error };

  try {
    if (error !== null && typeof error === "object") {
      const { name, message } = error as { name?: unknown; message?: unknown };

      if (typeof name === "string" && name.length > 0) details.type = name;

      if (typeof message === "string") details.message = message;
    }
  } catch {
    return details;
  }

  return details;
}
