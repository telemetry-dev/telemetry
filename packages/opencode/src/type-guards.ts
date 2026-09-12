export type JsonObject = { [key: string]: JsonValue | undefined };

export type JsonValue = string | number | boolean | null | undefined | JsonValue[] | JsonObject;

export function isJsonObject<T>(value: T): value is T & JsonObject {
  return value !== null && value !== undefined && !Array.isArray(value) && value instanceof Object;
}

export function asJsonObject<T>(value: T): (T & JsonObject) | undefined {
  return isJsonObject(value) ? value : undefined;
}
