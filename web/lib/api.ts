// Server-side data access. The browser never calls the gateway directly: server
// components use these helpers, and client components go through the Next route
// handler at /api/pk/* (see app/api/pk/[...path]/route.ts). Either way the
// gateway URL stays on the server.

export const API_URL = process.env.API_URL || "http://127.0.0.1:8787";

export async function apiGet<T>(path: string): Promise<T> {
  const res = await fetch(`${API_URL}/api/v1${path}`, { cache: "no-store" });
  if (!res.ok) {
    throw new Error(`Gateway responded ${res.status} for ${path}`);
  }
  return (await res.json()) as T;
}

/** Fetch that degrades gracefully: returns a fallback plus an error message
 *  instead of throwing, so a page can render a "can't reach the gateway" state. */
export async function apiGetOr<T>(
  path: string,
  fallback: T,
): Promise<{ data: T; error: string | null }> {
  try {
    return { data: await apiGet<T>(path), error: null };
  } catch (e) {
    return { data: fallback, error: e instanceof Error ? e.message : String(e) };
  }
}
