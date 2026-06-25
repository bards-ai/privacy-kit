// Browser-side data access. Client components fetch through the same-origin
// proxy at /api/pk/* (see app/api/pk/[...path]/route.ts), which forwards to the
// gateway server-side. Mirrors apiGet in lib/api.ts but runs in the browser.

export async function clientGet<T>(path: string): Promise<T> {
  const res = await fetch(`/api/pk${path}`, { cache: "no-store" });
  if (!res.ok) {
    throw new Error(`Gateway responded ${res.status} for ${path}`);
  }
  return (await res.json()) as T;
}
