import { type NextRequest } from "next/server";

// Same-origin proxy: the browser calls /api/pk/<path>, we forward to the gateway
// at API_URL/api/v1/<path> server-side. Keeps the gateway URL off the client and
// avoids CORS. Used for client-initiated reads, mutations, and file downloads.
const API_URL = process.env.API_URL || "http://127.0.0.1:8787";

async function handler(
  req: NextRequest,
  ctx: { params: { path: string[] } },
): Promise<Response> {
  const path = ctx.params.path.join("/");
  const url = `${API_URL}/api/v1/${path}${req.nextUrl.search}`;

  const init: RequestInit = { method: req.method, cache: "no-store" };
  if (req.method !== "GET" && req.method !== "HEAD") {
    init.body = await req.text();
    init.headers = { "content-type": req.headers.get("content-type") ?? "application/json" };
  }

  let upstream: Response;
  try {
    upstream = await fetch(url, init);
  } catch (e) {
    return new Response(
      JSON.stringify({ error: e instanceof Error ? e.message : "gateway unreachable" }),
      { status: 502, headers: { "content-type": "application/json" } },
    );
  }

  const headers = new Headers();
  for (const h of ["content-type", "content-disposition"]) {
    const v = upstream.headers.get(h);
    if (v) headers.set(h, v);
  }
  return new Response(upstream.body, { status: upstream.status, headers });
}

export const GET = handler;
export const POST = handler;
export const DELETE = handler;
export const dynamic = "force-dynamic";
