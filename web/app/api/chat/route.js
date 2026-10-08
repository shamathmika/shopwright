export const dynamic = "force-dynamic";

export async function POST(request) {
  const upstream = await fetch(`${process.env.AGENT_URL ?? "http://localhost:8010"}/chat/stream`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: await request.text(),
  });
  return new Response(upstream.body, {
    status: upstream.status,
    headers: { "Content-Type": "text/event-stream", "Cache-Control": "no-cache" },
  });
}
