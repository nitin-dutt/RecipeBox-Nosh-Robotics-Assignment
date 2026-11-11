import { parseSse } from "./sse";
import type { ChatEvent, RecipeFull, TasteProfile } from "./types";

const BASE = "/api"; // proxied to the FastAPI service in dev, so the browser talks to one origin

export async function streamChat(message: string, sessionId: string | null, onEvent: (e: ChatEvent) => void, vegetarian = false, taste: TasteProfile = {}, signal?: AbortSignal) {
  const res = await fetch(`${BASE}/chat`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ message, session_id: sessionId, vegetarian, taste }),
    signal,
  });
  if (!res.ok || !res.body) throw new Error(`The server answered ${res.status}.`);
  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  for (;;) {
    const { value, done } = await reader.read();
    buffer += decoder.decode(value ?? new Uint8Array(), { stream: !done });
    const { events, rest } = parseSse(done ? buffer + "\n\n" : buffer);
    buffer = rest;
    events.forEach(onEvent);
    if (done) break;
  }
}

export async function fetchRecipe(id: string, scale = 1, taste: TasteProfile = {}): Promise<RecipeFull> {
  const q = new URLSearchParams({ id, scale: String(scale) });
  for (const [axis, level] of Object.entries(taste)) if (level) q.set(axis, level);
  const res = await fetch(`${BASE}/recipe?${q.toString()}`);
  if (!res.ok) throw new Error("Could not load the recipe.");
  return res.json();
}

export async function resetSession(sessionId: string) {
  await fetch(`${BASE}/reset?session_id=${encodeURIComponent(sessionId)}`, { method: "POST" }).catch(() => undefined);
}
