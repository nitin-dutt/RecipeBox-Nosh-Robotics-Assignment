import type { ChatEvent } from "./types";

/**
 * The API streams Server-Sent Events with NAMED events:  `event: sources` then `data: {...}`, frames separated by a blank line.
 * Network chunks do not respect frame boundaries, so this keeps the unfinished tail and returns it for the next chunk.
 * A malformed frame is skipped, never fatal.
 */
export function parseSse(buffer: string): { events: ChatEvent[]; rest: string } {
  const events: ChatEvent[] = [];
  const frames = buffer.replace(/\r\n/g, "\n").split("\n\n");
  const rest = frames.pop() ?? "";
  for (const frame of frames) {
    let name = "";
    let data = "";
    for (const line of frame.split("\n")) {
      if (line.startsWith("event:")) name = line.slice(6).trim();
      else if (line.startsWith("data:")) data += line.slice(5).trim();
    }
    if (!name || !data) continue;
    let obj: unknown;
    try {
      obj = JSON.parse(data);
    } catch {
      continue;
    }
    const e = toEvent(name, obj);
    if (e) events.push(e);
  }
  return { events, rest };
}

function toEvent(name: string, obj: unknown): ChatEvent | null {
  if (typeof obj !== "object" || obj === null) return null;
  const o = obj as Record<string, any>;
  switch (name) {
    case "session":
      return typeof o.session_id === "string" ? { type: "session", sessionId: o.session_id } : null;
    case "active_filters":
      return { type: "active_filters", filters: o };
    case "sources":
      return Array.isArray(obj) ? { type: "sources", sources: obj as any } : null;
    case "computation":
      return { type: "computation", computation: o as any };
    case "web":
      return Array.isArray(obj) ? { type: "web", hits: obj as any } : null;
    case "token":
      return typeof o.text === "string" ? { type: "token", text: o.text } : null;
    case "error":
      return { type: "error", message: String(o.message ?? "Something went wrong.") };
    case "done":
      return { type: "done", info: { intent: String(o.intent ?? ""), status: String(o.status ?? ""), notes: o.notes ?? [], suggest: o.suggest ?? null } };
    default:
      return null;
  }
}
