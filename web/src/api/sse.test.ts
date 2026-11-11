import { describe, expect, it } from "vitest";
import { parseSse } from "./sse";

const frame = (name: string, data: unknown) => `event: ${name}\ndata: ${JSON.stringify(data)}\n\n`;

describe("parseSse", () => {
  it("parses named events in order", () => {
    const { events, rest } = parseSse(
      frame("session", { session_id: "abc" }) + frame("sources", [{ n: 1, recipe_id: "r", title: "T", source: "ar", evidence: [] }]) +
        frame("token", { text: "hi [1]" }) + frame("done", { intent: "search", status: "ok", notes: [] }),
    );
    expect(rest).toBe("");
    expect(events.map((e) => e.type)).toEqual(["session", "sources", "token", "done"]);
    expect(events[0]).toEqual({ type: "session", sessionId: "abc" });
  });

  it("keeps an unfinished tail for the next chunk", () => {
    const first = parseSse(frame("token", { text: "a" }) + 'event: token\ndata: {"te');
    expect(first.events).toHaveLength(1);
    const second = parseSse(first.rest + 'xt":"b"}\n\n');
    expect(second.events).toEqual([{ type: "token", text: "b" }]);
  });

  it("skips malformed or unknown frames instead of failing", () => {
    const { events } = parseSse("event: token\ndata: {not json}\n\n" + frame("mystery", {}) + frame("error", { message: "boom" }));
    expect(events).toEqual([{ type: "error", message: "boom" }]);
  });
});
