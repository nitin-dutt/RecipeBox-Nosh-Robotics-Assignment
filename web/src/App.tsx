import { useCallback, useEffect, useRef, useState } from "react";
import { resetSession, streamChat } from "./api/client";
import type { ChatEvent, ChatMessage, Level, TasteProfile } from "./api/types";
import { Evidence, FilterTags, WebList } from "./components/Panels";
import { RecipeWindow, type OpenRecipe } from "./components/RecipeWindow";
import { RichText } from "./components/RichText";
import { TasteMenu } from "./components/TasteMenu";

const EXAMPLES: { task: string; q: string }[] = [
  { task: "Find", q: "Show me penne pasta recipes" },
  { task: "Count", q: "What percentage of recipes use both garlic and onion?" },
  { task: "Match", q: "I have eggs, onions, tomatoes and rice. What can I cook?" },
  { task: "Swap", q: "Make the first banana bread egg-free" },
  { task: "Limits", q: "How many calories are in lasagna?" },
];

let nextId = 1;

type Theme = "light" | "dark";

/** Remembered in this browser only; if storage is blocked the choice simply is not remembered. */
function readStored(key: string): string | null {
  try {
    return localStorage.getItem(key);
  } catch {
    return null;
  }
}
function writeStored(key: string, value: string) {
  try {
    localStorage.setItem(key, value);
  } catch {
    /* private mode or blocked storage: fine */
  }
}

/** The saved taste profile; anything that is not a known level is dropped. */
function readTaste(): TasteProfile {
  try {
    const raw = JSON.parse(readStored("taste") ?? "{}") as Record<string, unknown>;
    const out: TasteProfile = {};
    for (const axis of ["sweet", "salt", "spice"] as const) {
      const v = raw[axis];
      if (v === "low" || v === "medium" || v === "high") out[axis] = v as Level;
    }
    return out;
  } catch {
    return {};
  }
}

function useTheme(): [Theme, () => void] {
  const [theme, setTheme] = useState<Theme>(() => {
    const saved = readStored("theme");
    if (saved === "light" || saved === "dark") return saved;
    return typeof window.matchMedia === "function" && window.matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light";
  });
  useEffect(() => {
    document.documentElement.dataset.theme = theme;
  }, [theme]);
  const toggle = useCallback(() => {
    setTheme((t) => {
      const next = t === "dark" ? "light" : "dark";
      writeStored("theme", next);
      return next;
    });
  }, []);
  return [theme, toggle];
}

function StatusNote({ m }: { m: ChatMessage }) {
  const note =
    m.status === "refused"
      ? { text: "The dataset cannot answer this, so nothing was guessed.", tone: "bg-warn-soft text-warn" }
      : m.status === "no_results"
        ? {
            text:
              m.intent === "substitute"
                ? "Only partly adapted: some swaps have weak evidence. Read the notes before you cook."
                : "No full match. No filter was loosened silently.",
            tone: "bg-warn-soft text-warn",
          }
        : m.status === "needs_input"
          ? { text: "One more detail from you is needed.", tone: "bg-accent-soft text-accent" }
          : null;
  return note ? <p className={`rounded-sm px-2.5 py-1.5 text-sm ${note.tone}`}>{note.text}</p> : null;
}

function Assistant({ m, onOpen, onSuggest }: { m: ChatMessage; onOpen: (r: OpenRecipe) => void; onSuggest: (m: ChatMessage) => void }) {
  const [open, setOpen] = useState(false); // the evidence dropdown is closed until the user asks for it
  const hasEvidence = m.sources.length > 0 || !!m.computation;
  const openN = (n: number) => {
    const s = m.sources.find((x) => x.n === n);
    if (s) onOpen({ id: s.recipe_id, title: s.title, source: s.source, view: s.view });
  };
  return (
    <div className="space-y-3">
      {m.pending && m.content === "" && (
        <p className="text-muted" role="status">
          Looking through the recipes…
        </p>
      )}
      {m.error ? <p className="text-danger">{m.content}</p> : m.content && <RichText text={m.content} sources={m.sources} onCite={openN} onOpen={openN} />}
      <StatusNote m={m} />
      {m.suggest === "turn_off_vegetarian" && (
        <button
          type="button"
          onClick={() => onSuggest(m)}
          className="rounded-md bg-brand px-3.5 py-2 text-sm font-medium text-brand-ink hover:brightness-95"
        >
          Turn off Vegetarian only and ask again
        </button>
      )}
      {(m.web?.length ?? 0) > 0 && <WebList hits={m.web ?? []} />}
      {hasEvidence && (
        <div>
          <button
            type="button"
            onClick={() => setOpen((o) => !o)}
            aria-expanded={open}
            className="text-sm font-medium text-accent hover:underline"
          >
            {open ? "Hide evidence" : "Evidence"}
            {!open && (
              <span className="font-normal text-muted">
                {" "}
                · {m.sources.length} recipe{m.sources.length === 1 ? "" : "s"}
                {m.web?.length ? `, ${m.web.length} web page${m.web.length === 1 ? "" : "s"}` : ""}, how it was calculated
              </span>
            )}
            <span aria-hidden> {open ? "−" : "+"}</span>
          </button>
          {open && (
            <div className="mt-3 rounded-md border border-line bg-panel p-4">
              <Evidence sources={m.sources} computation={m.computation} onOpen={onOpen} />
            </div>
          )}
        </div>
      )}
    </div>
  );
}

function Switch({ label, checked, onChange, hint }: { label: string; checked: boolean; onChange: (v: boolean) => void; hint?: string }) {
  return (
    <button
      type="button"
      role="switch"
      aria-checked={checked}
      title={hint}
      onClick={() => onChange(!checked)}
      className="flex items-center gap-2 rounded-md px-2 py-1.5 text-sm hover:bg-accent-soft"
    >
      <span className={`relative inline-block h-5 w-9 rounded-full border ${checked ? "border-ok bg-ok" : "border-muted bg-transparent"}`} aria-hidden>
        <span className={`absolute top-0.5 size-3.5 rounded-full transition-all ${checked ? "left-[1.1rem] bg-panel" : "left-0.5 bg-muted"}`} />
      </span>
      {label}
    </button>
  );
}

export default function App() {
  const [theme, toggleTheme] = useTheme();
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [filters, setFilters] = useState<ChatMessage["filters"]>();
  const [vegetarian, setVegetarianState] = useState<boolean>(() => readStored("vegetarian") === "1");
  const [taste, setTasteState] = useState<TasteProfile>(readTaste);
  const [input, setInput] = useState("");
  const [busy, setBusy] = useState(false);
  const [openRecipe, setOpenRecipe] = useState<OpenRecipe | null>(null);
  const session = useRef<string | null>(null);
  const bottom = useRef<HTMLDivElement>(null);

  useEffect(() => {
    bottom.current?.scrollIntoView?.({ behavior: "smooth", block: "end" });
  }, [messages]);

  const patchLast = useCallback((fn: (m: ChatMessage) => ChatMessage) => {
    setMessages((ms) => ms.map((m, i) => (i === ms.length - 1 ? fn(m) : m)));
  }, []);

  const onEvent = useCallback(
    (e: ChatEvent) => {
      switch (e.type) {
        case "session":
          session.current = e.sessionId;
          break;
        case "active_filters":
          setFilters(e.filters);
          break;
        case "sources":
          patchLast((m) => ({ ...m, sources: e.sources }));
          break;
        case "computation":
          patchLast((m) => ({ ...m, computation: e.computation }));
          break;
        case "web":
          patchLast((m) => ({ ...m, web: e.hits }));
          break;
        case "token":
          patchLast((m) => ({ ...m, content: m.content + e.text }));
          break;
        case "error":
          patchLast((m) => ({ ...m, content: e.message, error: true }));
          break;
        case "done":
          patchLast((m) => ({ ...m, pending: false, status: e.info.status, intent: e.info.intent, suggest: e.info.suggest }));
          break;
      }
    },
    [patchLast],
  );

  const send = useCallback(
    async (text: string, override?: { vegetarian?: boolean }) => {
      const q = text.trim();
      if (!q || busy) return;
      const uid = nextId++;
      const aid = nextId++;
      setInput("");
      setBusy(true);
      setMessages((ms) => [
        ...ms,
        { id: uid, role: "user", content: q, sources: [] },
        { id: aid, role: "assistant", content: "", sources: [], pending: true },
      ]);
      try {
        await streamChat(q, session.current, onEvent, override?.vegetarian ?? vegetarian, taste); // the toggle and the taste profile travel with every message
      } catch (err) {
        patchLast((m) => ({ ...m, content: err instanceof Error ? err.message : "Could not reach the server.", error: true }));
      } finally {
        patchLast((m) => ({ ...m, pending: false }));
        setBusy(false);
      }
    },
    [busy, onEvent, patchLast, vegetarian, taste],
  );

  const setVegetarian = (v: boolean) => {
    setVegetarianState(v);
    writeStored("vegetarian", v ? "1" : "0");
    // show the new setting in the filter tags at once; the server confirms it with the next answer
    setFilters((f) => ({ ...(f ?? {}), vegetarian_only: v || undefined, restriction_mode: f?.restriction_mode ?? "strict" }));
  };

  const setTaste = (t: TasteProfile) => {
    setTasteState(t);
    writeStored("taste", JSON.stringify(t));
  };

  /** "Turn off Vegetarian only and ask again": flip the switch and send the question that was blocked, with the switch off. */
  const turnOffVegAndAsk = (m: ChatMessage) => {
    const i = messages.findIndex((x) => x.id === m.id);
    const question = messages.slice(0, i).reverse().find((x) => x.role === "user")?.content;
    setVegetarian(false);
    if (question) void send(question, { vegetarian: false });
  };

  const reset = async () => {
    if (session.current) await resetSession(session.current);
    session.current = null;
    setMessages([]);
    setFilters(vegetarian ? { vegetarian_only: true, restriction_mode: "strict" } : undefined);
    setOpenRecipe(null);
  };

  const closeRecipe = useCallback(() => setOpenRecipe(null), []);

  return (
    <div className="flex h-full flex-col">
      <header className="border-b border-line">
        <div className="mx-auto flex max-w-[1500px] flex-wrap items-center justify-between gap-x-6 gap-y-2 px-5 py-3 lg:px-10">
          <div className="flex items-baseline gap-3">
            <h1 className="font-serif text-2xl font-semibold tracking-tight">Recipe Box</h1>
            <p className="hidden text-sm text-muted md:block">Ask about 125,000 recipes. Every answer shows the recipes behind it.</p>
          </div>
          <div className="flex items-center gap-1">
            <Switch label="Vegetarian only" checked={vegetarian} onChange={setVegetarian} hint="No meat, fish or egg. Dairy is fine." />
            <TasteMenu value={taste} onChange={setTaste} />
            <button
              type="button"
              onClick={toggleTheme}
              aria-label={`Switch to ${theme === "dark" ? "light" : "dark"} theme`}
              className="rounded-md px-2.5 py-1.5 text-sm hover:bg-accent-soft"
            >
              {theme === "dark" ? "Light theme" : "Dark theme"}
            </button>
            <button type="button" onClick={reset} className="rounded-md px-2.5 py-1.5 text-sm hover:bg-accent-soft">
              New chat
            </button>
          </div>
        </div>
        <div className="mx-auto flex max-w-[1500px] items-start gap-3 px-5 pb-3 lg:px-10">
          <span className="pt-0.5 text-xs font-semibold text-muted">Filters</span>
          <FilterTags filters={filters} />
        </div>
      </header>

      <main className="min-h-0 flex-1 overflow-y-auto overflow-x-hidden" aria-live="polite">
        <div className="mx-auto w-full max-w-[1100px] space-y-8 px-5 py-8 lg:px-10">
          {messages.length === 0 && (
            <div>
              <h2 className="font-serif text-3xl font-semibold tracking-tight">What are you cooking?</h2>
              <p className="mt-2 max-w-prose text-muted">
                Search recipes, count and compare them, find dinners from what is in your fridge, or ask for a swap. Start with one of these, or type your own.
              </p>
              <ul className="mt-6 border-t border-line">
                {EXAMPLES.map((ex) => (
                  <li key={ex.q} className="border-b border-line">
                    <button type="button" onClick={() => send(ex.q)} className="grid w-full grid-cols-[5rem_minmax(0,1fr)] gap-x-2 py-3 text-left hover:bg-accent-soft">
                      <span className="text-xs font-semibold uppercase tracking-wide text-accent">{ex.task}</span>
                      <span>{ex.q}</span>
                    </button>
                  </li>
                ))}
              </ul>
            </div>
          )}
          {messages.map((m) =>
            m.role === "user" ? (
              <div key={m.id} className="flex justify-end">
                <p className="max-w-[80%] whitespace-pre-wrap rounded-md bg-brand px-4 py-2.5 text-brand-ink">{m.content}</p>
              </div>
            ) : (
              <Assistant key={m.id} m={m} onOpen={setOpenRecipe} onSuggest={turnOffVegAndAsk} />
            ),
          )}
          <div ref={bottom} />
        </div>
      </main>

      <form
        className="border-t border-line"
        onSubmit={(e) => {
          e.preventDefault();
          void send(input);
        }}
      >
        <div className="mx-auto flex w-full max-w-[1100px] gap-2 px-5 pb-4 pt-3 lg:px-10">
          <label htmlFor="msg" className="sr-only">
            Your message
          </label>
          <input
            id="msg"
            value={input}
            onChange={(e) => setInput(e.target.value)}
            maxLength={2000}
            placeholder="Ask about recipes, ingredients or swaps"
            className="min-h-12 min-w-0 flex-1 rounded-md border border-line bg-panel px-4 outline-none focus:border-accent"
          />
          <button
            type="submit"
            disabled={busy || input.trim() === ""}
            className="min-h-12 rounded-md bg-brand px-6 font-medium text-brand-ink hover:brightness-95 disabled:cursor-not-allowed disabled:opacity-50"
          >
            Send
          </button>
        </div>
      </form>

      {openRecipe && <RecipeWindow key={openRecipe.id} recipe={openRecipe} taste={taste} onClose={closeRecipe} />}
    </div>
  );
}
