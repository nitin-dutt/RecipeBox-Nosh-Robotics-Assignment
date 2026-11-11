import { useEffect, useRef, useState } from "react";
import { fetchRecipe } from "../api/client";
import type { RecipeFull, RecipeView, TasteProfile } from "../api/types";

export interface OpenRecipe {
  id: string;
  title: string | null;
  source: string;
  /** Set for an adapted recipe: the window shows this sheet instead of loading the stored original. */
  view?: RecipeView | null;
}

const SOURCE_NAMES: Record<string, string> = { ar: "Allrecipes", epi: "Epicurious", fn: "Food Network" };

/** Instructions are stored one step per line for some sources and as one paragraph for others: show lines as numbered steps. */
function steps(text: string): string[] {
  return text.split(/\n+/).map((l) => l.trim()).filter(Boolean);
}

function Section({ title, count, open, onToggle, children, className = "" }: {
  title: string;
  count?: number;
  open: boolean;
  onToggle: () => void;
  children: React.ReactNode;
  className?: string;
}) {
  const id = `sec-${title.toLowerCase()}`;
  return (
    <section className={`flex min-h-0 flex-col rounded-md border border-line ${className}`}>
      <h3 className="shrink-0">
        <button
          type="button"
          onClick={onToggle}
          aria-expanded={open}
          aria-controls={id}
          className="flex w-full items-center justify-between gap-2 px-3.5 py-2.5 text-left text-sm font-semibold hover:bg-accent-soft"
        >
          <span>
            {title}
            {count != null && <span className="ml-1.5 font-normal text-muted">({count})</span>}
          </span>
          <span className="text-muted" aria-hidden>
            {open ? "−" : "+"}
          </span>
        </button>
      </h3>
      {open && (
        <div id={id} className="min-h-0 flex-1 overflow-y-auto border-t border-line px-3.5 py-3">
          {children}
        </div>
      )}
    </section>
  );
}

/**
 * The recipe, in a floating window, so the user can follow it while the chat stays on the page.
 * Name and source at the top left, ingredients beside them, instructions below; both sections collapse, and the window can be
 * enlarged. Escape or the close button dismisses it and focus goes back to the recipe that opened it.
 */
const PORTIONS: { f: number; label: string }[] = [
  { f: 0.5, label: "½×" },
  { f: 1, label: "1×" },
  { f: 2, label: "2×" },
  { f: 3, label: "3×" },
];

export function RecipeWindow({ recipe, taste = {}, onClose }: { recipe: OpenRecipe; taste?: TasteProfile; onClose: () => void }) {
  const tasteKey = JSON.stringify(taste);
  const [data, setData] = useState<RecipeFull | null>(null);
  const [error, setError] = useState("");
  const [attempt, setAttempt] = useState(0);
  const [big, setBig] = useState(false);
  const [showIng, setShowIng] = useState(true);
  const [showIns, setShowIns] = useState(true);
  const [showTaste, setShowTaste] = useState(true);
  const [scale, setScale] = useState(1);
  const closeRef = useRef<HTMLButtonElement>(null);
  const opener = useRef<Element | null>(document.activeElement);

  useEffect(() => {
    let live = true;
    setError("");
    if (recipe.view) {
      // an adapted recipe is shown from the answer itself: it does not exist in the database
      const v = recipe.view;
      setData({
        id: recipe.id, title: recipe.title, source: recipe.source, scale: 1, steps: v.instructions.length,
        ingredients: v.ingredients.map((i) => ({ text: i.text, original: i.original, scaled: i.status === "changed" })),
        instructions: v.instructions.join("\n"), taste: { profile: {}, baking: false, note: "", adjustments: [] },
      });
      return () => {
        live = false;
      };
    }
    fetchRecipe(recipe.id, scale, taste)
      .then((r) => live && setData(r))
      .catch((e: Error) => live && setError(e.message));
    return () => {
      live = false;
    };
  }, [recipe.id, attempt, scale, tasteKey]);

  useEffect(() => {
    closeRef.current?.focus();
    const onKey = (e: KeyboardEvent) => e.key === "Escape" && onClose();
    window.addEventListener("keydown", onKey);
    const back = opener.current as HTMLElement | null;
    return () => {
      window.removeEventListener("keydown", onKey);
      back?.focus?.();
    };
  }, [onClose]);

  const title = data?.title ?? recipe.title ?? "Recipe";
  const adjustments = data?.taste?.adjustments ?? [];
  const profileSet = Object.keys(taste).length > 0;
  const view = recipe.view ?? null;            // set when this window shows an ADAPTED recipe
  const lines = data ? steps(data.instructions) : [];
  return (
    <div className="fixed inset-0 z-20 flex items-center justify-center bg-black/30 p-3 sm:p-6" onMouseDown={(e) => e.target === e.currentTarget && onClose()}>
      <div
        role="dialog"
        aria-modal="true"
        aria-label={`Recipe: ${title}`}
        className={`flex flex-col rounded-lg border border-line bg-panel ${big ? "h-full w-full" : "h-[min(820px,92vh)] w-[min(1240px,100%)]"}`}
      >
        <div className="flex shrink-0 items-center justify-end gap-1 border-b border-line px-2 py-1.5">
          <button type="button" onClick={() => setBig((b) => !b)} className="rounded-sm px-2 py-1 text-xs text-muted hover:bg-accent-soft hover:text-ink" aria-pressed={big}>
            {big ? "Make smaller" : "Make larger"}
          </button>
          <button ref={closeRef} type="button" onClick={onClose} className="rounded-sm px-2 py-1 text-xs text-muted hover:bg-accent-soft hover:text-ink">
            Close
          </button>
        </div>

        {error ? (
          <div className="p-6 text-sm">
            <p className="text-danger">{error}</p>
            <button type="button" onClick={() => setAttempt((a) => a + 1)} className="mt-3 rounded-md bg-brand px-3.5 py-2 font-medium text-brand-ink">
              Try again
            </button>
          </div>
        ) : !data ? (
          <p className="p-6 text-sm text-muted" role="status">
            Loading the recipe…
          </p>
        ) : (
          <div className="grid min-h-0 flex-1 gap-3 p-4 md:grid-cols-[minmax(0,2fr)_minmax(0,3fr)] md:grid-rows-[auto_minmax(0,1fr)_auto]">
            <div className="min-w-0 self-start px-1 pt-1">
              <h2 className="font-serif text-2xl font-semibold leading-tight tracking-tight [overflow-wrap:anywhere]">
                {title}
                {view && <span className="ml-2 align-middle rounded-sm bg-accent-soft px-2 py-0.5 font-sans text-xs font-semibold text-accent">Adapted</span>}
              </h2>
              <p className="mt-2 text-sm text-muted">
                {SOURCE_NAMES[data.source] ?? data.source} · {data.ingredients.length} ingredients
              </p>
              <p className="mt-0.5 break-all text-xs text-muted">ID {data.id}</p>
              {view && <p className="mt-2 text-xs text-muted">This is the adapted version, not the stored recipe. Changed lines are bold; removed lines are struck through.</p>}
              <div className={view ? "hidden" : "mt-3"} role="group" aria-label="Portions">
                <p className="mb-1 text-xs font-semibold text-muted">Portions</p>
                <div className="flex gap-1">
                  {PORTIONS.map((p) => (
                    <button
                      key={p.f}
                      type="button"
                      onClick={() => setScale(p.f)}
                      aria-pressed={scale === p.f}
                      aria-label={`Scale amounts ${p.f} times`}
                      className={`rounded-sm border px-2.5 py-1 text-xs ${scale === p.f ? "border-accent bg-accent-soft font-semibold text-accent" : "border-line text-muted hover:bg-accent-soft"}`}
                    >
                      {p.label}
                    </button>
                  ))}
                </div>
                {scale !== 1 && <p className="mt-1 text-xs text-muted">Only the ingredient amounts change. Times and pan sizes are not scaled.</p>}
              </div>
            </div>

            <Section title="Ingredients" count={data.ingredients.length} open={showIng} onToggle={() => setShowIng((o) => !o)} className="max-h-60 md:max-h-[24rem]">
              <ul className="space-y-1 text-sm">
                {data.ingredients.map((i, k) => (
                  <li key={k} title={i.original ?? undefined} className="grid grid-cols-[1rem_minmax(0,1fr)]">
                    <span className="text-muted" aria-hidden>
                      –
                    </span>
                    {view?.ingredients[k]?.status === "removed" ? (
                      <span>
                        <span className="text-muted line-through">{i.text}</span>{" "}
                        <span className="rounded-sm bg-danger-soft px-1.5 py-0.5 text-xs text-danger">removed</span>
                      </span>
                    ) : (
                      <span className={i.scaled ? "font-semibold" : undefined}>{i.text}</span>
                    )}
                  </li>
                ))}
              </ul>
            </Section>

            <Section title="Cooking instructions" open={showIns} onToggle={() => setShowIns((o) => !o)} className="md:col-span-2">
              {lines.length > 1 ? (
                <ol className="space-y-3 text-[15px] leading-relaxed">
                  {lines.map((l, k) => (
                    <li key={k} className="grid grid-cols-[1.75rem_minmax(0,1fr)]">
                      <span className="font-semibold text-accent tabular-nums">{k + 1}.</span>
                      <span>{l}</span>
                    </li>
                  ))}
                </ol>
              ) : (
                <p className="whitespace-pre-wrap text-[15px] leading-relaxed">{data.instructions || "This recipe has no instructions in the dataset."}</p>
              )}
            </Section>

            {view ? (
              <Section title="What was changed" open={showTaste} onToggle={() => setShowTaste((o) => !o)} className="max-h-52 md:col-span-2">
                <ul className="list-disc space-y-1 pl-5 text-sm">
                  {view.notes.map((n, k) => (
                    <li key={k}>{n}</li>
                  ))}
                  {view.disclaimer && <li className="text-muted">{view.disclaimer}</li>}
                </ul>
              </Section>
            ) : (
            <Section title="Your taste" open={showTaste} onToggle={() => setShowTaste((o) => !o)} className="max-h-52 md:col-span-2">
              {!profileSet ? (
                <p className="text-sm text-muted">Set your sweet, salt and spice level from the Taste menu in the header to see how to adjust this recipe.</p>
              ) : (
                <div className="space-y-2 text-sm">
                  <p className="text-xs text-muted">{data.taste?.note}</p>
                  {adjustments.length > 0 && (
                    <ul className="space-y-1">
                      {adjustments.map((a, k) => (
                        <li key={k} className="grid grid-cols-[4.5rem_minmax(0,1fr)] gap-x-2">
                          <span className="text-xs font-semibold uppercase tracking-wide text-accent">{a.axis}</span>
                          <span>
                            {a.line} <span className="text-muted">→</span> <strong>{a.change}</strong>
                            {a.suggested ? <span>: about {a.suggested}</span> : <span className="text-muted"> ({a.advice})</span>}
                          </span>
                        </li>
                      ))}
                    </ul>
                  )}
                </div>
              )}
            </Section>
            )}
          </div>
        )}
      </div>
    </div>
  );
}
