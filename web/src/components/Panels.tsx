import { useState } from "react";
import type { ActiveFilters, Computation, Source, WebHit } from "../api/types";
import type { OpenRecipe } from "./RecipeWindow";

const SOURCE_NAMES: Record<string, string> = { ar: "Allrecipes", epi: "Epicurious", fn: "Food Network" };

const TAG = "rounded-sm px-1.5 py-0.5 text-xs";

/** The filters now in force. Nothing is hidden: what the bot filters on is always visible. */
export function FilterTags({ filters }: { filters: ActiveFilters | undefined }) {
  const tags: { label: string; tone: string }[] = [];
  if (filters) {
    if (filters.query) tags.push({ label: `search: ${filters.query}`, tone: "bg-accent-soft text-accent" });
    for (const x of (filters.include_ingredients as string[]) ?? []) tags.push({ label: `with ${x}`, tone: "bg-accent-soft text-accent" });
    for (const x of (filters.exclude_ingredients as string[]) ?? []) tags.push({ label: `no ${x}`, tone: "bg-danger-soft text-danger" });
    for (const x of (filters.exclude_classes as string[]) ?? []) tags.push({ label: `no ${x.replace("_", " ")}`, tone: "bg-danger-soft text-danger" });
    if (filters.max_steps != null) tags.push({ label: `max ${filters.max_steps} steps`, tone: "bg-warn-soft text-warn" });
    if (filters.vegetarian_only) tags.push({ label: "vegetarian only (no egg)", tone: "bg-ok-soft text-ok" });
    if (filters.available) tags.push({ label: `I have: ${filters.available}`, tone: "bg-accent-soft text-accent" });
    tags.push({
      label: filters.restriction_mode === "lenient" ? "lenient: possible matches allowed" : "strict: uncertain matches excluded",
      tone: "border border-line text-muted",
    });
  }
  if (tags.length === 0) return <p className="text-xs text-muted">No filters yet. Anything you ask for or rule out will show up here.</p>;
  return (
    <ul className="flex flex-wrap gap-1.5" aria-label="Active filters">
      {tags.map((c, i) => (
        <li key={i} className={`${TAG} ${c.tone}`}>
          {c.label}
        </li>
      ))}
    </ul>
  );
}

function SourceRow({ s, onOpen }: { s: Source; onOpen: (r: OpenRecipe) => void }) {
  return (
    <li data-source={s.n} className="min-w-0 border-b border-line py-2.5 text-sm [overflow-wrap:anywhere]">
      <button
        type="button"
        onClick={() => onOpen({ id: s.recipe_id, title: s.title, source: s.source, view: s.view })}
        aria-label={`Open recipe ${s.n}: ${s.title ?? "untitled"}`}
        className="group grid w-full grid-cols-[1.5rem_minmax(0,1fr)_auto] items-baseline gap-x-1 text-left"
      >
        <span className="font-semibold text-accent tabular-nums">{s.n}</span>
        <span className="min-w-0">
          <span className="block font-medium group-hover:text-accent group-hover:underline">{s.title ?? "(untitled)"}</span>
          <span className="block text-xs text-muted">
            {SOURCE_NAMES[s.source] ?? (s.source || "recipe")} · ID {s.recipe_id}
          </span>
        </span>
        <span className="flex items-center text-xs font-medium text-accent">
          {s.kind === "adapted" && <span className="mr-2 rounded-sm bg-accent-soft px-1.5 py-0.5 font-semibold">Adapted</span>}
          Open
        </span>
      </button>
      {s.evidence.length > 0 && (
        <ul className="mt-1 space-y-0.5 pl-6 text-xs text-muted">
          {s.evidence.map((e, i) => (
            <li key={i}>“{e}”</li>
          ))}
        </ul>
      )}
    </li>
  );
}

export function SourceList({ sources, onOpen }: { sources: Source[]; onOpen: (r: OpenRecipe) => void }) {
  if (sources.length === 0) return <p className="text-sm text-muted">This answer did not use any recipes.</p>;
  return (
    <ul className="border-t border-line">
      {sources.map((s) => (
        <SourceRow key={s.n} s={s} onOpen={onOpen} />
      ))}
    </ul>
  );
}

/** "How this was calculated": the brief asks every statistic to state its filters, N and calculation. */
export function CalculationDetails({ c }: { c: Computation }) {
  return (
    <div className="space-y-2 text-xs">
      <dl className="grid grid-cols-[7rem_minmax(0,1fr)] gap-x-2 gap-y-1">
        <dt className="text-muted">Tool</dt>
        <dd>
          <code>{c.tool}</code>
        </dd>
        <dt className="text-muted">Recipes analysed (N)</dt>
        <dd>{c.n_analyzed}</dd>
        {c.denominator != null && (
          <>
            <dt className="text-muted">Denominator</dt>
            <dd>{c.denominator}</dd>
          </>
        )}
        <dt className="text-muted">Population</dt>
        <dd>{c.population}</dd>
        <dt className="text-muted">Method</dt>
        <dd>{c.method}</dd>
      </dl>
      {c.funnel.length > 0 && (
        <div>
          <p className="font-semibold">Filter funnel</p>
          <ol className="ml-4 list-decimal">
            {c.funnel.map((f, i) => (
              <li key={i}>
                {f.label}: {f.remaining} recipes remain
              </li>
            ))}
          </ol>
        </div>
      )}
      {c.notes.length > 0 && (
        <ul className="list-disc pl-4 text-muted">
          {c.notes.map((n, i) => (
            <li key={i}>{n}</li>
          ))}
        </ul>
      )}
      <details>
        <summary className="cursor-pointer text-muted hover:text-ink">SQL</summary>
        <pre className="mt-1 overflow-x-auto bg-accent-soft p-2 text-[11px]">{c.sql}</pre>
      </details>
    </div>
  );
}

/** Web pages found because the dataset's evidence was weak. Shown in the answer as rows with a "Web" badge, outside the
 *  dataset; links open in a new tab. */
export function WebList({ hits }: { hits: WebHit[] }) {
  const safe = hits.filter((h) => /^https?:[/][/]/.test(h.url));
  return (
    <section aria-label="From the web">
      <h3 className="mb-1 text-sm font-semibold">From the web ({hits.length})</h3>
      <p className="mb-2 text-xs text-muted">Not from the dataset and not checked. Read them before relying on them.</p>
      <ul className="space-y-2">
        {safe.map((h, i) => (
          <li key={i} className="min-w-0">
            <a
              href={h.url}
              target="_blank"
              rel="noopener noreferrer"
              className="group grid grid-cols-[minmax(0,1fr)_auto] items-baseline gap-x-3 rounded-md border border-dashed border-line bg-panel px-3.5 py-2.5 hover:border-accent hover:bg-accent-soft"
            >
              <span className="min-w-0 [overflow-wrap:anywhere]">
                <span className="block font-medium group-hover:text-accent">{h.title}</span>
                <span className="block text-sm text-muted">
                  {h.domain} · for {h.ingredient}
                </span>
                {h.snippet && <span className="block text-xs text-muted">{h.snippet}</span>}
              </span>
              <span className="flex items-center text-sm font-medium text-accent">
                <span className="mr-2 rounded-sm bg-warn-soft px-2 py-0.5 text-xs font-semibold text-warn">Web</span>
                Visit ›
              </span>
            </a>
          </li>
        ))}
      </ul>
    </section>
  );
}

/** Everything that backs one answer: the recipes it used and how it was calculated. */
export function Evidence({ sources, computation, onOpen }: { sources: Source[]; computation?: Computation; onOpen: (r: OpenRecipe) => void }) {
  const [calcOpen, setCalcOpen] = useState(true);
  return (
    <div className="space-y-5">
      <section>
        <h3 className="mb-2 text-sm font-semibold">Recipes used ({sources.length})</h3>
        <SourceList sources={sources} onOpen={onOpen} />
      </section>
      {computation && (
        <section>
          <h3 className="mb-2 text-sm font-semibold">
            <button type="button" onClick={() => setCalcOpen((o) => !o)} aria-expanded={calcOpen} className="hover:text-accent">
              How this was calculated {calcOpen ? "−" : "+"}
            </button>
          </h3>
          {calcOpen && <CalculationDetails c={computation} />}
        </section>
      )}
    </div>
  );
}
