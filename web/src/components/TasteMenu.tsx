import { useEffect, useRef, useState } from "react";
import type { Level, TasteProfile } from "../api/types";

const AXES: { key: "sweet" | "salt" | "spice"; label: string }[] = [
  { key: "sweet", label: "Sweet" },
  { key: "salt", label: "Salt" },
  { key: "spice", label: "Spice" },
];
const LEVELS: { key: Level; label: string }[] = [
  { key: "low", label: "Low" },
  { key: "medium", label: "Medium" },
  { key: "high", label: "High" },
];

/** A short label for the header button: what is set, or just "Taste". */
export function tasteSummary(t: TasteProfile): string {
  const set = AXES.filter((a) => t[a.key] && t[a.key] !== "medium");
  return set.length ? set.map((a) => `${a.label} ${t[a.key]}`).join(", ") : "Taste";
}

/**
 * The user's taste profile: sweet, salt and spice, each low / medium / high. It is a setting like the vegetarian switch:
 * remembered in this browser, sent with every message, and used by the recipe window to say how much to reduce or increase.
 */
export function TasteMenu({ value, onChange }: { value: TasteProfile; onChange: (t: TasteProfile) => void }) {
  const [open, setOpen] = useState(false);
  const box = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (!open) return;
    const onKey = (e: KeyboardEvent) => e.key === "Escape" && setOpen(false);
    const onDown = (e: MouseEvent) => box.current && !box.current.contains(e.target as Node) && setOpen(false);
    window.addEventListener("keydown", onKey);
    window.addEventListener("mousedown", onDown);
    return () => {
      window.removeEventListener("keydown", onKey);
      window.removeEventListener("mousedown", onDown);
    };
  }, [open]);

  return (
    <div ref={box} className="relative">
      <button
        type="button"
        onClick={() => setOpen((o) => !o)}
        aria-expanded={open}
        aria-haspopup="true"
        className="rounded-md px-2.5 py-1.5 text-sm hover:bg-accent-soft"
      >
        {tasteSummary(value)} <span aria-hidden>{open ? "▴" : "▾"}</span>
      </button>
      {open && (
        <div role="group" aria-label="Your taste" className="absolute right-0 z-30 mt-1 w-72 rounded-md border border-line bg-panel p-3 shadow-sm">
          <p className="mb-2 text-xs text-muted">Set your taste once. Recipes then show how much to reduce or increase the sweet, salty and hot ingredients.</p>
          {AXES.map((a) => (
            <fieldset key={a.key} className="mb-2 last:mb-0">
              <legend className="mb-1 text-xs font-semibold">{a.label}</legend>
              <div className="grid grid-cols-3 gap-1" role="radiogroup" aria-label={a.label}>
                {LEVELS.map((l) => {
                  const on = (value[a.key] ?? "medium") === l.key;
                  return (
                    <button
                      key={l.key}
                      type="button"
                      role="radio"
                      aria-checked={on}
                      onClick={() => onChange({ ...value, [a.key]: l.key })}
                      className={`rounded-sm border px-2 py-1 text-xs ${on ? "border-accent bg-accent-soft font-semibold text-accent" : "border-line text-muted hover:bg-accent-soft"}`}
                    >
                      {l.label}
                    </button>
                  );
                })}
              </div>
            </fieldset>
          ))}
          <button type="button" onClick={() => onChange({})} className="mt-2 text-xs text-muted underline hover:text-ink">
            Reset to medium
          </button>
        </div>
      )}
    </div>
  );
}
