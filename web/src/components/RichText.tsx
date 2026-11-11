import { Fragment, type ReactNode } from "react";
import type { Source } from "../api/types";

/**
 * Renders an answer. Deliberately NOT a general markdown renderer: it understands paragraphs, "- " bullets, "1. " lines,
 * **bold** and `code`, and turns [n] markers into citation marks. Everything is built from React elements, so text from the
 * model or the dataset can never become HTML or script (React escapes it).
 */
const INLINE = /(\*\*[^*\n]+\*\*|`[^`\n]+`|\[\d+(?:\s*,\s*\d+)*\])/g;
const NUMBERED = /^\s*(\d+)[.)]\s+(.*)$/;
const BULLET = /^\s*[-*]\s+(.*)$/;

interface Props {
  text: string;
  sources?: Source[];
  onCite?: (n: number) => void;
  /** Open recipe n in the recipe window. When given, a numbered line that cites a known recipe becomes a clickable row. */
  onOpen?: (n: number) => void;
}

const FIRST_CITE = /\[(\d+)\]/;

function inline(text: string, sources: Source[] | undefined, onCite: ((n: number) => void) | undefined, keyPrefix: string): ReactNode[] {
  const known = new Set((sources ?? []).map((s) => s.n));
  return text.split(INLINE).map((part, i) => {
    const key = `${keyPrefix}-${i}`;
    if (part.startsWith("**") && part.endsWith("**") && part.length > 4) return <strong key={key}>{part.slice(2, -2)}</strong>;
    if (part.startsWith("`") && part.endsWith("`") && part.length > 2)
      return (
        <code key={key} className="bg-accent-soft px-1 py-0.5 text-[0.9em]">
          {part.slice(1, -1)}
        </code>
      );
    if (/^\[\d/.test(part)) {
      const numbers = part.slice(1, -1).split(",").map((x) => Number(x.trim()));
      return (
        <Fragment key={key}>
          {numbers.map((n) =>
            known.has(n) && onCite ? (
              <button
                key={n}
                type="button"
                onClick={() => onCite(n)}
                aria-label={`Go to source ${n}`}
                className="mx-0.5 inline-flex min-w-[1.25rem] items-center justify-center rounded-sm border border-accent/40 px-1 text-xs font-semibold text-accent hover:bg-accent hover:text-accent-ink"
              >
                {n}
              </button>
            ) : (
              <span key={n} className="mx-0.5 text-xs text-muted" title="This number does not match any shown source">
                [{n}]
              </span>
            ),
          )}
        </Fragment>
      );
    }
    return <Fragment key={key}>{part}</Fragment>;
  });
}

/** Section headings the backend writes on a line of their own. "list" sections take the bullets or numbered lines after them;
 *  "rest" sections (the instruction text) take everything up to the next heading. */
const SECTION = /^(Updated ingredient list|Revised instructions|Ingredients|Instructions|Adjusted to your taste)\b([^:]*):\s*$/;
const REST_SECTIONS = new Set(["Instructions"]);
const REMOVED = /^\(removed:\s*(.*)\)$/;

function Card({ title, note, children }: { title: string; note: string; children: ReactNode }) {
  return (
    <details open className="rounded-md border border-line bg-panel">
      <summary className="flex cursor-pointer items-baseline gap-2 px-3.5 py-2.5 text-sm font-semibold hover:bg-accent-soft">
        <span>{title}</span>
        {note && <span className="font-normal text-muted">{note}</span>}
      </summary>
      <div className="space-y-1.5 border-t border-line px-3.5 py-3 text-[15px]">{children}</div>
    </details>
  );
}

export function RichText({ text, sources, onCite, onOpen }: Props) {
  const known = new Set((sources ?? []).map((s) => s.n));
  const blocks = text.split(/\n{2,}/).filter((b) => b.trim() !== "");

  const line = (l: string, li: number | string, bi: number) => {
    const key = `${bi}-${li}`;
    const num = NUMBERED.exec(l);
    if (num && onOpen) {
      const c = FIRST_CITE.exec(num[2]);
      const n = c ? Number(c[1]) : NaN;
      if (c && known.has(n)) {
        const title = num[2].slice(0, c.index).trim();
        const detail = num[2].slice(c.index + c[0].length).trim();
        return (
          <div key={key} className="grid grid-cols-[1.75rem_minmax(0,1fr)] items-start gap-x-1 py-0.5">
            <span className="pt-2.5 text-muted tabular-nums">{num[1]}.</span>
            <button
              type="button"
              onClick={() => onOpen(n)}
              aria-label={`Open recipe ${n}: ${title}`}
              className="group grid w-full grid-cols-[minmax(0,1fr)_auto] items-baseline gap-x-3 rounded-md border border-line bg-panel px-3.5 py-2.5 text-left hover:border-accent hover:bg-accent-soft"
            >
              <span className="min-w-0">
                <span className="block font-medium group-hover:text-accent">{title}</span>
                {detail && <span className="block text-sm text-muted">{detail}</span>}
              </span>
              <span className="flex items-center text-sm font-medium text-accent">
                {sources?.find((s) => s.n === n)?.kind === "adapted" && (
                  <span className="mr-2 rounded-sm bg-accent-soft px-2 py-0.5 text-xs font-semibold text-accent">Adapted</span>
                )}
                Open recipe ›
              </span>
            </button>
          </div>
        );
      }
    }
    if (num)
      return (
        <div key={key} className="grid grid-cols-[1.75rem_minmax(0,1fr)] gap-x-1">
          <span className="text-muted tabular-nums">{num[1]}.</span>
          <span>{inline(num[2], sources, onCite, key)}</span>
        </div>
      );
    const bul = BULLET.exec(l);
    if (bul) {
      const gone = REMOVED.exec(bul[1]);
      return (
        <div key={key} className="grid grid-cols-[1.25rem_minmax(0,1fr)]">
          <span className="text-muted" aria-hidden>
            –
          </span>
          {gone ? (
            <span>
              <span className="text-muted line-through">{gone[1]}</span>{" "}
              <span className="rounded-sm bg-danger-soft px-1.5 py-0.5 text-xs text-danger">removed</span>
            </span>
          ) : (
            <span>{inline(bul[1], sources, onCite, key)}</span>
          )}
        </div>
      );
    }
    return <p key={key}>{inline(l, sources, onCite, key)}</p>;
  };

  /** One block of lines -> plain lines, with each recognised section grouped into a collapsible card. */
  const renderBlock = (block: string, bi: number) => {
    const lines = block.split("\n");
    const out: ReactNode[] = [];
    for (let i = 0; i < lines.length; i++) {
      const head = SECTION.exec(lines[i].trim());
      if (!head) {
        out.push(line(lines[i], i, bi));
        continue;
      }
      const rest = REST_SECTIONS.has(head[1]);
      const body: string[] = [];
      let j = i + 1;
      while (j < lines.length && !SECTION.test(lines[j].trim()) && (rest || NUMBERED.test(lines[j]) || BULLET.test(lines[j]))) {
        body.push(lines[j]);
        j++;
      }
      i = j - 1;
      out.push(
        <Card key={`${bi}-s${i}`} title={head[1]} note={head[2].replace(/^\s*\((.*)\)\s*$/, "$1")}>
          {rest && body.length > 1
            ? body.map((b, k) => (
                <div key={k} className="grid grid-cols-[1.75rem_minmax(0,1fr)] gap-x-1 leading-relaxed">
                  <span className="font-semibold text-accent tabular-nums">{k + 1}.</span>
                  <span>{b}</span>
                </div>
              ))
            : body.map((b, k) => (rest ? <p key={k} className="leading-relaxed">{b}</p> : line(b, `${i}-${k}`, bi)))}
        </Card>,
      );
    }
    return out;
  };

  return (
    <div className="space-y-3 leading-relaxed">
      {blocks.map((block, bi) => (
        <div key={bi} className="space-y-1.5">
          {renderBlock(block, bi)}
        </div>
      ))}
    </div>
  );
}
