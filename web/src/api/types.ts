/** The adapted recipe as the recipe window shows it: each ingredient with its status, the revised steps, and notes. */
export interface RecipeView {
  ingredients: { text: string; original: string | null; status: "same" | "changed" | "removed" }[];
  instructions: string[];
  notes: string[];
  disclaimer?: string;
}

export interface Source {
  n: number;
  recipe_id: string;
  title: string | null;
  source: string;
  evidence: string[];
  kind?: "recipe" | "adapted";
  view?: RecipeView | null;
}

export interface FunnelStep {
  label: string;
  remaining: number;
}

export interface Computation {
  tool: string;
  description: string;
  population: string;
  filters: Record<string, unknown>;
  funnel: FunnelStep[];
  n_analyzed: number;
  denominator: number | null;
  method: string;
  sql: string;
  notes: string[];
}

/** A web page found for weak evidence. Untrusted: shown as a link with a short snippet, never as a fact. */
export interface WebHit {
  ingredient: string;
  title: string;
  url: string;
  snippet: string;
  domain: string;
}

export type Level = "low" | "medium" | "high";
export type TasteProfile = Partial<Record<"sweet" | "salt" | "spice", Level>>;

export interface TasteAdjustment {
  axis: "sweet" | "salt" | "spice";
  level: Level;
  line: string;
  suggested: string | null;
  change: string;
  advice: string;
}

export interface TasteBox {
  profile: TasteProfile;
  baking: boolean;
  note: string;
  adjustments: TasteAdjustment[];
}

export type ActiveFilters = Record<string, unknown>;

export interface DoneInfo {
  intent: string;
  status: string;
  notes: string[];
  suggest?: string | null;
}

export type ChatEvent =
  | { type: "session"; sessionId: string }
  | { type: "active_filters"; filters: ActiveFilters }
  | { type: "sources"; sources: Source[] }
  | { type: "computation"; computation: Computation }
  | { type: "web"; hits: WebHit[] }
  | { type: "token"; text: string }
  | { type: "error"; message: string }
  | { type: "done"; info: DoneInfo };

export interface ChatMessage {
  id: number;
  role: "user" | "assistant";
  content: string;
  sources: Source[];
  computation?: Computation;
  web?: WebHit[];
  filters?: ActiveFilters;
  status?: string;
  intent?: string;
  suggest?: string | null;
  error?: boolean;
  pending?: boolean;
}

export interface RecipeFull {
  id: string;
  title: string | null;
  source: string;
  scale: number;
  ingredients: { text: string; original: string | null; scaled: boolean }[];
  instructions: string;
  steps: number;
  taste: TasteBox;
}
