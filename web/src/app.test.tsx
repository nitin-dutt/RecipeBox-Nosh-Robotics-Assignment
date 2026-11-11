import { render, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import App from "./App";
import { RichText } from "./components/RichText";

const frame = (name: string, data: unknown) => `event: ${name}\ndata: ${JSON.stringify(data)}\n\n`;

const RECIPE = {
  id: "ab/c.d", title: "Dairy Free Penne", source: "ar", steps: 2,
  ingredients: [{ text: "1 pound penne", original: "1 pound penne" }, { text: "4 tomatoes, chopped", original: null }],
  instructions: "Boil the penne.\nToss with tomatoes.",
};

const CHAT =
  frame("session", { session_id: "s1" }) +
  frame("active_filters", { query: "penne", exclude_classes: ["dairy"], restriction_mode: "strict" }) +
  frame("sources", [{ n: 1, recipe_id: "ab/c.d", title: "Dairy Free Penne", source: "ar", evidence: ["1 pound penne"] }]) +
  frame("computation", { tool: "search_recipes", description: "d", population: "all", filters: {}, funnel: [{ label: "start", remaining: 9 }], n_analyzed: 9, denominator: 9, method: "m", sql: "SELECT 1", notes: [] }) +
  frame("token", { text: "Here you go.\n\n1. Dairy Free Penne [1] (2 steps, 2 ingredients, source ar) - matched: 1 pound penne" }) +
  frame("done", { intent: "search", status: "ok", notes: [] });

/** One fake server: /chat streams `chat`, /recipe answers from the queue. Returns the recorded calls. */
function server(chat = CHAT, recipes: Response[] = []) {
  const calls: { url: string; body?: any }[] = [];
  vi.stubGlobal("fetch", vi.fn(async (url: string, init?: RequestInit) => {
    calls.push({ url, body: init?.body ? JSON.parse(String(init.body)) : undefined });
    return url.includes("/recipe?") ? (recipes.shift() ?? new Response(JSON.stringify(RECIPE), { status: 200 })) : new Response(chat, { status: 200 });
  }));
  return calls;
}

async function ask(text = "penne no dairy") {
  await userEvent.type(screen.getByLabelText("Your message"), `${text}{Enter}`);
}

beforeEach(() => {
  localStorage.clear();
  delete document.documentElement.dataset.theme;
});
afterEach(() => vi.unstubAllGlobals());

describe("RichText", () => {
  const sources = [{ n: 1, recipe_id: "r1", title: "Penne", source: "ar", evidence: [] }];

  it("turns known [n] markers into buttons and leaves unknown ones as plain text", async () => {
    const onCite = vi.fn();
    render(<RichText text={"Penne [1] and Soup [7]"} sources={sources} onCite={onCite} />);
    await userEvent.click(screen.getByRole("button", { name: "Go to source 1" }));
    expect(onCite).toHaveBeenCalledWith(1);
    expect(screen.queryByRole("button", { name: "Go to source 7" })).toBeNull();
  });

  it("renders a numbered line that cites a recipe as a clickable row", async () => {
    const onOpen = vi.fn();
    render(<RichText text={"1. Penne [1] (8 steps) - matched: penne"} sources={sources} onOpen={onOpen} />);
    const row = screen.getByRole("button", { name: "Open recipe 1: Penne" });
    expect(within(row).getByText("(8 steps) - matched: penne")).toBeInTheDocument();
    await userEvent.click(row);
    expect(onOpen).toHaveBeenCalledWith(1);
  });

  it("never turns text into HTML", () => {
    render(<RichText text={"<img src=x onerror=alert(1)> **bold**"} />);
    expect(document.querySelector("img")).toBeNull();
    expect(screen.getByText("bold").tagName).toBe("STRONG");
  });
});

describe("answers, evidence and recipes", () => {
  it("shows the active filters and keeps the evidence closed until it is asked for", async () => {
    server();
    render(<App />);
    await ask();
    await screen.findByRole("button", { name: "Open recipe 1: Dairy Free Penne" });
    expect(screen.getByText("no dairy")).toBeInTheDocument();
    expect(screen.getByText("search: penne")).toBeInTheDocument();
    expect(screen.queryByRole("complementary")).toBeNull();                       // no separate evidence panel any more
    expect(screen.queryByText("SELECT 1")).toBeNull();
    const toggle = screen.getByRole("button", { name: /^Evidence/ });
    expect(toggle).toHaveAttribute("aria-expanded", "false");
    await userEvent.click(toggle);
    expect(screen.getByText("SELECT 1")).toBeInTheDocument();
    expect(screen.getByText(/ID ab\/c\.d/)).toBeInTheDocument();
  });

  it("opens the recipe window from the listed recipe, with name, ingredients and numbered steps", async () => {
    const calls = server();
    render(<App />);
    await ask();
    await userEvent.click(await screen.findByRole("button", { name: "Open recipe 1: Dairy Free Penne" }));
    const dlg = await screen.findByRole("dialog");
    expect(calls.some((c) => c.url.includes("/recipe?id=ab%2Fc.d"))).toBe(true);   // IDs with "/" travel as a query value
    await within(dlg).findByRole("heading", { name: "Dairy Free Penne" });
    expect(within(dlg).getByText("4 tomatoes, chopped")).toBeInTheDocument();
    expect(within(dlg).getByText("Boil the penne.")).toBeInTheDocument();
    expect(within(dlg).getByText("2.")).toBeInTheDocument();
  });

  it("opens the same window from a recipe in the evidence dropdown", async () => {
    server();
    render(<App />);
    await ask();
    await userEvent.click(await screen.findByRole("button", { name: /^Evidence/ }));
    const rows = screen.getAllByRole("button", { name: /Open recipe 1/ });
    await userEvent.click(rows[rows.length - 1]);
    expect(await screen.findByRole("dialog")).toBeInTheDocument();
  });

  it("collapses each section of the window, and closes it with Escape", async () => {
    server();
    render(<App />);
    await ask();
    await userEvent.click(await screen.findByRole("button", { name: "Open recipe 1: Dairy Free Penne" }));
    const dlg = await screen.findByRole("dialog");
    await within(dlg).findByText("1 pound penne");
    await userEvent.click(within(dlg).getByRole("button", { name: /Ingredients/ }));
    expect(within(dlg).queryByText("1 pound penne")).toBeNull();
    await userEvent.click(within(dlg).getByRole("button", { name: /Cooking instructions/ }));
    expect(within(dlg).queryByText("Boil the penne.")).toBeNull();
    await userEvent.keyboard("{Escape}");
    await waitFor(() => expect(screen.queryByRole("dialog")).toBeNull());
  });

  it("shows an error with a retry instead of a blank window", async () => {
    server(CHAT, [new Response("nope", { status: 500 })]);
    render(<App />);
    await ask();
    await userEvent.click(await screen.findByRole("button", { name: "Open recipe 1: Dairy Free Penne" }));
    const dlg = await screen.findByRole("dialog");
    await within(dlg).findByText("Could not load the recipe.");
    await userEvent.click(within(dlg).getByRole("button", { name: "Try again" }));
    await within(dlg).findByRole("heading", { name: "Dairy Free Penne" });
  });

  it("marks a refusal clearly", async () => {
    server(frame("token", { text: "I can't answer that." }) + frame("done", { intent: "unsupported", status: "refused", notes: [] }));
    render(<App />);
    await ask("calories?");
    await waitFor(() => expect(screen.getByText(/nothing was guessed/)).toBeInTheDocument());
  });

  it("reports a network failure instead of hanging", async () => {
    vi.stubGlobal("fetch", vi.fn(async () => { throw new Error("offline"); }));
    render(<App />);
    await ask("hi");
    await waitFor(() => expect(screen.getByText("offline")).toBeInTheDocument());
  });
});

describe("toggles", () => {
  it("sends the vegetarian switch with every message and shows it as a filter tag", async () => {
    const calls = server();
    render(<App />);
    const sw = screen.getByRole("switch", { name: "Vegetarian only" });
    expect(sw).toHaveAttribute("aria-checked", "false");
    await userEvent.click(sw);
    expect(sw).toHaveAttribute("aria-checked", "true");
    expect(screen.getByText("vegetarian only (no egg)")).toBeInTheDocument();
    await ask("noodles");
    await screen.findByRole("button", { name: /Open recipe 1/ });
    expect(calls.find((c) => c.url.endsWith("/chat"))?.body).toMatchObject({ message: "noodles", vegetarian: true });
    await userEvent.click(sw);
    await ask("noodles again");
    await waitFor(() => expect(calls.filter((c) => c.url.endsWith("/chat"))).toHaveLength(2));
    expect(calls.filter((c) => c.url.endsWith("/chat"))[1].body.vegetarian).toBe(false);
  });

  it("remembers the vegetarian choice in this browser", async () => {
    server();
    const { unmount } = render(<App />);
    await userEvent.click(screen.getByRole("switch", { name: "Vegetarian only" }));
    unmount();
    render(<App />);
    expect(screen.getByRole("switch", { name: "Vegetarian only" })).toHaveAttribute("aria-checked", "true");
  });

  it("switches the theme, sets it on the page and remembers it", async () => {
    server();
    const { unmount } = render(<App />);
    expect(document.documentElement.dataset.theme).toBe("light");               // jsdom has no dark preference
    await userEvent.click(screen.getByRole("button", { name: "Switch to dark theme" }));
    expect(document.documentElement.dataset.theme).toBe("dark");
    expect(localStorage.getItem("theme")).toBe("dark");
    unmount();
    render(<App />);
    expect(document.documentElement.dataset.theme).toBe("dark");
    expect(screen.getByRole("button", { name: "Switch to light theme" })).toBeInTheDocument();
  });
});

const TASTE_RECIPE = {
  ...RECIPE,
  scale: 1,
  ingredients: [{ text: "1 teaspoon cayenne pepper", original: "1 teaspoon cayenne pepper", scaled: false }],
  taste: {
    profile: { spice: "low" },
    baking: false,
    note: "A rule of thumb, not from the dataset: low is about half.",
    adjustments: [{ axis: "spice", level: "low", line: "1 teaspoon cayenne pepper", suggested: "1/2 teaspoon cayenne pepper", change: "Reduce by about 50%", advice: "use less" }],
  },
};

describe("taste, portions and web results", () => {
  it("saves the taste profile, sends it with every message and remembers it", async () => {
    const calls = server();
    const { unmount } = render(<App />);
    await userEvent.click(screen.getByRole("button", { name: /^Taste/ }));
    await userEvent.click(within(screen.getByRole("radiogroup", { name: "Spice" })).getByRole("radio", { name: "Low" }));
    expect(screen.getByRole("button", { name: /^Spice low/ })).toBeInTheDocument();
    expect(JSON.parse(localStorage.getItem("taste") ?? "{}")).toEqual({ spice: "low" });
    await ask("noodles");
    await screen.findByRole("button", { name: /Open recipe 1/ });
    expect(calls.find((c) => c.url.endsWith("/chat"))?.body.taste).toEqual({ spice: "low" });
    unmount();
    render(<App />);
    expect(screen.getByRole("button", { name: /^Spice low/ })).toBeInTheDocument();
  });

  it("ignores a damaged saved profile", () => {
    localStorage.setItem("taste", JSON.stringify({ spice: "extreme", salt: "high", sweet: 5 }));
    render(<App />);
    expect(screen.getByRole("button", { name: /^Salt high/ })).toBeInTheDocument();
  });

  it("shows the taste box in the recipe window and rescales the amounts from the portion buttons", async () => {
    localStorage.setItem("taste", JSON.stringify({ spice: "low" }));
    const calls = server(CHAT, [new Response(JSON.stringify(TASTE_RECIPE), { status: 200 }), new Response(JSON.stringify({ ...TASTE_RECIPE, scale: 2 }), { status: 200 })]);
    render(<App />);
    await ask();
    await userEvent.click(await screen.findByRole("button", { name: "Open recipe 1: Dairy Free Penne" }));
    const dlg = await screen.findByRole("dialog");
    await within(dlg).findByText(/Reduce by about 50%/);
    expect(within(dlg).getByText(/about 1\/2 teaspoon cayenne pepper/)).toBeInTheDocument();
    expect(calls.find((c) => c.url.includes("/recipe?"))?.url).toContain("spice=low");
    await userEvent.click(within(dlg).getByRole("button", { name: "Scale amounts 2 times" }));
    await waitFor(() => expect(calls.filter((c) => c.url.includes("/recipe?")).at(-1)?.url).toContain("scale=2"));
    expect(within(dlg).getByRole("button", { name: "Scale amounts 2 times" })).toHaveAttribute("aria-pressed", "true");
  });

  it("asks the user to set a taste when there is none", async () => {
    server();
    render(<App />);
    await ask();
    await userEvent.click(await screen.findByRole("button", { name: "Open recipe 1: Dairy Free Penne" }));
    const dlg = await screen.findByRole("dialog");
    await within(dlg).findByText(/Set your sweet, salt and spice level/);
  });

  it("shows web pages as rows in the answer with a Web badge, safe links only", async () => {
    const chat =
      frame("web", [
        { ingredient: "egg", title: "Egg replacers", url: "https://example.com/eggs", snippet: "Use applesauce.", domain: "example.com" },
        { ingredient: "egg", title: "Bad link", url: "javascript:alert(1)", snippet: "", domain: "" },
      ]) + CHAT;
    server(chat);
    render(<App />);
    await ask("swap the egg");
    const link = await screen.findByRole("link", { name: /Egg replacers/ });          // visible without opening the evidence
    expect(screen.getByText("From the web (2)")).toBeInTheDocument();
    expect(screen.getByText(/Not from the dataset and not checked/)).toBeInTheDocument();
    expect(within(link).getByText("Web")).toBeInTheDocument();
    expect(link).toHaveAttribute("href", "https://example.com/eggs");
    expect(link).toHaveAttribute("target", "_blank");
    expect(link).toHaveAttribute("rel", expect.stringContaining("noopener"));
    expect(screen.queryByRole("link", { name: /Bad link/ })).toBeNull();
  });

  it("badges the adapted recipe and opens the ADAPTED version, not the stored original", async () => {
    const view = {
      ingredients: [
        { text: "2 cups flour", original: null, status: "same" },
        { text: "2 eggs, beaten", original: null, status: "removed" },
        { text: "1 cup applesauce", original: "1 egg", status: "changed" },
      ],
      instructions: ["Mix.", "Bake."],
      notes: ["egg: removed, no replacement applied (dataset evidence weak)."],
    };
    const chat =
      frame("sources", [{ n: 1, recipe_id: "ab/c.d", title: "Banana Bread", source: "ar", evidence: [], kind: "adapted", view }]) +
      frame("token", { text: "1. Banana Bread [1] (the recipe being adapted)" }) +
      frame("done", { intent: "substitute", status: "no_results", notes: [] });
    const calls = server(chat);
    render(<App />);
    await ask("make it egg-free");
    const row = await screen.findByRole("button", { name: "Open recipe 1: Banana Bread" });
    expect(within(row).getByText("Adapted")).toBeInTheDocument();
    await userEvent.click(row);
    const dlg = await screen.findByRole("dialog");
    expect(within(dlg).getAllByText("Adapted").length).toBeGreaterThan(0);
    expect(within(dlg).getByText("2 eggs, beaten")).toHaveClass("line-through");
    expect(within(dlg).getByText("removed")).toBeInTheDocument();
    expect(within(dlg).getByText("1 cup applesauce")).toHaveClass("font-semibold");
    expect(within(dlg).getByText("Bake.")).toBeInTheDocument();
    expect(within(dlg).getByText(/removed, no replacement applied/)).toBeInTheDocument();
    expect(calls.some((c) => c.url.includes("/recipe?"))).toBe(false);                 // nothing was fetched: this recipe is not stored
  });
});

describe("recipe answers are laid out as cards", () => {
  const sources = [{ n: 1, recipe_id: "r1", title: "Banana Bread", source: "ar", evidence: [] }];
  const ADAPTED = [
    "1. Banana Bread [1] (the recipe being adapted)",
    "Partly adapted. egg: weak.",
    "Updated ingredient list (changed lines in bold):",
    "- 2 cups flour",
    "- (removed: 2 eggs, beaten)",
    "- **1 cup applesauce** (replaces: milk)",
    "Revised instructions:",
    "1. Mix.",
    "2. Bake.",
    "The instructions were edited by simple word replacement.",
  ].join("\n");

  it("shows the adapted recipe as a clickable row plus collapsible ingredient and instruction cards", async () => {
    const onOpen = vi.fn();
    render(<RichText text={ADAPTED} sources={sources} onOpen={onOpen} />);
    await userEvent.click(screen.getByRole("button", { name: "Open recipe 1: Banana Bread" }));
    expect(onOpen).toHaveBeenCalledWith(1);
    const cards = document.querySelectorAll("details");
    expect(cards).toHaveLength(2);
    expect(cards[0].querySelector("summary")?.textContent).toContain("Updated ingredient list");
    expect(cards[0].open && cards[1].open).toBe(true);
    expect(within(cards[1] as HTMLElement).getByText("Bake.")).toBeInTheDocument();
    expect(screen.getByText(/edited by simple word replacement/).closest("details")).toBeNull();     // the note stays outside the card
  });

  it("marks a removed ingredient as removed instead of letting it vanish", () => {
    render(<RichText text={ADAPTED} sources={sources} onOpen={() => undefined} />);
    expect(screen.getByText("removed")).toBeInTheDocument();
    expect(screen.getByText("2 eggs, beaten")).toHaveClass("line-through");
    expect(screen.getByText("1 cup applesauce").tagName).toBe("STRONG");
  });

  it("groups the instruction text of a scaled recipe into numbered steps", () => {
    const text = "1. Banana Bread [1] (scaled by 2x)\nIngredients (changed amounts in bold):\n- **4 cups flour**\nInstructions (unchanged):\nMix well.\nBake it.\nAdjusted to your taste (salt low):\n- 1 tsp salt: Reduce by about 50%";
    render(<RichText text={text} sources={sources} onOpen={() => undefined} />);
    const cards = document.querySelectorAll("details");
    expect(cards).toHaveLength(3);
    expect(within(cards[1] as HTMLElement).getByText("Mix well.")).toBeInTheDocument();
    expect(cards[1].textContent).toContain("2.");
  });

  it("says a substitution is only partly adapted instead of 'no full match'", async () => {
    server(frame("token", { text: "1. Banana Bread [1] (the recipe being adapted)" }) + frame("done", { intent: "substitute", status: "no_results", notes: [] }));
    render(<App />);
    await ask("make the first one egg-free");
    await waitFor(() => expect(screen.getByText(/Only partly adapted/)).toBeInTheDocument());
    expect(screen.queryByText(/No full match/)).toBeNull();
  });
});

describe("vegetarian mode with a non-vegetarian request", () => {
  const BLOCKED =
    frame("token", { text: "Vegetarian only is on, and it leaves out meat, fish and egg, so I will not search for chicken while it is on. Turn the switch off to see these recipes." }) +
    frame("done", { intent: "search", status: "needs_input", notes: [], suggest: "turn_off_vegetarian" });

  it("offers a button that turns the switch off and asks the same question again", async () => {
    const calls = server(BLOCKED);
    render(<App />);
    await userEvent.click(screen.getByRole("switch", { name: "Vegetarian only" }));
    await ask("butter chicken recipe");
    const button = await screen.findByRole("button", { name: "Turn off Vegetarian only and ask again" });
    await userEvent.click(button);
    expect(screen.getByRole("switch", { name: "Vegetarian only" })).toHaveAttribute("aria-checked", "false");
    await waitFor(() => expect(calls.filter((c) => c.url.endsWith("/chat"))).toHaveLength(2));
    const second = calls.filter((c) => c.url.endsWith("/chat"))[1].body;
    expect(second).toMatchObject({ message: "butter chicken recipe", vegetarian: false });    // the retry goes out with the switch already off
  });

  it("does not show the button for a normal answer", async () => {
    server();
    render(<App />);
    await ask();
    await screen.findByRole("button", { name: /Open recipe 1/ });
    expect(screen.queryByRole("button", { name: /Turn off Vegetarian only/ })).toBeNull();
  });
});
