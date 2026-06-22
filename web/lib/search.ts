/**
 * Web search for the Researcher agent.
 *
 * Mirrors tools/toolkit.py: prefer Tavily when a key is available (best quality,
 * 1,000 free searches/month). On serverless there is no reliable keyless option,
 * so without a Tavily key we simply skip web search and let the model rely on its
 * own knowledge — the UI makes this state explicit.
 */

export interface SearchResult {
  title: string;
  url: string;
  content: string;
}

/** Search the web via Tavily. Returns [] on any failure (caller degrades gracefully). */
export async function tavilySearch(
  query: string,
  apiKey: string,
  maxResults = 5
): Promise<SearchResult[]> {
  try {
    const res = await fetch("https://api.tavily.com/search", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        api_key: apiKey,
        query,
        max_results: Math.min(maxResults, 10),
        search_depth: "basic",
      }),
    });
    if (!res.ok) return [];
    const data = await res.json();
    const results = (data?.results ?? []) as Array<{
      title?: string;
      url?: string;
      content?: string;
    }>;
    return results.map((r) => ({
      title: r.title ?? "",
      url: r.url ?? "",
      content: r.content ?? "",
    }));
  } catch {
    return [];
  }
}

/** Format search results into the same numbered layout the Python tool produces. */
export function formatResults(results: SearchResult[]): string {
  if (!results.length) return "No results found.";
  return results
    .map((r, i) => `[${i + 1}] ${r.title}\n${r.url}\n${r.content}\n`)
    .join("\n");
}
