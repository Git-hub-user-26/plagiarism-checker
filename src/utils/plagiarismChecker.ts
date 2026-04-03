// src/utils/plagiarismChecker.ts

export interface PlagiarismSource {
  title: string;
  url: string;
  matchPercentage: number;
  aiConfidence: "high" | "medium" | "low"; // added by AI verification layer
}

export interface PlagiarismResult {
  plagiarismPercentage: number;
  sources: PlagiarismSource[];
  cached?: boolean; // true if the result came from backend cache
}

// Point this at your Render backend URL in production
const BACKEND_URL =
  import.meta.env.VITE_API_URL || "http://localhost:5000";

const TIMEOUT_MS = 60_000; // 60s — Render cold starts can be slow

/**
 * Check text for plagiarism.
 * Includes:
 *  - Request timeout (AbortController)
 *  - One automatic retry on network failure
 *  - Structured error messages for the UI
 */
export async function checkPlagiarism(
  text: string,
  attempt = 1
): Promise<PlagiarismResult> {
  const controller = new AbortController();
  const timeoutId = setTimeout(() => controller.abort(), TIMEOUT_MS);

  try {
    const response = await fetch(`${BACKEND_URL}/check`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },  
      body: JSON.stringify({ text }),
      signal: controller.signal,
    });

    clearTimeout(timeoutId);

    // Handle non-2xx responses with clean error messages
    if (!response.ok) {
      const body = await response.json().catch(() => ({}));
      const message =
        body?.error ||
        (response.status === 429
          ? "Too many requests — please wait a moment."
          : response.status === 400
          ? "Invalid input. Please check your text and try again."
          : "Server error. Please try again.");
      throw new Error(message);
    }

    const data: PlagiarismResult = await response.json();
    return data;

  } catch (err: unknown) {
    clearTimeout(timeoutId);

    const isAbort =
      err instanceof Error && err.name === "AbortError";
    const isNetwork =
      err instanceof TypeError && err.message.includes("fetch");

    // Retry once on network errors or timeouts (not on validation/server errors)
    if ((isAbort || isNetwork) && attempt === 1) {
      console.warn("Request failed, retrying once...");
      return checkPlagiarism(text, 2);
    }

    if (isAbort) {
      throw new Error(
        "The request timed out. The server may be starting up — please try again."
      );
    }

    // Re-throw structured errors from above, or wrap unknown ones
    if (err instanceof Error) throw err;
    throw new Error("An unexpected error occurred.");
  }
}