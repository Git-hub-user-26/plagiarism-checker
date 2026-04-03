import natural from "natural";
import sw from "stopword";
import express from "express";
import cors from "cors";
import dotenv from "dotenv";
import axios from "axios";
import { GoogleGenerativeAI } from "@google/generative-ai";
import rateLimit from "express-rate-limit";

dotenv.config();

const app = express();
app.use(cors());
app.use(express.json({ limit: "50kb" }));

// ─────────────────────────────────────────────
// RATE LIMITER — 5 checks per IP per minute
// ─────────────────────────────────────────────
const checkLimiter = rateLimit({
  windowMs: 60 * 1000,
  max: 5,
  message: { error: "Too many requests. Please wait a moment and try again." },
  standardHeaders: true,
  legacyHeaders: false,
});

// ─────────────────────────────────────────────
// IN-MEMORY CACHE
// ─────────────────────────────────────────────
const cache = new Map();
const CACHE_TTL_MS = 10 * 60 * 1000;

function getCached(text) {
  const key = text.trim().slice(0, 200);
  const entry = cache.get(key);
  if (!entry) return null;
  if (Date.now() - entry.timestamp > CACHE_TTL_MS) { cache.delete(key); return null; }
  return entry.result;
}

function setCache(text, result) {
  const key = text.trim().slice(0, 200);
  cache.set(key, { result, timestamp: Date.now() });
  if (cache.size > 100) cache.delete(cache.keys().next().value);
}

// ─────────────────────────────────────────────
// GEMINI CLIENT
// ─────────────────────────────────────────────
const genAI = new GoogleGenerativeAI(process.env.GEMINI_API_KEY);
const geminiModel = genAI.getGenerativeModel({ model: "gemini-1.5-flash-latest" });
const tokenizer = new natural.WordTokenizer();

if (!process.env.GEMINI_API_KEY) console.error("❌ GEMINI_API_KEY missing");
if (!process.env.SERP_API_KEY)   console.error("❌ SERP_API_KEY missing");

// ─────────────────────────────────────────────
// TEST ROUTE
// ─────────────────────────────────────────────
app.get("/", (req, res) => res.send("Smart Plagiarism Checker is running."));

// ─────────────────────────────────────────────
// NLP HELPERS
// ─────────────────────────────────────────────

function preprocess(text) {
  const words = tokenizer.tokenize(text.toLowerCase()) || [];
  return sw.removeStopwords(words);
}

function chunkText(text, chunkSize = 50, overlap = 10) {
  const words = text.trim().split(/\s+/);
  const chunks = [];
  let i = 0;
  while (i < words.length) {
    const chunk = words.slice(i, i + chunkSize).join(" ");
    if (chunk.trim()) chunks.push(chunk);
    i += chunkSize - overlap;
  }
  return chunks;
}

function buildTFVector(words) {
  const freq = {};
  for (const w of words) freq[w] = (freq[w] || 0) + 1;
  const total = words.length || 1;
  for (const w in freq) freq[w] /= total;
  return freq;
}

function cosineSimilarity(vecA, vecB) {
  const allKeys = new Set([...Object.keys(vecA), ...Object.keys(vecB)]);
  let dot = 0, magA = 0, magB = 0;
  for (const key of allKeys) {
    const a = vecA[key] || 0;
    const b = vecB[key] || 0;
    dot += a * b;
    magA += a * a;
    magB += b * b;
  }
  if (magA === 0 || magB === 0) return 0;
  return dot / (Math.sqrt(magA) * Math.sqrt(magB));
}

async function fetchPageText(url, fallback = "") {
  try {
    const response = await axios.get(url, {
      timeout: 4000,
      headers: { "User-Agent": "Mozilla/5.0 (compatible; PlagiarismChecker/1.0)" },
      maxContentLength: 500_000,
    });
    const html = String(response.data || "");
    const plain = html
      .replace(/<script[\s\S]*?<\/script>/gi, "")
      .replace(/<style[\s\S]*?<\/style>/gi, "")
      .replace(/<[^>]+>/g, " ")
      .replace(/\s+/g, " ")
      .trim();
    const extracted = plain.slice(0, 3000);
    return extracted.length > 100 ? extracted : fallback;
  } catch {
    return fallback;
  }
}

function getBestMatchForChunk(chunk, serpResults) {
  const chunkVec = buildTFVector(preprocess(chunk));
  let best = null;
  for (const item of serpResults) {
    const pageText = item._fetchedText || item.snippet || item.title || "";
    const similarity = cosineSimilarity(chunkVec, buildTFVector(preprocess(pageText)));
    if (!best || similarity > best.similarity) {
      best = { url: item.link, title: item.title, similarity, pageText };
    }
  }
  return best;
}

// ─────────────────────────────────────────────
// COSINE → PLAGIARISM SCORE MAPPING
//
// Converts raw cosine similarity (0–1) into a
// human-readable plagiarism percentage.
//
// Cosine is not linear for plagiarism:
//   < 0.45  → likely just topic overlap → 0–15%
//   0.45–0.6 → possible match → 15–50%
//   0.6–0.75 → strong match → 50–80%
//   > 0.75  → near-exact copy → 80–100%
// ─────────────────────────────────────────────
function cosineToScore(similarity) {
  if (similarity < 0.45) return Math.round(similarity * 33);          // 0–15%
  if (similarity < 0.60) return Math.round(15 + (similarity - 0.45) * 233); // 15–50%
  if (similarity < 0.75) return Math.round(50 + (similarity - 0.60) * 200); // 50–80%
  return Math.round(80 + (similarity - 0.75) * 80);                   // 80–100%
}

// ─────────────────────────────────────────────
// AI REFINEMENT — Gemini adjusts the cosine score
//
// Instead of a binary pass/fail, Gemini returns
// a multiplier (0.0 – 1.2) that adjusts the score:
//   - Clearly not plagiarism → multiplier ~0.1 (reduces score heavily)
//   - Ambiguous / common phrases → multiplier ~0.5
//   - Confirmed plagiarism → multiplier ~1.0
//   - Paraphrased plagiarism (cosine missed) → multiplier ~1.2
//
// If Gemini fails for any reason, multiplier = 1.0
// (cosine score is used as-is — still meaningful)
// ─────────────────────────────────────────────
async function getAIMultiplier(chunk, matchedPageText, sourceUrl) {
  try {
    const prompt = `You are a plagiarism detection assistant. Analyze if the USER TEXT was copied or paraphrased from the SOURCE TEXT.

USER TEXT:
"""
${chunk}
"""

SOURCE TEXT (from ${sourceUrl}):
"""
${matchedPageText.slice(0, 800)}
"""

Return ONLY a JSON object with this exact structure, no markdown:
{
  "multiplier": <number between 0.0 and 1.2>,
  "label": "not_plagiarized" | "common_phrase" | "paraphrased" | "copied",
  "reason": "<one short sentence>"
}

Multiplier guide:
- 0.0 to 0.2 → clearly NOT plagiarism (unrelated content, coincidental overlap)
- 0.3 to 0.5 → common knowledge phrase or very generic sentence
- 0.6 to 0.9 → likely paraphrased or partially copied
- 1.0         → confirmed copy (exact or near-exact wording)
- 1.1 to 1.2  → paraphrased plagiarism (different words, clearly same copied structure/ideas)`;

    const result = await geminiModel.generateContent(prompt);
    const raw = result.response.text().trim();
    const clean = raw.replace(/```json|```/g, "").trim();
    const parsed = JSON.parse(clean);

    const multiplier = parseFloat(parsed.multiplier);
    if (isNaN(multiplier) || multiplier < 0 || multiplier > 1.2) {
      throw new Error("Invalid multiplier value from Gemini");
    }

    console.log(`  🤖 Gemini: label=${parsed.label}, multiplier=${multiplier}, reason="${parsed.reason}"`);
    return { multiplier, label: parsed.label, reason: parsed.reason };

  } catch (err) {
    console.warn(`  ⚠️ Gemini unavailable (${err.message.slice(0, 60)}...) — using cosine score directly`);
    // Gemini failed → use cosine score as-is (multiplier = 1.0)
    return { multiplier: 1.0, label: "unknown", reason: "AI unavailable" };
  }
}

// ─────────────────────────────────────────────
// CONSTANTS
// ─────────────────────────────────────────────
const COSINE_MIN_THRESHOLD = 0.40; // below this → skip entirely (clean)
const MAX_SERP_RESULTS = 5;
const MAX_CHUNKS = 6;

// ─────────────────────────────────────────────
// PLAGIARISM ROUTE
// ─────────────────────────────────────────────
app.post("/check", checkLimiter, async (req, res) => {
  const { text } = req.body;

  if (!text || typeof text !== "string" || text.trim().length < 30) {
    return res.status(400).json({ error: "Please provide at least 30 characters of text to check." });
  }
  if (text.trim().length > 5000) {
    return res.status(400).json({ error: "Text too long. Please limit to 5000 characters per check." });
  }
  if (!process.env.SERP_API_KEY) {
    return res.status(500).json({ error: "SERP API key is not configured." });
  }

  const cached = getCached(text);
  if (cached) {
    console.log("📦 Cache hit");
    return res.json({ ...cached, cached: true });
  }

  try {
    // ── Step 1: Chunk the text ──
    const allChunks = chunkText(text, 50, 10);
    const step = Math.max(1, Math.floor(allChunks.length / MAX_CHUNKS));
    const selectedChunks = allChunks.filter((_, i) => i % step === 0).slice(0, MAX_CHUNKS);

    console.log(`\n📄 Checking ${selectedChunks.length} chunks...`);

    const sourceMap = new Map();
    const chunkScores = []; // individual score per chunk (0–100)

    // ── Step 2: Process each chunk ──
    for (let i = 0; i < selectedChunks.length; i++) {
      const chunk = selectedChunks[i];
      console.log(`\n🔍 Chunk ${i + 1}/${selectedChunks.length}`);

      const queryWords = preprocess(chunk);
      const query = queryWords.slice(0, 10).join(" ");
      if (!query) { chunkScores.push(0); continue; }

      // SERP search
      let serpResults = [];
      try {
        const serpResponse = await axios.get("https://serpapi.com/search", {
          params: { q: query, api_key: process.env.SERP_API_KEY, engine: "google" },
          timeout: 8000,
        });
        serpResults = (serpResponse.data.organic_results || []).slice(0, MAX_SERP_RESULTS);
        console.log(`  SERP: ${serpResults.length} results`);
      } catch (err) {
        console.warn(`  ❌ SERP failed: ${err.message}`);
        chunkScores.push(0);
        continue;
      }

      // Fetch real page content in parallel
      await Promise.all(
        serpResults.map(async (item) => {
          item._fetchedText = await fetchPageText(item.link, item.snippet || item.title || "");
        })
      );

      // Best cosine match
      const best = getBestMatchForChunk(chunk, serpResults);
      if (!best || best.similarity < COSINE_MIN_THRESHOLD) {
        console.log(`  Cosine: ${best ? best.similarity.toFixed(3) : "N/A"} — below threshold, clean`);
        chunkScores.push(0);
        continue;
      }

      console.log(`  Cosine: ${best.similarity.toFixed(3)} → ${best.url}`);

      // Convert cosine → base score
      const baseScore = cosineToScore(best.similarity);
      console.log(`  Base score from cosine: ${baseScore}%`);

      // AI refinement
      const ai = await getAIMultiplier(chunk, best.pageText, best.url);
      const finalScore = Math.min(100, Math.round(baseScore * ai.multiplier));
      console.log(`  Final chunk score: ${finalScore}% (${baseScore} × ${ai.multiplier})`);

      chunkScores.push(finalScore);

      // Track source if score is meaningful
      if (finalScore >= 20) {
        if (!sourceMap.has(best.url)) {
          sourceMap.set(best.url, {
            title: best.title,
            url: best.url,
            highestScore: finalScore,
            aiLabel: ai.label,
          });
        } else {
          const existing = sourceMap.get(best.url);
          existing.highestScore = Math.max(existing.highestScore, finalScore);
        }
      }
    }

    // ── Step 3: Overall plagiarism % = average of all chunk scores ──
    const plagiarismPercentage = chunkScores.length > 0
      ? Math.min(100, Math.round(chunkScores.reduce((a, b) => a + b, 0) / chunkScores.length))
      : 0;

    console.log(`\n✅ Chunk scores: [${chunkScores.join(", ")}]`);
    console.log(`✅ Final plagiarism: ${plagiarismPercentage}%\n`);

    // ── Step 4: Format sources ──
    const sources = Array.from(sourceMap.values())
      .sort((a, b) => b.highestScore - a.highestScore)
      .map((s) => ({
        title: s.title,
        url: s.url,
        matchPercentage: s.highestScore,
        aiLabel: s.aiLabel, // "copied" | "paraphrased" | "common_phrase" | "not_plagiarized"
      }));

    const result = { plagiarismPercentage, sources };
    setCache(text, result);
    res.json(result);

  } catch (err) {
    console.error("Plagiarism check error:", err.message);
    res.status(500).json({ error: "Something went wrong. Please try again." });
  }
});

// ─────────────────────────────────────────────
app.listen(5000, () => console.log("Server running on http://localhost:5000"));