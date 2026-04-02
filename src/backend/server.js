import natural from "natural";
import sw from "stopword";
import express from "express";
import cors from "cors";
import dotenv from "dotenv";
import axios from "axios";

dotenv.config();

const app = express();
app.use(cors());
app.use(express.json());

// TEST ROUTE
app.get("/", (req, res) => {
  res.send("Backend is running");
});

// PLAGIARISM ROUTE
app.post("/check", async (req, res) => {
  const { text } = req.body;

  try {
    const apiKey = process.env.SERP_API_KEY;

    // Take first 10-15 words as search query
    const tokenizer = new natural.WordTokenizer();

// Tokenize
let words = tokenizer.tokenize(text.toLowerCase());

// Remove stopwords
words = sw.removeStopwords(words);

// Create cleaner query
const query = words.slice(0, 12).join(" ");

    const response = await axios.get("https://serpapi.com/search", {
      params: {
        q: query,
        api_key: apiKey,
        engine: "google"
      }
    });

    const results = response.data.organic_results || [];

    const TfIdf = natural.TfIdf;

const sources = results.slice(0, 5).map((item) => {
  const textWords = text.toLowerCase().split(/\W+/);
  const snippetWords = (item.snippet || item.title).toLowerCase().split(/\W+/);

  const setA = new Set(textWords);
  const setB = new Set(snippetWords);

  const intersection = new Set([...setA].filter(x => setB.has(x)));
  const union = new Set([...setA, ...setB]);

  const similarity = union.size === 0 ? 0 : intersection.size / union.size;

  const matchPercentage = Math.round(similarity * 100);

  return {
    title: item.title,
    url: item.link,
    matchPercentage
  };
});

    const plagiarismPercentage = sources.length > 0
      ? Math.min(80, sources.length * 20)
      : 0;

    res.json({
      sources,
      plagiarismPercentage
    });

  } catch (err) {
    console.error(err);
    res.status(500).json({ error: "API failed" });
  }
});

app.listen(5000, () => {
  console.log("Server running on http://localhost:5000");
});