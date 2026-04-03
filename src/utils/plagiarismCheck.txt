
import { Source } from "@/components/SourceLink";
import { PlagiarismMethodType, DatabaseSourceType } from "@/components/PlagiarismMethod";
import { SearchApiConfig } from "./googleSearchApi";
import { fetchApiSearchResults } from "./search/searchHandler";
import { checkAgainstDatabase } from "./database/databaseHandler";
import { checkAgainstUploadedDocuments } from "./fileProcessing/fileHandler";
import { processUploadedFiles } from "./textProcessing";

// Main function to check plagiarism with enhanced accuracy
export const checkPlagiarism = async (
  text: string,
  method,
  options = {}
) => {
  try {
    const res = await fetch("https://plagiarism-backend-tmp4.onrender.com/check", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({ text }),
    });

    const backendData = await res.json();

    console.log("Backend response:", backendData);

    return {
      sources: backendData.sources || [],
      plagiarismPercentage: backendData.plagiarismPercentage || 0,
    };

  } catch (error) {
    console.error("Backend error:", error);

    return {
      sources: [],
      plagiarismPercentage: 0,
    };
  }
};

// Export other utility functions that might be needed elsewhere
export { calculateTextSimilarity } from './similarity';
export { splitTextIntoChunks } from './textProcessing';
