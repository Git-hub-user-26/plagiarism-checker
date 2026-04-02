
import { Source } from "@/components/SourceLink";
import { SearchApiConfig, searchGoogleApi, enhanceSearchResults } from "../googleSearchApi";
import { simulateSearchApiResults } from "./simulatedSearch";
import { splitTextIntoChunks } from "../textProcessing";
import { calculateTextSimilarity } from "../similarity";

// Real API search results from Google Custom Search API
export const fetchApiSearchResults = async (
  query: string, 
  searchApiConfig?: SearchApiConfig
): Promise<Source[]> => {
  // Skip API call if no configuration is provided
  if (!searchApiConfig?.apiKey || !searchApiConfig?.searchEngineId) {
    console.log("Using simulated search results as no API key was provided");
    return simulateSearchApiResults(query);
  }

  try {
    console.log("Using real Google API search with provided configuration");
    
    // Split the query into smaller chunks for more accurate search
    const chunks = splitTextIntoChunks(query, 200);
    console.log(`Searching with ${chunks.length} query chunks using Google API`);
    
    // Process each chunk with actual Google API search
    const allResults: Source[] = [];
    
    // Only use first few chunks to minimize API usage and ensure reasonable response time
    const chunksToProcess = chunks.slice(0, Math.min(chunks.length, 3));
    
    for (const chunk of chunksToProcess) {
      if (chunk.length < 50) continue; // Skip very small chunks
      
      console.log(`Searching chunk of length ${chunk.length} with Google API`);
      
      // Get actual results from Google API for this chunk
      const chunkResults = await searchGoogleApi(chunk, searchApiConfig);
      
      console.log(`Google API returned ${chunkResults.length} results for chunk`);
      
      if (chunkResults.length === 0) {
        // No results found for this chunk
        continue;
      }
      
      // Calculate match percentages based on similarity algorithms
      const enhancedResults = enhanceSearchResults(
        chunkResults,
        query, // Use the full query for better similarity matching
        calculateTextSimilarity
      );
      
      allResults.push(...enhancedResults);
    }
    
    // If no results found at all, return empty array (will result in 0% plagiarism)
    if (allResults.length === 0) {
      console.log("No matching sources found in Google search");
      return [];
    }
    
    // Remove duplicates by URL
    const uniqueResults = allResults.filter((result, index, self) =>
      index === self.findIndex((r) => r.url === result.url)
    );
    
    console.log(`After deduplication, found ${uniqueResults.length} unique sources`);
    
    // Sort by match percentage (descending)
    return uniqueResults
      .sort((a, b) => b.matchPercentage - a.matchPercentage)
      .slice(0, 5); // Return top 5 results
  } catch (error) {
    console.error("Error fetching search results:", error);
    // Fallback to simulated results on error
    console.log("Falling back to simulated search results due to API error");
    return simulateSearchApiResults(query);
  }
};
