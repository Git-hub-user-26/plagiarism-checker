
import { Source } from "@/components/SourceLink";

interface GoogleSearchResponse {
  items?: {
    title: string;
    link: string;
    snippet: string;
  }[];
  error?: {
    message: string;
  };
}

export interface SearchApiConfig {
  apiKey: string;
  searchEngineId: string;
}

/**
 * Performs a real Google Custom Search API query
 */
export const searchGoogleApi = async (
  query: string,
  config: SearchApiConfig
): Promise<Source[]> => {
  if (!config.apiKey || !config.searchEngineId) {
    console.error("Google API key or Search Engine ID not provided");
    throw new Error("Google API configuration missing");
  }

  try {
    const url = new URL("https://www.googleapis.com/customsearch/v1");
    url.searchParams.append("key", config.apiKey);
    url.searchParams.append("cx", config.searchEngineId);
    url.searchParams.append("q", query);

    const response = await fetch(url.toString());
    
    if (!response.ok) {
      const errorData = await response.json();
      console.error("Google API error:", errorData);
      throw new Error(`Google API error: ${errorData.error?.message || response.statusText}`);
    }

    const data: GoogleSearchResponse = await response.json();
    
    if (!data.items || data.items.length === 0) {
      return [];
    }

    // Convert Google search results to Source format
    return data.items.map(item => ({
      url: item.link,
      title: item.title,
      matchPercentage: 0, // Will be calculated later
      snippet: item.snippet
    }));
  } catch (error) {
    console.error("Failed to fetch from Google API:", error);
    throw error;
  }
};

/**
 * Process search results to calculate match percentages
 */
export const enhanceSearchResults = (
  results: Source[],
  originalText: string,
  calculateTextSimilarity: (text1: string, text2: string) => number
): Source[] => {
  return results.map(result => {
    // Calculate similarity between search result snippet and original text
    const similarity = calculateTextSimilarity(
      originalText,
      result.snippet || ""
    );
    
    return {
      ...result,
      matchPercentage: similarity
    };
  });
};
