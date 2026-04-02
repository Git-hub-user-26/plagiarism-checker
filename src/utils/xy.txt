
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
  method: PlagiarismMethodType,
  options: {
    databaseSourceType?: DatabaseSourceType;
    uploadedFiles?: File[];
    studentFiles?: File[];
    searchApiConfig?: SearchApiConfig;
  } = {}
): Promise<{ sources: Source[]; plagiarismPercentage: number }> => {
  // Simulate a processing delay
  await new Promise(resolve => setTimeout(resolve, 1000));
  
  let sources: Source[] = [];
  const { 
    databaseSourceType = 'research', 
    uploadedFiles = [], 
    studentFiles = [],
    searchApiConfig
  } = options;
  
  // Process uploaded files first if there are any
  let contentToCheck = text;
  let extractedFromFiles = false;
  
  if (uploadedFiles.length > 0) {
    console.log(`Processing ${uploadedFiles.length} uploaded files for content extraction`);
    // Extract and add text from uploaded files
    const extractedText = await processUploadedFiles(uploadedFiles);
    
    // If user provided no text but uploaded files, use the extracted text
    // If user provided both text and files, append the extracted text
    if (text.trim().length === 0) {
      contentToCheck = extractedText;
      extractedFromFiles = true;
    } else {
      contentToCheck = `${text}\n\n${extractedText}`;
    }
    
    console.log(`Extracted ${extractedText.length} characters from uploaded files`);
  }
  
  // Skip processing if no content to check
  if (!contentToCheck || contentToCheck.trim().length === 0) {
    return { sources: [], plagiarismPercentage: 0 };
  }
  
  if (method === "searchEngine") {
    console.log(`Using search engine method with API config present: ${!!searchApiConfig?.apiKey}`);
    
    // Use real or simulated API search based on availability of searchApiConfig
    sources = await fetchApiSearchResults(contentToCheck, searchApiConfig);
    
    console.log(`Found ${sources.length} sources from search API`);
    
    // Check against uploaded documents if present (different from the documents being checked)
    if (studentFiles.length > 0) {
      console.log(`Checking against ${studentFiles.length} student files`);
      const uploadedResults = await checkAgainstUploadedDocuments(contentToCheck, studentFiles);
      sources = [...sources, ...uploadedResults];
      
      console.log(`Found ${uploadedResults.length} additional sources from student files`);
    }
  } else {
    // For database method, use the improved database comparison
    sources = checkAgainstDatabase(contentToCheck, databaseSourceType);
    
    // Check against student files if provided
    if (studentFiles.length > 0) {
      const studentResults = await checkAgainstUploadedDocuments(contentToCheck, studentFiles);
      sources = [...sources, ...studentResults];
    }
  }
  
  // Remove duplicate sources by URL
  sources = sources.filter((source, index, self) =>
    index === self.findIndex((s) => s.url === source.url)
  );
  
  // FIXED: Ensure plagiarism percentage is always 0 when no sources are found
  let plagiarismPercentage = 0;
  
  if (sources.length > 0) {
    // Sort sources by match percentage (highest first)
    sources.sort((a, b) => b.matchPercentage - a.matchPercentage);
    
    // Get top 3 sources (or fewer if less available)
    const topSources = sources.slice(0, 3);
    
    // Calculate weighted average with weights for top sources
    const weights = [0.6, 0.25, 0.15]; // Weights for 1st, 2nd, and 3rd sources
    let weightSum = 0;
    let weightedScore = 0;
    
    topSources.forEach((source, index) => {
      const weight = weights[index] || 0.1; // Use lower weight for any sources beyond top 3
      weightedScore += source.matchPercentage * weight;
      weightSum += weight;
    });
    
    // Normalize based on actual weights used
    let baseScore = weightSum > 0 ? weightedScore / weightSum : 0;
    
    // Apply source count multiplier (more sources = higher plagiarism likelihood)
    // Use a more moderate scaling based on number of sources
    const sourceCountMultiplier = Math.min(1 + (sources.length / 10), 1.5);
    
    // Calculate final percentage with scaling
    plagiarismPercentage = Math.min(
      Math.round(baseScore * sourceCountMultiplier),
      95 // Cap at 95%
    );
    
    // REMOVED the minimum thresholds to allow the score to be more directly
    // based on actual similarity rather than enforcing minimums
    
    // If we extracted from files, add a note to the first source
    if (extractedFromFiles) {
      sources[0].title = `${sources[0].title} (from uploaded document)`;
    }
  }
  // No need for an else block here since plagiarismPercentage is initialized to 0
  
  console.log(`Final plagiarism percentage: ${plagiarismPercentage}% with ${sources.length} sources`);
  
  return {
    sources: sources.slice(0, 5), // Return top 5 sources
    plagiarismPercentage
  };
};

// Export other utility functions that might be needed elsewhere
export { calculateTextSimilarity } from './similarity';
export { splitTextIntoChunks } from './textProcessing';
