
import { Source } from "@/components/SourceLink";
import { calculateTextSimilarity } from "../similarity";
import { splitTextIntoChunks, extractTextFromFile } from "../textProcessing";

// Improved check against uploaded student documents
export const checkAgainstUploadedDocuments = async (text: string, files: File[]): Promise<Source[]> => {
  const results: Source[] = [];
  const chunks = splitTextIntoChunks(text, 150);
  
  for (const file of files) {
    const fileContent = await extractTextFromFile(file);
    let bestSimilarity = 0;
    
    // Check each chunk against the document to find best match
    for (const chunk of chunks) {
      if (chunk.length < 20) continue; // Skip very short chunks
      const similarity = calculateTextSimilarity(chunk, fileContent);
      bestSimilarity = Math.max(bestSimilarity, similarity);
    }
    
    // Significantly lower the threshold to ensure documents are included
    if (bestSimilarity > 5) {
      // Apply a stronger boost to document match scores
      const boostedScore = Math.min(bestSimilarity * 1.5, 95);
      
      results.push({
        url: `#document-${file.name}`,
        title: file.name,
        matchPercentage: Math.round(boostedScore),
        snippet: fileContent.substring(0, 150) + '...'
      });
    }
  }
  
  return results.sort((a, b) => b.matchPercentage - a.matchPercentage);
};
