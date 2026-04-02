
import { Source } from "@/components/SourceLink";
import { DatabaseSourceType } from "@/components/PlagiarismMethod";
import { calculateTextSimilarity } from "../similarity";
import { 
  academicDatabase, 
  studentDocumentsDatabase, 
  researchPapersDatabase, 
  educationalContentDatabase, 
  booksJournalsDatabase 
} from "./mockDatabases";
import { splitTextIntoChunks } from "../textProcessing";

// Get the appropriate database based on the source type
export const getDatabaseBySourceType = (sourceType: DatabaseSourceType) => {
  switch (sourceType) {
    case 'research':
      return researchPapersDatabase;
    case 'educational':
      return educationalContentDatabase;
    case 'assignments':
      return studentDocumentsDatabase;
    case 'books':
      return booksJournalsDatabase;
    default:
      return academicDatabase;
  }
};

// Check text against the selected database with improved accuracy
export const checkAgainstDatabase = (text: string, databaseSourceType: DatabaseSourceType = 'research'): Source[] => {
  const database = getDatabaseBySourceType(databaseSourceType);
  const results: Source[] = [];
  
  // Split text into chunks for more accurate comparison
  const chunks = splitTextIntoChunks(text, 150);
  
  database.forEach(document => {
    const documentContent = document.content;
    let bestSimilarity = 0;
    
    // Check each chunk against the document to find best match
    for (const chunk of chunks) {
      if (chunk.length < 20) continue; // Skip very short chunks
      const similarity = calculateTextSimilarity(chunk, documentContent);
      bestSimilarity = Math.max(bestSimilarity, similarity);
    }
    
    if (bestSimilarity > 30) {
      results.push({
        url: `https://example.com/document/${document.id}`,
        title: document.title,
        matchPercentage: bestSimilarity,
        snippet: documentContent.substring(0, 150) + '...'
      });
    }
  });
  
  return results.sort((a, b) => b.matchPercentage - a.matchPercentage);
};
