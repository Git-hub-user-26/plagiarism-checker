
import { calculateJaccardSimilarity } from './jaccard';
import { calculateTFIDF } from './tfidf';

/**
 * Choose the best similarity score based on multiple algorithms
 */
export const calculateTextSimilarity = (text1: string, text2: string): number => {
  const jaccardScore = calculateJaccardSimilarity(text1, text2);
  const tfidfScore = calculateTFIDF(text1, text2);
  
  // Enhance the method selection:
  // - More aggressively trust higher scores to better detect plagiarism
  // - Apply a stronger boost for even minimal matches
  
  // Get highest score and apply a significant boost
  const highestScore = Math.max(jaccardScore, tfidfScore);
  
  // Apply a much stronger boost to all scores to emphasize plagiarism detection
  // Even minimal matches should be prominently flagged
  if (highestScore > 15) {
    // Apply stronger logarithmic scaling boost
    const boost = 15 + Math.log10(highestScore + 20) * 15;
    return Math.min(highestScore + boost, 95);
  }
  
  // Apply minimum baseline for any match at all
  if (highestScore > 0) {
    return Math.max(highestScore, 25); // Set minimum score for any match
  }
  
  return highestScore;
};

export { calculateJaccardSimilarity, calculateTFIDF };
