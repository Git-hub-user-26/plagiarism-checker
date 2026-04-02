
/**
 * TF-IDF implementation for better text similarity matching
 */
export const calculateTFIDF = (text1: string, text2: string): number => {
  // Convert texts to lowercase and split into words
  const words1 = text1.toLowerCase().split(/\s+/).filter(word => word.length > 2);
  const words2 = text2.toLowerCase().split(/\s+/).filter(word => word.length > 2);
  
  // Calculate term frequency for both documents
  const tf1: Record<string, number> = {};
  const tf2: Record<string, number> = {};
  
  words1.forEach(word => {
    tf1[word] = (tf1[word] || 0) + 1;
  });
  
  words2.forEach(word => {
    tf2[word] = (tf2[word] || 0) + 1;
  });
  
  // Get all unique terms
  const allTerms = new Set([...Object.keys(tf1), ...Object.keys(tf2)]);
  
  // Calculate document frequency
  const df: Record<string, number> = {};
  allTerms.forEach(term => {
    df[term] = ((term in tf1) ? 1 : 0) + ((term in tf2) ? 1 : 0);
  });
  
  // Normalize term frequencies
  const maxTf1 = Math.max(...Object.values(tf1), 1);
  const maxTf2 = Math.max(...Object.values(tf2), 1);
  
  const normalizedTf1: Record<string, number> = {};
  const normalizedTf2: Record<string, number> = {};
  
  allTerms.forEach(term => {
    normalizedTf1[term] = term in tf1 ? tf1[term] / maxTf1 : 0;
    normalizedTf2[term] = term in tf2 ? tf2[term] / maxTf2 : 0;
  });
  
  // Calculate TF-IDF vectors
  const tfidf1: Record<string, number> = {};
  const tfidf2: Record<string, number> = {};
  
  allTerms.forEach(term => {
    const idf = Math.log(2 / df[term]);
    tfidf1[term] = normalizedTf1[term] * idf;
    tfidf2[term] = normalizedTf2[term] * idf;
  });
  
  // Calculate cosine similarity
  let dotProduct = 0;
  let magnitude1 = 0;
  let magnitude2 = 0;
  
  allTerms.forEach(term => {
    dotProduct += tfidf1[term] * tfidf2[term];
    magnitude1 += tfidf1[term] * tfidf1[term];
    magnitude2 += tfidf2[term] * tfidf2[term];
  });
  
  magnitude1 = Math.sqrt(magnitude1);
  magnitude2 = Math.sqrt(magnitude2);
  
  if (magnitude1 === 0 || magnitude2 === 0) return 0;
  
  // Enhanced scaling to better reflect plagiarism severity
  const similarity = dotProduct / (magnitude1 * magnitude2);
  // Reduced power factor and increased multiplier for higher scores
  const scaledSimilarity = Math.pow(similarity, 0.5) * 120;
  
  return Math.min(Math.round(scaledSimilarity), 95); // Kept cap at 95% but significantly boosted base score
};
