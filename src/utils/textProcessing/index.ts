
/**
 * Utility functions for processing text
 */

// Split text into smaller, more manageable chunks for better analysis
export const splitTextIntoChunks = (text: string, chunkSize: number = 200): string[] => {
  if (!text || text.length === 0) return [];
  
  const words = text.split(/\s+/);
  const chunks: string[] = [];
  
  for (let i = 0; i < words.length; i += chunkSize) {
    const chunk = words.slice(i, i + chunkSize).join(' ');
    if (chunk.length > 20) { // Only include chunks with meaningful content
      chunks.push(chunk);
    }
  }
  
  return chunks;
};

// Extract text content from uploaded files
export const extractTextFromFile = async (file: File): Promise<string> => {
  return new Promise((resolve) => {
    // In a real implementation, this would use PDF.js, mammoth.js, etc.
    // For this demo, we're simulating text extraction
    
    const reader = new FileReader();
    
    reader.onload = (e) => {
      let content = "";
      const fileName = file.name.toLowerCase();
      
      // For txt files, we can read directly
      if (fileName.endsWith('.txt')) {
        content = e.target?.result as string || "";
      } else {
        // For other files, we simulate extraction
        content = `Content extracted from ${file.name}. In a real implementation, 
        this would contain the actual text extracted from the document using the appropriate 
        library. For PDFs this would use PDF.js, for Word documents mammoth.js, etc.
        The content would be fully analyzable for plagiarism detection.`;
      }
      
      resolve(content);
    };
    
    reader.onerror = () => {
      resolve(`Failed to read ${file.name}. Please try again.`);
    };
    
    // Read as text for txt files, as array buffer for others
    if (file.name.toLowerCase().endsWith('.txt')) {
      reader.readAsText(file);
    } else {
      // In a real implementation, we would handle different file types
      // Here we just read as text for simplicity
      reader.readAsText(file);
    }
  });
};

// Process all files to extract their text content
export const processUploadedFiles = async (files: File[]): Promise<string> => {
  if (!files || files.length === 0) return '';
  
  const fileContents: string[] = [];
  
  for (const file of files) {
    const content = await extractTextFromFile(file);
    fileContents.push(content);
  }
  
  return fileContents.join('\n\n');
};
