
import { Source } from "@/components/SourceLink";

// Simulate search API results with more realistic data
export const simulateSearchApiResults = (query: string): Source[] => {
  const normalizedQuery = query.toLowerCase();
  
  // Common plagiarism topics with realistic data
  if (normalizedQuery.includes("machine learning") || normalizedQuery.includes("artificial intelligence")) {
    return [
      {
        url: "https://en.wikipedia.org/wiki/Machine_learning",
        title: "Machine learning - Wikipedia",
        matchPercentage: Math.min(75 + Math.floor(Math.random() * 17), 92),
        snippet: "Machine learning is a branch of artificial intelligence and computer science which focuses on the use of data and algorithms to imitate the way that humans learn, gradually improving its accuracy."
      },
      {
        url: "https://www.ibm.com/cloud/learn/machine-learning",
        title: "What is Machine Learning? | IBM",
        matchPercentage: Math.min(65 + Math.floor(Math.random() * 20), 92),
        snippet: "Machine learning is a form of AI that enables a system to learn from data rather than through explicit programming."
      }
    ];
  }
  
  if (normalizedQuery.includes("climate change") || normalizedQuery.includes("global warming")) {
    return [
      {
        url: "https://www.un.org/en/climatechange",
        title: "Climate Change | United Nations",
        matchPercentage: Math.min(70 + Math.floor(Math.random() * 22), 92),
        snippet: "Climate change refers to long-term shifts in temperatures and weather patterns. These shifts may be natural, but since the 1800s, human activities have been the main driver of climate change, primarily due to the burning of fossil fuels."
      },
      {
        url: "https://climate.nasa.gov/",
        title: "Climate Change: Vital Signs of the Planet – NASA",
        matchPercentage: Math.min(65 + Math.floor(Math.random() * 20), 92),
        snippet: "The current warming trend is of particular significance because it is unequivocally the result of human activity since the mid-20th century."
      }
    ];
  }
  
  if (normalizedQuery.includes("web development") || normalizedQuery.includes("frontend")) {
    return [
      {
        url: "https://developer.mozilla.org/en-US/docs/Learn",
        title: "Learn web development | MDN",
        matchPercentage: Math.min(68 + Math.floor(Math.random() * 20), 92),
        snippet: "Modern web development encompasses a variety of technologies and methodologies including responsive design, progressive web apps, and component-based architecture."
      },
      {
        url: "https://www.w3schools.com/",
        title: "W3Schools Online Web Tutorials",
        matchPercentage: Math.min(60 + Math.floor(Math.random() * 20), 92),
        snippet: "Web development refers to building, creating, and maintaining websites. It includes aspects such as web design, web publishing, web programming, and database management."
      }
    ];
  }
  
  // Generate some generic results for other queries
  return [
    {
      url: `https://example.com/result-${Math.floor(Math.random() * 1000)}`,
      title: `Search Result for "${query.substring(0, 30)}${query.length > 30 ? '...' : ''}"`,
      matchPercentage: Math.min(50 + Math.floor(Math.random() * 42), 92),
      snippet: `This is a simulated search result for your query. In a real implementation, this would be an actual matching passage from the web using Bing Search API. The text might contain terms like: ${normalizedQuery.split(' ').slice(0, 5).join(', ')}...`
    },
    {
      url: `https://example.org/result-${Math.floor(Math.random() * 1000)}`,
      title: `Related Information on "${query.substring(0, 25)}${query.length > 25 ? '...' : ''}"`,
      matchPercentage: Math.min(45 + Math.floor(Math.random() * 35), 92),
      snippet: `Another simulated result showing how the enhanced plagiarism detection would display found matches with TF-IDF and Jaccard similarity algorithms for improved accuracy.`
    }
  ];
};
