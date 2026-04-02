
import React from "react";
import { motion } from "framer-motion";
import { ExternalLink, FileText } from "lucide-react";

export interface Source {
  url: string;
  title: string;
  matchPercentage: number;
  snippet?: string;
}

interface SourceLinkProps {
  source: Source;
  index: number;
}

const SourceLink: React.FC<SourceLinkProps> = ({ source, index }) => {
  // Check if the source is from an uploaded document
  const isDocumentSource = source.url.startsWith('#document-');
  
  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.3, delay: index * 0.1 }}
      className="glass-morphism rounded-lg p-4 hover:bg-secondary/40 transition-all duration-300"
    >
      <div className="flex flex-col gap-2">
        <div className="flex justify-between items-start">
          {isDocumentSource ? (
            <div className="text-primary font-medium flex items-center group">
              <FileText className="mr-2 h-4 w-4" />
              <span className="line-clamp-1">{source.title}</span>
            </div>
          ) : (
            <a
              href={source.url}
              target="_blank"
              rel="noopener noreferrer"
              className="text-primary hover:text-primary/80 transition-colors duration-200 font-medium flex items-center group"
            >
              <span className="line-clamp-1">{source.title}</span>
              <ExternalLink className="ml-1 h-3 w-3 opacity-70 group-hover:opacity-100 transition-opacity duration-200" />
            </a>
          )}
          <div className="glass-morphism bg-secondary/30 px-2 py-0.5 rounded-full text-xs font-medium text-foreground">
            {source.matchPercentage}% Match
          </div>
        </div>
        
        {source.snippet && (
          <div className="text-sm text-muted-foreground border-l-2 border-primary/30 pl-3 mt-1">
            <p className="line-clamp-2">{source.snippet}</p>
          </div>
        )}
        
        <div className="text-xs text-muted-foreground truncate mt-1">
          {isDocumentSource ? (
            <span className="italic">Uploaded document reference</span>
          ) : (
            source.url
          )}
        </div>
      </div>
    </motion.div>
  );
};

export default SourceLink;
