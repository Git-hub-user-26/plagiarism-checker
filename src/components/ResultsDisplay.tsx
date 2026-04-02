
import React from "react";
import { motion, AnimatePresence } from "framer-motion";
import SourceLink, { Source } from "./SourceLink";
import { PlagiarismMethodType } from "./PlagiarismMethod";
import { CircleDashed, AlertTriangle, CheckCircle } from "lucide-react";

interface ResultsDisplayProps {
  isVisible: boolean;
  sources: Source[];
  plagiarismPercentage: number;
  methodType: PlagiarismMethodType;
}

const ResultsDisplay: React.FC<ResultsDisplayProps> = ({
  isVisible,
  sources,
  plagiarismPercentage,
  methodType,
}) => {
  if (!isVisible) return null;

  const getPlagiarismSeverity = (percentage: number) => {
    if (percentage === 0) return "none";
    if (percentage < 15) return "low";
    if (percentage < 40) return "medium";
    return "high";
  };

  const severity = getPlagiarismSeverity(plagiarismPercentage);

  const severityConfig = {
    none: {
      color: "bg-emerald-500",
      icon: <CheckCircle className="h-6 w-6 text-emerald-500" />,
      message: "No plagiarism detected",
    },
    low: {
      color: "bg-emerald-500",
      icon: <CheckCircle className="h-6 w-6 text-emerald-500" />,
      message: "Low plagiarism detected",
    },
    medium: {
      color: "bg-amber-500",
      icon: <AlertTriangle className="h-6 w-6 text-amber-500" />,
      message: "Moderate plagiarism detected",
    },
    high: {
      color: "bg-rose-500",
      icon: <AlertTriangle className="h-6 w-6 text-rose-500" />,
      message: "High plagiarism detected",
    },
  };

  return (
    <AnimatePresence>
      {isVisible && (
        <motion.div
          initial={{ opacity: 0, height: 0 }}
          animate={{ opacity: 1, height: "auto" }}
          exit={{ opacity: 0, height: 0 }}
          transition={{ duration: 0.5, ease: "easeInOut" }}
          className="w-full max-w-3xl mx-auto mt-10 overflow-hidden"
        >
          <div className="glass-morphism rounded-xl p-6 border border-white/5">
            <div className="flex flex-col md:flex-row gap-6 items-center mb-6">
              <div className="relative flex-shrink-0">
                <div className="w-32 h-32 rounded-full glass-morphism flex items-center justify-center">
                  <svg className="w-full h-full" viewBox="0 0 100 100">
                    <circle
                      className="text-secondary/50"
                      strokeWidth="8"
                      stroke="currentColor"
                      fill="transparent"
                      r="40"
                      cx="50"
                      cy="50"
                    />
                    {plagiarismPercentage > 0 && (
                      <circle
                        className={`${severityConfig[severity].color}`}
                        strokeWidth="8"
                        strokeDasharray={`${plagiarismPercentage * 2.51} 251`}
                        strokeLinecap="round"
                        stroke="currentColor"
                        fill="transparent"
                        r="40"
                        cx="50"
                        cy="50"
                        transform="rotate(-90 50 50)"
                      />
                    )}
                  </svg>
                  <div className="absolute inset-0 flex flex-col items-center justify-center">
                    <span className="text-2xl font-bold text-gradient">
                      {plagiarismPercentage}%
                    </span>
                    <span className="text-xs text-muted-foreground">
                      {plagiarismPercentage > 0 ? "Plagiarized" : "Original"}
                    </span>
                  </div>
                </div>
              </div>

              <div className="flex-1 text-center md:text-left">
                <div className="flex items-center justify-center md:justify-start gap-2 mb-2">
                  {severityConfig[severity].icon}
                  <h3 className="font-semibold text-lg">
                    {severityConfig[severity].message}
                  </h3>
                </div>

                <p className="text-muted-foreground text-sm mb-4">
                  {methodType === "searchEngine"
                    ? "Results based on search engine comparison"
                    : "Results based on our academic database comparison"}
                </p>

                <div className="text-xs inline-flex items-center glass-morphism bg-secondary/20 px-3 py-1.5 rounded-full">
                  <CircleDashed className="mr-1 h-3 w-3" />
                  {sources.length} matching {sources.length === 1 ? "source" : "sources"} found
                </div>
              </div>
            </div>

            {sources.length > 0 ? (
              <div className="space-y-3 mt-6">
                <h4 className="text-sm font-medium mb-3">Matching Sources:</h4>
                {sources.map((source, index) => (
                  <SourceLink key={index} source={source} index={index} />
                ))}
              </div>
            ) : (
              <div className="text-center py-6 text-muted-foreground">
                No matching sources found
              </div>
            )}
          </div>
        </motion.div>
      )}
    </AnimatePresence>
  );
};

export default ResultsDisplay;
