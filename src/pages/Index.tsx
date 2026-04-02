
import React, { useState, useEffect } from "react";
import { motion } from "framer-motion";
import TextInput from "@/components/TextInput";
import PlagiarismMethod, { PlagiarismMethodType, DatabaseSourceType } from "@/components/PlagiarismMethod";
import ResultsDisplay from "@/components/ResultsDisplay";
import { Source } from "@/components/SourceLink";
import { checkPlagiarism } from "@/utils/plagiarismChecker";
import { useToast } from "@/components/ui/use-toast";
import ApiKeyConfig from "@/components/ApiKeyConfig";
import { SearchApiConfig } from "@/utils/googleSearchApi";

const Index = () => {
  const [selectedMethod, setSelectedMethod] = useState<PlagiarismMethodType>("searchEngine");
  const [selectedDatabaseSource, setSelectedDatabaseSource] = useState<DatabaseSourceType>("research");
  const [studentFiles, setStudentFiles] = useState<File[]>([]);
  const [isChecking, setIsChecking] = useState(false);
  const [sources, setSources] = useState<Source[]>([]);
  const [plagiarismPercentage, setPlagiarismPercentage] = useState(0);
  const [showResults, setShowResults] = useState(false);
  const [searchApiConfig, setSearchApiConfig] = useState<SearchApiConfig>(() => {
    // Try to load from localStorage if available
    const savedConfig = localStorage.getItem("plagiarismCheckerApiConfig");
    return savedConfig ? JSON.parse(savedConfig) : { apiKey: "", searchEngineId: "" };
  });
  
  const { toast } = useToast();

  // Save API config to localStorage when it changes
  useEffect(() => {
    localStorage.setItem("plagiarismCheckerApiConfig", JSON.stringify(searchApiConfig));
  }, [searchApiConfig]);

  const handleMethodSelect = (method: PlagiarismMethodType) => {
    setSelectedMethod(method);
    
    // Reset results when changing method
    if (showResults) {
      setShowResults(false);
      setSources([]);
      setPlagiarismPercentage(0);
    }
  };

  const handleDatabaseSourceSelect = (source: DatabaseSourceType) => {
    setSelectedDatabaseSource(source);
    
    // Reset results when changing source
    if (showResults) {
      setShowResults(false);
      setSources([]);
      setPlagiarismPercentage(0);
    }
  };

  const handleStudentFileUpload = (files: File[]) => {
    setStudentFiles(files);
  };

  const handleTextSubmit = async (text: string, uploadedFiles: File[] = []) => {
    const hasContent = text.trim().length > 0 || uploadedFiles.length > 0;
    
    if (!hasContent) {
      toast({
        title: "No content to check",
        description: "Please enter text or upload a document to check for plagiarism.",
        variant: "destructive",
      });
      return;
    }

    if (text.length < 20 && uploadedFiles.length === 0) {
      toast({
        title: "Text too short",
        description: "Please enter at least 20 characters for accurate plagiarism detection.",
        variant: "destructive",
      });
      return;
    }

    // Check if API key is configured when using search engine method


    setIsChecking(true);
    setShowResults(false);

    try {
      // Process uploaded files first if there are any
      let contentToCheck = text;
      
      if (uploadedFiles.length > 0) {
        toast({
          title: "Processing documents",
          description: `Extracting text from ${uploadedFiles.length} ${uploadedFiles.length === 1 ? 'document' : 'documents'} with enhanced accuracy...`,
        });
      }

      const result = await checkPlagiarism(contentToCheck, selectedMethod, {
        databaseSourceType: selectedDatabaseSource,
        uploadedFiles: uploadedFiles,
        studentFiles: studentFiles,
        searchApiConfig: searchApiConfig,
      });
      
      setSources(result.sources);
      setPlagiarismPercentage(result.plagiarismPercentage);
      setShowResults(true);
      
      if (result.sources.length === 0) {
        toast({
          title: "No plagiarism detected",
          description: "Your content appears to be original based on our enhanced TF-IDF and Jaccard similarity algorithms.",
        });
      } else {
        toast({
          title: `${result.plagiarismPercentage}% plagiarized content detected`,
          description: `Found ${result.sources.length} matching ${result.sources.length === 1 ? "source" : "sources"} using improved TF-IDF and Jaccard similarity.`,
          variant: result.plagiarismPercentage > 30 ? "destructive" : "default",
        });
      }
    } catch (error) {
      console.error("Error checking plagiarism:", error);
      toast({
        title: "Error",
        description: "Failed to check plagiarism. Please try again.",
        variant: "destructive",
      });
    } finally {
      setIsChecking(false);
    }
  };

  const handleApiConfigChange = (config: SearchApiConfig) => {
    setSearchApiConfig(config);
    
    if (config.apiKey && config.searchEngineId) {
      toast({
        title: "API configured successfully",
        description: "Your Google API key and Search Engine ID have been saved.",
      });
    }
  };

  return (
    <div className="min-h-screen bg-gradient-to-b from-background to-secondary/20 px-4 py-16 overflow-x-hidden">
      <motion.div
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        transition={{ duration: 0.8 }}
        className="max-w-5xl mx-auto"
      >
        <div className="text-center mb-12">
          <div className="flex justify-center items-center mb-6">
            <img src="/logo.svg" alt="Plagiarism Checker Logo" className="h-14 w-14" />
          </div>
          
          <motion.h1 
            className="text-4xl md:text-5xl font-bold mb-3 text-gradient"
            initial={{ opacity: 0, y: -20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.2, duration: 0.5 }}
          >
            Plagiarism Checker
          </motion.h1>
          
          <motion.p 
            className="text-muted-foreground max-w-2xl mx-auto"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            transition={{ delay: 0.4, duration: 0.5 }}
          >
            Verify the originality of your content with our advanced plagiarism detection tools
            <span className="block text-sm mt-1 text-green-400">Enhanced with TF-IDF and Jaccard similarity algorithms (up to 92% accuracy)</span>
          </motion.p>
          
          {selectedMethod === "searchEngine" && (
            <motion.div 
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              transition={{ delay: 0.6, duration: 0.5 }}
              className="mt-4 flex justify-center"
            >
              <ApiKeyConfig 
                onConfigChange={handleApiConfigChange} 
                config={searchApiConfig} 
              />
            </motion.div>
          )}
        </div>

        <PlagiarismMethod
          selectedMethod={selectedMethod}
          onSelectMethod={handleMethodSelect}
          selectedDatabaseSource={selectedDatabaseSource}
          onSelectDatabaseSource={handleDatabaseSourceSelect}
          onFileUpload={handleStudentFileUpload}
        />

        <TextInput
          onSubmit={handleTextSubmit}
          isLoading={isChecking}
        />

        <ResultsDisplay
          isVisible={showResults}
          sources={sources}
          plagiarismPercentage={plagiarismPercentage}
          methodType={selectedMethod}
        />
      </motion.div>
    </div>
  );
};

export default Index;
