
import React, { useState } from "react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { motion } from "framer-motion";
import { SearchApiConfig } from "@/utils/googleSearchApi";
import { Settings, X } from "lucide-react";

interface ApiKeyConfigProps {
  onConfigChange: (config: SearchApiConfig) => void;
  config: SearchApiConfig;
}

const ApiKeyConfig: React.FC<ApiKeyConfigProps> = ({ onConfigChange, config }) => {
  const [isOpen, setIsOpen] = useState(false);
  const [apiKey, setApiKey] = useState(config.apiKey || "");
  const [searchEngineId, setSearchEngineId] = useState(config.searchEngineId || "");

  const handleSave = () => {
    onConfigChange({
      apiKey,
      searchEngineId
    });
    setIsOpen(false);
  };

  const hasConfig = config.apiKey && config.searchEngineId;

  return (
    <div className="mb-6 relative">
      <Button 
        variant={hasConfig ? "default" : "outline"} 
        className="flex items-center gap-2"
        onClick={() => setIsOpen(!isOpen)}
        size="sm"
      >
        <Settings className="h-4 w-4" />
        {hasConfig ? "API Configured" : "Configure Google API"}
      </Button>
      
      {isOpen && (
        <motion.div
          initial={{ opacity: 0, y: -10 }}
          animate={{ opacity: 1, y: 0 }}
          className="absolute top-full left-0 mt-2 p-4 bg-card border rounded-lg shadow-md z-10 w-full md:w-96"
        >
          <div className="flex justify-between items-center mb-4">
            <h3 className="font-medium">Google Search API Configuration</h3>
            <Button variant="ghost" size="icon" onClick={() => setIsOpen(false)}>
              <X className="h-4 w-4" />
            </Button>
          </div>
          
          <div className="space-y-4">
            <div className="space-y-2">
              <Label htmlFor="apiKey">Google API Key</Label>
              <Input
                id="apiKey"
                type="password"
                value={apiKey}
                onChange={(e) => setApiKey(e.target.value)}
                placeholder="Enter your Google API Key"
              />
            </div>
            
            <div className="space-y-2">
              <Label htmlFor="searchEngineId">Search Engine ID (CSE ID)</Label>
              <Input
                id="searchEngineId"
                value={searchEngineId}
                onChange={(e) => setSearchEngineId(e.target.value)}
                placeholder="Enter your Custom Search Engine ID"
              />
            </div>
            
            <div className="text-xs text-muted-foreground">
              <p>Your API key and Search Engine ID are stored in your browser only and are not sent to our servers.</p>
            </div>
            
            <div className="flex justify-end gap-2 pt-2">
              <Button variant="outline" size="sm" onClick={() => setIsOpen(false)}>
                Cancel
              </Button>
              <Button size="sm" onClick={handleSave}>
                Save Configuration
              </Button>
            </div>
          </div>
        </motion.div>
      )}
    </div>
  );
};

export default ApiKeyConfig;
