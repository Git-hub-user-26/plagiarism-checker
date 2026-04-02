import React, { useState, useRef, useEffect } from "react";
import { Textarea } from "@/components/ui/textarea";
import { Button } from "@/components/ui/button";
import { Search, Upload, FileText, FileCode } from "lucide-react";
import { motion, AnimatePresence } from "framer-motion";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";

interface TextInputProps {
  onSubmit: (text: string, files?: File[]) => void;
  isLoading: boolean;
}

const TextInput: React.FC<TextInputProps> = ({ onSubmit, isLoading }) => {
  const [text, setText] = useState("");
  const [files, setFiles] = useState<File[]>([]);
  const textareaRef = useRef<HTMLTextAreaElement>(null);
  const [charCount, setCharCount] = useState(0);
  const maxChars = 5000;
  const fileInputRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    setCharCount(text.length);
  }, [text]);

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    const hasValidInput = text.trim() || files.length > 0;
    
    if (hasValidInput && !isLoading) {
      onSubmit(text, files);
    }
  };

  const handlePaste = async () => {
    try {
      const clipboardText = await navigator.clipboard.readText();
      setText(clipboardText);
    } catch (err) {
      console.error("Failed to read clipboard:", err);
    }
  };

  const handleClear = () => {
    setText("");
    setFiles([]);
    if (textareaRef.current) {
      textareaRef.current.focus();
    }
  };

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files.length > 0) {
      const newFiles = Array.from(e.target.files);
      setFiles(prev => [...prev, ...newFiles]);
    }
  };

  const removeFile = (index: number) => {
    setFiles(files.filter((_, i) => i !== index));
  };

  const handleDrop = (e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    
    if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
      const droppedFiles = Array.from(e.dataTransfer.files).filter(file => {
        const ext = file.name.split('.').pop()?.toLowerCase();
        return ext === 'pdf' || ext === 'doc' || ext === 'docx';
      });
      
      if (droppedFiles.length > 0) {
        setFiles(prev => [...prev, ...droppedFiles]);
      }
    }
  };

  const preventDefaults = (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
  };

  const getFileIcon = (fileName: string) => {
    const extension = fileName.split('.').pop()?.toLowerCase();
    if (extension === 'pdf') {
      return <FileText className="h-4 w-4 text-rose-400" />;
    }
    return <FileCode className="h-4 w-4 text-blue-400" />;
  };

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.5 }}
      className="w-full max-w-3xl mx-auto mt-8"
    >
      <form onSubmit={handleSubmit} className="space-y-4">
        <Tabs defaultValue="text" className="w-full">
          <TabsList className="grid w-full grid-cols-2 mb-2">
            <TabsTrigger value="text">Enter Text</TabsTrigger>
            <TabsTrigger value="document">Upload Document</TabsTrigger>
          </TabsList>
          
          <TabsContent value="text">
            <div className="relative">
              <Textarea
                ref={textareaRef}
                value={text}
                onChange={(e) => setText(e.target.value)}
                placeholder="Paste or type text to check for plagiarism..."
                className="min-h-[200px] resize-y glass-morphism bg-secondary/20 border-secondary/30 focus:border-primary/50 text-foreground placeholder:text-muted-foreground transition-all duration-300"
                maxLength={maxChars}
              />
              <div className="absolute bottom-2 right-2 text-xs text-muted-foreground">
                {charCount}/{maxChars}
              </div>
            </div>
          </TabsContent>
          
          <TabsContent value="document">
            <div 
              className="border-2 border-dashed border-secondary/40 rounded-lg p-8 text-center cursor-pointer hover:bg-secondary/10 transition-colors duration-200 min-h-[200px] flex flex-col items-center justify-center"
              onDrop={handleDrop}
              onDragOver={preventDefaults}
              onDragEnter={preventDefaults}
              onDragLeave={preventDefaults}
              onClick={() => fileInputRef.current?.click()}
            >
              <input
                type="file"
                ref={fileInputRef}
                className="hidden"
                accept=".doc,.docx,.pdf"
                onChange={handleFileChange}
                multiple
              />
              <Upload className="h-10 w-10 text-muted-foreground mb-4" />
              <h3 className="text-lg font-medium mb-1">Upload Documents</h3>
              <p className="text-sm text-muted-foreground mb-4">
                Drag and drop your files here or click to browse
              </p>
              <Button 
                type="button"
                variant="outline"
                onClick={(e) => {
                  e.stopPropagation();
                  fileInputRef.current?.click();
                }}
                className="bg-secondary/30"
              >
                Choose Files
              </Button>
              <p className="text-xs text-muted-foreground mt-3">
                Supports: PDF, Word (.doc, .docx)
              </p>
            </div>
            
            {files.length > 0 && (
              <div className="mt-4 space-y-2">
                <h4 className="text-sm font-medium">Uploaded Files:</h4>
                <div className="space-y-2 max-h-[200px] overflow-y-auto pr-2">
                  {files.map((file, index) => (
                    <div key={index} className="flex items-center justify-between bg-secondary/20 p-2 rounded">
                      <div className="flex items-center">
                        {getFileIcon(file.name)}
                        <span className="ml-2 text-sm truncate max-w-[200px]">{file.name}</span>
                        <span className="ml-2 text-xs text-muted-foreground">
                          ({(file.size / 1024).toFixed(0)} KB)
                        </span>
                      </div>
                      <Button 
                        type="button" 
                        variant="ghost" 
                        size="sm"
                        onClick={(e) => {
                          e.stopPropagation();
                          removeFile(index);
                        }}
                        className="h-6 w-6 p-0"
                      >
                        ✕
                      </Button>
                    </div>
                  ))}
                </div>
              </div>
            )}
          </TabsContent>
        </Tabs>

        <div className="flex flex-wrap gap-3 justify-between items-center">
          <div className="flex gap-2">
            <Button 
              type="button" 
              variant="outline" 
              onClick={handlePaste}
              className="glass-morphism bg-secondary/20 hover:bg-secondary/40"
            >
              Paste
            </Button>
            <Button 
              type="button" 
              variant="outline" 
              onClick={handleClear}
              className="glass-morphism bg-secondary/20 hover:bg-secondary/40"
            >
              Clear
            </Button>
          </div>

          <Button 
            type="submit" 
            disabled={(text.trim() === "" && files.length === 0) || isLoading}
            className="gradient-border bg-primary/80 hover:bg-primary text-primary-foreground"
          >
            <AnimatePresence mode="wait">
              {isLoading ? (
                <motion.div
                  key="loading"
                  initial={{ opacity: 0 }}
                  animate={{ opacity: 1 }}
                  exit={{ opacity: 0 }}
                  className="flex items-center"
                >
                  <div className="mr-2 h-4 w-4 animate-spin rounded-full border-2 border-primary-foreground border-t-transparent"></div>
                  Checking...
                </motion.div>
              ) : (
                <motion.div
                  key="check"
                  initial={{ opacity: 0 }}
                  animate={{ opacity: 1 }}
                  exit={{ opacity: 0 }}
                  className="flex items-center"
                >
                  <Search className="mr-2 h-4 w-4" />
                  Check Plagiarism
                </motion.div>
              )}
            </AnimatePresence>
          </Button>
        </div>
      </form>
    </motion.div>
  );
};

export default TextInput;
