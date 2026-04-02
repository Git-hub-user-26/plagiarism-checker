
import React, { useState } from "react";
import { motion } from "framer-motion";
import { Search, Database, Upload } from "lucide-react";
import { cn } from "@/lib/utils";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { 
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";

export type PlagiarismMethodType = "searchEngine" | "database";
export type DatabaseSourceType = "research" | "educational" | "assignments" | "books";

interface PlagiarismMethodProps {
  selectedMethod: PlagiarismMethodType;
  onSelectMethod: (method: PlagiarismMethodType) => void;
  selectedDatabaseSource?: DatabaseSourceType;
  onSelectDatabaseSource?: (source: DatabaseSourceType) => void;
  onFileUpload?: (files: File[]) => void;
}

const PlagiarismMethod: React.FC<PlagiarismMethodProps> = ({
  selectedMethod,
  onSelectMethod,
  selectedDatabaseSource = "research",
  onSelectDatabaseSource = () => {},
  onFileUpload = () => {},
}) => {
  const [studentFiles, setStudentFiles] = useState<File[]>([]);

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files.length > 0) {
      const filesArray = Array.from(e.target.files);
      setStudentFiles(filesArray);
      onFileUpload(filesArray);
    }
  };

  return (
    <div className="w-full max-w-lg mx-auto mt-8">
      <div className="bg-card/30 glass-morphism rounded-xl p-1 grid grid-cols-2 gap-2">
        <MethodOption
          id="searchEngine"
          title="Search Engine"
          description="Compares text with web content to find exact matches"
          icon={<Search className="h-6 w-6" />}
          isSelected={selectedMethod === "searchEngine"}
          onClick={() => onSelectMethod("searchEngine")}
        />
        <MethodOption
          id="database"
          title="Database"
          description="Checks against our curated database of academic content"
          icon={<Database className="h-6 w-6" />}
          isSelected={selectedMethod === "database"}
          onClick={() => onSelectMethod("database")}
        />
      </div>

      {selectedMethod === "database" && (
        <motion.div
          initial={{ opacity: 0, y: -10 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.3 }}
          className="mt-4 glass-morphism rounded-lg p-4"
        >
          <Tabs defaultValue="sources" className="w-full">
            <TabsList className="grid w-full grid-cols-2 mb-4">
              <TabsTrigger value="sources">Database Sources</TabsTrigger>
              <TabsTrigger value="upload">Upload Students' Work</TabsTrigger>
            </TabsList>
            
            <TabsContent value="sources" className="space-y-4">
              <div className="space-y-2">
                <label className="text-sm text-muted-foreground">Select Database Source</label>
                <Select 
                  value={selectedDatabaseSource} 
                  onValueChange={(value) => onSelectDatabaseSource(value as DatabaseSourceType)}
                >
                  <SelectTrigger className="w-full bg-background/50">
                    <SelectValue placeholder="Select source" />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="research">Published Research Papers</SelectItem>
                    <SelectItem value="educational">Educational Content</SelectItem>
                    <SelectItem value="assignments">Previously Submitted Assignments</SelectItem>
                    <SelectItem value="books">Books and Journals</SelectItem>
                  </SelectContent>
                </Select>
              </div>
            </TabsContent>
            
            <TabsContent value="upload">
              <div className="space-y-2">
                <label className="text-sm text-muted-foreground">
                  Upload student documents to check for similarities
                </label>
                <div className="border-2 border-dashed border-secondary/40 rounded-lg p-6 text-center cursor-pointer hover:bg-secondary/10 transition-colors duration-200">
                  <input
                    type="file"
                    multiple
                    accept=".doc,.docx,.pdf"
                    className="hidden"
                    id="student-file-upload"
                    onChange={handleFileChange}
                  />
                  <label htmlFor="student-file-upload" className="cursor-pointer flex flex-col items-center">
                    <Upload className="h-8 w-8 text-muted-foreground mb-2" />
                    <p className="text-sm font-medium">Click to upload documents</p>
                    <p className="text-xs text-muted-foreground mt-1">
                      Supports: PDF, Word (.doc, .docx)
                    </p>
                  </label>
                </div>
                
                {studentFiles.length > 0 && (
                  <div className="mt-3">
                    <p className="text-xs text-muted-foreground mb-2">Uploaded files:</p>
                    <div className="space-y-1">
                      {studentFiles.map((file, index) => (
                        <div key={index} className="text-sm bg-secondary/20 px-3 py-2 rounded flex justify-between items-center">
                          <span className="truncate">{file.name}</span>
                          <span className="text-xs text-muted-foreground">
                            {(file.size / 1024).toFixed(0)} KB
                          </span>
                        </div>
                      ))}
                    </div>
                  </div>
                )}
              </div>
            </TabsContent>
          </Tabs>
        </motion.div>
      )}
    </div>
  );
};

interface MethodOptionProps {
  id: PlagiarismMethodType;
  title: string;
  description: string;
  icon: React.ReactNode;
  isSelected: boolean;
  onClick: () => void;
}

const MethodOption: React.FC<MethodOptionProps> = ({
  id,
  title,
  description,
  icon,
  isSelected,
  onClick,
}) => {
  return (
    <motion.button
      whileHover={{ scale: 1.02 }}
      whileTap={{ scale: 0.98 }}
      onClick={onClick}
      className={cn(
        "relative flex flex-col items-center justify-center p-4 rounded-lg transition-all duration-200 text-center h-36",
        isSelected
          ? "bg-secondary text-foreground"
          : "text-muted-foreground hover:bg-secondary/30"
      )}
    >
      {isSelected && (
        <motion.div
          layoutId="selectedMethod"
          className="absolute inset-0 rounded-lg bg-secondary"
          initial={false}
          transition={{ type: "spring", duration: 0.6 }}
        />
      )}
      <span className="relative z-10 mb-2">{icon}</span>
      <span className="relative z-10 font-semibold mb-1">{title}</span>
      <p className="relative z-10 text-xs opacity-80">{description}</p>
    </motion.button>
  );
};

export default PlagiarismMethod;
