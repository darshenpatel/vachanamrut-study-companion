export interface Message {
  id: string;
  content: string;
  isUser: boolean;
  timestamp: Date;
  citations?: Citation[];
  relatedThemes?: string[];
}

export interface Citation {
  reference: string;
  passage: string;
  pageNumber?: number;
  relevanceScore?: number;
}

export interface ChatRequest {
  message: string;
  theme?: string;
  context?: string[];
}

export interface ChatResponse {
  response: string;
  citations: Citation[];
  relatedThemes: string[];
  timestamp: string;
}

export interface Theme {
  name: string;
  description?: string;
  keywords?: string[];
}

export interface ThemeDetail extends Theme {
  relatedPassages: string[];
  relatedThemes: string[];
}