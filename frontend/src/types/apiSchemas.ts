import { z } from 'zod';

export const CitationSchema = z.object({
  reference: z.string(),
  passage: z.string(),
  pageNumber: z.number().optional(),
  relevanceScore: z.number().optional(),
});

export const ChatResponseSchema = z.object({
  response: z.string(),
  citations: z.array(CitationSchema),
  relatedThemes: z.array(z.string()),
  timestamp: z.string(),
});

export const ThemeDetailSchema = z.object({
  name: z.string(),
  description: z.string(),
  keywords: z.array(z.string()),
  relatedPassages: z.array(z.string()),
  relatedThemes: z.array(z.string()),
});

export type ChatResponseParsed = z.infer<typeof ChatResponseSchema>;
export type ThemeDetailParsed = z.infer<typeof ThemeDetailSchema>; 