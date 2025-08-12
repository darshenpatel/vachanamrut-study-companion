import type { ChatRequest, ChatResponse, ThemeDetail } from '@/types';
import { z } from 'zod';
import { ChatResponseSchema, ThemeDetailSchema } from '@/types/apiSchemas';

// Use environment variable for API URL
const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';

// Debug logging
console.log('Environment variables:', import.meta.env);
console.log('API_BASE_URL:', API_BASE_URL);

class ApiService {
  private async request<T>(endpoint: string, options?: RequestInit): Promise<T> {
    const url = `${API_BASE_URL}/api${endpoint}`;
    
    const response = await fetch(url, {
      headers: {
        'Content-Type': 'application/json',
        ...options?.headers,
      },
      ...options,
    });

    if (!response.ok) {
      throw new Error(`API request failed: ${response.status} ${response.statusText}`);
    }

    return response.json();
  }

  async sendMessage(request: ChatRequest): Promise<ChatResponse> {
    const raw = await this.request<unknown>('/chat/', {
      method: 'POST',
      body: JSON.stringify(request),
    });
    const parsed = ChatResponseSchema.parse(raw);
    return parsed as ChatResponse;
  }

  async getThemes(): Promise<string[]> {
    const raw = await this.request<unknown>('/themes/');
    const schema = z.array(z.string());
    return schema.parse(raw);
  }

  async getThemeDetail(themeName: string): Promise<ThemeDetail> {
    const raw = await this.request<unknown>(`/themes/${themeName}`);
    const parsed = ThemeDetailSchema.parse(raw);
    return parsed as ThemeDetail;
  }

  async healthCheck(): Promise<{ status: string; message: string }> {
    const url = `${API_BASE_URL}/health`;
    const response = await fetch(url);
    if (!response.ok) {
      throw new Error(`Health check failed: ${response.status}`);
    }
    return response.json();
  }
}

export const apiService = new ApiService();