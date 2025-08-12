import { useState, useCallback } from 'react';
import type { Message, ChatRequest } from '@/types';
import { apiService } from '@/services/api';

export const useChat = () => {
  const [messages, setMessages] = useState<Message[]>([
    {
      id: '1',
      content: "Welcome to the Vachanamrut Study Companion! I'm here to help you explore the spiritual teachings. Feel free to ask me any questions about devotion, faith, surrender, or any other aspect of the Vachanamrut.",
      isUser: false,
      timestamp: new Date(),
    }
  ]);
  const [isLoading, setIsLoading] = useState(false);
  const [selectedTheme, setSelectedTheme] = useState<string | undefined>();
  const [error, setError] = useState<string | null>(null);

  const sendMessage = useCallback(async (content: string) => {
    if (!content.trim() || isLoading) return;

    // Clear any previous error
    setError(null);

    // Add user message
    const userMessage: Message = {
      id: Date.now().toString(),
      content,
      isUser: true,
      timestamp: new Date(),
    };

    setMessages(prev => [...prev, userMessage]);
    setIsLoading(true);

    try {
      const request: ChatRequest = {
        message: content,
        theme: selectedTheme,
      };

      const response = await apiService.sendMessage(request);

      // Add AI response
      const aiMessage: Message = {
        id: (Date.now() + 1).toString(),
        content: response.response,
        isUser: false,
        timestamp: new Date(response.timestamp),
        citations: response.citations,
        relatedThemes: response.relatedThemes,
      };

      setMessages(prev => [...prev, aiMessage]);
    } catch (err) {
      console.error('Failed to send message:', err);
      setError('We encountered an error while contacting the server. Please try again.');
      // Add error message
      const errorMessage: Message = {
        id: (Date.now() + 1).toString(),
        content: 'I apologize, but I encountered an error while processing your request. Please try again.',
        isUser: false,
        timestamp: new Date(),
      };

      setMessages(prev => [...prev, errorMessage]);
    } finally {
      setIsLoading(false);
    }
  }, [selectedTheme, isLoading]);

  const clearMessages = useCallback(() => {
    setMessages([{
      id: '1',
      content: "Welcome to the Vachanamrut Study Companion! I'm here to help you explore the spiritual teachings. Feel free to ask me any questions about devotion, faith, surrender, or any other aspect of the Vachanamrut.",
      isUser: false,
      timestamp: new Date(),
    }]);
    setError(null);
  }, []);

  const clearError = useCallback(() => setError(null), []);

  return {
    messages,
    isLoading,
    selectedTheme,
    setSelectedTheme,
    sendMessage,
    clearMessages,
    error,
    clearError,
  };
};