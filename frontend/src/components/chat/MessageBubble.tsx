import React from 'react';
import { cn, formatTimestamp, formatReference } from '@/utils/formatting';
import type { Message } from '@/types';

interface MessageBubbleProps {
  message: Message;
  className?: string;
}

export const MessageBubble: React.FC<MessageBubbleProps> = ({
  message,
  className
}) => {
  return (
    <div className={cn('animate-fade-in', className)}>
      {message.isUser ? (
        // User message - right aligned
        <div className="flex justify-end mb-6">
          <div className="flex flex-col items-end max-w-md">
            <div className="message-bubble-user">
              <p className="text-sm leading-relaxed whitespace-pre-wrap">
                {message.content}
              </p>
            </div>
            <time className="text-xs text-gray-400 mt-1 px-2">
              {formatTimestamp(message.timestamp)}
            </time>
          </div>
        </div>
      ) : (
        // Assistant message - left aligned
        <div className="flex justify-start mb-8">
          <div className="flex flex-col max-w-3xl">
            <div className="flex items-start space-x-3">
              {/* Avatar */}
              <div className="flex-shrink-0 w-8 h-8 bg-gradient-to-br from-purple-600 to-purple-700 rounded-lg flex items-center justify-center">
                <svg className="w-4 h-4 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 6.253v13m0-13C10.832 5.477 9.246 5 7.5 5S4.168 5.477 3 6.253v13C4.168 18.477 5.754 18 7.5 18s3.332.477 4.5 1.253m0-13C13.168 5.477 14.754 5 16.5 5c1.746 0 3.332.477 4.5 1.253v13C19.832 18.477 18.246 18 16.5 18c-1.746 0-3.332.477-4.5 1.253" />
                </svg>
              </div>
              
              {/* Message content */}
              <div className="flex-1 min-w-0">
                <div className="message-bubble-assistant">
                  <div className="text-sm text-gray-900 leading-relaxed whitespace-pre-wrap">
                    {message.content}
                  </div>
                </div>
                
                {/* Citations */}
                {message.citations && message.citations.length > 0 && (
                  <div className="mt-4 space-y-3">
                    {message.citations.map((citation, index) => (
                      <div key={index} className="citation animate-slide-up">
                        <div className="citation-reference">
                          {formatReference(citation.reference)}
                          {citation.pageNumber && ` (p. ${citation.pageNumber})`}
                          {citation.relevanceScore && (
                            <span className="ml-2 text-purple-600">
                              {Math.round(citation.relevanceScore * 100)}% match
                            </span>
                          )}
                        </div>
                        <blockquote className="citation-text mt-1">
                          "{citation.passage}"
                        </blockquote>
                      </div>
                    ))}
                  </div>
                )}

                {/* Related Themes */}
                {message.relatedThemes && message.relatedThemes.length > 0 && (
                  <div className="mt-3 flex flex-wrap gap-1.5">
                    {message.relatedThemes.map((theme, index) => (
                      <span key={index} className="theme-badge">
                        {theme}
                      </span>
                    ))}
                  </div>
                )}

                {/* Timestamp */}
                <time className="text-xs text-gray-400 mt-2 block">
                  {formatTimestamp(message.timestamp)}
                </time>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};