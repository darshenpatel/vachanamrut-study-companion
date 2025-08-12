import React from 'react';
import { cn } from '@/utils/formatting';

interface TypingIndicatorProps {
  className?: string;
}

export const TypingIndicator: React.FC<TypingIndicatorProps> = ({
  className
}) => {
  return (
    <div className={cn('flex items-center space-x-2 p-4', className)}>
      <div className="flex items-center space-x-1">
        <div className="bg-gray-100 rounded-2xl rounded-bl-md px-4 py-3">
          <div className="flex space-x-1">
            <div className="w-2 h-2 bg-gray-400 rounded-full animate-bounce [animation-delay:-0.3s]"></div>
            <div className="w-2 h-2 bg-gray-400 rounded-full animate-bounce [animation-delay:-0.15s]"></div>
            <div className="w-2 h-2 bg-gray-400 rounded-full animate-bounce"></div>
          </div>
        </div>
      </div>
      <span className="text-xs text-gray-500">AI is thinking...</span>
    </div>
  );
};