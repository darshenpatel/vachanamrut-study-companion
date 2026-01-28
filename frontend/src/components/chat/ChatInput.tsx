import React, { useState, useRef, useEffect } from 'react';
import { cn } from '@/utils/formatting';

interface ChatInputProps {
  onSendMessage: (message: string) => void;
  disabled?: boolean;
  placeholder?: string;
  className?: string;
}

export const ChatInput: React.FC<ChatInputProps> = ({
  onSendMessage,
  disabled = false,
  placeholder = "Ask a question about spiritual guidance...",
  className
}) => {
  const [message, setMessage] = useState('');
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    const currentValue = (textareaRef.current?.value ?? message).trim();
    if (!currentValue || disabled) return;

    onSendMessage(currentValue);
    setMessage('');
    if (textareaRef.current) {
      // Keep the DOM value in sync even if some environments don't trigger React's onChange reliably.
      textareaRef.current.value = '';
      textareaRef.current.style.height = '44px';
    }
  };

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSubmit(e);
    }
  };

  // Auto-resize textarea
  useEffect(() => {
    if (textareaRef.current) {
      textareaRef.current.style.height = '44px';
      const scrollHeight = textareaRef.current.scrollHeight;
      textareaRef.current.style.height = Math.min(scrollHeight, 120) + 'px';
    }
  }, [message]);

  return (
    <form onSubmit={handleSubmit} className={cn('relative', className)}>
      <div className="relative flex items-end space-x-3">
        <div className="flex-1 relative">
          <textarea
            ref={textareaRef}
            value={message}
            onChange={(e) => setMessage(e.target.value)}
            onInput={(e) => setMessage((e.target as HTMLTextAreaElement).value)}
            onKeyDown={handleKeyDown}
            placeholder={placeholder}
            disabled={disabled}
            rows={1}
            className="input resize-none min-h-[44px] max-h-[120px] py-3 pr-12"
            style={{ height: '44px' }}
          />
          
          {/* Send button inside input */}
          <button
            type="submit"
            disabled={disabled || !message.trim()}
            className={cn(
              'absolute right-2 top-1/2 -translate-y-1/2 w-8 h-8 rounded-lg flex items-center justify-center transition-all',
              message.trim() && !disabled
                ? 'bg-purple-600 text-white hover:bg-purple-700 shadow-sm'
                : 'bg-gray-100 text-gray-400 cursor-not-allowed'
            )}
          >
            {disabled ? (
              <div className="w-4 h-4 border-2 border-gray-300 border-t-gray-600 rounded-full animate-spin" />
            ) : (
              <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 19l9 2-9-18-9 18 9-2zm0 0v-8" />
              </svg>
            )}
          </button>
        </div>
      </div>
      
      {/* Helper text */}
      <p className="text-xs text-gray-400 mt-2 px-1">
        Press Enter to send, Shift + Enter for new line
      </p>
    </form>
  );
};