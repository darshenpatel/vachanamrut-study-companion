import React, { useState, useEffect } from 'react';
import { cn } from '@/utils/formatting';
import { apiService } from '@/services/api';

interface ThemeSelectorProps {
  selectedTheme?: string;
  onThemeSelect: (theme: string | undefined) => void;
  className?: string;
}

export const ThemeSelector: React.FC<ThemeSelectorProps> = ({
  selectedTheme,
  onThemeSelect,
  className
}) => {
  const [themes, setThemes] = useState<string[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const [isExpanded, setIsExpanded] = useState(false);

  useEffect(() => {
    const loadThemes = async () => {
      setIsLoading(true);
      try {
        const themeList = await apiService.getThemes();
        setThemes(themeList);
      } catch (error) {
        console.error('Failed to load themes:', error);
      } finally {
        setIsLoading(false);
      }
    };

    loadThemes();
  }, []);

  const handleThemeSelect = (theme: string) => {
    if (selectedTheme === theme) {
      onThemeSelect(undefined); // Deselect if same theme
    } else {
      onThemeSelect(theme);
    }
    setIsExpanded(false);
  };

  const formatThemeName = (theme: string) => {
    return theme.charAt(0).toUpperCase() + theme.slice(1);
  };

  return (
    <div className={cn('relative', className)}>
      {/* Theme selector button */}
      <button
        onClick={() => setIsExpanded(!isExpanded)}
        className={cn(
          'btn btn-secondary min-w-[120px] justify-between text-sm',
          selectedTheme && 'border-purple-300 bg-purple-50 text-purple-700'
        )}
        disabled={isLoading}
      >
        <span className="flex items-center truncate">
          {isLoading ? (
            <div className="w-4 h-4 border-2 border-gray-300 border-t-gray-600 rounded-full animate-spin mr-2" />
          ) : (
            <svg className="w-4 h-4 mr-2 text-purple-500" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M7 7h.01M7 3h5c.512 0 1.024.195 1.414.586l7 7a2 2 0 010 2.828l-7 7a2 2 0 01-2.828 0l-7-7A1.994 1.994 0 013 12V7a4 4 0 014-4z" />
            </svg>
          )}
          {selectedTheme ? formatThemeName(selectedTheme) : 'All Themes'}
        </span>
        
        <svg
          className={cn(
            'w-4 h-4 transition-transform ml-2 flex-shrink-0',
            isExpanded ? 'rotate-180' : 'rotate-0'
          )}
          fill="none"
          stroke="currentColor"
          viewBox="0 0 24 24"
        >
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
        </svg>
      </button>

      {/* Dropdown menu */}
      {isExpanded && (
        <div className="absolute top-full left-0 right-0 mt-2 card shadow-lg z-50 max-h-60 overflow-y-auto animate-fade-in">
          {/* All themes option */}
          <button
            onClick={() => handleThemeSelect('')}
            className={cn(
              'w-full px-3 py-2 text-left text-sm hover:bg-gray-50 flex items-center transition-colors',
              !selectedTheme && 'bg-purple-50 text-purple-700'
            )}
          >
            <svg className="w-4 h-4 mr-3 text-gray-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 11H5m14 0a2 2 0 012 2v6a2 2 0 01-2 2H5a2 2 0 01-2-2v-6a2 2 0 012-2m14 0V9a2 2 0 00-2-2M5 11V9a2 2 0 012-2m0 0V5a2 2 0 012-2h6a2 2 0 012 2v2M7 7h10" />
            </svg>
            <span>All Themes</span>
            {!selectedTheme && (
              <svg className="w-4 h-4 ml-auto text-purple-600" fill="currentColor" viewBox="0 0 24 24">
                <path d="M9 16.17L4.83 12l-1.42 1.41L9 19 21 7l-1.41-1.41z" />
              </svg>
            )}
          </button>

          {/* Individual themes */}
          {themes.map((theme) => (
            <button
              key={theme}
              onClick={() => handleThemeSelect(theme)}
              className={cn(
                'w-full px-3 py-2 text-left text-sm hover:bg-gray-50 flex items-center transition-colors border-t border-gray-100',
                selectedTheme === theme && 'bg-purple-50 text-purple-700'
              )}
            >
              <svg className="w-4 h-4 mr-3 text-purple-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M7 7h.01M7 3h5c.512 0 1.024.195 1.414.586l7 7a2 2 0 010 2.828l-7 7a2 2 0 01-2.828 0l-7-7A1.994 1.994 0 013 12V7a4 4 0 014-4z" />
              </svg>
              <span>{formatThemeName(theme)}</span>
              {selectedTheme === theme && (
                <svg className="w-4 h-4 ml-auto text-purple-600" fill="currentColor" viewBox="0 0 24 24">
                  <path d="M9 16.17L4.83 12l-1.42 1.41L9 19 21 7l-1.41-1.41z" />
                </svg>
              )}
            </button>
          ))}

          {themes.length === 0 && !isLoading && (
            <div className="px-3 py-4 text-center text-sm text-gray-500">
              No themes available
            </div>
          )}
        </div>
      )}

      {/* Click outside to close */}
      {isExpanded && (
        <div
          className="fixed inset-0 z-40"
          onClick={() => setIsExpanded(false)}
        />
      )}
    </div>
  );
};