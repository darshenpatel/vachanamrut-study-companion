# Vachanamrut Study Companion - Frontend

React + TypeScript + Vite frontend for the Vachanamrut Study Companion.

## UI Overview

The frontend features a modern sidebar-based layout inspired by Ramp/Linear design patterns.

### Layout Structure

```
┌──────────────────────────────────────────────────┐
│ ┌─────────┐ ┌──────────────────────────────────┐ │
│ │ Sidebar │ │ Main Content Area                │ │
│ │ (240px) │ │                                  │ │
│ │         │ │ Views:                           │ │
│ │ • Home  │ │ - Home (Quick Search, Topics)    │ │
│ │ • Topics│ │ - Topics (All 6 categories)      │ │
│ │ • Search│ │ - Topic Detail (Questions)       │ │
│ │         │ │ - Search (Chat interface)        │ │
│ │ Quick   │ │                                  │ │
│ │ Access  │ │                                  │ │
│ │ ─────── │ │                                  │ │
│ │ ⌘K      │ │                                  │ │
│ └─────────┘ └──────────────────────────────────┘ │
└──────────────────────────────────────────────────┘
```

### Features

- **Sidebar Navigation**: Home, Topics, Search views
- **Command Palette**: ⌘K (Cmd+K) for quick topic navigation
- **6 Topic Categories**:
  - Daily Guidance
  - Path of Devotion
  - Mind & Emotions
  - Relationships & Society
  - Detachment & Renunciation
  - Theological Concepts
- **Chat Interface**: Full conversation with citations and theme badges

## Tech Stack

- **React 18+** with TypeScript
- **Vite** for fast development and building
- **Tailwind CSS** for styling
- **lucide-react** for icons
- **class-variance-authority** for component variants
- **Zod** for API response validation

## Development

```bash
# Install dependencies
npm install

# Start development server
npm run dev

# Build for production
npm run build

# Run linter
npm run lint
```

## Key Dependencies

```json
{
  "dependencies": {
    "react": "^19.1.0",
    "react-dom": "^19.1.0",
    "lucide-react": "latest",
    "class-variance-authority": "latest",
    "clsx": "^2.1.1",
    "tailwind-merge": "^3.3.1",
    "zod": "^3.23.8"
  }
}
```

## Component Architecture

### UI Components (`src/components/ui/`)

- `Button.tsx` - Variants: default, destructive, outline, secondary, ghost, link
- `Input.tsx` - Text input with focus states
- `Badge.tsx` - Variants: default, secondary, destructive, outline
- `ErrorBoundary.tsx` - Error handling wrapper
- `LoadingSpinner.tsx` - Loading indicator

### Design Tokens (`src/index.css`)

```css
:root {
  --background: #fafafa;
  --foreground: #0f0f0f;
  --surface: #ffffff;
  --border: rgba(0, 0, 0, 0.1);
  --text-secondary: #737373;
  --radius: 0.625rem;
}
```

## ESLint Configuration

For production, update the configuration to enable type-aware lint rules. See the original Vite template documentation for details.
