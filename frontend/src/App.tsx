import { ChatInterface } from '@/components/chat/ChatInterface';
import { ErrorBoundary } from '@/components/ui/ErrorBoundary';

function App() {
  return (
    <ErrorBoundary>
      <div className="min-h-screen bg-gray-50">
        <div className="max-w-5xl mx-auto">
          <ChatInterface />
        </div>
      </div>
    </ErrorBoundary>
  );
}

export default App;
