import { useState, useEffect, useRef } from 'react';
import './App.css';

interface Message {
  id: string;
  type: 'player' | 'dm';
  content: string;
  timestamp: Date;
}

interface GameState {
  current_location: string;
  active_npcs: string[];
  party_status: string;
  recent_events: string[];
}

interface PendingAction {
  type: string;
  dice_type: string;
  stat: string;
  skill: string;
  dc: number;
}

const formatTime = (date: Date): string => {
  return date.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
};

function App() {
  const [input, setInput] = useState('');
  const [messages, setMessages] = useState<Message[]>([
    {
      id: "",
      type: 'dm',
      content: 'Welcome, adventurer! You find yourself in The Yawning Portal Tavern. The air is thick with smoke and the murmur of whispered rumors. Durnan, the grizzled barkeep, polishes a tankard behind the counter. What do you do?',
      timestamp: new Date(),
    },
  ]);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [gameState, setGameState] = useState<GameState>({
    current_location: 'The Yawning Portal Tavern',
    active_npcs: ['Durnan the Barkeep'],
    party_status: 'Healthy',
    recent_events: [],
  });
  const [isConnecting, setIsConnecting] = useState(true);
  const scrollRef = useRef<HTMLDivElement>(null);
  const [pendingAction, setPendingAction] = useState<PendingAction | null>(null);

  useEffect(() => {
    // Simulate initial connection
    const timer = setTimeout(() => {
      setIsConnecting(false);
    }, 1000);
    return () => clearTimeout(timer);
  }, []);

  useEffect(() => {
    // Auto-scroll to bottom when new messages arrive
    if (scrollRef.current) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
    }
  }, [messages]);

  const handleSend = async () => {
    if (!input.trim() || isLoading) return;

    const playerInput = input.trim();
    setInput('');
    setError(null);

    // Add player message
    const playerMessage: Message = {
      id: crypto.randomUUID(),
      type: 'player',
      content: playerInput,
      timestamp: new Date(),
    };
    setMessages((prev) => [...prev, playerMessage]);
    setIsLoading(true);

    try {
      const response = await fetch('http://localhost:5000/api/game', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ input: playerInput }),
      });

      if (!response.ok) {
        throw new Error(`Server responded with ${response.status}`);
      }

      const data = await response.json();

      // Add DM response
      const dmMessage: Message = {
        id: crypto.randomUUID(), // Stop using Date.now()!
        type: 'dm',
        content: data.response,
        timestamp: new Date(),
      };
      setMessages((prev) => [...prev, dmMessage]);

      // Update game state
      if (data.game_state) {
        setGameState(data.game_state);
      }

      // Check for tool calls
      if (data.pending_action) {
        setPendingAction(data.pending_action);
      }
    } catch (err) {
      console.error('Error sending message:', err);
      setError(err instanceof Error ? err.message : 'Could not connect to the server');
    } finally {
      setIsLoading(false);
    }
  };

  const handleDiceRoll = async () => {
    if (!pendingAction) return;

    // 1. Calculate the roll (Standard 1d20 for now. You can add character modifiers later)
    const roll = Math.floor(Math.random() * 20) + 1;
    const isSuccess = roll >= pendingAction.dc;

    // 2. Build the System Prompt
    const systemPayload = `[SYSTEM: ROLL_RESOLUTION | SKILL: ${pendingAction.skill} | ROLL: ${roll} | DC: ${pendingAction.dc} | RESULT: ${isSuccess ? 'SUCCESS' : 'FAILURE'}]`;

    // 3. Clear the pending action so the UI resets
    setPendingAction(null);
    setIsLoading(true);

    // 4. Send the resolution to the backend invisibly (don't show the system tag in the UI)
    const playerMessage: Message = {
      id: crypto.randomUUID(),
      type: 'player',
      content: `*Rolls a ${roll} for ${pendingAction.skill}*`,
      timestamp: new Date(),
    };
    setMessages((prev) => [...prev, playerMessage]);

    try {
      const response = await fetch('http://localhost:5000/api/game', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ input: systemPayload }),
      });
      
      const data = await response.json();
      
      setMessages((prev) => [...prev, {
        id: crypto.randomUUID(),
        type: 'dm',
        content: data.response,
        timestamp: new Date(),
      }]);
      
      if (data.game_state) setGameState(data.game_state);

    } catch (err) {
      console.error('Error resolving roll:', err);
      setError('Failed to resolve dice roll.');
    } finally {
      setIsLoading(false);
    }
  };

  const handleKeyPress = (e: React.KeyboardEvent<HTMLInputElement>) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      handleSend();
    }
  };

  if (isConnecting) {
    return (
      <div className="loading-overlay">
        <div className="loading-content">
          <h2>Preparing Your Adventure</h2>
          <div className="loading-dice">⚔️</div>
        </div>
      </div>
    );
  }

  return (
    <div className="app-container">
      <header className="app-header">
        <h1>RAGnarok</h1>
        <p className="subtitle">A Dungeons & Dragons Adventure</p>
      </header>

      <main className="main-content">
        <aside className="side-panel">
          <div className="state-card">
            <h3>Location</h3>
            <div className="state-item">
              <span className="state-value">{gameState.current_location}</span>
            </div>
          </div>

          <div className="state-card">
            <h3>Active NPCs</h3>
            {gameState.active_npcs.length > 0 ? (
              <ul className="npc-list">
                {gameState.active_npcs.map((npc, index) => (
                  <li key={index}>{npc}</li>
                ))}
              </ul>
            ) : (
              <p style={{ fontStyle: 'italic', color: 'var(--text-muted)' }}>
                No one nearby...
              </p>
            )}
          </div>

          <div className="state-card">
            <h3>Party Status</h3>
            <div className="state-item">
              <span className="state-value">{gameState.party_status}</span>
            </div>
          </div>

          <div className="state-card">
            <h3>Recent Events</h3>
            {gameState.recent_events.length > 0 ? (
              <ul className="npc-list">
                {gameState.recent_events.slice(-3).map((event, index) => (
                  <li key={index} style={{ fontSize: '0.85rem' }}>
                    {event.length > 50 ? event.substring(0, 50) + '...' : event}
                  </li>
                ))}
              </ul>
            ) : (
              <p style={{ fontStyle: 'italic', color: 'var(--text-muted)' }}>
                Your story begins...
              </p>
            )}
          </div>
        </aside>

        <div className="narrative-panel">
          <div className="narrative-scroll" ref={scrollRef}>
            {messages.map((message) => (
              <div
                key={message.id}
                className={`message ${message.type === 'player' ? 'player-action' : ''}`}
              >
                <span className="timestamp">{formatTime(message.timestamp)}</span>
                <span className="sender">
                  {message.type === 'player' ? 'You' : 'Dungeon Master'}
                </span>
                <div className={`content ${message.type === 'dm' ? 'dm-response' : ''}`}>
                  {message.content}
                </div>
              </div>
            ))}
            {isLoading && (
              <div className="message">
                <div className="typing-indicator">
                  <span></span>
                  <span></span>
                  <span></span>
                </div>
              </div>
            )}
            {error && (
              <div className="message">
                <div className="error-message">
                  {error} — Make sure the server is running on port 5000
                </div>
              </div>
            )}
          </div>

          <div className="input-area">
            {pendingAction ? (
              <div className="dice-action-container" style={{ width: '100%', textAlign: 'center', padding: '1rem', backgroundColor: '#2a2a2a', borderRadius: '8px', border: '1px solid #ff4444' }}>
                <p style={{ margin: '0 0 10px 0', color: '#fff' }}>
                  The DM demands a <strong>{pendingAction.stat} ({pendingAction.skill})</strong> check.
                  <br/>
                  <span style={{ fontSize: '0.9em', color: '#aaa' }}>Difficulty Class: {pendingAction.dc}</span>
                </p>
                <button 
                  onClick={handleDiceRoll}
                  style={{ padding: '10px 20px', fontSize: '1.2rem', backgroundColor: '#ff4444', color: 'white', border: 'none', borderRadius: '4px', cursor: 'pointer', fontWeight: 'bold' }}
                >
                  🎲 Roll {pendingAction.dice_type}
                </button>
              </div>
            ) : (
              <>
                <input
                  type="text"
                  value={input}
                  onChange={(e) => setInput(e.target.value)}
                  onKeyDown={handleKeyPress}
                  placeholder="What do you do?"
                  disabled={isLoading}
                  autoFocus
                />
                <button
                  className="send-button"
                  onClick={handleSend}
                  disabled={isLoading || !input.trim()}
                >
                  {isLoading ? 'Sending...' : 'Action'}
                </button>
              </>
            )}
          </div>
        </div>
      </main>
    </div>
  );
}

export default App;