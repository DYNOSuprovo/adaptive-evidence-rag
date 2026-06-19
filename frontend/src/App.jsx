import { useState } from 'react';
import axios from 'axios';
import { Search, Loader2, Sparkles, BrainCircuit, Activity, CheckCircle2 } from 'lucide-react';
import { motion, AnimatePresence } from 'framer-motion';
import './index.css';

function App() {
  const [query, setQuery] = useState('');
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);

  const handleSearch = async (e) => {
    e.preventDefault();
    if (!query.trim()) return;

    setLoading(true);
    setError(null);
    try {
      // Connect to FastAPI backend
      const response = await axios.post('http://localhost:8000/api/query', {
        question: query,
        num_documents: 5
      });
      setResult(response.data);
    } catch (err) {
      setError('Failed to connect to the RAG backend. Ensure the API is running.');
      console.error(err);
    } finally {
      setLoading(false);
    }
  };

  const ScoreCard = ({ title, score, icon: Icon }) => (
    <div className="score-card">
      <div className="score-header">
        <Icon size={18} className="score-icon" />
        <h3>{title}</h3>
      </div>
      <div className="score-value-container">
        <span className="score-value">{(score * 100).toFixed(1)}%</span>
      </div>
      <div className="progress-bar-bg">
        <motion.div 
          className="progress-bar-fill" 
          initial={{ width: 0 }}
          animate={{ width: `${score * 100}%` }}
          transition={{ duration: 1, delay: 0.2 }}
        />
      </div>
    </div>
  );

  return (
    <div className={`app-container ${result ? 'has-results' : ''}`}>
      <motion.div 
        className="search-section"
        layout
        transition={{ type: "spring", stiffness: 100, damping: 20 }}
      >
        <div className="logo">
          <BrainCircuit size={40} className="logo-icon" />
          <h1>Adaptive Evidence RAG</h1>
        </div>
        <p className="subtitle">Intelligent, self-evaluating retrieval pipeline.</p>
        
        <form onSubmit={handleSearch} className="search-bar-container">
          <div className={`search-bar ${loading ? 'loading' : ''}`}>
            <Search size={20} className="search-icon" />
            <input
              type="text"
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              placeholder="Ask anything (e.g., What is machine learning?)"
              disabled={loading}
            />
            <button type="submit" disabled={!query.trim() || loading}>
              {loading ? <Loader2 size={20} className="spin" /> : <Sparkles size={20} />}
            </button>
          </div>
        </form>
        {error && <div className="error-message">{error}</div>}
      </motion.div>

      <AnimatePresence>
        {result && !loading && (
          <motion.div 
            className="results-section"
            initial={{ opacity: 0, y: 30 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -30 }}
            transition={{ duration: 0.5 }}
          >
            <div className="metadata-banner">
              <span><strong>Original Query:</strong> {result.question}</span>
              <span><strong>Optimized Query:</strong> {result.query_used}</span>
            </div>

            <div className="dashboard">
              <div className="dashboard-grid">
                <ScoreCard title="Independence" score={result.independence_score} icon={Activity} />
                <ScoreCard title="Utility" score={result.utility_score} icon={Sparkles} />
                <ScoreCard title="Stability" score={result.stability_score} icon={CheckCircle2} />
              </div>
              
              <div className="overall-score-card">
                <div className="overall-content">
                  <h3>Overall Pipeline Quality</h3>
                  <div className="overall-value">{(result.overall_quality * 100).toFixed(1)}<span className="percent">%</span></div>
                  <p>Filtered from {result.metadata?.num_original || '?'} to {result.metadata?.num_filtered || '?'} highly relevant documents.</p>
                </div>
              </div>
            </div>

            <div className="evidence-section">
              <h2>Retrieved Evidence</h2>
              {result.filtered_documents.length === 0 ? (
                <div className="no-evidence">No documents passed the quality filters.</div>
              ) : (
                <div className="evidence-list">
                  {result.filtered_documents.map((doc, idx) => (
                    <motion.div 
                      key={idx} 
                      className="evidence-card"
                      initial={{ opacity: 0, x: -20 }}
                      animate={{ opacity: 1, x: 0 }}
                      transition={{ delay: 0.4 + (idx * 0.1) }}
                    >
                      <div className="evidence-number">{idx + 1}</div>
                      <p className="evidence-text">{doc}</p>
                    </motion.div>
                  ))}
                </div>
              )}
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}

export default App;
