import React, { useState } from 'react';
import axios from 'axios';

const SummarizeSyllabus = () => {
  const [file, setFile] = useState(null);
  const [summary, setSummary] = useState('');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');

  const handleFileChange = (event) => {
    setFile(event.target.files[0]);
  };

  const handleSummarize = async () => {
    if (!file) {
      setError('Please select a file first.');
      return;
    }

    setLoading(true);
    setError('');
    try {
      const formData = new FormData();
      formData.append('file', file);

      const response = await axios.post('http://localhost:5000/summarize_entire_syllabus', formData, {
        headers: { 'Content-Type': 'multipart/form-data' },
      });

      setSummary(response.data.summary);
    } catch (err) {
      setError('Error summarizing syllabus.');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div>
      <input type="file" accept=".pdf" onChange={handleFileChange} />
      <button onClick={handleSummarize} disabled={loading}>
        {loading ? 'Summarizing...' : 'Summarize Syllabus'}
      </button>
      {error && <p style={{ color: 'red' }}>{error}</p>}
      {summary && <p>{summary}</p>}
    </div>
  );
};

export default SummarizeSyllabus;
