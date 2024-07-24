// src/components/CompareSummaries.js
import React, { useState } from 'react';
import axios from 'axios';
import "../Chatbot.css";

const CompareSummaries = () => {
  const [keyword, setKeyword] = useState('');
  const [file, setFile] = useState(null);
  const [result, setResult] = useState({ model_summary: '', pdf_summary: '' });

  const handleKeywordChange = (e) => {
    setKeyword(e.target.value);
  };

  const handleFileChange = (e) => {
    setFile(e.target.files[0]);
  };

  const handleSubmit = async (e) => {
    e.preventDefault();

    const formData = new FormData();
    formData.append('keyword', keyword);
    formData.append('file', file);

    try {
      const response = await axios.post('http://localhost:5000/compare_summaries', formData, {
        headers: {
          'Content-Type': 'multipart/form-data'
        }
      });
      setResult(response.data);
    } catch (error) {
      console.error('There was an error comparing the summaries!', error);
    }
  };

  return (
    <div className="chatbot-container">
      <h2>Compare Summaries by Keyword:</h2>
      <form onSubmit={handleSubmit}>
        <input
          type="text"
          value={keyword}
          onChange={handleKeywordChange}
          placeholder="Enter keyword"
          className="chatbot-input"
        />
        <input
          type="file"
          onChange={handleFileChange}
          className="file-choose"
        />
        <button type="submit" className="chatbot-button">Compare</button>
      </form>
      <h2>Comparison Results:</h2>
      <div>
        <h3>Model Summary:</h3>
        <p>{result.model_summary}</p>
        <h3>PDF Summary:</h3>
        <p>{result.pdf_summary}</p>
      </div>
    </div>
  );
};

export default CompareSummaries;
