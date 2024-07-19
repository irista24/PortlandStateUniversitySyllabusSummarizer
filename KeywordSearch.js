// src/components/KeywordSearch.js
import React, { useState } from 'react';
import axios from 'axios';
import "../Chatbot.css";

const KeywordSearch = () => {
  const [keyword, setKeyword] = useState('');
  const [results, setResults] = useState([]);

  const onKeywordChange = (e) => {
    setKeyword(e.target.value);
  };

  const onSubmit = async (e) => {
    e.preventDefault();

    try {
      const response = await axios.post('http://localhost:5000/search_keyword', { keyword });
      setResults(response.data.instances);
    } catch (err) {
      console.error(err);
      setResults([]);
    }
  };

  return (
    <div className="chatbot-container">
      <h2>Search by Keyword:</h2>
      <form onSubmit={onSubmit}>
        <input
          type="text"
          value={keyword}
          onChange={onKeywordChange}
          placeholder="Enter keyword"
          className="chatbot-input"
        />
        <button type="submit" className="chatbot-button">Search</button>
      </form>
      <h2>Search Results:</h2>
      <div>
        {results.length > 0 ? (
          results.map((result, index) => (
            <div key={index} className="chatbot-message">
              <p>{result}</p>
            </div>
          ))
        ) : (
          <p></p>
        )}
      </div>
    </div>
  );
};

export default KeywordSearch;
