import React, { useState } from 'react';
import axios from 'axios';

const CompareSummaries = () => {
  const [text, setText] = useState('');
  const [keyword, setKeyword] = useState('');
  const [userSummary, setUserSummary] = useState('');
  const [modelSummary, setModelSummary] = useState('');

  const handleCompare = async () => {
    try {
      const response = await axios.post('http://127.0.0.1:5000/compare_summaries', { text, keyword });
      setUserSummary(response.data.user_summary);
      setModelSummary(response.data.model_summary);
    } catch (error) {
      console.error('Error comparing summaries:', error);
    }
  };

  return (
    <div>
      <h2>Compare Summaries</h2>
      <input type="text" value={keyword} onChange={(e) => setKeyword(e.target.value)} placeholder="Enter keyword" />
      <button onClick={handleCompare}>Compare</button>
      <h3>User Summary</h3>
      <pre>{userSummary}</pre>
      <h3>Model Summary</h3>
      <pre>{modelSummary}</pre>
    </div>
  );
};

export default CompareSummaries;
