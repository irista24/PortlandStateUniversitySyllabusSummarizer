import React, { useState } from 'react';
import axios from 'axios';

const CompareSummaries = () => {
  const [file, setFile] = useState(null);
  const [keywords, setKeywords] = useState([]);
  const [selectedKeyword, setSelectedKeyword] = useState('');
  const [userSummary, setUserSummary] = useState('');
  const [modelSummary, setModelSummary] = useState('');

  const handleFileChange = (e) => {
    setFile(e.target.files[0]);
  };

  const handleKeywordGeneration = async () => {
    if (!file) return;

    const formData = new FormData();
    formData.append('file', file);

    try {
      const response = await axios.post('http://127.0.0.1:5000/extract_keywords', formData, {
        headers: { 'Content-Type': 'multipart/form-data' },
      });
      setKeywords(response.data.keywords);
    } catch (error) {
      console.error('Error extracting keywords:', error.response ? error.response.data : error.message);
    }
  };

  const handleKeywordChange = (e) => {
    setSelectedKeyword(e.target.value);
  };

  const handleCompareSummaries = async () => {
    if (!file || !selectedKeyword) return;

    const formData = new FormData();
    formData.append('file', file);
    formData.append('keyword', selectedKeyword);

    try {
      const response = await axios.post('http://127.0.0.1:5000/compare_summaries', formData, {
        headers: { 'Content-Type': 'multipart/form-data' },
      });
      setUserSummary(response.data.user_summary);
      setModelSummary(response.data.model_summary);
    } catch (error) {
      console.error('Error comparing summaries:', error.response ? error.response.data : error.message);
      alert('Failed to compare summaries. Please check the console for details.');
    }
  };

  const styles = {
    container: {
      maxWidth: '800px',
      margin: 'auto',
      padding: '20px',
      borderRadius: '8px',
      boxShadow: '0px 4px 8px rgba(0, 0, 0, 0.1)',
      textAlign: 'center', // Center text and elements
  
    },
    select: {
      display: 'block',
      width: '100%',
      maxWidth: '400px',
      margin: '10px auto', // Center dropdown
      padding: '10px',
      borderRadius: '4px',
      border: '1px solid #6d8d24',
    },
    button: {
      backgroundColor: '#6d8d24',
      color: '#fff',
      padding: '10px 20px',
      border: 'none',
      borderRadius: '4px',
      cursor: 'pointer',
      transition: 'background-color 0.3s ease',
      margin: '10px auto', // Center button
      display: 'inline-block', // Ensure button is centered
    },
    buttonHover: {
      backgroundColor: '#4c6a1f',
    },
    summariesContainer: {
      marginTop: '20px',
      backgroundColor: '#f8f9fa',
      padding: '15px',
      borderRadius: '4px',
      border: '1px solid #ddd',
    },
    summaryBox: {
      margin: '10px auto', // Center summary box
      padding: '10px',
      borderRadius: '4px',
      border: '1px solid #ddd',
      backgroundColor: '#fff',
      maxWidth: '800px',
    },
    heading: {
      color: '#6d8d24',
    },
  };

  return (
    <div style={styles.container}>
      <h3 style={styles.header}>Compare a keyword summary of your syllabus with our model keyword summary</h3>
      <input type="file" onChange={handleFileChange} />
      <button
        onClick={handleKeywordGeneration}
        style={styles.button}
        onMouseOver={(e) => e.target.style.backgroundColor = styles.buttonHover.backgroundColor}
        onMouseOut={(e) => e.target.style.backgroundColor = styles.button.backgroundColor}
      >
        Generate Keywords
      </button>

      {keywords.length > 0 && (
        <div>
          <label htmlFor="keywordDropdown" style={styles.heading}>Select Keyword:</label>
          <select
            id="keywordDropdown"
            value={selectedKeyword}
            onChange={handleKeywordChange}
            style={styles.select}
          >
            <option value="">Select a keyword</option>
            {keywords.map((keyword, index) => (
              <option key={index} value={keyword}>
                {keyword}
              </option>
            ))}
          </select>

          <button
            onClick={handleCompareSummaries}
            style={styles.button}
            onMouseOver={(e) => e.target.style.backgroundColor = styles.buttonHover.backgroundColor}
            onMouseOut={(e) => e.target.style.backgroundColor = styles.button.backgroundColor}
          >
            Compare Summaries
          </button>

          {userSummary && (
            <div style={styles.summariesContainer}>
              <h3 style={styles.heading}>User Summary:</h3>
              <p>{userSummary}</p>
            </div>
          )}

          {modelSummary && (
            <div style={styles.summariesContainer}>
              <h3 style={styles.heading}>Model Summary:</h3>
              <p>{modelSummary}</p>
            </div>
          )}
        </div>
      )}
    </div>
  );
};

export default CompareSummaries;
