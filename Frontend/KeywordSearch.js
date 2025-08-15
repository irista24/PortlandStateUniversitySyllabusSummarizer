import React, { useState } from 'react';
import axios from 'axios';

function KeywordSearch() {
  const [selectedKeyword, setSelectedKeyword] = useState('');
  const [summaries, setSummaries] = useState([]);

  // List of predefined keywords
  const keywords = [
    'Instructor', 'Email', 'Office', 'Late Work', 'Course Description', 'Objective', 
    'Materials', 'Grade', 'Week', 'Location', 'Grading', 'Calendar', 'Expectations', 
    'Resources', 'Attendance', 'Academic Integrity', 'Technology'
  ];

  // Handle dropdown selection change
  const handleKeywordChange = (e) => {
    setSelectedKeyword(e.target.value);
  };

  // Handle search button click
  const handleSearch = async () => {
    if (selectedKeyword) {
      try {
        const response = await axios.post('http://localhost:5000/search_keyword', {
          keyword: selectedKeyword
        }, {
          headers: { 'Content-Type': 'application/json' },
        });

        setSummaries(response.data.summaries || []);
      } catch (error) {
        console.error("Error searching for keyword:", error);
      }
    } else {
      console.error("No keyword selected.");
    }
  };

  const styles = {
    container: {
      maxWidth: '800px',
      margin: 'auto',
      padding: '20px',
      borderRadius: '8px',
      boxShadow: '0px 4px 8px rgba(0, 0, 0, 0.1)',
      backgroundColor: '#6d8d24', // PSU Green
      color: '#FFFFFF', // White
      textAlign: 'center', // Center text and elements
    },
    select: {
      display: 'block',
      width: '100%',
      maxWidth: '400px',
      margin: '10px auto', // Center dropdown
      padding: '10px',
      borderRadius: '4px',
      border: '1px solid #ccc',
      backgroundColor: '#FFFFFF', // White
      color: '#000000', // Black text
    },
    button: {
      backgroundColor: '#6d8d24', // PSU Green
      color: '#FFFFFF', // White
      padding: '10px 20px',
      border: 'none',
      borderRadius: '4px',
      cursor: 'pointer',
      transition: 'background-color 0.3s ease',
      margin: '10px auto', // Center button
      display: 'inline-block', // Ensure button is centered
    },
    buttonHover: {
      backgroundColor: '#4a5b19', // Darker PSU Green
    },
    summariesContainer: {
      marginTop: '20px',
      backgroundColor: '#4a5b19', // Darker PSU Green
      padding: '15px',
      borderRadius: '4px',
      textAlign: 'center', // Center text in summaries
    },
    summaryItem: {
      margin: '10px auto', // Center summary item
      padding: '10px',
      borderRadius: '4px',
      border: '1px solid #ddd',
      backgroundColor: '#6d8d24', // PSU Green
      color: '#FFFFFF', // White
      maxWidth: '600px', // Limit width for readability
    },
    noSummaries: {
      color: '#FFD700', // Gold
    },
  };

  return (
    <div style={styles.container}>
      <h3>Search Summaries by Keyword</h3>
      {/* Dropdown for selecting a keyword */}
      <select value={selectedKeyword} onChange={handleKeywordChange} style={styles.select}>
        <option value="">Select a keyword</option>
        {keywords.map((keyword, index) => (
          <option key={index} value={keyword}>{keyword}</option>
        ))}
      </select>

      {/* Search button */}
      <button
        onClick={handleSearch}
        style={styles.button}
        onMouseOver={(e) => e.target.style.backgroundColor = styles.buttonHover.backgroundColor}
        onMouseOut={(e) => e.target.style.backgroundColor = styles.button.backgroundColor}
      >
        Search
      </button>

      {/* Display the fetched summaries */}
      <div style={styles.summariesContainer}>
        {summaries.length > 0 ? (
          <ul>
            {summaries.map((summary, index) => (
              <li key={index} style={styles.summaryItem}>{summary}</li>
            ))}
          </ul>
        ) : (
          <p style={styles.noSummaries}>No summaries found.</p>
        )}
      </div>
    </div>
  );
}

export default KeywordSearch;
