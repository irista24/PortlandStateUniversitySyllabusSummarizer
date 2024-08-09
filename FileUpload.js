import React, { useState } from 'react';
import axios from 'axios';

const FileUpload = () => {
  const [file, setFile] = useState(null);
  const [message, setMessage] = useState('');
  const [summaries, setSummaries] = useState({});
  const [fileSummary, setFileSummary] = useState('');

  const handleFileChange = (event) => {
    setFile(event.target.files[0]);
  };

  const handleSubmit = async (event) => {
    event.preventDefault();
    if (!file) {
      setMessage('Please select a file first.');
      return;
    }

    const formData = new FormData();
    formData.append('file', file);

    try {
      const response = await axios.post('http://localhost:5000/upload', formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
      });
      setMessage(`File uploaded successfully: ${response.data.message}`);
      setSummaries(response.data.summaries || {});
      setFileSummary(response.data.file_summary || '');
    } catch (error) {
      setMessage(`Error uploading file: ${error.message}`);
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
    input: {
      display: 'block',
      width: '100%',
      maxWidth: '400px',
      margin: '10px auto', // Center input
      padding: '10px',
      borderRadius: '4px',
      border: '1px solid #6d8d24',
      boxSizing: 'border-box',
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
    message: {
      marginTop: '10px',
      color: '#d9534f',
    },
    summariesContainer: {
      marginTop: '20px',
      backgroundColor: '#f8f9fa',
      padding: '15px',
      borderRadius: '4px',
      border: '1px solid #ddd',
      textAlign: 'center', // Center text in summaries
    },
    summaryBox: {
      margin: '10px auto', // Center summary box
      padding: '10px',
      borderRadius: '4px',
      border: '1px solid #ddd',
      backgroundColor: '#fff',
      maxWidth: '800px',
    },
  };

  return (
    <div style={styles.container}>
        <h3 style={styles.header}>Upload a syllabus and see a summary of it</h3>
      <form onSubmit={handleSubmit}>
        <input
          type="file"
          onChange={handleFileChange}
          style={styles.input}
        />
        <button
          type="submit"
          style={styles.button}
          onMouseOver={(e) => e.target.style.backgroundColor = styles.buttonHover.backgroundColor}
          onMouseOut={(e) => e.target.style.backgroundColor = styles.button.backgroundColor}
        >
          Upload
        </button>
      </form>
      {message && <p style={styles.message}>{message}</p>}
      {fileSummary && (
        <div style={styles.summariesContainer}>
          <h2>File Summary:</h2>
          <div style={styles.summaryBox}>
            <p>{fileSummary}</p>
          </div>
        </div>
      )}
    </div>
  );
};

export default FileUpload;
