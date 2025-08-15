import React, { useState } from 'react';
import axios from 'axios';
import "../Chatbot.css";

const PdfSummary = () => {
  const [file, setFile] = useState(null);
  const [summary, setSummary] = useState('');

  const handleFileChange = (event) => {
    setFile(event.target.files[0]);
  };

  const handleUpload = async () => {
    if (!file) {
      alert("Please select a file first.");
      return;
    }

    const formData = new FormData();
    formData.append('file', file);

    try {
      const response = await axios.post('http://localhost:5000/summarize_pdf', formData, {
        headers: {
          'Content-Type': 'multipart/form-data'
        }
      });

      setSummary(response.data.summary);
    } catch (error) {
      console.error('There was an error uploading the file!', error);
      alert('File upload failed.');
    }
  };

  return (
    <div className="chatbot-container">
      <h2>Upload a PDF to Summarize</h2>
      <input type="file" onChange={handleFileChange} />
      <button onClick={handleUpload}>Upload</button>
      {summary && <div className="chatbot-message"><p>{summary}</p></div>}
    </div>
  );
};

export default PdfSummary;
