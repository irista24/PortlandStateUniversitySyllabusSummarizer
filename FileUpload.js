import React, { useState } from 'react';
import axios from 'axios';

const FileUpload = ({ onUploadSuccess }) => {
  const [file, setFile] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');

  const handleFileChange = (event) => {
    setFile(event.target.files[0]);
  };

  const handleUpload = async () => {
    if (!file) {
      setError('Please select a file first.');
      return;
    }

    setLoading(true);
    setError('');
    try {
      const formData = new FormData();
      formData.append('file', file);

      const response = await axios.post('http://localhost:5000/upload', formData, {
        headers: { 'Content-Type': 'multipart/form-data' },
      });

      onUploadSuccess(response.data.syllabus_id);
      setFile(null);
    } catch (err) {
      setError('Error uploading file.');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div>
      <input type="file" accept=".pdf" onChange={handleFileChange} />
      <button onClick={handleUpload} disabled={loading}>
        {loading ? 'Uploading...' : 'Upload PDF'}
      </button>
      {error && <p style={{ color: 'red' }}>{error}</p>}
    </div>
  );
};

export default FileUpload;

// // src/components/FileUpload.js
// import React, { useState } from 'react';
// import axios from 'axios';
// import "../Chatbot.css";

// const FileUpload = ({ onUploadSuccess }) => {
//   const [file, setFile] = useState(null);

//   const handleFileChange = (event) => {
//     setFile(event.target.files[0]);
//   };

//   const handleUpload = async () => {
//     if (!file) {
//       alert("Please select a file first.");
//       return;
//     }

//     const formData = new FormData();
//     formData.append('file', file);

//     try {
//       const response = await axios.post('http://localhost:5000/upload', formData, {
//         headers: {
//           'Content-Type': 'multipart/form-data'
//         }
//       });

//       alert('File uploaded successfully.');
//       onUploadSuccess(response.data);  // handle the response data if needed
//     } catch (error) {
//       console.error('There was an error uploading the file!', error);
//     }
//   };

//   return (
//     <div className="chatbot-container">
//       <h2>Upload a PDF:</h2>
//       <input type="file" onChange={handleFileChange} className="file-choose" />
//       <button onClick={handleUpload} className="chatbot-upload">Upload</button>
//     </div>
//   );
// };

// export default FileUpload;
