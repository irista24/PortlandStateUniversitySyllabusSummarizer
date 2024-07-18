import React, { useState } from 'react';
import axios from 'axios';
import "../Chatbot.css";

const FileUpload = ({ onUploadSuccess }) => {
  const [file, setFile] = useState(null);

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
      const response = await axios.post('http://localhost:5000/upload', formData, {
        headers: {
          'Content-Type': 'multipart/form-data'
        }
      });

      alert('File uploaded successfully.');
      onUploadSuccess(response.data);  // handle the response data if needed
    } catch (error) {
      console.error('There was an error uploading the file!', error);
    }
  };

  return (
    <div className="chatbot-container">
      <h2>Upload a PDF:</h2>
      <input type="file" onChange={handleFileChange} className = "file-chose" />
      <button onClick={handleUpload} className = "chatbot-upload">Upload</button>
    </div>
  );
};

export default FileUpload;

// import React, { useState } from 'react';
// import axios from 'axios';
// import "./App.css";
// const FileUpload = () => {
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
//       console.log(response.data);  // handle the response data if needed
//     } catch (error) {
//       console.error('There was an error uploading the file!', error);
//       alert('File upload failed.');
//     }
//   };

//   return (
//     <div>
//       <h2>Upload a PDF</h2>
//       <input type="file" onChange={handleFileChange} />
//       <button onClick={handleUpload}>Upload</button>
//     </div>
//   );
// };

// export default FileUpload;