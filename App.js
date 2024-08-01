import React from 'react';
import FileUpload from './components/FileUpload';
import KeywordSearch from './components/KeywordSearch';
import SummarizeSyllabus from './components/SummarizeSyllabus';
import './Chatbot.css'
const App = () => {
  const handleUploadSuccess = (syllabusId) => {
    console.log(`File uploaded successfully with syllabus ID: ${syllabusId}`);
  };

  return (
    <div>
      <h1>Syllabus Application</h1>
      <FileUpload onUploadSuccess={handleUploadSuccess} />
      <KeywordSearch />
      <SummarizeSyllabus />
    </div>
  );
};

export default App;
// import React, { useState } from 'react';
// import axios from 'axios';

// function App() {
//   const [file, setFile] = useState(null);
//   const [keyword, setKeyword] = useState('');
//   const [response, setResponse] = useState(null);
//   const [error, setError] = useState(null);

//   const handleFileChange = (event) => {
//     setFile(event.target.files[0]);
//   };

//   const handleKeywordChange = (event) => {
//     setKeyword(event.target.value);
//   };

//   const handleUpload = async () => {
//     if (!file) {
//       alert('Please select a file');
//       return;
//     }

//     const formData = new FormData();
//     formData.append('file', file);

//     try {
//       const result = await axios.post('http://localhost:5000/upload', formData, {
//         headers: { 'Content-Type': 'multipart/form-data' }
//       });
//       setResponse(result.data);
//     } catch (error) {
//       setError(error.response ? error.response.data.error : error.message);
//     }
//   };

//   const handleSearchKeyword = async () => {
//     if (!keyword) {
//       alert('Please enter a keyword');
//       return;
//     }

//     try {
//       const result = await axios.post('http://localhost:5000/search_keyword', { keyword });
//       setResponse(result.data);
//     } catch (error) {
//       setError(error.response ? error.response.data.error : error.message);
//     }
//   };

//   return (
//     <div className="App">
//       <h1>Syllabus Management</h1>

//       <div>
//         <h2>Upload Syllabus PDF</h2>
//         <input type="file" onChange={handleFileChange} />
//         <button onClick={handleUpload}>Upload</button>
//       </div>

//       <div>
//         <h2>Search Keyword</h2>
//         <input type="text" value={keyword} onChange={handleKeywordChange} placeholder="Enter keyword" />
//         <button onClick={handleSearchKeyword}>Search</button>
//       </div>

//       {response && (
//         <div>
//           <h2>Response</h2>
//           <pre>{JSON.stringify(response, null, 2)}</pre>
//         </div>
//       )}

//       {error && (
//         <div>
//           <h2>Error</h2>
//           <pre>{error}</pre>
//         </div>
//       )}
//     </div>
//   );
// }

// export default App;



// // import React, { useState } from 'react';
// // import FileUpload from './components/FileUpload';
// // import QuestionForm from './components/QuestionForm';
// // import AnswerDisplay from './components/AnswerDisplay';
// // import KeywordSearch from './components/KeywordSearch';
// // import CompareSummaries from './components/CompareSummaries';
// // import "./Chatbot.css";

// // const App = () => {
// //   const [answer, setAnswer] = useState('');

// //   return (
// //     <div className="chatbot-wrapper">
// //       <FileUpload onUploadSuccess={(data) => console.log(data)} />
// //       <QuestionForm setAnswer={setAnswer} />
// //       {answer && <AnswerDisplay answer={answer} />}
// //       <KeywordSearch />
// //       <CompareSummaries />
// //     </div>
// //   );
// // };

// // export default App;
