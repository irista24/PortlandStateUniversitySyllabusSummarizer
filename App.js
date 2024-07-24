import React, { useState } from 'react';
import FileUpload from './components/FileUpload';
import QuestionForm from './components/QuestionForm';
import AnswerDisplay from './components/AnswerDisplay';

function App() {
  const [uploadedFile, setUploadedFile] = useState(null);
  const [question, setQuestion] = useState('');
  const [answer, setAnswer] = useState('');
  const [loading, setLoading] = useState(false);

  const handleFileUpload = (file) => {
    setUploadedFile(file);
  };

  const handleQuestionSubmit = async () => {
    setLoading(true);
    // Implement API call to backend to get answer based on uploadedFile and question
    // Example using Axios:
    /*
    const formData = new FormData();
    formData.append('file', uploadedFile);
    formData.append('question', question);
    const response = await axios.post('/api/answer', formData);
    setAnswer(response.data.answer);
    */
    // Replace above placeholder with actual API call logic
    setLoading(false);
  };

  return (
    <div className="App">
      <h1>PDF Syllabus Q&A</h1>
      <FileUpload onFileUpload={handleFileUpload} />
      <QuestionForm
        question={question}
        setQuestion={setQuestion}
        onQuestionSubmit={handleQuestionSubmit}
        disabled={!uploadedFile}
      />
      {loading && <p>Loading...</p>}
      {answer && <AnswerDisplay answer={answer} />}
    </div>
  );
}

export default App;
