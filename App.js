import React, { useState } from 'react';
import { BrowserRouter as Router, Route, Routes } from 'react-router-dom';
import FileUpload from './components/FileUpload';
import QuestionForm from './components/QuestionForm';
import AnswerDisplay from './components/AnswerDisplay';

const App = () => {
  const [answer, setAnswer] = useState('');

  return (
    <Router>
      <div>
        <h1>Syllabus App</h1>
        <Routes>
          <Route path="/upload" element={<FileUpload />} />
          <Route path="/ask" element={<QuestionForm setAnswer={setAnswer} />} />
        </Routes>
        {answer && <AnswerDisplay answer={answer} />}
      </div>
    </Router>
  );
};

export default App;
