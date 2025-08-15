
import React, { useState } from 'react';
import FileUpload from './components/FileUpload';
import KeywordSearch from './components/KeywordSearch';
import CompareSummaries from './components/CompareSummaries';
import AnswerQuestions from './components/AnswerQuestion';
import Header from './components/Header';
const App = () => {
  const [answer, setAnswer] = useState('');

  return (
    <div className="chatbot-wrapper">
      <Header/>
      <FileUpload onUploadSuccess={(data) => console.log(data)} />
      <KeywordSearch />
      <CompareSummaries />
      <AnswerQuestions/>
    </div>
  );
};

export default App;
