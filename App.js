//src/App.js
import React, { useState } from 'react';
import FileUpload from './components/FileUpload';
import QuestionForm from './components/QuestionForm';
import AnswerDisplay from './components/AnswerDisplay';
import "./Chatbot.css";

const App = () => {
  const [answer, setAnswer] = useState('');

  return (
    <div className="chatbot-wrapper">
      <FileUpload />
      <QuestionForm setAnswer={setAnswer} />
      {answer && <AnswerDisplay answer={answer} />}
    </div>
  );
};

export default App;

// // src/App.js
// import React, { useState } from 'react';
// import { BrowserRouter as Router, Route, Routes } from 'react-router-dom';
// import FileUpload from './components/FileUpload';
// import QuestionForm from './components/QuestionForm';
// import AnswerDisplay from './components/AnswerDisplay';
// import "./App.css";
// function App() {
//   const [answer, setAnswer] = useState('');

//   return (
//     <Router>
//       <div>
//         <h1>Syllabus App</h1>
//         <Routes>
//           <Route path="/upload" element={<FileUpload />} />
//           <Route
//             path="/answer"
//             element={<>
//               <QuestionForm setAnswer={setAnswer} />
//               <AnswerDisplay answer={answer} />
//             </>}
//           />
//         </Routes>
//       </div>
//     </Router>
//   );
// }

// export default App;

