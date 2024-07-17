import React, { useState } from 'react';
import axios from 'axios';
import AnswerDisplay from './AnswerDisplay';

const QuestionForm = () => {
  const [question, setQuestion] = useState('');
  const [answer, setAnswer] = useState('');

  const onQuestionChange = (e) => {
    setQuestion(e.target.value);
  };

  const onSubmit = async (e) => {
    e.preventDefault();

    try {
      const response = await axios.post('http://localhost:5000/answer', { question });
      setAnswer(response.data.answer);
    } catch (err) {
      console.error(err);
      setAnswer('Failed to get answer');
    }
  };

  return (
    <div>
      <h2>Ask a Question:</h2>
      <form onSubmit={onSubmit}>
        <input
          type="text"
          value={question}
          onChange={onQuestionChange}
          placeholder="Enter your question"
        />
        <button type="submit">Ask</button>
      </form>
      <h2>Answer:</h2>
      {answer && <AnswerDisplay answer={answer} />}
    </div>
  );
};

export default QuestionForm;

// import React, { useState } from 'react';
// import axios from 'axios';
// import AnswerDisplay from './AnswerDisplay';

// const QuestionForm = () => {
//   const [question, setQuestion] = useState('');
//   const [answer, setAnswer] = useState('');

//   const onQuestionChange = (e) => {
//     setQuestion(e.target.value);
//   };

//   const onSubmit = async (e) => {
//     e.preventDefault();

//     try {
//       const response = await axios.post('http://localhost:5000/answer', { question });
//       setAnswer(response.data.answer);
//     } catch (err) {
//       console.error(err);
//       setAnswer('Failed to get answer');
//     }
//   };

//   return (
//     <div>
//       <h2>Ask a Question</h2>
//       <form onSubmit={onSubmit}>
//         <input
//           type="text"
//           value={question}
//           onChange={onQuestionChange}
//           placeholder="Enter your question"
//         />
//         <button type="submit">Ask</button>
//       </form>
//       <AnswerDisplay answer={answer} />
//     </div>
//   );
// };

// export default QuestionForm;
