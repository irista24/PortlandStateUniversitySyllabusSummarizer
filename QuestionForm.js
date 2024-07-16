import React, { useState } from 'react';
import axios from 'axios';

const QuestionForm = ({ setAnswer }) => {
  const [question, setQuestion] = useState('');

  const handleQuestionChange = (event) => {
    setQuestion(event.target.value);
  };

  const handleQuestionSubmit = async (event) => {
    event.preventDefault();

    try {
      const response = await axios.post('http://localhost:5000/ask', { question });
      setAnswer(response.data.answer);
    } catch (error) {
      console.error('There was an error asking the question!', error);
      setAnswer('There was an error asking the question.');
    }
  };

  return (
    <div>
      <h2>Ask a Question</h2>
      <form onSubmit={handleQuestionSubmit}>
        <input
          type="text"
          value={question}
          onChange={handleQuestionChange}
          placeholder="Enter your question"
          required
        />
        <button type="submit">Ask</button>
      </form>
    </div>
  );
};

export default QuestionForm;
