import React, { useState } from 'react';
import axios from 'axios';
import AnswerDisplay from './AnswerDisplay';
import "../Chatbot.css";

const QuestionForm = ({ setAnswer }) => {
  const [question, setQuestion] = useState('');

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
    <div className="chatbot-container">
      <h2>Ask a Question:</h2>
      <form onSubmit={onSubmit}>
        <input
          type="text"
          value={question}
          onChange={onQuestionChange}
          placeholder="Enter your question"
          className="chatbot-input"
        />
        <button type="submit" className="chatbot-button">Generate Answer</button>
      </form>
    </div>
  );
};

export default QuestionForm;
