import React from 'react';
import "../Chatbot.css";

const AnswerDisplay = ({ answer }) => {
  return (
    <div className="chatbot-container">
      <h2>Answer:</h2>
      <p className="chatbot-message">{answer}</p>
    </div>
  );
};

export default AnswerDisplay;

// import React from 'react';
// import "./App.css";
// const AnswerDisplay = ({ answer }) => {
//   return (
//     <div>
//       <h2></h2>
//       <p>{answer}</p>
//     </div>
//   );
// };

// export default AnswerDisplay;
