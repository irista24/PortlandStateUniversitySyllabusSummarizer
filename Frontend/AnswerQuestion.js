import React, { useState } from 'react';
import axios from 'axios';

const AnswerQuestion = () => {
    const [keyword, setKeyword] = useState('');
    const [answer, setAnswer] = useState('');
    
    const onKeywordChange = (e) => {
        setKeyword(e.target.value);
    };

    const getAnswer = async () => {
        try {
            const response = await axios.post('http://localhost:5000/answer', { keyword });
            setAnswer(response.data.answer);
        } catch (error) {
            console.error('Error fetching answer:', error);
        }
    };

    return (
        <div style={styles.container}>
            <h3 style={styles.header}>Ask a Question</h3>
            
            <input 
                type="text" 
                value={keyword} 
                onChange={onKeywordChange} 
                placeholder="Enter keyword" 
                style={styles.input}
            />
            <button onClick={getAnswer} style={styles.button}>Get Answer</button>
            
            {answer && <div style={styles.answer}>{answer}</div>}
        </div>
    );
};

// Define your PSU theme styles
const styles = {
    container: {
        backgroundColor: '#6d8d24', // PSU Green color
        color: 'white',
        padding: '20px',
        borderRadius: '5px',
        maxWidth: '600px',
        margin: '0 auto'
    },
    header: {
        textAlign: 'center',
        marginBottom: '20px'
    },
    button: {
        backgroundColor: '#f4f4f4', // Light button color
        color: '#6d8d24', // PSU Green color
        border: 'none',
        padding: '10px 20px',
        borderRadius: '5px',
        cursor: 'pointer',
        marginRight: '10px'
    },
    input: {
        padding: '10px',
        borderRadius: '5px',
        border: '1px solid #ddd',
        marginBottom: '10px'
    },
    answer: {
        marginTop: '20px',
        fontSize: '18px'
    }
};

export default AnswerQuestion;
