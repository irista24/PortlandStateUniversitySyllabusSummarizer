import React, { useState } from 'react';
import axios from 'axios';

const KeywordSearch = () => {
  const [keyword, setKeyword] = useState('');
  const [summaries, setSummaries] = useState([]);
  const [error, setError] = useState('');

  const handleSearch = async () => {
    setError(''); // Clear previous errors
    try {
      const response = await axios.post('http://localhost:5000/search_keyword', { keyword });

      if (response.data && Array.isArray(response.data.summaries)) {
        setSummaries(response.data.summaries);
      } else {
        setSummaries([]);
        setError('No summaries available.');
      }
    } catch (error) {
      console.error('Error searching keyword:', error);
      setSummaries([]);
      setError('An error occurred while searching the keyword.');
    }
  };

  return (
    <div>
      <input
        type="text"
        value={keyword}
        onChange={(e) => setKeyword(e.target.value)}
        placeholder="Enter keyword"
      />
      <button onClick={handleSearch}>Search</button>
      {error && <div style={{ color: 'red' }}>{error}</div>}
      <div>
        <h3>Summaries:</h3>
        {summaries.length > 0 ? (
          <ul>
            {summaries.map((summary, index) => (
              <li key={index}>{summary}</li>
            ))}
          </ul>
        ) : (
          <p>No summaries found.</p>
        )}
      </div>
    </div>
  );
};

export default KeywordSearch;



// import React, { useState } from 'react';
// import axios from 'axios';

// const KeywordSearch = () => {
//   const [keyword, setKeyword] = useState('');
//   const [results, setResults] = useState([]);
//   const [loading, setLoading] = useState(false);
//   const [error, setError] = useState('');

//   const handleSearch = async () => {
//     if (!keyword) {
//       setError('Please enter a keyword.');
//       return;
//     }

//     setLoading(true);
//     setError('');
//     try {
//       const response = await axios.post('http://localhost:5000/search_keyword', {
//         keyword: keyword,
//       });
//       setResults(response.data.instances);
//     } catch (err) {
//       setError('Error searching for keyword.');
//     } finally {
//       setLoading(false);
//     }
//   };

//   return (
//     <div>
//       <input
//         type="text"
//         value={keyword}
//         onChange={(e) => setKeyword(e.target.value)}
//         placeholder="Enter keyword"
//       />
//       <button onClick={handleSearch} disabled={loading}>
//         {loading ? 'Searching...' : 'Search Keyword'}
//       </button>
//       {error && <p style={{ color: 'red' }}>{error}</p>}
//       <ul>
//         {results.map((result, index) => (
//           <li key={index}>{result}</li>
//         ))}
//       </ul>
//     </div>
//   );
// };

// export default KeywordSearch;
