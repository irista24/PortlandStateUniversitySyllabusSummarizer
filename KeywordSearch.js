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
//   const [summaries, setSummaries] = useState([]);
//   const [error, setError] = useState('');

//   const handleSearch = async () => {
//     setError(''); // Clear previous errors
//     try {
//       const response = await axios.post('http://localhost:5000/search_keyword', { keyword });

//       if (response.data && Array.isArray(response.data.summaries)) {
//         setSummaries(response.data.summaries);
//       } else {
//         setSummaries([]);
//         setError('No summaries available.');
//       }
//     } catch (error) {
//       console.error('Error searching keyword:', error);
//       setSummaries([]);
//       setError('An error occurred while searching the keyword.');
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
//       <button onClick={handleSearch}>Search</button>
//       {error && <div style={{ color: 'red' }}>{error}</div>}
//       <div>
//         <h3>Summaries:</h3>
//         {summaries.length > 0 ? (
//           <ul>
//             {summaries.map((summary, index) => (
//               <li key={index}>{summary}</li>
//             ))}
//           </ul>
//         ) : (
//           <p>No summaries found.</p>
//         )}
//       </div>
//     </div>
//   );
// };

// export default KeywordSearch;
