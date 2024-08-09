import React from 'react';

// Define the styles for the header component
const headerStyle = {
  textAlign: 'center', // Center-aligns the text
  color: '#6d8d24',    // PSU Green color
  fontSize: '2rem',    // Font size for the header
  marginBottom: '20px', // Spacing below the header
  padding: '10px',     // Padding around the header text
};

// Header component
const Header = () => {
  return (
    <header style={headerStyle}>
      Portland State University Syllabus Application
    </header>
  );
};

export default Header;
