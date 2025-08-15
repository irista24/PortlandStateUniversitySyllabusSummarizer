import React from 'react';

const headerStyle = {
  textAlign: 'center', 
  color: '#6d8d24',    
  fontSize: '2rem',    
  marginBottom: '20px', 
  padding: '10px',     
};

const Header = () => {
  return (
    <header style={headerStyle}>
      Portland State University Syllabus Application
    </header>
  );
};

export default Header;
