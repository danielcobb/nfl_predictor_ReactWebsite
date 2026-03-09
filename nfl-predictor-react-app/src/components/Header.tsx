const Header = () => {
  return (
    <header className="site-header">
      <div className="logo-lockup">
        <div className="logo-icon-wrap">
          <svg
            className="logo-icon"
            viewBox="0 0 32 32"
            fill="none"
            xmlns="http://www.w3.org/2000/svg"
          >
            {/* Football outline */}
            <ellipse
              cx="16"
              cy="16"
              rx="11.5"
              ry="7"
              stroke="#f59e0b"
              strokeWidth="1.7"
            />
            {/* Center lace horizontal */}
            <line
              x1="4.5"
              y1="16"
              x2="27.5"
              y2="16"
              stroke="#f59e0b"
              strokeWidth="1.4"
            />
            {/* Center lace vertical */}
            <line
              x1="16"
              y1="9.2"
              x2="16"
              y2="22.8"
              stroke="#f59e0b"
              strokeWidth="1.4"
            />
            {/* Side stitches */}
            <line
              x1="12"
              y1="11.8"
              x2="12"
              y2="20.2"
              stroke="#f59e0b"
              strokeWidth="1"
              strokeOpacity="0.5"
            />
            <line
              x1="20"
              y1="11.8"
              x2="20"
              y2="20.2"
              stroke="#f59e0b"
              strokeWidth="1"
              strokeOpacity="0.5"
            />
          </svg>
        </div>

        <div className="logo-text-block">
          <span className="logo-nfl-tag">NFL</span>
          <span className="logo-wordmark">PREDICTOR</span>
        </div>
      </div>

      <p className="header-tagline">ML-powered game outcome predictions</p>
    </header>
  );
};

export default Header;
