import styles from '../styles/RefreshButton.module.css';

export default function RefreshButton({ onClick, isLoading }) {
  return (
    <button 
      className={styles.refreshButton} 
      onClick={onClick}
      disabled={isLoading}
    >
      {isLoading ? 'Refreshing...' : '🔄 Refresh Data Now'}
    </button>
  );
} 