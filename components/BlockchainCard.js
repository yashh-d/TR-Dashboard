import { useMemo } from 'react';
import dynamic from 'next/dynamic';
import styles from '../styles/BlockchainCard.module.css';

// Import Plotly dynamically to avoid SSR issues
const Plot = dynamic(() => import('react-plotly.js'), { ssr: false });

export default function BlockchainCard({ blockchain, tvlData, priceData }) {
  // Calculate TVL metrics
  const tvlMetrics = useMemo(() => {
    if (!tvlData || tvlData.length === 0) return null;
    
    const currentTvl = tvlData[tvlData.length - 1].tvl;
    let monthlyChange = 0;
    
    if (tvlData.length > 30) {
      const monthAgoTvl = tvlData[tvlData.length - 31].tvl;
      monthlyChange = ((currentTvl - monthAgoTvl) / monthAgoTvl) * 100;
    }
    
    return { currentTvl, monthlyChange };
  }, [tvlData]);

  // Calculate price metrics
  const priceMetrics = useMemo(() => {
    if (!priceData || priceData.length === 0) return null;
    
    const currentPrice = priceData[priceData.length - 1].price;
    let monthlyChange = 0;
    
    if (priceData.length > 30) {
      const monthAgoPrice = priceData[priceData.length - 31].price;
      monthlyChange = ((currentPrice - monthAgoPrice) / monthAgoPrice) * 100;
    }
    
    return { currentPrice, monthlyChange };
  }, [priceData]);

  // TVL chart configuration
  const tvlChartConfig = useMemo(() => {
    if (!tvlData || tvlData.length === 0) return null;
    
    return {
      data: [
        {
          x: tvlData.map(d => d.date),
          y: tvlData.map(d => d.tvl),
          type: 'scatter',
          mode: 'lines',
          name: 'TVL',
          line: { color: '#3498db', width: 2 },
          fill: 'tozeroy',
          fillcolor: 'rgba(52, 152, 219, 0.2)'
        }
      ],
      layout: {
        height: 400,
        margin: { l: 50, r: 20, t: 30, b: 50 },
        paper_bgcolor: 'white',
        plot_bgcolor: 'white',
        xaxis: {
          title: "Date",
          showgrid: true,
          gridcolor: 'rgba(230, 230, 230, 0.8)',
          tickfont: { color: '#000000' },
          titlefont: { color: '#000000' }
        },
        yaxis: {
          title: "TVL (USD)",
          showgrid: true,
          gridcolor: 'rgba(230, 230, 230, 0.8)',
          tickprefix: "$",
          tickfont: { color: '#000000' },
          titlefont: { color: '#000000' }
        },
        hovermode: "x unified"
      }
    };
  }, [tvlData]);

  // Price chart configuration
  const priceChartConfig = useMemo(() => {
    if (!priceData || priceData.length === 0) return null;
    
    return {
      data: [
        {
          x: priceData.map(d => d.date),
          y: priceData.map(d => d.price),
          type: 'scatter',
          mode: 'lines',
          name: 'Price',
          line: { color: '#2ecc71', width: 2 },
          fill: 'tozeroy',
          fillcolor: 'rgba(46, 204, 113, 0.2)'
        }
      ],
      layout: {
        height: 400,
        margin: { l: 50, r: 20, t: 30, b: 50 },
        paper_bgcolor: 'white',
        plot_bgcolor: 'white',
        xaxis: {
          title: "Date",
          showgrid: true,
          gridcolor: 'rgba(230, 230, 230, 0.8)',
          tickfont: { color: '#000000' },
          titlefont: { color: '#000000' }
        },
        yaxis: {
          title: "Price (USD)",
          showgrid: true,
          gridcolor: 'rgba(230, 230, 230, 0.8)',
          tickprefix: "$",
          tickfont: { color: '#000000' },
          titlefont: { color: '#000000' }
        },
        hovermode: "x unified"
      }
    };
  }, [priceData]);

  return (
    <div className={styles.card}>
      <h2 className={styles.blockchainTitle}>{blockchain}</h2>
      
      <div className={styles.chartsContainer}>
        {/* TVL Section */}
        <div className={styles.chartSection}>
          <h3 className={styles.chartTitle}>Total Value Locked (TVL)</h3>
          
          {tvlData && tvlData.length > 0 ? (
            <>
              <div className={styles.chartWrapper}>
                <Plot
                  data={tvlChartConfig.data}
                  layout={tvlChartConfig.layout}
                  config={{ responsive: true }}
                />
              </div>
              
              {tvlMetrics && (
                <div className={styles.metricsContainer}>
                  <div className={styles.metricCard}>
                    <h4>Current TVL</h4>
                    <h2>${tvlMetrics.currentTvl.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}</h2>
                  </div>
                  
                  <div className={styles.metricCard}>
                    <h4>30-Day Change</h4>
                    <h2 className={tvlMetrics.monthlyChange >= 0 ? styles.positive : styles.negative}>
                      {tvlMetrics.monthlyChange >= 0 ? '+' : ''}{tvlMetrics.monthlyChange.toFixed(2)}%
                    </h2>
                  </div>
                </div>
              )}
            </>
          ) : (
            <div className={styles.noData}>No TVL data available for {blockchain}</div>
          )}
        </div>
        
        {/* Price Section */}
        <div className={styles.chartSection}>
          <h3 className={styles.chartTitle}>Price (USD)</h3>
          
          {priceData && priceData.length > 0 ? (
            <>
              <div className={styles.chartWrapper}>
                <Plot
                  data={priceChartConfig.data}
                  layout={priceChartConfig.layout}
                  config={{ responsive: true }}
                />
              </div>
              
              {priceMetrics && (
                <div className={styles.metricsContainer}>
                  <div className={styles.metricCard}>
                    <h4>Current Price</h4>
                    <h2>${priceMetrics.currentPrice.toLocaleString(undefined, { minimumFractionDigits: 4, maximumFractionDigits: 4 })}</h2>
                  </div>
                  
                  <div className={styles.metricCard}>
                    <h4>30-Day Change</h4>
                    <h2 className={priceMetrics.monthlyChange >= 0 ? styles.positive : styles.negative}>
                      {priceMetrics.monthlyChange >= 0 ? '+' : ''}{priceMetrics.monthlyChange.toFixed(2)}%
                    </h2>
                  </div>
                </div>
              )}
            </>
          ) : (
            <div className={styles.noData}>No price data available for {blockchain}</div>
          )}
        </div>
      </div>
    </div>
  );
} 