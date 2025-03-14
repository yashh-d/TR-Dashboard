import Head from 'next/head';
import { useState, useEffect } from 'react';
import axios from 'axios';
import dynamic from 'next/dynamic';
import { format } from 'date-fns';
import styles from '../styles/Dashboard.module.css';

// Import components
import BlockchainCard from '../components/BlockchainCard';
import RefreshButton from '../components/RefreshButton';

// Constants
const BLOCKCHAIN_MAPPING = {
  "Aptos": {"defillama": "aptos", "coingecko": "aptos"},
  "Avalanche": {"defillama": "Avalanche", "coingecko": "avalanche-2"},
  "Core DAO": {"defillama": "core", "coingecko": "coredaoorg"},
  "Flow": {"defillama": "flow", "coingecko": "flow"},
  "Injective": {"defillama": "injective", "coingecko": "injective-protocol"},
  "Optimism": {"defillama": "optimism", "coingecko": "optimism"},
  "Polygon": {"defillama": "polygon", "coingecko": "matic-network"},
  "XRP/XRPL": {"defillama": "XRPL", "coingecko": "ripple"},
  "Sei": {"defillama": "sei", "coingecko": "sei-network"}
};

export default function Home() {
  const [blockchainData, setBlockchainData] = useState({});
  const [lastUpdated, setLastUpdated] = useState(null);
  const [isLoading, setIsLoading] = useState(true);

  // Fetch data for all blockchains
  const fetchAllData = async () => {
    setIsLoading(true);
    const newData = {};
    const fetchPromises = [];

    for (const [blockchain, ids] of Object.entries(BLOCKCHAIN_MAPPING)) {
      fetchPromises.push(
        fetchBlockchainData(blockchain, ids.defillama, ids.coingecko)
          .then(data => {
            newData[blockchain] = data;
          })
      );
    }

    await Promise.all(fetchPromises);
    setBlockchainData(newData);
    setLastUpdated(new Date());
    setIsLoading(false);
  };

  // Fetch data for a single blockchain
  const fetchBlockchainData = async (blockchain, defiLlamaId, coinGeckoId) => {
    try {
      // Fetch TVL data
      const tvlResponse = await axios.get(`https://api.llama.fi/v2/historicalChainTvl/${defiLlamaId}`);
      const tvlData = tvlResponse.data.map(item => ({
        date: new Date(item.date * 1000),
        tvl: item.tvl
      }));

      // Fetch price data
      const priceResponse = await axios.get(
        `https://api.coingecko.com/api/v3/coins/${coinGeckoId}/market_chart?vs_currency=usd&days=90&interval=daily`
      );
      const priceData = priceResponse.data.prices.map(item => ({
        date: new Date(item[0]),
        price: item[1]
      }));

      return {
        tvlData,
        priceData
      };
    } catch (error) {
      console.error(`Error fetching data for ${blockchain}:`, error);
      return {
        tvlData: [],
        priceData: []
      };
    }
  };

  // Initial data fetch
  useEffect(() => {
    fetchAllData();
    
    // Set up hourly refresh
    const intervalId = setInterval(fetchAllData, 3600000);
    return () => clearInterval(intervalId);
  }, []);

  return (
    <div className={styles.container}>
      <Head>
        <title>Blockchain Metrics Dashboard</title>
        <meta name="description" content="Dashboard showing blockchain metrics" />
        <link rel="icon" href="/favicon.ico" />
      </Head>

      <main className={styles.main}>
        <h1 className={styles.title}>Token Relations Dashboard 📊</h1>
        
        {lastUpdated && (
          <p className={styles.lastUpdated}>
            Last updated: {format(lastUpdated, 'yyyy-MM-dd HH:mm:ss')} UTC
          </p>
        )}

        <RefreshButton onClick={fetchAllData} isLoading={isLoading} />

        <div className={styles.grid}>
          {Object.entries(blockchainData).map(([blockchain, data]) => (
            <BlockchainCard 
              key={blockchain}
              blockchain={blockchain}
              tvlData={data.tvlData || []}
              priceData={data.priceData || []}
            />
          ))}
        </div>
      </main>
    </div>
  );
} 