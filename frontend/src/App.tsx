import { useEffect, useState } from "react";
import AccountOverview from "./screens/AccountOverview";
import DashboardHealth from "./screens/DashboardHealth";
import ExecuteTrades from "./screens/ExecuteTrades";
import MLPipeline from "./screens/MLPipeline";
import PlaceholderScreen from "./screens/PlaceholderScreen";
import PositionsMonitoring from "./screens/PositionsMonitoring";
import ScreenerOverview from "./screens/ScreenerOverview";
import TradesOverview from "./screens/TradesOverview";

const tabLabels = [
  "Dashboard",
  "Account",
  "Trades",
  "Positions",
  "Execute",
  "Screener",
  "ML Pipeline",
];

const toHash = (label: string) => `#${label.toLowerCase().replace(/\s+/g, "-")}`;

const labelFromHash = (hash: string) => {
  const normalized = hash.replace("#", "").toLowerCase();
  return (
    tabLabels.find((label) => label.toLowerCase().replace(/\s+/g, "-") === normalized) ?? null
  );
};

export default function App() {
  const [activeTab, setActiveTab] = useState("Dashboard");

  useEffect(() => {
    const initial = labelFromHash(window.location.hash);
    if (initial) {
      setActiveTab(initial);
    }

    const handleHashChange = () => {
      const next = labelFromHash(window.location.hash);
      if (next) {
        setActiveTab(next);
      }
    };

    window.addEventListener("hashchange", handleHashChange);
    return () => window.removeEventListener("hashchange", handleHashChange);
  }, []);

  const handleTabSelect = (label: string) => {
    setActiveTab(label);
    window.location.hash = toHash(label);
  };

  if (activeTab === "Dashboard") {
    return <DashboardHealth activeTab={activeTab} onTabSelect={handleTabSelect} />;
  }
  if (activeTab === "Account") {
    return <AccountOverview activeTab={activeTab} onTabSelect={handleTabSelect} />;
  }
  if (activeTab === "Trades") {
    return <TradesOverview activeTab={activeTab} onTabSelect={handleTabSelect} />;
  }
  if (activeTab === "Positions") {
    return <PositionsMonitoring activeTab={activeTab} onTabSelect={handleTabSelect} />;
  }
  if (activeTab === "Execute") {
    return <ExecuteTrades activeTab={activeTab} onTabSelect={handleTabSelect} />;
  }
  if (activeTab === "Screener") {
    return <ScreenerOverview activeTab={activeTab} onTabSelect={handleTabSelect} />;
  }
  if (activeTab === "ML Pipeline") {
    return <MLPipeline activeTab={activeTab} onTabSelect={handleTabSelect} />;
  }
  return (
    <PlaceholderScreen title={activeTab} activeTab={activeTab} onTabSelect={handleTabSelect} />
  );
}
