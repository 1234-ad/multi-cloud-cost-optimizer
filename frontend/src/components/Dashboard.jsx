/**
 * Multi-Cloud Cost Optimizer Dashboard
 * ===================================
 * 
 * Main dashboard component with real-time cost monitoring,
 * optimization recommendations, and multi-cloud resource visibility.
 */

import React, { useState, useEffect, useCallback } from 'react';
import {
  Box,
  Grid,
  Card,
  CardContent,
  Typography,
  Button,
  Chip,
  Alert,
  CircularProgress,
  Tabs,
  Tab,
  IconButton,
  Tooltip,
  Switch,
  FormControlLabel
} from '@mui/material';
import {
  TrendingUp,
  TrendingDown,
  CloudQueue,
  AttachMoney,
  Speed,
  Warning,
  CheckCircle,
  Refresh,
  Settings,
  Download,
  Notifications
} from '@mui/icons-material';
import {
  LineChart,
  Line,
  AreaChart,
  Area,
  BarChart,
  Bar,
  PieChart,
  Pie,
  Cell,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip as RechartsTooltip,
  Legend,
  ResponsiveContainer
} from 'recharts';

// Custom hooks
import { useApi } from '../hooks/useApi';
import { useWebSocket } from '../hooks/useWebSocket';

// Components
import MetricCard from './MetricCard';
import OptimizationRecommendations from './OptimizationRecommendations';
import ResourceInventory from './ResourceInventory';
import CostForecast from './CostForecast';
import AlertsPanel from './AlertsPanel';

// Constants
const CLOUD_PROVIDERS = {
  aws: { name: 'AWS', color: '#FF9900', icon: '☁️' },
  azure: { name: 'Azure', color: '#0078D4', icon: '🔷' },
  gcp: { name: 'GCP', color: '#4285F4', icon: '🟦' }
};

const REFRESH_INTERVAL = 30000; // 30 seconds

const Dashboard = () => {
  // State management
  const [activeTab, setActiveTab] = useState(0);
  const [autoRefresh, setAutoRefresh] = useState(true);
  const [lastUpdated, setLastUpdated] = useState(new Date());
  const [selectedTimeRange, setSelectedTimeRange] = useState('7d');
  
  // API hooks
  const { data: dashboardData, loading: dashboardLoading, error: dashboardError, refetch: refetchDashboard } = useApi('/api/v1/dashboard/summary');
  const { data: realtimeMetrics, loading: metricsLoading } = useApi('/api/v1/metrics/realtime');
  const { data: recommendations, loading: recommendationsLoading } = useApi('/api/v1/recommendations');
  const { data: costTrend, loading: trendLoading } = useApi(`/api/v1/costs/trend?period=${selectedTimeRange}`);
  
  // WebSocket for real-time updates
  const { connected: wsConnected, lastMessage } = useWebSocket('/ws/dashboard');
  
  // Auto-refresh effect
  useEffect(() => {
    if (!autoRefresh) return;
    
    const interval = setInterval(() => {
      refetchDashboard();
      setLastUpdated(new Date());
    }, REFRESH_INTERVAL);
    
    return () => clearInterval(interval);
  }, [autoRefresh, refetchDashboard]);
  
  // Handle WebSocket messages
  useEffect(() => {
    if (lastMessage) {
      // Update real-time data based on WebSocket messages
      console.log('WebSocket update:', lastMessage);
    }
  }, [lastMessage]);
  
  // Manual refresh handler
  const handleRefresh = useCallback(() => {
    refetchDashboard();
    setLastUpdated(new Date());
  }, [refetchDashboard]);
  
  // Tab change handler
  const handleTabChange = (event, newValue) => {
    setActiveTab(newValue);
  };
  
  // Render loading state
  if (dashboardLoading) {
    return (
      <Box display=\"flex\" justifyContent=\"center\" alignItems=\"center\" minHeight=\"400px\">
        <CircularProgress size={60} />
        <Typography variant=\"h6\" sx={{ ml: 2 }}>
          Loading dashboard...
        </Typography>
      </Box>
    );
  }
  
  // Render error state
  if (dashboardError) {
    return (
      <Alert severity=\"error\" sx={{ m: 2 }}>
        Failed to load dashboard data: {dashboardError.message}
        <Button onClick={handleRefresh} sx={{ ml: 2 }}>
          Retry
        </Button>
      </Alert>
    );
  }
  
  const {
    total_monthly_cost = 0,
    cost_savings = 0,
    savings_percentage = 0,
    active_resources = 0,
    optimization_opportunities = 0,
    alerts_count = 0,
    top_spending_services = [],
    cost_trend = [],
    cloud_distribution = {}
  } = dashboardData || {};
  
  return (
    <Box sx={{ flexGrow: 1, p: 3 }}>
      {/* Header */}
      <Box display=\"flex\" justifyContent=\"space-between\" alignItems=\"center\" mb={3}>
        <Typography variant=\"h4\" component=\"h1\" fontWeight=\"bold\">
          Multi-Cloud Cost Optimizer
        </Typography>
        
        <Box display=\"flex\" alignItems=\"center\" gap={2}>
          {/* Connection status */}
          <Chip
            icon={wsConnected ? <CheckCircle /> : <Warning />}
            label={wsConnected ? 'Connected' : 'Disconnected'}
            color={wsConnected ? 'success' : 'warning'}
            size=\"small\"
          />
          
          {/* Auto-refresh toggle */}
          <FormControlLabel
            control={
              <Switch
                checked={autoRefresh}
                onChange={(e) => setAutoRefresh(e.target.checked)}
                size=\"small\"
              />
            }
            label=\"Auto-refresh\"
          />
          
          {/* Manual refresh */}
          <Tooltip title=\"Refresh data\">
            <IconButton onClick={handleRefresh} disabled={dashboardLoading}>
              <Refresh />
            </IconButton>
          </Tooltip>
          
          {/* Settings */}
          <Tooltip title=\"Settings\">
            <IconButton>
              <Settings />
            </IconButton>
          </Tooltip>
        </Box>
      </Box>
      
      {/* Last updated */}
      <Typography variant=\"body2\" color=\"text.secondary\" mb={2}>
        Last updated: {lastUpdated.toLocaleTimeString()}
      </Typography>
      
      {/* Key Metrics */}
      <Grid container spacing={3} mb={4}>
        <Grid item xs={12} sm={6} md={2}>
          <MetricCard
            title=\"Monthly Cost\"
            value={`$${total_monthly_cost.toLocaleString()}`}
            icon={<AttachMoney />}
            color=\"primary\"
            trend={cost_trend.length > 1 ? 
              ((cost_trend[cost_trend.length - 1]?.cost - cost_trend[cost_trend.length - 2]?.cost) / cost_trend[cost_trend.length - 2]?.cost * 100) : 0
            }
          />
        </Grid>
        
        <Grid item xs={12} sm={6} md={2}>
          <MetricCard
            title=\"Cost Savings\"
            value={`$${cost_savings.toLocaleString()}`}
            icon={<TrendingDown />}
            color=\"success\"
            trend={savings_percentage}
            subtitle={`${savings_percentage.toFixed(1)}% saved`}
          />
        </Grid>
        
        <Grid item xs={12} sm={6} md={2}>
          <MetricCard
            title=\"Active Resources\"
            value={active_resources.toLocaleString()}
            icon={<CloudQueue />}
            color=\"info\"
          />
        </Grid>
        
        <Grid item xs={12} sm={6} md={2}>
          <MetricCard
            title=\"Opportunities\"
            value={optimization_opportunities}
            icon={<Speed />}
            color=\"warning\"
            subtitle=\"Optimization available\"
          />
        </Grid>
        
        <Grid item xs={12} sm={6} md={2}>
          <MetricCard
            title=\"Alerts\"
            value={alerts_count}
            icon={<Warning />}
            color={alerts_count > 0 ? \"error\" : \"success\"}
            subtitle={alerts_count > 0 ? \"Needs attention\" : \"All good\"}
          />
        </Grid>
        
        <Grid item xs={12} sm={6} md={2}>
          <MetricCard
            title=\"Efficiency\"
            value=\"92%\"
            icon={<TrendingUp />}
            color=\"success\"
            subtitle=\"Resource utilization\"
          />
        </Grid>
      </Grid>
      
      {/* Main Content Tabs */}
      <Card>
        <Box sx={{ borderBottom: 1, borderColor: 'divider' }}>
          <Tabs value={activeTab} onChange={handleTabChange} aria-label=\"dashboard tabs\">
            <Tab label=\"Overview\" />
            <Tab label=\"Cost Analysis\" />
            <Tab label=\"Optimization\" />
            <Tab label=\"Resources\" />
            <Tab label=\"Forecasting\" />
            <Tab label=\"Alerts\" />
          </Tabs>
        </Box>
        
        <CardContent>
          {/* Overview Tab */}
          {activeTab === 0 && (
            <Grid container spacing={3}>
              {/* Cost Trend Chart */}
              <Grid item xs={12} md={8}>
                <Typography variant=\"h6\" gutterBottom>
                  Cost Trend (Last 30 Days)
                </Typography>
                <ResponsiveContainer width=\"100%\" height={300}>
                  <AreaChart data={cost_trend}>
                    <CartesianGrid strokeDasharray=\"3 3\" />
                    <XAxis dataKey=\"date\" />
                    <YAxis />
                    <RechartsTooltip formatter={(value) => [`$${value.toLocaleString()}`, 'Cost']} />
                    <Area type=\"monotone\" dataKey=\"cost\" stroke=\"#8884d8\" fill=\"#8884d8\" fillOpacity={0.3} />
                  </AreaChart>
                </ResponsiveContainer>
              </Grid>
              
              {/* Cloud Distribution */}
              <Grid item xs={12} md={4}>
                <Typography variant=\"h6\" gutterBottom>
                  Cloud Distribution
                </Typography>
                <ResponsiveContainer width=\"100%\" height={300}>
                  <PieChart>
                    <Pie
                      data={Object.entries(cloud_distribution).map(([provider, data]) => ({
                        name: CLOUD_PROVIDERS[provider]?.name || provider,
                        value: data.cost,
                        color: CLOUD_PROVIDERS[provider]?.color
                      }))}
                      cx=\"50%\"
                      cy=\"50%\"
                      labelLine={false}
                      label={({ name, percentage }) => `${name} ${(percentage || 0).toFixed(0)}%`}
                      outerRadius={80}
                      fill=\"#8884d8\"
                      dataKey=\"value\"
                    >
                      {Object.entries(cloud_distribution).map(([provider], index) => (
                        <Cell key={`cell-${index}`} fill={CLOUD_PROVIDERS[provider]?.color || '#8884d8'} />
                      ))}
                    </Pie>
                    <RechartsTooltip formatter={(value) => [`$${value.toLocaleString()}`, 'Cost']} />
                  </PieChart>
                </ResponsiveContainer>
              </Grid>
              
              {/* Top Spending Services */}
              <Grid item xs={12}>
                <Typography variant=\"h6\" gutterBottom>
                  Top Spending Services
                </Typography>
                <ResponsiveContainer width=\"100%\" height={250}>
                  <BarChart data={top_spending_services}>
                    <CartesianGrid strokeDasharray=\"3 3\" />
                    <XAxis dataKey=\"service\" />
                    <YAxis />
                    <RechartsTooltip formatter={(value) => [`$${value.toLocaleString()}`, 'Cost']} />
                    <Bar dataKey=\"cost\" fill=\"#8884d8\" />
                  </BarChart>
                </ResponsiveContainer>
              </Grid>
            </Grid>
          )}
          
          {/* Cost Analysis Tab */}
          {activeTab === 1 && (
            <Box>
              <Typography variant=\"h6\" gutterBottom>
                Detailed Cost Analysis
              </Typography>
              {/* Cost analysis components would go here */}
              <Alert severity=\"info\">
                Detailed cost analysis features coming soon...
              </Alert>
            </Box>
          )}
          
          {/* Optimization Tab */}
          {activeTab === 2 && (
            <OptimizationRecommendations 
              recommendations={recommendations}
              loading={recommendationsLoading}
            />
          )}
          
          {/* Resources Tab */}
          {activeTab === 3 && (
            <ResourceInventory />
          )}
          
          {/* Forecasting Tab */}
          {activeTab === 4 && (
            <CostForecast />
          )}
          
          {/* Alerts Tab */}
          {activeTab === 5 && (
            <AlertsPanel />
          )}
        </CardContent>
      </Card>
      
      {/* Real-time Metrics Footer */}
      {realtimeMetrics && (
        <Card sx={{ mt: 3 }}>
          <CardContent>
            <Typography variant=\"h6\" gutterBottom>
              Real-time Metrics
            </Typography>
            <Grid container spacing={2}>
              <Grid item xs={12} sm={3}>
                <Typography variant=\"body2\" color=\"text.secondary\">
                  Current Hourly Spend
                </Typography>
                <Typography variant=\"h6\">
                  ${realtimeMetrics.current_hourly_spend}
                </Typography>
              </Grid>
              <Grid item xs={12} sm={3}>
                <Typography variant=\"body2\" color=\"text.secondary\">
                  Active Instances
                </Typography>
                <Typography variant=\"h6\">
                  {Object.values(realtimeMetrics.active_instances || {}).reduce((a, b) => a + b, 0)}
                </Typography>
              </Grid>
              <Grid item xs={12} sm={3}>
                <Typography variant=\"body2\" color=\"text.secondary\">
                  Avg CPU Utilization
                </Typography>
                <Typography variant=\"h6\">
                  {realtimeMetrics.cpu_utilization?.average || 0}%
                </Typography>
              </Grid>
              <Grid item xs={12} sm={3}>
                <Typography variant=\"body2\" color=\"text.secondary\">
                  Storage Usage
                </Typography>
                <Typography variant=\"h6\">
                  {(realtimeMetrics.storage_usage?.total_gb || 0).toLocaleString()} GB
                </Typography>
              </Grid>
            </Grid>
          </CardContent>
        </Card>
      )}
    </Box>
  );
};

export default Dashboard;