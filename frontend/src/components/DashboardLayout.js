import React, { useState, useEffect } from 'react';
import { Outlet, useNavigate, useLocation } from 'react-router-dom';
import {
  Drawer,
  AppBar,
  Toolbar,
  List,
  ListItem,
  ListItemIcon,
  ListItemText,
  IconButton,
  Typography,
  Avatar,
  Box,
  Menu,
  MenuItem,
  Chip
} from '@mui/material';
import {
  Menu as MenuIcon,
  Dashboard as DashboardIcon,
  BugReport as DiseaseIcon,
  Warning as AlertsIcon,
  Person as ProfileIcon,
  Settings as SettingsIcon,
  Help as HelpIcon,
  Agriculture as AgricultureIcon,
  ExitToApp as LogoutIcon,
  Language as LanguageIcon
} from '@mui/icons-material';
import { useAuth } from '../context/AuthContext';
import { useTranslation } from '../context/LanguageContext';

const drawerWidth = 240;

function DashboardLayout() {
  const { userProfile, logout } = useAuth();
  const { t, language } = useTranslation();
  const navigate = useNavigate();
  const location = useLocation();
  const [mobileOpen, setMobileOpen] = useState(false);
  const [anchorEl, setAnchorEl] = useState(null);

  // Debug logs
  console.log('🔍 DashboardLayout - Current language:', language);
  console.log('🔍 DashboardLayout - Dashboard translation:', t('dashboard'));
  console.log('🔍 DashboardLayout - localStorage:', localStorage.getItem('grapeGuardSettings'));

  const handleDrawerToggle = () => {
    setMobileOpen(!mobileOpen);
  };

  const handleMenu = (event) => {
    setAnchorEl(event.currentTarget);
  };

  const handleClose = () => {
    setAnchorEl(null);
  };

  const handleLogout = async () => {
    try {
      await logout();
      navigate('/login');
    } catch (error) {
      console.error('Logout failed:', error);
    }
    handleClose();
  };

  const handleNavigation = (path) => {
    navigate(path);
    setMobileOpen(false);
  };

  // Menu items with translations
  const menuItems = [
    { text: t('dashboard'), icon: DashboardIcon, path: '/dashboard' },
    { text: t('diseaseDetection'), icon: DiseaseIcon, path: '/detection' },
    { text: t('alerts'), icon: AlertsIcon, path: '/alerts' },
    { text: t('profile'), icon: ProfileIcon, path: '/profile' },
    { text: t('settings'), icon: SettingsIcon, path: '/settings' },
    { text: t('help'), icon: HelpIcon, path: '/help' },
  ];

  console.log('🔍 Menu items:', menuItems); // Debug menu items

  // Page titles with translations
  const getPageTitle = () => {
    const path = location.pathname;
    const titleMap = {
      '/dashboard': t('welcomeToGrapeGuard'),
      '/detection': t('diseaseDetection'),
      '/alerts': t('alerts'),
      '/profile': t('profile'),
      '/settings': t('settings'),
      '/help': t('help')
    };
    return titleMap[path] || 'Grape Guard';
  };

  // Language display helper
  const getLanguageDisplay = () => {
    const languageMap = {
      english: 'Eng',
      hindi: 'हिं',
      marathi: 'मर'
    };
    return languageMap[language] || 'Eng';
  };

  const drawer = (
    <div>
      {/* Logo */}
      <Box p={2} display="flex" alignItems="center">
        <div style={{
          background: 'linear-gradient(to right, #8b5cf6, #22c55e)',
          width: '2rem',
          height: '2rem',
          borderRadius: '50%',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          marginRight: '0.75rem'
        }}>
          <AgricultureIcon style={{ color: 'white', fontSize: '1rem' }} />
        </div>
        <Typography variant="h6" style={{ fontWeight: 'bold', color: '#1f2937' }}>
          Grape Guard
        </Typography>
      </Box>

      {/* User Profile */}
      <Box p={2} style={{ backgroundColor: '#f9fafb', margin: '1rem' }}>
        <Box display="flex" alignItems="center" mb={1}>
          <Avatar style={{ backgroundColor: '#22c55e', width: '3rem', height: '3rem', marginRight: '0.75rem' }}>
            {userProfile?.fullName?.charAt(0) || 'U'}
          </Avatar>
          <Box>
            <Typography variant="body1" style={{ fontWeight: 600 }}>
              {userProfile?.fullName || 'Demo User'}
            </Typography>
          </Box>
        </Box>
      </Box>

      {/* Navigation Menu */}
      <List>
        {menuItems.map((item) => {
          const isActive = location.pathname === item.path;
          return (
            <ListItem 
              button 
              key={item.path}
              onClick={() => handleNavigation(item.path)}
              style={{ 
                backgroundColor: isActive ? '#f0fdf4' : 'transparent',
                borderRight: isActive ? '3px solid #22c55e' : 'none',
                margin: '0.25rem 0'
              }}
            >
              <ListItemIcon style={{ color: isActive ? '#22c55e' : '#6b7280' }}>
                <item.icon />
              </ListItemIcon>
              <ListItemText 
                primary={item.text} 
                style={{ color: isActive ? '#22c55e' : '#1f2937' }}
              />
            </ListItem>
          );
        })}
      </List>
    </div>
  );

  return (
    <Box sx={{ display: 'flex' }}>
      {/* App Bar */}
      <AppBar
        position="fixed"
        sx={{
          width: { sm: `calc(100% - ${drawerWidth}px)` },
          ml: { sm: `${drawerWidth}px` },
          backgroundColor: 'white',
          color: '#1f2937',
          boxShadow: '0 1px 3px rgba(0,0,0,0.1)'
        }}
      >
        <Toolbar>
          <IconButton
            color="inherit"
            aria-label="open drawer"
            edge="start"
            onClick={handleDrawerToggle}
            sx={{ mr: 2, display: { sm: 'none' } }}
          >
            <MenuIcon />
          </IconButton>
          
          <Typography variant="h6" component="h1" sx={{ flexGrow: 1, fontWeight: 600 }}>
            {getPageTitle()}
          </Typography>

          {/* Language Display & Profile */}
          <Box display="flex" alignItems="center" gap={1}>
            <LanguageIcon style={{ color: '#6b7280' }} />
            <Chip 
              label={getLanguageDisplay()} 
              size="small" 
              style={{ 
                backgroundColor: '#f3f4f6', 
                color: '#374151',
                fontWeight: 600
              }} 
            />
            
            <IconButton color="inherit" onClick={handleMenu}>
              <Avatar style={{ backgroundColor: '#8b5cf6', width: '2rem', height: '2rem' }}>
                {userProfile?.fullName?.charAt(0) || 'U'}
              </Avatar>
            </IconButton>
          </Box>
          
          <Menu anchorEl={anchorEl} open={Boolean(anchorEl)} onClose={handleClose}>
            <MenuItem onClick={handleLogout}>
              <LogoutIcon style={{ marginRight: '0.5rem' }} />
              {t('logout')}
            </MenuItem>
          </Menu>
        </Toolbar>
      </AppBar>

      {/* Sidebar */}
      <Box
        component="nav"
        sx={{ width: { sm: drawerWidth }, flexShrink: { sm: 0 } }}
      >
        <Drawer
          variant="temporary"
          open={mobileOpen}
          onClose={handleDrawerToggle}
          ModalProps={{ keepMounted: true }}
          sx={{
            display: { xs: 'block', sm: 'none' },
            '& .MuiDrawer-paper': { boxSizing: 'border-box', width: drawerWidth },
          }}
        >
          {drawer}
        </Drawer>
        <Drawer
          variant="permanent"
          sx={{
            display: { xs: 'none', sm: 'block' },
            '& .MuiDrawer-paper': { boxSizing: 'border-box', width: drawerWidth },
          }}
          open
        >
          {drawer}
        </Drawer>
      </Box>

      {/* Main Content */}
      <Box
        component="main"
        sx={{
          flexGrow: 1,
          p: 3,
          width: { sm: `calc(100% - ${drawerWidth}px)` },
          backgroundColor: '#f8fafc',
          minHeight: '100vh'
        }}
      >
        <Toolbar />
        <Outlet />
      </Box>
    </Box>
  );
}

export default DashboardLayout;