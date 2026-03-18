/// <reference types="vitest" />
import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';
import path from 'path';

const srcRoot = path.resolve(__dirname, './src');
const muiRoot = path.resolve(__dirname, './node_modules/@mui');
const emotionRoot = path.resolve(__dirname, './node_modules/@emotion');

export default defineConfig({
  plugins: [react()],
  resolve: {
    alias: [
      { find: /^@mui\/material$/, replacement: path.join(muiRoot, 'material/index.js') },
      { find: /^@mui\/material\/(.*)$/, replacement: path.join(muiRoot, 'material/$1') },
      { find: /^@mui\/system$/, replacement: path.join(muiRoot, 'system/index.js') },
      { find: /^@mui\/system\/(.*)$/, replacement: path.join(muiRoot, 'system/$1') },
      { find: /^@mui\/styled-engine$/, replacement: path.join(muiRoot, 'styled-engine/index.js') },
      { find: /^@mui\/styled-engine\/(.*)$/, replacement: path.join(muiRoot, 'styled-engine/$1') },
      {
        find: /^@mui\/icons-material$/,
        replacement: path.join(muiRoot, 'icons-material/index.js'),
      },
      {
        find: /^@mui\/icons-material\/(.*)$/,
        replacement: path.join(muiRoot, 'icons-material/$1'),
      },
      {
        find: /^@emotion\/react$/,
        replacement: path.join(emotionRoot, 'react/dist/emotion-react.esm.js'),
      },
      {
        find: /^@emotion\/styled$/,
        replacement: path.join(emotionRoot, 'styled/dist/emotion-styled.esm.js'),
      },
      { find: /^@mui\/material\/node\//, replacement: '@mui/material/' },
      { find: /^@mui\/system\/node\//, replacement: '@mui/system/' },
      { find: /^@mui\/styled-engine\/node\//, replacement: '@mui/styled-engine/' },
      { find: /^@\//, replacement: `${srcRoot}/` },
      { find: /^@components\//, replacement: `${srcRoot}/components/` },
      { find: /^@pages\//, replacement: `${srcRoot}/pages/` },
      { find: /^@hooks\//, replacement: `${srcRoot}/hooks/` },
      { find: /^@services\//, replacement: `${srcRoot}/services/` },
      { find: /^@store\//, replacement: `${srcRoot}/store/` },
      { find: /^@utils\//, replacement: `${srcRoot}/utils/` },
      { find: /^@types\//, replacement: `${srcRoot}/types/` },
      { find: /^@styles\//, replacement: `${srcRoot}/styles/` },
    ],
    conditions: ['browser'],
  },
  server: {
    port: 5173,
    proxy: {
      '/api': {
        target: 'http://localhost:3000',
        changeOrigin: true,
        ws: true, // Enable WebSocket support for /api endpoints (including pipeline)
      },
    },
  },
  build: {
    outDir: 'dist',
    sourcemap: true,
    rollupOptions: {
      output: {
        manualChunks: {
          vendor: ['react', 'react-dom'],
          mui: ['@mui/material', '@mui/icons-material'],
          charts: ['recharts', '@mui/x-charts'],
          redux: ['@reduxjs/toolkit', 'react-redux'],
        },
      },
    },
  },
  optimizeDeps: {
    include: [
      '@mui/material',
      '@mui/icons-material',
      '@mui/system',
      '@emotion/react',
      '@emotion/styled',
    ],
    exclude: ['@mui/x-date-pickers'],
  },
  ssr: {
    noExternal: [
      '@mui/material',
      '@mui/icons-material',
      '@mui/system',
      '@mui/styled-engine',
      '@emotion/react',
      '@emotion/styled',
    ],
  },
  test: {
    globals: true,
    environment: 'jsdom',
    setupFiles: ['./src/test/setup.ts'],
    css: false,
    server: {
      deps: {
        inline: [
          /@mui\/material/,
          /@mui\/icons-material/,
          /@mui\/system/,
          /@mui\/styled-engine/,
          /@emotion\/react/,
          /@emotion\/styled/,
        ],
      },
    },
    coverage: {
      provider: 'v8',
      reporter: ['text', 'json', 'html', 'lcov'],
      exclude: [
        'node_modules/',
        'src/test/',
        '**/*.d.ts',
        '**/*.config.*',
        '**/index.ts',
        'src/main.tsx',
        'src/vite-env.d.ts',
      ],
      thresholds: {
        global: {
          branches: 80,
          functions: 80,
          lines: 80,
          statements: 80,
        },
      },
    },
    testTimeout: 10000,
    hookTimeout: 10000,
  },
});
