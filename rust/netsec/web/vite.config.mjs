import {defineConfig} from 'vite';
export default defineConfig({build:{outDir:'dist',assetsDir:'static',target:'es2022',sourcemap:false},server:{host:'127.0.0.1'}});
