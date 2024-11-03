declare module 'remark-collapse';

/// <reference types="astro/client" />

declare module '*.md' {
    const content: any;
    export default content;
  }