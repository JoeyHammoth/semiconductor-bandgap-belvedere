import adapter from '@sveltejs/adapter-static';
import { sveltekit } from '@sveltejs/kit/vite';
import { defineConfig } from 'vite';

export default defineConfig({
  plugins: [
    sveltekit({
      adapter: adapter({
        pages: 'build',
        assets: 'build',
        fallback: '404.html',
        precompress: false,
        strict: true,
      }),
      paths: {
        base: (process.env.BASE_PATH ?? '') as '' | `/${string}`,
      },
      prerender: {
        handleHttpError: 'warn',
      },
    }),
  ],
  build: {
    target: 'es2022',
  },
  worker: {
    format: 'es',
  },
  define: {
    __COMMIT_HASH__: JSON.stringify(process.env.GITHUB_SHA ?? 'dev'),
    __BUILD_TIME__: JSON.stringify(new Date().toISOString()),
  },
});
