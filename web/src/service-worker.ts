/// <reference lib="webworker" />
import { assets, immutable, prerendered } from '$app/manifest';
import { version } from '$app/env';

const worker = self as unknown as ServiceWorkerGlobalScope;
const cacheName = `bgo-${version}`;
const entries = [...immutable, ...assets, ...prerendered] as Array<string | { path?: string }>;
const precache = [
  ...new Set(
    entries
      .map((entry) => (typeof entry === 'string' ? entry : entry.path))
      .filter((path): path is string => Boolean(path))
      .filter((path) => !path.split('/').some((segment) => segment.startsWith('.'))),
  ),
];
const offlineFallback = new URL('404.html', worker.location.href).href;

worker.addEventListener('install', (event) => {
  event.waitUntil(
    caches
      .open(cacheName)
      .then((cache) => cache.addAll(precache))
      .then(() => worker.skipWaiting()),
  );
});

worker.addEventListener('activate', (event) => {
  event.waitUntil(
    caches
      .keys()
      .then((keys) =>
        Promise.all(
          keys
            .filter((key) => key.startsWith('bgo-') && key !== cacheName)
            .map((key) => caches.delete(key)),
        ),
      )
      .then(() => worker.clients.claim()),
  );
});

worker.addEventListener('fetch', (event) => {
  const request = event.request;
  if (request.method !== 'GET') return;
  const url = new URL(request.url);
  if (url.origin !== worker.location.origin) return;

  event.respondWith(
    caches.match(request, { ignoreVary: true }).then((cached) => {
      if (cached) return cached;
      return fetch(request)
        .then((response) => {
          if (response.ok) {
            const copy = response.clone();
            void caches.open(cacheName).then((cache) => cache.put(request, copy));
          }
          return response;
        })
        .catch(() =>
          caches.match(offlineFallback, { ignoreVary: true }).then((fallback) => fallback ?? Response.error()),
        );
    }),
  );
});
