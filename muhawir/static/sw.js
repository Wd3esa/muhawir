// مُحاور as an installed app. Questions and answers always go to the server and are never stored here;
// only the page itself and its fonts and icons are kept, so the app opens when the connection is poor.
const SHELL = "muhawir-shell-v1";
const FILES = ["/", "/manifest.webmanifest", "/icons/icon-192.png", "/fonts/NotoSansArabic-Regular.ttf",
  "/fonts/NotoSansArabic-Bold.ttf"];

self.addEventListener("install", (event) => {
  event.waitUntil(caches.open(SHELL).then((cache) => cache.addAll(FILES)).then(() => self.skipWaiting()));
});

self.addEventListener("activate", (event) => {
  event.waitUntil(caches.keys()
    .then((keys) => Promise.all(keys.filter((key) => key !== SHELL).map((key) => caches.delete(key))))
    .then(() => self.clients.claim()));
});

self.addEventListener("fetch", (event) => {
  const url = new URL(event.request.url);
  if (event.request.method !== "GET" || url.origin !== self.location.origin) return;
  if (url.pathname.startsWith("/api/") || url.pathname === "/tts") return;  // never cached
  // the newest page first; the kept copy only when the network fails
  event.respondWith(fetch(event.request).then((response) => {
    if (response.ok && FILES.includes(url.pathname === "/" ? "/" : url.pathname)) {
      const copy = response.clone();
      caches.open(SHELL).then((cache) => cache.put(url.pathname === "/" ? "/" : event.request, copy));
    }
    return response;
  }).catch(() => caches.match(url.pathname === "/" || event.request.mode === "navigate" ? "/" : event.request)));
});
