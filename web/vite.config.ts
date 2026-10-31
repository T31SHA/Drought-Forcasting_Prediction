import { defineConfig } from "vite";
import data from "./src/data/forecast.json";
import { prerender } from "./src/static";
import type { Payload } from "./src/types";

export default defineConfig({
  plugins: [
    {
      name: "prerender-forecast",
      transformIndexHtml: (html) => prerender(html, data as unknown as Payload),
    },
  ],
  build: { target: "es2022", cssMinify: true, reportCompressedSize: true },
});
