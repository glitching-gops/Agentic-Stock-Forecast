import path from "node:path";

import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  /*
   * The Next.js app lives in `web/` inside a Python repo. Without this,
   * Turbopack walks up past the repo root looking for a lockfile and picks up
   * unrelated ones from the home directory.
   */
  turbopack: { root: path.join(__dirname) },

  // Fail the build on a type error rather than shipping one.
  typescript: { ignoreBuildErrors: false },
};

export default nextConfig;
