import type { NextConfig } from "next";

const flaskOrigin = process.env.API_PROXY_TARGET ?? "http://localhost:5000";

const nextConfig: NextConfig = {
  async rewrites() {
    return [
      {
        source: "/api/:path*",
        destination: `${flaskOrigin}/api/:path*`,
      },
    ];
  },
};

export default nextConfig;
