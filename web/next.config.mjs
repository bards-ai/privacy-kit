/** @type {import('next').NextConfig} */
const nextConfig = {
  // Self-contained server bundle for a slim Docker runtime image.
  output: "standalone",
  reactStrictMode: true,
  // The frontend ships no ESLint config of its own; type-checking still runs.
  eslint: { ignoreDuringBuilds: true },
};

export default nextConfig;
