import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "StreamRecall · Visual Memory Agent",
  description: "Ask the past without replaying it.",
};

export default function RootLayout({ children }: Readonly<{ children: React.ReactNode }>) {
  return (
    <html lang="zh-CN">
      <body>{children}</body>
    </html>
  );
}
