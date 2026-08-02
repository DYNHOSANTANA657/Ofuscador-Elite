import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "Ofuscador Elite",
  description: "Processamento local de vídeo e voz com controle de antifase.",
  icons: { icon: "/favicon.svg", shortcut: "/favicon.svg" },
  openGraph: {
    title: "Ofuscador Elite",
    description: "Processamento local de vídeo e voz com controle de antifase.",
    images: [{ url: "/og.png", width: 1536, height: 1024 }],
  },
};

export default function RootLayout({ children }: Readonly<{ children: React.ReactNode }>) {
  return (
    <html lang="pt-BR">
      <body>{children}</body>
    </html>
  );
}
