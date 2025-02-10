import type { Metadata } from 'next';
import { Geist, Geist_Mono } from 'next/font/google';
import './globals.css';
import { FreshFinderProvider } from './context';

const geistSans = Geist({
  variable: '--font-geist-sans',
  subsets: ['latin'],
});

const geistMono = Geist_Mono({
  variable: '--font-geist-mono',
  subsets: ['latin'],
});

export const metadata: Metadata = {
  title: 'FreshFinder',
  description: 'Find the freshest food in your area',
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <body className={`${geistSans.variable} ${geistMono.variable} antialiased`}>
        <FreshFinderProvider>
          <div className="min-h-screen flex flex-col items-center bg-white dark:bg-gray-900">
            {children}
          </div>
        </FreshFinderProvider>
      </body>
    </html>
  );
}
