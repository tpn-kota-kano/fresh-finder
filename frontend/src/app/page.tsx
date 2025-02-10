'use client';

import { useRouter } from 'next/navigation';

export default function Page1() {
  const router = useRouter();

  const handleStart = () => {
    router.push('/page2');
  };

  return (
    <main className="flex flex-col items-center justify-center w-full px-4 py-8">
      <h1 className="text-3xl font-bold mb-8 text-gray-800 dark:text-gray-100">FreshFinder</h1>
      <button
        onClick={handleStart}
        className="px-6 py-3 bg-blue-600 text-white rounded hover:bg-blue-700 font-medium"
      >
        はじめる
      </button>
    </main>
  );
}
