'use client';

import { useRouter } from 'next/navigation';
import Image from 'next/image';
import { useFreshFinder } from '../context';

export default function Page4() {
  const router = useRouter();
  const { resultImage } = useFreshFinder();

  const handleFinish = () => {
    router.push('/');
  };

  return (
    <main className="flex flex-col items-center w-full px-4 py-8">
      <h2 className="text-xl font-bold mb-4 text-gray-800 dark:text-gray-100">解析結果</h2>
      {resultImage ? (
        <div className="mb-6">
          <Image
            src={resultImage}
            alt="解析結果画像"
            width={400}
            height={400}
            className="object-contain border border-gray-300 rounded"
          />
        </div>
      ) : (
        <p className="text-gray-700 dark:text-gray-200 mb-6">結果画像がありません</p>
      )}
      <button
        onClick={handleFinish}
        className="px-6 py-3 bg-blue-600 text-white rounded hover:bg-blue-700 font-medium"
      >
        完了
      </button>
    </main>
  );
}
