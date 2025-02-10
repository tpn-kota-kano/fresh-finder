'use client';

import { useState, useEffect } from 'react';
import { useRouter } from 'next/navigation';
import { useFreshFinder } from '../context';

export default function Page3() {
  const router = useRouter();
  const {
    capturedImage,
    productType,
    setProductType,
    desiredInfo,
    setDesiredInfo,
    setResultImage,
  } = useFreshFinder();
  const [error, setError] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [rateLimited, setRateLimited] = useState(false);

  useEffect(() => {
    if (!productType) {
      setProductType('vegetables');
    }
  }, [productType, setProductType]);

  const handleAnalysis = async () => {
    if (desiredInfo.trim() === '') {
      setError('テキストボックスに入力が必要です。');
      return;
    }

    setError('');
    setIsLoading(true);

    const formData = new FormData();
    if (capturedImage) {
      formData.append('image', capturedImage);
    }
    formData.append('productType', productType);
    formData.append('desiredInfo', desiredInfo);

    try {
      const API_BASE_URL = process.env.NEXT_PUBLIC_API_BASE_URL || 'http://localhost:8000';
      const res = await fetch(`${API_BASE_URL}/genai-image-analysis`, {
        method: 'POST',
        body: formData,
      });

      if (res.status === 429) {
        throw new Error('Too many requests. Please wait a moment before retrying.');
      }
      if (!res.ok) {
        throw new Error('解析に失敗しました。');
      }

      const data = await res.json();
      const resultBase64 = data?.resultImage || '';
      setResultImage(resultBase64);

      router.push('/page4');
    } catch (err: unknown) {
      if (err instanceof Error) {
        setError(err.message);
        if (err.message.includes('Too many requests')) {
          setRateLimited(true);
          setTimeout(() => setRateLimited(false), 60000);
        }
      } else {
        setError('予期せぬエラーが発生しました。');
      }
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <main className="flex flex-col w-full max-w-xs mx-auto px-4 py-8">
      <h2 className="text-lg font-semibold mb-4 text-gray-800 dark:text-gray-100">
        撮影した商品の種類を選択してください
      </h2>
      <div className="mb-4">
        <label className="flex items-center mb-2">
          <input
            type="radio"
            name="productType"
            value="vegetables"
            checked={productType === 'vegetables'}
            onChange={(e) => setProductType(e.target.value)}
            className="mr-2"
          />
          野菜
        </label>
        <label className="flex items-center">
          <input
            type="radio"
            name="productType"
            value="meat"
            checked={productType === 'meat'}
            onChange={(e) => setProductType(e.target.value)}
            className="mr-2"
          />
          肉
        </label>
      </div>

      <h2 className="text-lg font-semibold mb-2 text-gray-800 dark:text-gray-100">
        どんな商品を選びたいですか？
      </h2>
      <input
        type="text"
        placeholder="例：鮮度が高い"
        value={desiredInfo}
        onChange={(e) => setDesiredInfo(e.target.value)}
        className="border border-gray-300 rounded w-full p-2 mb-2 text-black"
      />
      <div className="min-h-6">{error && <p className="text-red-500 text-center">{error}</p>}</div>

      <button
        onClick={handleAnalysis}
        disabled={isLoading || rateLimited}
        className="w-full py-2 mt-4 bg-blue-600 text-white rounded hover:bg-blue-700 font-medium flex items-center justify-center"
      >
        {isLoading ? (
          <>
            <svg
              className="animate-spin h-5 w-5 mr-2 text-white"
              xmlns="http://www.w3.org/2000/svg"
              fill="none"
              viewBox="0 0 24 24"
            >
              <circle
                className="opacity-25"
                cx="12"
                cy="12"
                r="10"
                stroke="currentColor"
                strokeWidth="4"
              ></circle>
              <path
                className="opacity-75"
                fill="currentColor"
                d="M4 12a8 8 0 018-8v4a4 4 0 00-4 4H4z"
              ></path>
            </svg>
            解析中...
          </>
        ) : (
          '解析開始'
        )}
      </button>
    </main>
  );
}
