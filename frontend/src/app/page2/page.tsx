'use client';

import { useRouter } from 'next/navigation';
import { useFreshFinder } from '../context';
import { useRef } from 'react';

// 画像を横幅800pxに圧縮する関数
function compressImage(file: File, maxWidth = 800): Promise<File> {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = (event) => {
      const img = new Image();
      img.onload = () => {
        // 元画像の横幅が既に maxWidth 以下ならそのまま返す
        if (img.width <= maxWidth) {
          resolve(file);
          return;
        }
        const scale = maxWidth / img.width;
        const newWidth = maxWidth;
        const newHeight = img.height * scale;
        const canvas = document.createElement('canvas');
        canvas.width = newWidth;
        canvas.height = newHeight;
        const ctx = canvas.getContext('2d');
        if (ctx) {
          ctx.drawImage(img, 0, 0, newWidth, newHeight);
          canvas.toBlob((blob) => {
            if (blob) {
              const compressedFile = new File([blob], file.name, { type: file.type });
              resolve(compressedFile);
            } else {
              reject(new Error('圧縮処理に失敗しました。Blobが取得できませんでした。'));
            }
          }, file.type);
        } else {
          reject(new Error('Canvas のコンテキストが取得できませんでした。'));
        }
      };
      img.onerror = () => reject(new Error('画像の読み込みに失敗しました。'));
      img.src = event.target?.result as string;
    };
    reader.onerror = () => reject(new Error('FileReader のエラーが発生しました。'));
    reader.readAsDataURL(file);
  });
}

export default function Page2() {
  const { setCapturedImage } = useFreshFinder();
  const router = useRouter();
  const fileInputRef = useRef<HTMLInputElement | null>(null);

  const handleFileChange = async (e: React.ChangeEvent<HTMLInputElement>) => {
    if (!e.target.files || e.target.files.length === 0) return;
    const file = e.target.files[0];
    try {
      // 画像を圧縮して横幅800pxにリサイズ
      const compressedFile = await compressImage(file, 800);
      setCapturedImage(compressedFile);
    } catch (error) {
      console.error('画像圧縮に失敗しました:', error);
      // 圧縮に失敗した場合は元の画像を利用する
      setCapturedImage(file);
    }
    // 画像選択後すぐに Page3 へ遷移
    router.push('/page3');
  };

  return (
    <main className="flex flex-col items-center justify-center w-full px-4 py-8 text-center">
      <h2 className="text-xl font-medium mb-6 text-gray-800 dark:text-gray-100">
        おすすめを知りたい商品を撮影してください
      </h2>
      <input
        ref={fileInputRef}
        type="file"
        accept="image/*"
        capture="environment" // スマホでカメラが起動するよう指定
        className="hidden" // 画面上には表示しない
        onChange={handleFileChange}
      />
      <button
        onClick={() => fileInputRef.current?.click()}
        className="px-4 py-2 bg-blue-500 text-white rounded hover:bg-blue-600"
      >
        撮影する
      </button>
    </main>
  );
}
