'use client';

import { createContext, useContext, useState } from 'react';

type FreshFinderContextType = {
  capturedImage: File | null; // 2ページ目で撮影 or 選択された画像
  setCapturedImage: (file: File | null) => void;
  productType: string; // 3ページ目で選択する種類 ("野菜" or "肉" など)
  setProductType: (value: string) => void;
  desiredInfo: string; // 3ページ目で入力する「どんな商品を選びたいか」
  setDesiredInfo: (value: string) => void;
  resultImage: string; // 解析結果（4ページ目で表示する画像, base64 など）
  setResultImage: (value: string) => void;
};

const FreshFinderContext = createContext<FreshFinderContextType>({
  capturedImage: null,
  setCapturedImage: () => {},
  productType: '',
  setProductType: () => {},
  desiredInfo: '',
  setDesiredInfo: () => {},
  resultImage: '',
  setResultImage: () => {},
});

export const useFreshFinder = () => useContext(FreshFinderContext);

export function FreshFinderProvider({ children }: { children: React.ReactNode }) {
  const [capturedImage, setCapturedImage] = useState<File | null>(null);
  const [productType, setProductType] = useState('');
  const [desiredInfo, setDesiredInfo] = useState('');
  const [resultImage, setResultImage] = useState('');

  return (
    <FreshFinderContext.Provider
      value={{
        capturedImage,
        setCapturedImage,
        productType,
        setProductType,
        desiredInfo,
        setDesiredInfo,
        resultImage,
        setResultImage,
      }}
    >
      {children}
    </FreshFinderContext.Provider>
  );
}
