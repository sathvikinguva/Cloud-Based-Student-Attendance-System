import React, { useRef, useEffect, useState } from 'react';
import { Camera } from 'lucide-react';

interface CameraProps {
  onCapture: (imageData: string) => void;
  duration?: number;
}

export default function CameraComponent({ onCapture, duration = 20000 }: CameraProps) {
  const videoRef = useRef<HTMLVideoElement>(null);
  const [timeLeft, setTimeLeft] = useState(duration / 1000);
  const [isActive, setIsActive] = useState(false);

  useEffect(() => {
    if (isActive) {
      const timer = setInterval(() => {
        setTimeLeft((time) => {
          if (time <= 1) {
            setIsActive(false);
            stopCamera();
            clearInterval(timer);
          }
          return time - 1;
        });
      }, 1000);

      return () => clearInterval(timer);
    }
  }, [isActive]);

  const startCamera = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ video: true });
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
      }
      setIsActive(true);
    } catch (err) {
      console.error('Error accessing camera:', err);
    }
  };

  const stopCamera = () => {
    if (videoRef.current?.srcObject) {
      const tracks = (videoRef.current.srcObject as MediaStream).getTracks();
      tracks.forEach(track => track.stop());
      videoRef.current.srcObject = null;
    }
  };

  const captureImage = () => {
    if (videoRef.current) {
      const canvas = document.createElement('canvas');
      canvas.width = videoRef.current.videoWidth;
      canvas.height = videoRef.current.videoHeight;
      canvas.getContext('2d')?.drawImage(videoRef.current, 0, 0);
      const imageData = canvas.toDataURL('image/jpeg');
      onCapture(imageData);
      stopCamera();
      setIsActive(false);
    }
  };

  return (
    <div className="relative">
      <video
        ref={videoRef}
        autoPlay
        playsInline
        className="w-full rounded-lg shadow-lg"
      />
      {!isActive ? (
        <button
          onClick={startCamera}
          className="mt-4 flex items-center justify-center gap-2 bg-blue-500 text-white py-2 px-4 rounded-md hover:bg-blue-600"
        >
          <Camera size={20} />
          Start Camera
        </button>
      ) : (
        <div className="mt-4 space-y-4">
          <div className="text-center text-lg font-semibold">
            Time remaining: {timeLeft}s
          </div>
          <button
            onClick={captureImage}
            className="w-full bg-green-500 text-white py-2 px-4 rounded-md hover:bg-green-600"
          >
            Capture
          </button>
        </div>
      )}
    </div>
  );
}