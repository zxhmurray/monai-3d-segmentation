/**
 * 文件上传组件
 */

import React, { useCallback } from 'react';
import { useDropzone } from 'react-dropzone';
import { Upload, X, File } from 'lucide-react';
import { useStore } from '../hooks/useStore';

export const UploadZone: React.FC = () => {
  const { uploadedFiles, addFile, removeFile, updateFile } = useStore();

  const onDrop = useCallback((acceptedFiles: File[]) => {
    acceptedFiles.forEach((file) => {
      const id = Math.random().toString(36).substring(7);
      addFile({
        id,
        name: file.name,
        size: file.size,
        status: 'pending',
      });
    });
  }, [addFile]);

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: {
      'application/nifti': ['.nii', '.nii.gz'],
    },
    multiple: true,
  });

  return (
    <div className="space-y-4">
      {/* 上传区域 */}
      <div
        {...getRootProps()}
        className={`
          border-2 border-dashed rounded-lg p-8 text-center cursor-pointer
          transition-colors duration-200
          ${isDragActive
            ? 'border-blue-500 bg-blue-50 dark:bg-blue-900/20'
            : 'border-gray-300 dark:border-gray-600 hover:border-blue-400'
          }
        `}
      >
        <input {...getInputProps()} />
        <Upload className="mx-auto h-12 w-12 text-gray-400" />
        <p className="mt-2 text-sm text-gray-600 dark:text-gray-400">
          拖拽 NIfTI 文件到这里，或点击选择
        </p>
        <p className="mt-1 text-xs text-gray-500">
          支持 .nii 和 .nii.gz 格式
        </p>
      </div>

      {/* 文件列表 */}
      {uploadedFiles.length > 0 && (
        <div className="space-y-2">
          <h3 className="text-sm font-medium text-gray-700 dark:text-gray-300">
            已选择的文件 ({uploadedFiles.length})
          </h3>
          <ul className="divide-y divide-gray-200 dark:divide-gray-700 rounded-lg border border-gray-200 dark:border-gray-700">
            {uploadedFiles.map((file) => (
              <li
                key={file.id}
                className="flex items-center justify-between p-3"
              >
                <div className="flex items-center space-x-3">
                  <File className="h-5 w-5 text-gray-400" />
                  <div>
                    <p className="text-sm font-medium text-gray-900 dark:text-gray-100">
                      {file.name}
                    </p>
                    <p className="text-xs text-gray-500">
                      {(file.size / 1024 / 1024).toFixed(2)} MB
                    </p>
                  </div>
                </div>
                <button
                  onClick={() => removeFile(file.id)}
                  className="text-gray-400 hover:text-red-500"
                >
                  <X className="h-5 w-5" />
                </button>
              </li>
            ))}
          </ul>
        </div>
      )}
    </div>
  );
};

export default UploadZone;
